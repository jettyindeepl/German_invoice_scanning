from __future__ import annotations

import os
import pickle

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv3Processor

from layout_approach import LayoutLMv3MCDropoutForNER, ner_ce_loss, collate_fn, mc_predict_ner

from bio_tagging import process_dataset
from datasets import load_dataset



class InvoiceQADataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class InvoiceTrainer:
    def __init__(
        self,
        model_name="microsoft/layoutlmv3-base",
        batch_size=4,
        lr=2e-5,
        head_lr=1e-3,
        dropout_p=0.1,
        entity_weight=20.0,
        max_length=512,
        stride=64,
        checkpoint_dir="checkpoints",
        train_cache_path="data/processed_samples.pkl",
        val_cache_path="data/processed_samples_val.pkl",
        device=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.head_lr = head_lr
        self.dropout_p = dropout_p
        self.entity_weight = entity_weight
        self.max_length = max_length
        self.stride = stride
        self.checkpoint_dir = checkpoint_dir
        self.train_cache_path = train_cache_path
        self.val_cache_path = val_cache_path
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )

        self.processor = None
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.n_train = 0
        self.n_val = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        print(f"device={self.device}  dropout_p={self.dropout_p}  stride={self.stride}  entity_weight={self.entity_weight}")

    def setup(self):
        self.processor = LayoutLMv3Processor.from_pretrained(self.model_name, apply_ocr=False)
        self.model = LayoutLMv3MCDropoutForNER(self.model_name, self.dropout_p).to(self.device)

    def _load_or_cache(self, dataset_split, cache_path):
        if os.path.exists(cache_path):
            print(f"Loading cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        df = dataset_split.to_pandas()
        samples = process_dataset(df)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"Cached {len(samples)} samples to {cache_path}")
        return samples

    def load_data(self, train_split, val_split):
        assert self.processor is not None, "Call setup() before load_data()."

        train_samples = [s for s in self._load_or_cache(train_split, self.train_cache_path) if s["found"]]
        val_samples   = [s for s in self._load_or_cache(val_split,   self.val_cache_path)   if s["found"]]

        self.n_train = len(train_samples)
        self.n_val   = len(val_samples)
        print(f"train={self.n_train}  val={self.n_val}")

        _collate = lambda batch: collate_fn(batch, processor=self.processor, max_length=self.max_length)

        self.train_loader = DataLoader(InvoiceQADataset(train_samples), batch_size=self.batch_size, shuffle=True,  collate_fn=_collate)
        self.val_loader   = DataLoader(InvoiceQADataset(val_samples),   batch_size=self.batch_size, shuffle=False, collate_fn=_collate)

    def _run_epoch(self, loader, train):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, correct, total, n_batches = 0.0, 0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bbox           = batch["bbox"].to(self.device)
                pixel_values   = batch["pixel_values"].to(self.device)
                labels         = batch["labels"].to(self.device)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, pixel_values=pixel_values)

                # truncate to text tokens only (model output includes image patch tokens)
                L = labels.shape[1]
                loss = ner_ce_loss(logits[:, :L], labels, self.entity_weight)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

                # F1
                preds = logits[:, :L].argmax(-1)
                for b in range(labels.shape[0]):
                    mask = labels[b] != -100
                    gold = labels[b][mask] > 0
                    if gold.sum() == 0:
                        continue
                    pred = preds[b][mask] > 0
                    tp = (pred & gold).sum().item()
                    denom = gold.sum().item() + pred.sum().item()
                    correct += (2 * tp / denom) if denom > 0 else 0.0
                    total   += 1

        return (total_loss / n_batches if n_batches else 0.0,
                correct / total        if total     else 0.0)

    def train(self, num_epochs=10):

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.optimizer is None:
            backbone_params = [p for p in self.model.layoutlmv3.parameters() if p.requires_grad]
            head_params = list(self.model.classifier.parameters())
            self.optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": self.lr},
                {"params": head_params,     "lr": self.head_lr},
            ])

        for epoch in range(1, num_epochs + 1):
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)

            ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(self.model.state_dict(), ckpt_path)
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch:>3}/{num_epochs} | train loss={tr_loss:.4f} f1={tr_acc:.4f} | val loss={val_loss:.4f} f1={val_acc:.4f}")

        print("Done.")

    def mc_uncertainty(self, batch, n_samples=20):
        return mc_predict_ner(self.model, batch, n_samples=n_samples, device=self.device)

    def predict(self, test_split, ckpt_path=None, n_samples=20):
        if ckpt_path:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            print(f"Loaded checkpoint: {ckpt_path}")
        samples = process_dataset(test_split.to_pandas())
        _collate = lambda batch: collate_fn(batch, processor=self.processor, max_length=self.max_length)
        loader = DataLoader(InvoiceQADataset(samples), batch_size=self.batch_size, shuffle=False, collate_fn=_collate)

        results = []
        for batch in loader:
            pred_labels, mean_probs, entropy = mc_predict_ner(
                self.model, batch, n_samples=n_samples, device=self.device
            )
            labels = batch["labels"]
            L = labels.shape[1]  # text tokens only (max_length)
            for b in range(pred_labels.shape[0]):
                mask = labels[b] != -100
                preds = pred_labels[b, :L][mask]
                probs = mean_probs[b, :L][mask]
                ent   = entropy[b, :L][mask]

                entity_mask = preds > 0
                if entity_mask.any():
                    mean_entropy = ent[entity_mask].mean().item()
                else:
                    mean_entropy = ent.mean().item()

                results.append({
                    "pred_labels": preds.cpu().tolist(),
                    "entropy":     round(mean_entropy, 4),
                })
        return results


    def plot(self, save_path=None):

        epochs = list(range(1, len(self.history["train_loss"]) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("MC-Dropout NER — Training Curves", fontsize=14, fontweight="bold")

        ax = axes[0]
        ax.plot(epochs, self.history["train_loss"], "b-o", label="Train")
        ax.plot(epochs, self.history["val_loss"],   "r-o", label="Val")
        ax.set(xlabel="Epoch", ylabel="Loss", title="Loss")
        ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.5); ax.set_xticks(epochs)

        ax = axes[1]
        ax.plot(epochs, [a * 100 for a in self.history["train_acc"]], "b-o", label="Train")
        ax.plot(epochs, [a * 100 for a in self.history["val_acc"]],   "r-o", label="Val")
        ax.set(xlabel="Epoch", ylabel="Entity Token F1 (%)", title="Token F1")
        ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.5); ax.set_xticks(epochs)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150)
        plt.show()


def main():
    ds = load_dataset("Aoschu/donut_model_data_for_german_invoice")

    trainer = InvoiceTrainer(dropout_p=0.1, entity_weight=20.0, head_lr=1e-3)
    trainer.setup()
    trainer.load_data(ds["train"], ds["validation"])
    trainer.train(num_epochs=10)
    trainer.plot(save_path="results/mc_dropout_curves.png")

    infer = InvoiceTrainer()
    infer.setup()
    results = infer.predict(ds["test"], ckpt_path="checkpoints/checkpoint_epoch_8.pt", n_samples=20)
    print(f"\nPredictions on {len(results)} test samples:")
    for i, r in enumerate(results):
        print(f"  [{i:>4}] entropy={r['entropy']:.4f}  labels={r['pred_labels']}")


if __name__ == "__main__":
    main()

