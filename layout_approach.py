from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LayoutLMv3Model, LayoutLMv3Processor


class LayoutLMv3MCDropoutForNER(nn.Module):
    # O=0, B=1, I=2

    def __init__(self, model_name="microsoft/layoutlmv3-base", dropout_p=0.1):
        super().__init__()
        self.layoutlmv3 = LayoutLMv3Model.from_pretrained(model_name)
        hidden_size = self.layoutlmv3.config.hidden_size

        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, 3)

        # freeze backbone, keep last encoder layer trainable
        self.layoutlmv3.requires_grad_(False)
        self.layoutlmv3.encoder.layer[-1].requires_grad_(True)

    def forward(self, input_ids, attention_mask, bbox, pixel_values):
        out = self.layoutlmv3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )
        return self.classifier(self.dropout(out.last_hidden_state))


def ner_ce_loss(logits, labels, entity_weight=5.0):
    B, L, C = logits.shape
    weights = torch.ones(C, device=logits.device)
    weights[1:] = entity_weight
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)
    return loss_fn(logits.view(B * L, C), labels.view(B * L))


def build_token_labels(word_ids, word_bio):
    labels = []
    seen = set()
    for wid in word_ids:
        if wid is None or wid in seen:
            labels.append(-100)
        else:
            seen.add(wid)
            labels.append(word_bio[wid] if wid < len(word_bio) else -100)
    return labels


def collate_fn(batch, processor, max_length=512):
    encodings, all_labels = [], []

    for sample in batch:
        n_words = len(sample["words"])
        word_bio = [0] * n_words
        if sample["found"] and sample["start_idx"] is not None:
            s, e = int(sample["start_idx"]), int(sample["end_idx"])
            for i in range(s, min(e + 1, n_words)):
                word_bio[i] = 1 if i == s else 2

        enc = processor(
            sample["image"],
            sample["question"],
            sample["words"],
            boxes=sample["boxes"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        encodings.append({k: enc[k] for k in enc.keys()})
        all_labels.append(build_token_labels(enc.word_ids(batch_index=0), word_bio))

    batch_out = {
        k: torch.cat([e[k] for e in encodings], dim=0) for k in encodings[0]
    }
    max_len = max(len(l) for l in all_labels)
    padded = [l + [-100] * (max_len - len(l)) for l in all_labels]
    batch_out["labels"] = torch.tensor(padded, dtype=torch.long)
    return batch_out


def mc_predict_ner(model, batch, n_samples=20, device="cpu"):
    model.train()
    all_probs = []

    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                bbox=batch["bbox"].to(device),
                pixel_values=batch["pixel_values"].to(device),
            )
            all_probs.append(torch.softmax(logits, dim=-1))

    mean_probs = torch.stack(all_probs).mean(0)
    entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(-1)
    return mean_probs.argmax(-1), mean_probs, entropy


