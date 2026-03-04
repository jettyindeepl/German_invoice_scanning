# clf

1. Create a conda env with python 3.9.6

2. pip install -r requirements.txt

3. Tesseract also needs to be installed at the OS level

4. Download parquet files from hugging face and put them under data/ folder.

Note:

Model training is commented out in invoice_trainer.py and
inference on test data is enabled.
The output looks like [sample index] entropy = float labels = [512 nos (0,1,2)]

