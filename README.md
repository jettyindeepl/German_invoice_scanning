# clf

1. Create a conda env with python 3.9.6

2. pip install -r requirements.txt

3. Tesseract also needs to be installed at the OS level

4. Download parquet files from hugging face and put them under data/ folder.

Note:

When main function in invoice_trainer.py is executed, the model trains and validates
and then gives a prediction in terminal in a format described below.

The output looks like [sample index] entropy = float labels = [512 nos (0,1,2)]

