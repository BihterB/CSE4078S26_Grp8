from datasets import load_dataset
import pandas as pd
import os

# This script is for preparing our project dataset.
# It downloads the mandatory Turkish legal QA dataset and checks its structure.

# Mandatory dataset for Group 8.
# Group 8 is an even-numbered group, so we use Renicames/turkish-law-chatbot.
DATASET_NAME = "Renicames/turkish-law-chatbot"


def main():
    # First, we load the dataset from Hugging Face.
    # If it was downloaded before, Hugging Face can use the cached version.
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully.")
    print(dataset)

    # Create data folder if it does not exist.
    # The CSV files will be saved inside this folder.
    os.makedirs("data", exist_ok=True)

    # Print available splits.
    # For this dataset, we expect train and test splits.
    print("\nAvailable splits:")
    print(dataset.keys())

    # Convert train and test splits to pandas DataFrames.
    # This makes it easier to inspect and save them as CSV files.
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Check column names.
    # In this dataset, "Soru" is the question and "Cevap" is the reference answer.
    print("\nTrain columns:")
    print(train_df.columns.tolist())

    print("\nTest columns:")
    print(test_df.columns.tolist())

    # Print dataset sizes.
    # Train split will be used later for fine-tuning.
    # Test split is only used for baseline and final evaluation.
    print("\nTrain size:", len(train_df))
    print("Test size:", len(test_df))

    # Show first examples from test split.
    # kendime not: burada test datasi dogru mu diye hizli bakiyorum.
    print("\nFirst 3 test examples:")
    print(test_df.head(3))

    # Save local CSV copies.
    # These files are ignored by .gitignore, because full dataset files should not be pushed to GitHub.
    # We keep them locally only for easier inspection.
    train_df.to_csv("data/train_split.csv", index=False, encoding="utf-8")
    test_df.to_csv("data/test_split.csv", index=False, encoding="utf-8")

    # Save a small subset for quick testing.
    # This is useful when we do not want to run models on all 1500 test examples.
    small_test_df = test_df.head(50)
    small_test_df.to_csv("data/test_subset_50.csv", index=False, encoding="utf-8")

    print("\nSaved files:")
    print("data/train_split.csv")
    print("data/test_split.csv")
    print("data/test_subset_50.csv")


if __name__ == "__main__":
    main()