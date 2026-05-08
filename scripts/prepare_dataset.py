from datasets import load_dataset
import pandas as pd
import os

# Mandatory dataset for Group 8.
# Group 8 is an even-numbered group, so we use Renicames/turkish-law-chatbot.
DATASET_NAME = "Renicames/turkish-law-chatbot"


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully.")
    print(dataset)

    # Create data folder if it does not exist
    os.makedirs("data", exist_ok=True)

    # Print available splits
    print("\nAvailable splits:")
    print(dataset.keys())

    # Convert train and test splits to pandas DataFrames
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    print("\nTrain columns:")
    print(train_df.columns.tolist())

    print("\nTest columns:")
    print(test_df.columns.tolist())

    print("\nTrain size:", len(train_df))
    print("Test size:", len(test_df))

    print("\nFirst 3 test examples:")
    print(test_df.head(3))

    # Save local CSV copies
    # These files are ignored by .gitignore, because full dataset files should not be pushed to GitHub.
    train_df.to_csv("data/train_split.csv", index=False, encoding="utf-8")
    test_df.to_csv("data/test_split.csv", index=False, encoding="utf-8")

    # Save a small subset for quick testing
    small_test_df = test_df.head(50)
    small_test_df.to_csv("data/test_subset_50.csv", index=False, encoding="utf-8")

    print("\nSaved files:")
    print("data/train_split.csv")
    print("data/test_split.csv")
    print("data/test_subset_50.csv")


if __name__ == "__main__":
    main()