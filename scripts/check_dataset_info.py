from datasets import load_dataset

# Small helper file for checking dataset info.
# It prints split names, row counts and column names.

DATASET_NAME = "Renicames/turkish-law-chatbot"


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded.")
    print(dataset)

    print("\nSplits:")
    print(list(dataset.keys()))

    train_data = dataset["train"]
    test_data = dataset["test"]

    print("\nTrain size:", len(train_data))
    print("Test size:", len(test_data))

    print("\nTrain columns:")
    print(train_data.column_names)

    print("\nTest columns:")
    print(test_data.column_names)

    # quick check, just to see data format
    print("\nFirst test question:")
    print(test_data[0]["Soru"])

    print("\nFirst test answer:")
    print(test_data[0]["Cevap"])


if __name__ == "__main__":
    main()
