from datasets import load_dataset
import pandas as pd
import os

# This helper script creates a small sample from the test split.
# It is useful for quickly looking at some questions and answers.

DATASET_NAME = "Renicames/turkish-law-chatbot"
OUTPUT_FILE = "data/sample_test_examples.csv"


def main():
    print("Loading dataset...")

    dataset = load_dataset(DATASET_NAME)
    test_data = dataset["test"]

    rows = []

    # take first 10 examples only
    for i in range(10):
        rows.append(
            {
                "index": i,
                "question": test_data[i]["Soru"],
                "reference_answer": test_data[i]["Cevap"],
            }
        )

    os.makedirs("data", exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Sample file saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
