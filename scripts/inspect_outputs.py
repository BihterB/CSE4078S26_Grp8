import pandas as pd

# This helper script prints a few model outputs.
# I used it for quickly comparing question, reference answer and model answer.

FILES = [
    "results/qwen_1_5b_baseline_outputs_25.csv",
    "results/gemma_2_2b_it_baseline_outputs_25.csv",
]


def print_examples(file_path, n=3):
    print("=" * 80)
    print("File:", file_path)
    print("=" * 80)

    df = pd.read_csv(file_path)

    for i in range(min(n, len(df))):
        row = df.iloc[i]

        print("\nExample", i + 1)
        print("-" * 80)
        print("Question:")
        print(row["question"])

        print("\nReference answer:")
        print(row["reference_answer"])

        print("\nModel answer:")
        print(row["model_answer"])
        print("-" * 80)


def main():
    for file_path in FILES:
        try:
            print_examples(file_path)
        except FileNotFoundError:
            print("File not found:", file_path)


if __name__ == "__main__":
    main()