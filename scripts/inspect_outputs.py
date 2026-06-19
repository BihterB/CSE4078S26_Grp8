import pandas as pd

# This helper prints few model outputs to terminal.
# quick look only, serious metric degil just inspect.

FILES = [
    "results/qwen_1_5b_baseline_outputs_1500.csv",
    "results/llama_3_2_3b_baseline_outputs_1500.csv",
    "results/llama_3_2_3b_finetuned_outputs_1500.csv",
]


def print_examples(file_path, n=3):
    # print first n examples from one csv for easy compare.
    print("=" * 80)
    print("File:", file_path)
    print("=" * 80)

    df = pd.read_csv(file_path)

    for i in range(min(n, len(df))):
        # row has question reference and model answer.
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
            # if output not generated yet just tell and continue.
            print("File not found:", file_path)


if __name__ == "__main__":
    main()
