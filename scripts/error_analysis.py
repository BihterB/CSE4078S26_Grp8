import argparse
import os

import pandas as pd


DEFAULT_FILES = {
    "qwen_baseline": "results/qwen_1_5b_baseline_outputs_1500.csv",
    "llama_baseline": "results/llama_3_2_3b_baseline_outputs_1500.csv",
    "llama_finetuned": "results/llama_3_2_3b_finetuned_outputs_1500.csv",
}

DEFAULT_OUTPUT_FILE = "results/error_analysis_examples.csv"


def load_outputs(label: str, path: str) -> pd.DataFrame:
    # load one model output csv and check required columns.
    # if file missing stop early, no silent wrong table.
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file not found: {path}")

    df = pd.read_csv(path)
    required = {"index", "question", "reference_answer", "model_answer", "error"}
    missing = required - set(df.columns)
    if missing:
        # columns wrong means probably old csv, buraya geri don bak.
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    return df[["index", "question", "reference_answer", "model_answer", "error"]].rename(
        columns={
            "model_answer": f"{label}_answer",
            "error": f"{label}_error",
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a side-by-side CSV for qualitative error analysis."
    )
    parser.add_argument("--qwen_file", default=DEFAULT_FILES["qwen_baseline"])
    parser.add_argument("--llama_file", default=DEFAULT_FILES["llama_baseline"])
    parser.add_argument("--finetuned_file", default=DEFAULT_FILES["llama_finetuned"])
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--num_examples", type=int, default=30)
    args = parser.parse_args()

    qwen = load_outputs("qwen_baseline", args.qwen_file)
    llama = load_outputs("llama_baseline", args.llama_file)
    finetuned = load_outputs("llama_finetuned", args.finetuned_file)

    # merge all model answers by same test index.
    # same question should stay side by side.
    merged = qwen.merge(
        llama[["index", "llama_baseline_answer", "llama_baseline_error"]],
        on="index",
        how="inner",
    ).merge(
        finetuned[["index", "llama_finetuned_answer", "llama_finetuned_error"]],
        on="index",
        how="inner",
    )

    # fixed evenly spaced examples, not random.
    # hocaya example verirken tekrar ayni rows gelsin.
    if len(merged) > args.num_examples:
        step = max(len(merged) // args.num_examples, 1)
        selected = merged.iloc[::step].head(args.num_examples).copy()
    else:
        selected = merged.copy()

    selected["notes"] = ""
    # notes column is for manual comments about error pattern.
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    selected.to_csv(args.output_file, index=False, encoding="utf-8")

    print(f"Saved {len(selected)} examples to {args.output_file}")


if __name__ == "__main__":
    main()
