import pandas as pd
import os
# Reads final metric summary and make smaller table for slides.
# sunum icin, summary_1500 degisirse tekrar run et.
INPUT_FILE = "results/summary_1500.csv"
OUTPUT_FILE = "results/presentation_result_table.csv"


def main():
    if not os.path.exists(INPUT_FILE):
        print("Summary file not found:", INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE)

    # choose only columns needed in presentation table.
    # too many columns makes slide crowded.
    table = df[
        [
            "model_name",
            "examples",
            "avg_rouge1",
            "avg_rouge2",
            "avg_rouge_l",
            "avg_bleu",
            "avg_chrf",
            "avg_tfidf_similarity",
            "avg_bertscore_f1",
            "error_count",
        ]
    ].copy()

    table = table.rename(
        columns={
            "model_name": "Model",
            "examples": "Examples",
            "avg_rouge1": "ROUGE-1",
            "avg_rouge2": "ROUGE-2",
            "avg_rouge_l": "ROUGE-L",
            "avg_bleu": "BLEU",
            "avg_chrf": "chrF",
            "avg_tfidf_similarity": "TF-IDF Similarity",
            "avg_bertscore_f1": "BERTScore F1",
            "error_count": "Errors",
        }
    )

    # write clean csv, PowerPoint script can read this later.
    table.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Presentation table:")
    print(table)

    print("\nSaved file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
