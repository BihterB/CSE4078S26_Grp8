import os
import re

import pandas as pd
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# This file is for calculating simple scores.
# We already generated answers with run_baseline_eval.py.
# Now we compare model_answer and reference_answer.
# Not perfect metric but enough for first progress part.

INPUT_FILES = [
    "results/qwen_1_5b_baseline_outputs_100.csv",
    "results/gemma_2_2b_it_baseline_outputs_100.csv",
]

# Summary table will be saved here.
OUTPUT_FILE = "results/baseline_summary.csv"


def normalize_text(text: str) -> str:
    """
    Small text cleaning before score calculation.
    """
    if pd.isna(text):
        return ""

    # lower case yapıyorum çünkü büyük küçük harf skoru bozmasın
    text = str(text).lower()

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def compute_rouge_l(reference: str, prediction: str, scorer) -> float:
    """
    ROUGE-L gives overlap score between real answer and model answer.
    """
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)

    # If model gives empty answer, score is 0.
    if not reference or not prediction:
        return 0.0

    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def compute_tfidf_similarity(reference: str, prediction: str) -> float:
    """
    Simple similarity score with TF-IDF.
    This is not deep semantic metric, but it gives extra comparison.
    """
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)

    if not reference or not prediction:
        return 0.0

    # burada sadece iki text icin vector olusturuyorum
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([reference, prediction])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return float(similarity)


def evaluate_file(input_path: str) -> dict:
    """
    Reads one CSV file and returns average scores for that model.
    """
    df = pd.read_csv(input_path)

    # scorer object for ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    rouge_scores = []
    similarity_scores = []

    # Go row by row.
    # Each row has one question, one reference answer and one model answer.
    for _, row in df.iterrows():
        reference = row["reference_answer"]
        prediction = row["model_answer"]

        rouge_l = compute_rouge_l(reference, prediction, scorer)
        similarity = compute_tfidf_similarity(reference, prediction)

        rouge_scores.append(rouge_l)
        similarity_scores.append(similarity)

    model_name = df["model_name"].iloc[0]
    example_count = len(df)

    # If there is an error message in a row, count it.
    error_count = df["error"].fillna("").astype(str).str.len().gt(0).sum()

    # Average scores for this model.
    # yuvarlama yaptim tablo daha temiz dursun diye
    return {
        "model_name": model_name,
        "examples": example_count,
        "avg_rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4),
        "avg_tfidf_similarity": round(sum(similarity_scores) / len(similarity_scores), 4),
        "error_count": int(error_count),
        "input_file": input_path,
    }


def main():
    summaries = []

    # Evaluate both model output files.
    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            print(f"Skipping missing file: {input_file}")
            continue

        print(f"Evaluating: {input_file}")
        summary = evaluate_file(input_file)
        summaries.append(summary)

    # If no file is found, stop.
    if not summaries:
        print("No valid input files found.")
        return

    os.makedirs("results", exist_ok=True)

    # Make one summary table for presentation/report.
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\nBaseline summary:")
    print(summary_df)

    print(f"\nSaved summary file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()