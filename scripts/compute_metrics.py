import argparse
import os
import re

import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# This script takes model output CSV files (from run_baseline_eval.py or eval_finetuned.py)
# and computes 7 metrics for each model. Results go into a summary CSV.
# Run it like: python scripts/compute_metrics.py --input_files results/a.csv results/b.csv --output_file results/summary.csv

DEFAULT_INPUT_FILES = [
    "results/qwen_1_5b_baseline_outputs_100.csv",
    "results/llama_3_2_3b_baseline_outputs_100.csv",
]

DEFAULT_OUTPUT_FILE = "results/baseline_summary.csv"


def normalize_text(text: str) -> str:
    # lowercase + extra boşlukları temizle, skorları tutarlı yapmak için
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_tfidf_similarity(reference: str, prediction: str) -> float:
    # TF-IDF vectors for two texts, then cosine similarity between them
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)
    if not reference or not prediction:
        return 0.0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([reference, prediction])
    return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])


def compute_bleu(references: list, predictions: list) -> float:
    # BLEU with effective_order=True so short outputs don't get unfairly penalized
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(predictions, [references])
    return round(result.score / 100, 4)  # sacrebleu returns 0-100, we normalize to 0-1


def compute_chrf(references: list, predictions: list) -> float:
    # chrF++ — karakter + kelime sırası birlikte bakıyor
    # Türkçe gibi eklemeli diller için BLEU'dan daha güvenilir
    chrf = CHRF(word_order=2)
    result = chrf.corpus_score(predictions, [references])
    return round(result.score / 100, 4)


def compute_bertscore(references: list, predictions: list) -> float:
    # xlm-roberta-base: 100+ dil destekliyor, Türkçe dahil
    # F1 skorunu alıyoruz — precision ve recall'ın harmonik ortalaması
    _, _, F1 = bert_score(predictions, references, lang="tr", model_type="xlm-roberta-base", verbose=False)
    return round(F1.mean().item(), 4)


def evaluate_file(input_path: str) -> dict:
    df = pd.read_csv(input_path)

    # rouge1: unigram overlap, rouge2: bigram, rougeL: longest common subsequence
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    tfidf_scores = []
    references = []
    predictions = []

    for _, row in df.iterrows():
        ref = row["reference_answer"]
        pred = row["model_answer"]
        ref_norm = normalize_text(ref)
        pred_norm = normalize_text(pred) if normalize_text(pred) else " "  # boş cevap varsa skor 0 olsun

        if ref_norm and pred_norm:
            s = scorer.score(ref_norm, pred_norm)
            rouge1_scores.append(s["rouge1"].fmeasure)
            rouge2_scores.append(s["rouge2"].fmeasure)
            rougel_scores.append(s["rougeL"].fmeasure)
        else:
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougel_scores.append(0.0)

        tfidf_scores.append(compute_tfidf_similarity(ref, pred))
        references.append(ref_norm)
        predictions.append(pred_norm)

    model_name = df["model_name"].iloc[0]
    # error column has the exception message if generation failed, empty string if ok
    error_count = df["error"].fillna("").astype(str).str.len().gt(0).sum()

    n = len(rougel_scores)
    print(f"  Computing BLEU + chrF...")
    avg_bleu = compute_bleu(references, predictions)
    avg_chrf = compute_chrf(references, predictions)

    print(f"  Computing BERTScore...")
    avg_bertscore = compute_bertscore(references, predictions)

    return {
        "model_name": model_name,
        "examples": len(df),
        "avg_rouge1": round(sum(rouge1_scores) / n, 4),
        "avg_rouge2": round(sum(rouge2_scores) / n, 4),
        "avg_rouge_l": round(sum(rougel_scores) / n, 4),
        "avg_bleu": avg_bleu,
        "avg_chrf": avg_chrf,
        "avg_tfidf_similarity": round(sum(tfidf_scores) / n, 4),
        "avg_bertscore_f1": avg_bertscore,
        "error_count": int(error_count),
        "input_file": input_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", default=DEFAULT_INPUT_FILES)
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    summaries = []

    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Skipping missing file: {input_file}")
            continue

        print(f"Evaluating: {input_file}")
        summary = evaluate_file(input_file)
        summaries.append(summary)

    if not summaries:
        print("No valid input files found.")
        return

    os.makedirs("results", exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_file, index=False, encoding="utf-8")

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {args.output_file}")


if __name__ == "__main__":
    main()