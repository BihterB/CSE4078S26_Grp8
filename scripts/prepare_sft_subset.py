from datasets import load_dataset
import json
import os

# This script prepares small SFT subset from train split.
# SFT means supervised fine-tuning, yani modele soru-cevap ornegi gostermek.

DATASET_NAME = "Renicames/turkish-law-chatbot"
OUTPUT_FILE = "data/sft_subset_500.jsonl"
SUBSET_SIZE = 500


def build_prompt(question: str) -> str:
    """
    Simple prompt format for SFT data.
    """
    return f"""Aşağıdaki Türkçe hukuk sorusunu kısa, açık ve doğru şekilde cevapla.

Soru:
{question}

Cevap:"""


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    # For SFT, we use train split.
    # Test split should stay only for evaluation.
    train_data = dataset["train"]

    print("Train size:", len(train_data))
    print("Preparing SFT subset size:", SUBSET_SIZE)

    os.makedirs("data", exist_ok=True)

    count = min(SUBSET_SIZE, len(train_data))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in range(count):
            question = train_data[i]["Soru"]
            answer = train_data[i]["Cevap"]

            item = {
                "index": i,
                "prompt": build_prompt(question),
                "response": answer,
            }

            # jsonl formatinda her satir bir training example oluyor
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("SFT subset saved:")
    print(OUTPUT_FILE)

    print("\nFirst example preview:")
    print("Prompt:")
    print(build_prompt(train_data[0]["Soru"]))
    print("Response:")
    print(train_data[0]["Cevap"])


if __name__ == "__main__":
    main()