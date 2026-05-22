"""
Evaluates a fine-tuned LoRA model on the test split.
Produces the same CSV format as run_baseline_eval.py so compute_metrics.py works on both.

Usage:
  python scripts/eval_finetuned.py \
    --base_model_name meta-llama/Llama-3.2-3B-Instruct \
    --finetuned_path models/finetuned \
    --model_short_name llama_3_2_3b_finetuned \
    --num_examples 100
"""

import argparse
import os
import time

import bitsandbytes  # must be imported before torch on Windows to avoid CUDA crash
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "Renicames/turkish-law-chatbot"
SYSTEM_PROMPT = "Aşağıdaki Türkçe hukuk sorusunu kısa, açık ve doğru şekilde cevapla."


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 100) -> str:
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt).
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def save_results(results, output_path):
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--finetuned_path", type=str, required=True)
    parser.add_argument("--model_short_name", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    output_path = f"results/{args.model_short_name}_outputs_{args.num_examples}.csv"

    # --- Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load base model + LoRA adapter ---
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        dtype=torch.float16,
        device_map="auto",
    )

    print("Applying LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, args.finetuned_path)
    model.eval()

    # --- Dataset ---
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    test_data = dataset["test"]
    print(f"Using {args.num_examples} test examples.")

    results = []
    start_time = time.time()

    try:
        for i in tqdm(range(args.num_examples)):
            question = test_data[i]["Soru"]
            reference_answer = test_data[i]["Cevap"]
            try:
                answer = generate_answer(model, tokenizer, question, args.max_new_tokens)
                error = ""
            except Exception as e:
                answer = ""
                error = str(e)

            results.append({
                "index": i,
                "model_name": f"{args.base_model_name}+LoRA",
                "question": question,
                "reference_answer": reference_answer,
                "model_answer": answer,
                "error": error,
            })
            save_results(results, output_path)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
        save_results(results, output_path)
        return

    elapsed = time.time() - start_time
    save_results(results, output_path)
    print(f"\nDone. {args.num_examples} examples in {elapsed:.1f}s")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
