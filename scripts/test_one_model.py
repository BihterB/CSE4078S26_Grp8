from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Small test script.
# I used this before main baseline script, just to see if model loads or not.

DATASET_NAME = "Renicames/turkish-law-chatbot"

# First model for quick local test.
# Qwen is not too big, so it was easier to try first.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def build_prompt(question):
    """
    Simple prompt for Turkish legal question.
    """
    return f"""Aşağıdaki Türkçe hukuk sorusunu kısa, açık ve doğru şekilde cevapla.

Soru:
{question}

Cevap:"""


def main():
    # Load dataset.
    # burada sadece test split ile deneme yapiyorum, train kullanmiyorum.
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    test_data = dataset["test"]

    # Load tokenizer.
    # Tokenizer text'i modelin anlayacagi sayilara ceviriyor.
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load model.
    # This is only inference, no fine-tuning here.
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None
    )

    # eval mode because we are only testing the model.
    model.eval()

    print("\nRunning test on 3 examples...\n")

    # Just 3 examples, because this file is only for quick check.
    # Real test is in run_baseline_eval.py
    for i in range(3):
        # Dataset columns:
        # Soru = question
        # Cevap = reference answer
        question = test_data[i]["Soru"]
        reference_answer = test_data[i]["Cevap"]

        prompt = build_prompt(question)

        # Convert prompt to tokens.
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate answer.
        # do_sample=False makes output more stable.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode model output.
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean prompt part if model repeats it.
        # kendime not: bazen model soruyu da cevabin icine tekrar koyuyor.
        model_answer = full_output.replace(prompt, "").strip()

        # Print all parts so I can compare quickly.
        print("=" * 80)
        print(f"Example {i + 1}")
        print("-" * 80)
        print("QUESTION:")
        print(question)
        print("\nREFERENCE ANSWER:")
        print(reference_answer)
        print("\nMODEL ANSWER:")
        print(model_answer)
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()