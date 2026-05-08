from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DATASET_NAME = "Renicames/turkish-law-chatbot"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def build_prompt(question):
    return f"""Aşağıdaki Türkçe hukuk sorusunu kısa, açık ve doğru şekilde cevapla.

Soru:
{question}

Cevap:"""


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    test_data = dataset["test"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None
    )

    model.eval()

    print("\nRunning test on 3 examples...\n")

    for i in range(3):
        question = test_data[i]["Soru"]
        reference_answer = test_data[i]["Cevap"]

        prompt = build_prompt(question)

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt part from the output if possible
        model_answer = full_output.replace(prompt, "").strip()

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