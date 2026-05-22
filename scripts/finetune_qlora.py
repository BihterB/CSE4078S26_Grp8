import argparse
import os
import bitsandbytes  # must be first on Windows
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DATASET_NAME = "Renicames/turkish-law-chatbot"
SYSTEM_PROMPT = "Aşağıdaki Türkçe hukuk sorusunu kısa, açık ve doğru şekilde cevapla."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model, e.g. meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="models/finetuned")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output_dir")
    args = parser.parse_args()

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    print(f"Model : {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs} | Batch: {args.batch_size} | Grad accum: {args.grad_accum}")
    print(f"LoRA  : r={args.lora_r}, alpha={args.lora_alpha}")

    # --- Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- QLoRA quantization config (4-bit) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # --- Load base model in 4-bit ---
    print("Loading model in 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # --- LoRA config ---
    # target_modules covers attention + MLP for Llama / Qwen / Phi architectures.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_NAME)
    train_data = dataset["train"]
    print(f"Train examples: {len(train_data)}")

    def format_example(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["Soru"]},
            {"role": "assistant", "content": example["Cevap"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    print("Formatting dataset...")
    train_dataset = train_data.map(
        format_example,
        remove_columns=train_data.column_names,
        desc="Formatting",
    )

    # --- Training config ---
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # --- Train ---
    print(f"\nStarting training ({len(train_dataset)} examples, {args.num_epochs} epochs)...")
    trainer.train(resume_from_checkpoint=args.resume or None)

    # --- Save ---
    print("\nSaving fine-tuned LoRA adapters...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
