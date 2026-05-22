# CSE4078S26_Grp8

CSE4078 Spring 2026 Term Project

## Project Topic

Evaluation and fine-tuning of small open-source Large Language Models for Turkish legal question answering.

## Group

Group 8

## Mandatory Dataset

Since Group 8 is an even-numbered group, we use:

- Renicames/turkish-law-chatbot

## Baseline Models

- Qwen/Qwen2.5-1.5B-Instruct
- meta-llama/Llama-3.2-3B-Instruct

> We initially evaluated google/gemma-2-2b-it but switched to Llama-3.2-3B-Instruct based on instructor feedback.

## Fine-tuned Model

We fine-tuned **Llama-3.2-3B-Instruct** using QLoRA (4-bit quantization, LoRA r=16) on the full training split of the SFT corpus (13,354 examples, 5 epochs).

Fine-tuned adapters are saved under `models/llama_3_2_3b_finetuned/`.

## Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | chrF | TF-IDF | BERTScore F1 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B (baseline) | 0.2299 | 0.1092 | 0.1956 | 0.0770 | 0.2472 | 0.1359 | 0.8548 |
| Llama-3.2-3B (baseline) | 0.2789 | 0.1515 | 0.2392 | 0.1039 | 0.3159 | 0.1969 | 0.8707 |
| Llama-3.2-3B (fine-tuned) | 0.6285 | 0.5233 | 0.5951 | 0.5433 | 0.6591 | 0.5408 | 0.9389 |

Evaluated on 100 examples from the test split of `Renicames/turkish-law-chatbot`.

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Baseline evaluation:**
```bash
python scripts/run_baseline_eval.py --model_name meta-llama/Llama-3.2-3B-Instruct --model_short_name llama_3_2_3b --num_examples 100
```

**Fine-tuning (QLoRA):**
```bash
python scripts/finetune_qlora.py --model_name meta-llama/Llama-3.2-3B-Instruct --output_dir models/llama_3_2_3b_finetuned --num_epochs 5
```

**Evaluate fine-tuned model:**
```bash
python scripts/eval_finetuned.py --base_model_name meta-llama/Llama-3.2-3B-Instruct --finetuned_path models/llama_3_2_3b_finetuned --model_short_name llama_3_2_3b_finetuned
```

**Compute all metrics:**
```bash
python scripts/compute_metrics.py --input_files results/qwen_1_5b_baseline_outputs_100.csv results/llama_3_2_3b_baseline_outputs_100.csv results/llama_3_2_3b_finetuned_outputs_100.csv --output_file results/full_summary.csv
```

## Important Rule

The test split is only used for evaluation. It is never used during training or fine-tuning.