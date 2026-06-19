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
| Qwen2.5-1.5B (baseline) | 0.2609 | 0.1271 | 0.2126 | 0.0688 | 0.2810 | 0.1648 | 0.8667 |
| Llama-3.2-3B (baseline) | 0.2896 | 0.1597 | 0.2506 | 0.1002 | 0.2802 | 0.1979 | 0.8740 |
| Llama-3.2-3B (fine-tuned) | 0.6186 | 0.5163 | 0.5874 | 0.4720 | 0.6038 | 0.5419 | 0.9369 |

Evaluated on 1500 examples from the test split of `Renicames/turkish-law-chatbot`.

## How to Run

**Install dependencies:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Baseline evaluation (1500 examples):**
```bash
python scripts/run_baseline_eval.py --model_name Qwen/Qwen2.5-1.5B-Instruct --model_short_name qwen_1_5b --num_examples 1500
python scripts/run_baseline_eval.py --model_name meta-llama/Llama-3.2-3B-Instruct --model_short_name llama_3_2_3b --num_examples 1500
```

**Fine-tuning (QLoRA, 5 epochs):**
```bash
python scripts/finetune_qlora.py --model_name meta-llama/Llama-3.2-3B-Instruct --output_dir models/llama_3_2_3b_finetuned --num_epochs 5
```

**Evaluate fine-tuned model (1500 examples):**
```bash
python scripts/eval_finetuned.py --base_model_name meta-llama/Llama-3.2-3B-Instruct --finetuned_path models/llama_3_2_3b_finetuned --model_short_name llama_3_2_3b_finetuned --num_examples 1500
```

**Compute all metrics:**
```bash
python scripts/compute_metrics.py \
  --input_files results/qwen_1_5b_baseline_outputs_1500.csv results/llama_3_2_3b_baseline_outputs_1500.csv results/llama_3_2_3b_finetuned_outputs_1500.csv \
  --output_file results/summary_1500.csv
```

**Create presentation and error-analysis helper tables:**
```bash
python scripts/make_result_table.py
python scripts/error_analysis.py
```

**Run Llama baseline + fine-tuned eval in sequence:**
```bash
python run_llama_evals.py
```

## Reports

LaTeX source files are in `reports/`. Compile with any LaTeX distribution (e.g. MiKTeX, Overleaf):

- `reports/CSE4078S26_Grp8_FinalReport.tex` - IEEE-format final report
- `reports/CSE4078S26_Grp8_DatasetLinks.tex` - dataset links document

The compiled presentation is at `reports/CSE4078S26_Grp8_FinalPresentation.pptx`.

## Important Rule

The test split is only used for evaluation. It is never used during training or fine-tuning.
