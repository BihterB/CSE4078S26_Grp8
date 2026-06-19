"""Runs Llama baseline then fine-tuned eval sequentially."""
import subprocess
import sys
import os
import time

wd = os.path.dirname(os.path.abspath(__file__))
python = sys.executable

steps = [
    {
        "name": "Llama baseline (1500)",
        "cmd": [
            python, "scripts/run_baseline_eval.py",
            "--model_name", "meta-llama/Llama-3.2-3B-Instruct",
            "--model_short_name", "llama_3_2_3b",
            "--num_examples", "1500",
        ],
    },
    {
        "name": "Llama fine-tuned (1500)",
        "cmd": [
            python, "scripts/eval_finetuned.py",
            "--base_model_name", "meta-llama/Llama-3.2-3B-Instruct",
            "--finetuned_path", os.path.join(wd, "models", "llama_3_2_3b_finetuned"),
            "--model_short_name", "llama_3_2_3b_finetuned",
            "--num_examples", "1500",
        ],
    },
]

for step in steps:
    print(f"\n{'='*60}")
    print(f"Starting: {step['name']}")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    result = subprocess.run(step["cmd"], cwd=wd)

    if result.returncode != 0:
        print(f"\nERROR: {step['name']} exited with code {result.returncode}")
        print("Stopping chain.")
        sys.exit(result.returncode)

    print(f"\nFinished: {step['name']} at {time.strftime('%H:%M:%S')}")
    # Short pause to let GPU memory fully release before next model load
    time.sleep(30)

print("\nAll Llama evals complete.")
