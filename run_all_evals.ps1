$venv = "C:\Users\bihte\OneDrive\Desktop\CSE4078S26_Grp8\venv\Scripts\python.exe"
$wd   = "C:\Users\bihte\OneDrive\Desktop\CSE4078S26_Grp8"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$qwenCSV = "$wd\results\qwen_1_5b_baseline_outputs_1500.csv"

Write-Host "Waiting for Qwen to finish (target: 1500 rows)..."
while ($true) {
    if (Test-Path $qwenCSV) {
        $count = (Import-Csv $qwenCSV).Count
        Write-Host "  $(Get-Date -Format 'HH:mm') -- $count / 1500 rows"
        if ($count -ge 1500) { break }
    }
    Start-Sleep -Seconds 120
}
Write-Host "Qwen done."

Write-Host "=== Starting Llama baseline eval ==="
& $venv "$wd\scripts\run_baseline_eval.py" --model_name "meta-llama/Llama-3.2-3B-Instruct" --model_short_name "llama_3_2_3b" --num_examples 1500 2>&1 | Tee-Object -FilePath "$wd\results\llama_baseline_log.txt"

Write-Host "=== Starting Llama fine-tuned eval ==="
& $venv "$wd\scripts\eval_finetuned.py" --base_model_name "meta-llama/Llama-3.2-3B-Instruct" --finetuned_path "$wd\models\llama_3_2_3b_finetuned" --model_short_name "llama_3_2_3b_finetuned" --num_examples 1500 2>&1 | Tee-Object -FilePath "$wd\results\llama_finetuned_log.txt"

Write-Host "=== All evals complete ==="
