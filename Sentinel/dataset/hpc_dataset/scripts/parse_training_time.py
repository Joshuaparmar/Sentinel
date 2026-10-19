import glob
import re
import json

MODELS = [
    "Autoformer", "Crossformer", "DLinear", "ETSformer", "FEDformer", "FiLM", "Informer", "Mamba",
    "Nonstationary_Transformer", "PatchTST", "Pyraformer", "Reformer", "TimesNet", "TimeXer", "Transformer", "TSMixer",
]

pattern_training_time = re.compile(r"training time:\s*([0-9.]+)h")
pattern_pred_len = re.compile(r"Evaluating .* with pred_len 1")

results = {}

for model in MODELS:
    log_files = glob.glob(f"logs/olcf_eval_*/*{model}*_eval.log")
    best_time = None
    for lf in log_files:
        with open(lf, "r", errors="ignore") as f:
            lines = f.readlines()
        inside_pred1 = False
        for line in lines:
            if not inside_pred1:
                if pattern_pred_len.search(line):
                    # Only consider lines until next '------ Evaluating' or end
                    inside_pred1 = True
                continue
            # If we encounter another Evaluating block, break
            if line.startswith("------ Evaluating") and "pred_len" in line:
                break
            m = pattern_training_time.search(line)
            if m:
                t_val = float(m.group(1))
                best_time = t_val
                break  # Found training time within pred_len 1 block
        if best_time is not None:
            break  # prefer first successful log
    results[model] = best_time

print(json.dumps(results, indent=2)) 