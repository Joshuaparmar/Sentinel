import os
import subprocess
import time
import json
from pathlib import Path

MODELS = [
    "Autoformer", "Crossformer", "DLinear", "ETSformer", "FEDformer", "FiLM", "Informer", "Mamba",
    "Nonstationary_Transformer", "PatchTST", "Pyraformer", "Reformer", "TimesNet", "TimeXer", "Transformer", "TSMixer",
]

# Environment shared by all calls (mirrors settings in run_all_olcf_eval.sh)
BASE_ENV = {
    "TASK_NAME": "long_term_forecast",
    "IS_TRAINING": "0",            # Inference-only
    "ROOT_PATH": "./data/olcf_task/",
    "DATA_PATH": "train.csv",
    "DATASET": "olcf",
    "FEATURES": "M",
    "SEQ_LEN": "60",
    "LABEL_LEN": "30",
    "ENC_IN": "28",
    "DEC_IN": "28",
    "C_OUT": "28",
    "DES": "InferTime",
    "ITR": "1",
    "PRED_LENS_STR": "1",          # single pred_len
}

results = {}

for model_name in MODELS:
    script_path = f"scripts/long_term_forecast/olcf_script/{model_name}.sh"
    if not Path(script_path).exists():
        results[model_name] = {
            "wall_h": None,
            "per_record_ms": None,
        }
        continue

    env = os.environ.copy()
    env.update(BASE_ENV)

    print(f"[INFO] Measuring inference time for {model_name}…", flush=True)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            ["bash", script_path], env=env, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        elapsed = time.perf_counter() - start

        # Attempt to extract the number of test samples from stdout (line like: 'test 714000')
        test_count = None
        for line in completed.stdout.splitlines():
            if line.lower().startswith("test "):
                try:
                    test_count = int(line.split()[1])
                except ValueError:
                    pass
        per_rec_ms = None
        if test_count and test_count > 0:
            per_rec_ms = round((elapsed / test_count) * 1000, 6)  # milliseconds

        results[model_name] = {
            "wall_h": round(elapsed / 3600, 3),
            "per_record_ms": per_rec_ms,
        }

        print(f"    → wall {results[model_name]['wall_h']} h  | test_count {test_count} | {results[model_name]['per_record_ms']} ms/record", flush=True)

    except subprocess.CalledProcessError as e:
        print(f"    ! Script failed — skipping", flush=True)
        results[model_name] = {
            "wall_h": None,
            "per_record_ms": None,
        }

# Persist raw numbers so that combine_results.py can consume them
out_file = Path("scripts/infer_time.json")
out_file.write_text(json.dumps(results, indent=2))
print(f"\n[OK] Inference-time statistics saved to {out_file}") 