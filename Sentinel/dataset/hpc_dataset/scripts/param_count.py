import os
import glob
import json
import torch

MODELS = [
    "Autoformer", "Crossformer", "DLinear", "ETSformer", "FEDformer", "FiLM", "Informer", "Mamba",
    "Nonstationary_Transformer", "PatchTST", "Pyraformer", "Reformer", "TimesNet", "TimeXer", "Transformer", "TSMixer",
]

RESULTS = {}
for model_name in MODELS:
    pattern = f"checkpoints/*_60_1_{model_name}_*/*.*"
    candidates = glob.glob(pattern)
    ckpt_path = None
    for file_path in candidates:
        if file_path.endswith((".pth", ".ckpt")):
            ckpt_path = file_path
            break
    if ckpt_path is None:
        RESULTS[model_name] = {
            "checkpoint": None,
            "param_count": None,
            "size_mb": None,
        }
        continue

    state = torch.load(ckpt_path, map_location="cpu")
    # Some checkpoints store state dict under the 'model' key
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format for {ckpt_path}")

    param_count = sum(p.numel() for p in state.values())
    size_mb = os.path.getsize(ckpt_path) / 1024 ** 2

    RESULTS[model_name] = {
        "checkpoint": ckpt_path,
        "param_count": int(param_count),
        "size_mb": size_mb,
    }

print(json.dumps(RESULTS, indent=2))