'''
# 1. parameter statistics (already done once)
python scripts/param_count.py  > scripts/param_data.json

# 2. training-time statistics (already done once)
python scripts/parse_training_time.py > scripts/train_data.json

# 3. inference-time statistics  ––  this step may take a while
python scripts/measure_inference_time.py        # saves infer_time.json

# 4. final consolidated report
python scripts/combine_results.py
'''

import json
from pathlib import Path
import pandas as pd

param_data = json.loads(Path('scripts/param_data.json').read_text()) if Path('scripts/param_data.json').exists() else None
train_data = json.loads(Path('scripts/train_data.json').read_text()) if Path('scripts/train_data.json').exists() else None
infer_data = json.loads(Path('scripts/infer_time.json').read_text()) if Path('scripts/infer_time.json').exists() else {}

if param_data is None or train_data is None:
    print('Missing intermediate data files (param_data.json or train_data.json). Run the helper scripts first.')
    exit()

models = sorted(param_data.keys())
rows = []
for m in models:
    row = {
        'Model': m,
        'Params': param_data[m]['param_count'],
        'Size(MB)': param_data[m]['size_mb'],
        'Training Time(h)': train_data.get(m),
        # 'Inference Time(h)': infer_data.get(m, {}).get('wall_h') if isinstance(infer_data.get(m), dict) else None,
        'Per-record Time(ms)': infer_data.get(m, {}).get('per_record_ms') if isinstance(infer_data.get(m), dict) else None,
    }
    rows.append(row)

print(pd.DataFrame(rows).to_markdown()) 