from pathlib import Path
import pandas as pd

# Project root = parent of scripts/
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "dataset" / "ClusterWise_mini" / "olcf_task" / "test_anomaly.parquet"
OUTPUT_FILE = BASE_DIR / "dataset" / "ClusterWise_mini" / "olcf_task" / "test_anomaly.csv"

print(f"Reading: {INPUT_FILE}")

df = pd.read_parquet(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

df.to_csv(OUTPUT_FILE, index=False)

print(f"CSV created successfully:")
print(OUTPUT_FILE)