from pathlib import Path
import pyarrow.parquet as pq

DATA_DIR = Path("../dataset/ClusterWise_mini/olcf_task")

files = [
    "train.parquet",
    "val.parquet",
    "test.parquet",
    "test_anomaly.parquet",
    "train_cluster.parquet",
    "val_cluster.parquet",
    "test_cluster.parquet",
    "train_mmd.parquet",
    "val_mmd.parquet",
    "test_mmd.parquet",
]

for filename in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    parquet_file = pq.ParquetFile(path)

    print("Rows:", parquet_file.metadata.num_rows)
    print("Row groups:", parquet_file.metadata.num_row_groups)
    print("Columns:", parquet_file.metadata.num_columns)

    print("\nSchema:")
    print(parquet_file.schema_arrow)

import pyarrow.parquet as pq

path = DATA_DIR / "train.parquet"

table = pq.read_table(
    path,
    columns=[
        "timestamp",
        "hostname",
        "node_state",
        "label",
        "xid",
    ]
)

df = table.to_pandas()

print("\n" + "=" * 70)
print("MAIN TRAIN DATA PROFILE")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False).sort_index())

print("\nXID distribution:")
print(df["xid"].value_counts(dropna=False).sort_index())

print("\nUnique hostnames:")
print(df["hostname"].nunique())

print("\nTop hostnames:")
print(df["hostname"].value_counts().head(20))

print("\nNode states:")
print(df["node_state"].value_counts(dropna=False))

print("\nTimestamp range:")
print("Minimum:", df["timestamp"].min())
print("Maximum:", df["timestamp"].max())

print("\nMissing values:")
print(df.isna().sum())

# ================================================================
# LABEL / XID PROFILE FOR OTHER DATA SPLITS
# ================================================================

files_to_profile = [
    "val.parquet",
    "test.parquet",
    "test_anomaly.parquet",
]

for filename in files_to_profile:

    path = DATA_DIR / filename

    table = pq.read_table(
        path,
        columns=[
            "timestamp",
            "hostname",
            "label",
            "xid",
        ]
    )

    df = table.to_pandas()

    print("\n" + "=" * 70)
    print(f"{filename.upper()} PROFILE")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False).sort_index())

    print("\nXID distribution:")
    print(df["xid"].value_counts(dropna=False).sort_index())

    print("\nUnique hostnames:")
    print(df["hostname"].nunique())

    print("\nTimestamp range:")
    print("Minimum:", df["timestamp"].min())
    print("Maximum:", df["timestamp"].max())

    print("\nMissing values:")
    print(df.isna().sum())