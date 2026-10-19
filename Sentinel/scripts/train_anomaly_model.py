import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "ClusterWise_mini" / "olcf_task"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

TRAIN_FILE = DATA_DIR / "train.parquet"
VAL_FILE = DATA_DIR / "val.parquet"
TEST_FILE = DATA_DIR / "test.parquet"
ANOMALY_FILE = DATA_DIR / "test_anomaly.parquet"


# Telemetry features used for the first SENTINEL model
BASE_FEATURES = [
    "p0_power",
    "p1_power",

    "gpu0_core_temp",
    "gpu0_mem_temp",
    "gpu1_core_temp",
    "gpu1_mem_temp",
    "gpu2_core_temp",
    "gpu2_mem_temp",
    "gpu3_core_temp",
    "gpu3_mem_temp",
    "gpu4_core_temp",
    "gpu4_mem_temp",
    "gpu5_core_temp",
    "gpu5_mem_temp",

    "ps0_input_power",
    "ps1_input_power",

    "p0_temp_mean",
    "p0_temp_min",
    "p0_temp_max",

    "p1_temp_mean",
    "p1_temp_min",
    "p1_temp_max",
]


# ================================================================
# TEMPORAL FEATURE ENGINEERING
# ================================================================

def create_temporal_features(df):
    """
    Create time-series features independently for each hostname.

    Features include:
        - Current telemetry values
        - First-order changes
        - Rolling mean
        - Rolling standard deviation
    """

    print("Creating temporal features...")

    df = df.copy()

    # ------------------------------------------------------------
    # Sort chronologically within each node
    # ------------------------------------------------------------

    df = df.sort_values(
        ["hostname", "timestamp"]
    ).reset_index(drop=True)

    feature_columns = []

    # ------------------------------------------------------------
    # Current-state features
    # ------------------------------------------------------------

    for col in BASE_FEATURES:
        feature_columns.append(col)

    # ------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------

    for col in BASE_FEATURES:

        # Change from previous 10-second observation
        delta_col = f"{col}_delta"

        df[delta_col] = (
            df.groupby("hostname")[col]
            .diff()
        )

        feature_columns.append(delta_col)

        # Rolling mean over previous observations
        rolling_mean_col = f"{col}_rolling_mean"

        df[rolling_mean_col] = (
            df.groupby("hostname")[col]
            .transform(
                lambda x: x.rolling(
                    window=6,
                    min_periods=1
                ).mean()
            )
        )

        feature_columns.append(rolling_mean_col)

        # Rolling standard deviation
        rolling_std_col = f"{col}_rolling_std"

        df[rolling_std_col] = (
            df.groupby("hostname")[col]
            .transform(
                lambda x: x.rolling(
                    window=6,
                    min_periods=1
                ).std()
            )
        )

        feature_columns.append(rolling_std_col)

    # ------------------------------------------------------------
    # Replace infinite values
    # ------------------------------------------------------------

    df[feature_columns] = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    # ------------------------------------------------------------
    # Fill temporal NaNs
    # ------------------------------------------------------------

    df[feature_columns] = (
        df.groupby("hostname")[feature_columns]
        .transform(lambda x: x.bfill().ffill())
    )

    # Any remaining NaNs
    df[feature_columns] = (
        df[feature_columns].fillna(0)
    )

    print(
        f"Created {len(feature_columns)} ML features."
    )

    return df, feature_columns


# ================================================================
# LOAD DATA
# ================================================================

def load_dataset(path):
    print(f"\nLoading: {path.name}")

    columns = [
        "timestamp",
        "hostname",
        *BASE_FEATURES,
        "label",
        "xid",
    ]

    df = pd.read_parquet(
        path,
        columns=columns
    )

    print(f"Rows loaded: {len(df):,}")

    return df


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("SENTINEL — TEMPORAL ANOMALY DETECTION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------

    train_df = load_dataset(TRAIN_FILE)

    # ------------------------------------------------------------
    # Create temporal features
    # ------------------------------------------------------------

    train_df, feature_columns = create_temporal_features(
        train_df
    )

    X_train = train_df[feature_columns]

    print("\nTraining feature matrix:")
    print(X_train.shape)

    # ------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------

    print("\nFitting feature scaler...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # ------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------

    print("\nTraining Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_scaled
    )

    print("Model training complete.")

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    model_path = MODEL_DIR / "sentinel_isolation_forest.joblib"
    scaler_path = MODEL_DIR / "sentinel_scaler.joblib"
    features_path = MODEL_DIR / "sentinel_features.joblib"

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    joblib.dump(
        feature_columns,
        features_path
    )

    print("\nSaved:")
    print(model_path)
    print(scaler_path)
    print(features_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()