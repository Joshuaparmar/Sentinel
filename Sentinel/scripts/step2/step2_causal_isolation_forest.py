import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "ClusterWise_mini" / "olcf_task"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

TRAIN_FILE = DATA_DIR / "train.parquet"
VAL_FILE = DATA_DIR / "val.parquet"
TEST_FILE = DATA_DIR / "test.parquet"
ANOMALY_FILE = DATA_DIR / "test_anomaly.parquet"


# Telemetry features used by SENTINEL
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


# Six 10-second samples = 60 seconds
ROLLING_WINDOW = 6


# ================================================================
# CAUSAL TEMPORAL FEATURE ENGINEERING
# ================================================================

def create_causal_temporal_features(df):
    """
    Create strictly causal temporal features.

    For every telemetry feature:

        1. Current value
        2. Delta from previous observation
        3. Rolling mean of PREVIOUS observations
        4. Rolling std of PREVIOUS observations

    IMPORTANT:
        The current observation is NOT included in rolling statistics.

    This makes the feature pipeline suitable for online / early-warning
    evaluation without future-looking leakage.
    """

    print("Creating CAUSAL temporal features...")

    df = df.copy()

    # ------------------------------------------------------------
    # Sort chronologically within each host
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

        grouped = df.groupby("hostname")[col]

        # --------------------------------------------------------
        # Previous-observation delta
        # --------------------------------------------------------

        delta_col = f"{col}_delta"

        df[delta_col] = grouped.diff()

        feature_columns.append(delta_col)

        # --------------------------------------------------------
        # CAUSAL rolling mean
        #
        # shift(1) means:
        #
        # feature at time t uses:
        # t-1, t-2, ..., t-6
        #
        # and NOT t itself.
        # --------------------------------------------------------

        rolling_mean_col = f"{col}_rolling_mean"

        df[rolling_mean_col] = (
            grouped
            .shift(1)
            .groupby(df["hostname"])
            .transform(
                lambda x: x.rolling(
                    window=ROLLING_WINDOW,
                    min_periods=1
                ).mean()
            )
        )

        feature_columns.append(rolling_mean_col)

        # --------------------------------------------------------
        # CAUSAL rolling standard deviation
        # --------------------------------------------------------

        rolling_std_col = f"{col}_rolling_std"

        df[rolling_std_col] = (
            grouped
            .shift(1)
            .groupby(df["hostname"])
            .transform(
                lambda x: x.rolling(
                    window=ROLLING_WINDOW,
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
    # IMPORTANT:
    #
    # DO NOT use bfill().
    #
    # bfill() could use future observations.
    #
    # Instead, use only past information.
    # ------------------------------------------------------------

    # Forward-fill values that become available later.
    df[feature_columns] = (
        df.groupby("hostname")[feature_columns]
        .transform(lambda x: x.ffill())
    )

    # ------------------------------------------------------------
    # Remaining NaNs occur only at the beginning of each host's
    # sequence where no historical observation exists.
    #
    # For those initial values, use zero.
    # ------------------------------------------------------------

    df[feature_columns] = (
        df[feature_columns]
        .fillna(0)
    )

    print(
        f"Created {len(feature_columns)} CAUSAL ML features."
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

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(f"Rows loaded: {len(df):,}")

    return df


# ================================================================
# SCORE DATA
# ================================================================

def score_dataset(
    df,
    scaler,
    model,
    feature_columns,
    dataset_name
):
    """
    Generate anomaly scores.

    Isolation Forest's decision_function has:

        higher = more normal

    Therefore we multiply by -1 so that:

        higher = more anomalous
    """

    print(
        f"\nScoring {dataset_name}..."
    )

    df_features, _ = create_causal_temporal_features(
        df
    )

    X = df_features[feature_columns]

    X_scaled = scaler.transform(X)

    # Higher = more anomalous
    anomaly_score = -model.decision_function(
        X_scaled
    )

    prediction = model.predict(
        X_scaled
    )

    result = df_features[
        [
            "timestamp",
            "hostname",
            "label",
            "xid",
        ]
    ].copy()

    result["anomaly_score"] = anomaly_score

    # Isolation Forest's native prediction:
    #
    #  1 = normal
    # -1 = anomaly
    #
    result["if_prediction"] = prediction

    print(
        f"{dataset_name}: {len(result):,} rows scored."
    )

    return result


# ================================================================
# SELECT VALIDATION THRESHOLD
# ================================================================

def select_validation_threshold(
    val_scores,
    target_false_positive_rate=0.01
):
    """
    Select an anomaly-score threshold using ONLY normal validation data.

    The threshold is chosen at the 99th percentile of normal validation
    scores, corresponding approximately to a 1% validation false-positive
    rate.

    IMPORTANT:
        No anomaly labels are used to select this threshold.
    """

    threshold = np.quantile(
        val_scores,
        1.0 - target_false_positive_rate
    )

    return threshold


# ================================================================
# EVALUATION
# ================================================================

def evaluate_predictions(
    scores,
    threshold,
    dataset_name
):
    """
    Evaluate anomaly detection against known labels.

    Labels:
        0 = normal
        1 = anomaly
    """

    y_true = scores["label"].astype(int).values

    y_pred = (
        scores["anomaly_score"].values >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # PR-AUC requires both classes.
    if len(np.unique(y_true)) == 2:
        pr_auc = average_precision_score(
            y_true,
            scores["anomaly_score"]
        )
    else:
        pr_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    normal_count = tn + fp

    false_positive_rate = (
        fp / normal_count
        if normal_count > 0
        else 0.0
    )

    print("\n" + "-" * 70)
    print(f"EVALUATION — {dataset_name}")
    print("-" * 70)

    print(f"Threshold:          {threshold:.6f}")
    print(f"Precision:          {precision:.6f}")
    print(f"Recall:             {recall:.6f}")
    print(f"F1:                 {f1:.6f}")
    print(f"PR-AUC:             {pr_auc:.6f}")
    print(f"False Positive Rate:{false_positive_rate:.6f}")

    print("\nConfusion Matrix:")
    print(
        f"TN={tn:,}  FP={fp:,}"
    )
    print(
        f"FN={fn:,}  TP={tp:,}"
    )

    return {
        "dataset": dataset_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "false_positive_rate": false_positive_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ================================================================
# EVENT-LEVEL EVALUATION
# ================================================================

def evaluate_event_detection(
    anomaly_scores,
    threshold
):
    """
    Evaluate detection at the XID-31 episode level.

    An episode is considered detected if at least one row within the
    known XID-31 episode crosses the anomaly threshold.

    This is important because 1,105 anomalous rows represent only
    85 actual synchronized episodes.
    """

    print("\n" + "-" * 70)
    print("EVENT-LEVEL EVALUATION")
    print("-" * 70)

    anomaly_scores = anomaly_scores.copy()

    anomaly_scores["detected"] = (
        anomaly_scores["anomaly_score"]
        >= threshold
    )

    # Only known anomalous rows
    anomaly_rows = anomaly_scores[
        anomaly_scores["label"] == 1
    ].copy()

    if anomaly_rows.empty:
        print("No anomalous events found.")
        return None

    # Group by host because the forensic analysis established
    # one XID-31 episode per affected host.
    event_detection = (
        anomaly_rows
        .groupby("hostname")["detected"]
        .any()
    )

    total_events = len(event_detection)

    detected_events = int(
        event_detection.sum()
    )

    missed_events = (
        total_events - detected_events
    )

    event_recall = (
        detected_events / total_events
        if total_events > 0
        else 0.0
    )

    print(
        f"Known XID-31 events: {total_events}"
    )

    print(
        f"Detected events:     {detected_events}"
    )

    print(
        f"Missed events:       {missed_events}"
    )

    print(
        f"Event recall:        {event_recall:.6f}"
    )

    return {
        "total_events": total_events,
        "detected_events": detected_events,
        "missed_events": missed_events,
        "event_recall": event_recall,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("SENTINEL — STEP 2")
    print("CAUSAL ISOLATION FOREST BASELINE")
    print("=" * 70)

    # ============================================================
    # 1. LOAD TRAINING DATA
    # ============================================================

    train_df = load_dataset(
        TRAIN_FILE
    )

    # ============================================================
    # 2. CREATE CAUSAL FEATURES
    # ============================================================

    train_df, feature_columns = (
        create_causal_temporal_features(
            train_df
        )
    )

    X_train = train_df[
        feature_columns
    ]

    print("\nTraining feature matrix:")
    print(
        X_train.shape
    )

    # ============================================================
    # 3. FIT SCALER — TRAIN ONLY
    # ============================================================

    print(
        "\nFitting scaler on TRAIN only..."
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # ============================================================
    # 4. TRAIN ISOLATION FOREST
    # ============================================================

    print(
        "\nTraining Isolation Forest..."
    )

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_scaled
    )

    print(
        "Model training complete."
    )

    # ============================================================
    # 5. SCORE VALIDATION DATA
    # ============================================================

    val_df = load_dataset(
        VAL_FILE
    )

    val_scores = score_dataset(
        val_df,
        scaler,
        model,
        feature_columns,
        "VALIDATION"
    )

    # ============================================================
    # 6. SELECT THRESHOLD USING NORMAL VALIDATION ONLY
    # ============================================================

    threshold = select_validation_threshold(
        val_scores["anomaly_score"],
        target_false_positive_rate=0.01
    )

    print("\nValidation threshold:")
    print(
        f"{threshold:.6f}"
    )

    validation_alarm_rate = (
        val_scores["anomaly_score"]
        >= threshold
    ).mean()
    
    print(
        f"Validation alarm rate: "
        f"{validation_alarm_rate:.6f}"
    )

    # ============================================================
    # 7. SCORE NORMAL TEST DATA
    # ============================================================

    test_df = load_dataset(
        TEST_FILE
    )

    test_scores = score_dataset(
        test_df,
        scaler,
        model,
        feature_columns,
        "NORMAL TEST"
    )

    # Normal test evaluation
    test_normal_results = evaluate_predictions(
        test_scores,
        threshold,
        "NORMAL TEST"
    )

    # ============================================================
    # 8. SCORE ANOMALOUS TEST DATA
    # ============================================================

    anomaly_df = load_dataset(
        ANOMALY_FILE
    )

    anomaly_scores = score_dataset(
        anomaly_df,
        scaler,
        model,
        feature_columns,
        "ANOMALY TEST"
    )

    anomaly_results = evaluate_predictions(
        anomaly_scores,
        threshold,
        "ANOMALY TEST"
    )

    # ============================================================
    # 9. EVENT-LEVEL EVALUATION
    # ============================================================

    event_results = evaluate_event_detection(
        anomaly_scores,
        threshold
    )

    # ============================================================
    # 10. SAVE SCORES
    # ============================================================

    val_scores.to_csv(
        REPORT_DIR / "step2_validation_scores.csv",
        index=False
    )

    test_scores.to_csv(
        REPORT_DIR / "step2_normal_test_scores.csv",
        index=False
    )

    anomaly_scores.to_csv(
        REPORT_DIR / "step2_anomaly_test_scores.csv",
        index=False
    )

    # ============================================================
    # 11. SAVE METRICS
    # ============================================================

    metrics = {
        "threshold": threshold,
        "validation_alarm_rate": validation_alarm_rate,
        "normal_test_precision": test_normal_results["precision"],
        "normal_test_recall": test_normal_results["recall"],
        "normal_test_f1": test_normal_results["f1"],
        "normal_test_pr_auc": test_normal_results["pr_auc"],
        "normal_test_fpr": test_normal_results["false_positive_rate"],
        "anomaly_test_precision": anomaly_results["precision"],
        "anomaly_test_recall": anomaly_results["recall"],
        "anomaly_test_f1": anomaly_results["f1"],
        "anomaly_test_pr_auc": anomaly_results["pr_auc"],
        "anomaly_test_fpr": anomaly_results["false_positive_rate"],
    }

    if event_results is not None:
        metrics.update({
            "total_xid31_events": event_results[
                "total_events"
            ],
            "detected_xid31_events": event_results[
                "detected_events"
            ],
            "missed_xid31_events": event_results[
                "missed_events"
            ],
            "event_recall": event_results[
                "event_recall"
            ],
        })

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        REPORT_DIR / "step2_causal_if_metrics.csv",
        index=False
    )

    # ============================================================
    # 12. SAVE MODEL
    # ============================================================

    model_path = (
        MODEL_DIR /
        "sentinel_causal_isolation_forest.joblib"
    )

    scaler_path = (
        MODEL_DIR /
        "sentinel_causal_scaler.joblib"
    )

    features_path = (
        MODEL_DIR /
        "sentinel_causal_features.joblib"
    )

    threshold_path = (
        MODEL_DIR /
        "sentinel_causal_threshold.joblib"
    )

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

    joblib.dump(
        threshold,
        threshold_path
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)

    print("\nSaved model:")
    print(model_path)

    print("\nSaved scaler:")
    print(scaler_path)

    print("\nSaved feature list:")
    print(features_path)

    print("\nSaved threshold:")
    print(threshold_path)

    print("\nSaved reports:")
    print(
        REPORT_DIR /
        "step2_validation_scores.csv"
    )

    print(
        REPORT_DIR /
        "step2_normal_test_scores.csv"
    )

    print(
        REPORT_DIR /
        "step2_anomaly_test_scores.csv"
    )

    print(
        REPORT_DIR /
        "step2_causal_if_metrics.csv"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
