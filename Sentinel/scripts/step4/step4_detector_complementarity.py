from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# ============================================================
# SENTINEL - STEP 4
# Detector Complementarity + Evidence Fusion
#
# Uses EXISTING Step 2 and Step 3 outputs.
# No model retraining.
#
# Alignment:
#   IF timestamp == LSTM end_time
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = BASE_DIR / "reports"
OUT_DIR = REPORT_DIR / "step4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

FILES = {
    "if_val": REPORT_DIR / "step2_validation_scores.csv",
    "if_normal": REPORT_DIR / "step2_normal_test_scores.csv",
    "if_anomaly": REPORT_DIR / "step2_anomaly_test_scores.csv",

    "lstm_val": REPORT_DIR / "step3_lstm_validation_scores.csv",
    "lstm_normal": REPORT_DIR / "step3_lstm_normal_test_scores.csv",
    "lstm_anomaly": REPORT_DIR / "step3_lstm_anomaly_test_scores.csv",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_csv(name):
    path = FILES[name]

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)

    print(f"Loaded {name}: {len(df):,} rows")

    return df


def prepare_if(df):
    """
    Standardize Isolation Forest output.

    IF:
      timestamp = point timestamp
      anomaly_score = higher means more anomalous
    """

    out = pd.DataFrame({
        "hostname": df["hostname"].astype(str),
        "time": pd.to_datetime(df["timestamp"]),
        "if_score": pd.to_numeric(
            df["anomaly_score"],
            errors="coerce",
        ),
    })

    if "label" in df.columns:
        out["label"] = pd.to_numeric(
            df["label"],
            errors="coerce",
        )

    return out.dropna(
        subset=["hostname", "time", "if_score"]
    )


def prepare_lstm(df):
    """
    Standardize LSTM output.

    LSTM scores are associated with end_time,
    i.e. the endpoint of the 60-step sequence.
    """

    out = pd.DataFrame({
        "hostname": df["hostname"].astype(str),
        "time": pd.to_datetime(df["end_time"]),

        "window_mse": pd.to_numeric(
            df["window_mse"],
            errors="coerce",
        ),

        "last_step_mse": pd.to_numeric(
            df["last_step_mse"],
            errors="coerce",
        ),
    })

    if "label" in df.columns:
        out["label"] = pd.to_numeric(
            df["label"],
            errors="coerce",
        )

    return out.dropna(
        subset=[
            "hostname",
            "time",
            "window_mse",
            "last_step_mse",
        ]
    )


def align(if_df, lstm_df):
    """
    Exact endpoint alignment:

        IF timestamp == LSTM end_time
        AND hostname matches
    """

    merged = pd.merge(
        if_df,
        lstm_df,
        on=["hostname", "time"],
        how="inner",
        suffixes=("_if", "_lstm"),
    )

    # Prefer the anomaly label from IF if available.
    if "label_if" in merged.columns:
        merged["label"] = merged["label_if"]
    elif "label_lstm" in merged.columns:
        merged["label"] = merged["label_lstm"]
    else:
        merged["label"] = np.nan

    return merged


def percentile_against_normal(values, reference):
    """
    Convert a score to its empirical percentile relative
    to NORMAL validation scores.

    Higher percentile = more anomalous.

    This is used to put IF and LSTM scores onto a comparable
    [0, 1] scale without fitting anything on the anomaly test.
    """

    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    reference = np.sort(reference)

    positions = np.searchsorted(
        reference,
        values,
        side="right",
    )

    return positions / len(reference)


def binary_metrics(y_true, prediction):

    y_true = np.asarray(y_true).astype(int)
    prediction = np.asarray(prediction).astype(int)

    tp = int(((y_true == 1) & (prediction == 1)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fn = int(((y_true == 1) & (prediction == 0)).sum())

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "FPR": fpr,
    }


def build_events(df):
    """
    Build XID-31 events from labelled rows.

    Events are grouped by hostname and contiguous timestamps
    with <=15 second gaps.
    """

    labelled = df[df["label"] == 1].copy()

    if labelled.empty:
        return pd.DataFrame()

    labelled = labelled.sort_values(
        ["hostname", "time"]
    )

    events = []
    event_id = 0

    for hostname, group in labelled.groupby(
        "hostname",
        sort=False,
    ):

        group = group.sort_values("time")

        times = group["time"].to_numpy()

        start = 0

        for i in range(1, len(times) + 1):

            split = (
                i == len(times)
                or (
                    (times[i] - times[i - 1])
                    / np.timedelta64(1, "s")
                    > 15
                )
            )

            if split:

                part = group.iloc[start:i]

                events.append({
                    "event_id": event_id,
                    "hostname": hostname,
                    "start_time": part["time"].min(),
                    "end_time": part["time"].max(),
                    "rows": len(part),
                })

                event_id += 1
                start = i

    return pd.DataFrame(events)


def event_recall(df, score_column, threshold):

    events = build_events(df)

    if events.empty:
        return {
            "events_detected": 0,
            "events_total": 0,
            "event_recall": 0.0,
        }, events

    detected = []

    for _, event in events.iterrows():

        mask = (
            (df["hostname"] == event["hostname"])
            & (df["time"] >= event["start_time"])
            & (df["time"] <= event["end_time"])
            & (df[score_column] >= threshold)
        )

        detected.append(bool(mask.any()))

    events["detected"] = detected

    count = int(events["detected"].sum())
    total = len(events)

    return {
        "events_detected": count,
        "events_total": total,
        "event_recall": count / total,
    }, events


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("SENTINEL - STEP 4: DETECTOR COMPLEMENTARITY")
    print("=" * 70)

    print(f"Project root: {BASE_DIR}")
    print(f"Output dir:   {OUT_DIR}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if_val_raw = load_csv("if_val")
    if_normal_raw = load_csv("if_normal")
    if_anomaly_raw = load_csv("if_anomaly")

    lstm_val_raw = load_csv("lstm_val")
    lstm_normal_raw = load_csv("lstm_normal")
    lstm_anomaly_raw = load_csv("lstm_anomaly")

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    if_val = prepare_if(if_val_raw)
    if_normal = prepare_if(if_normal_raw)
    if_anomaly = prepare_if(if_anomaly_raw)

    lstm_val = prepare_lstm(lstm_val_raw)
    lstm_normal = prepare_lstm(lstm_normal_raw)
    lstm_anomaly = prepare_lstm(lstm_anomaly_raw)

    # --------------------------------------------------------
    # Align
    # --------------------------------------------------------

    print("\nAligning detector outputs by:")
    print("  hostname + IF timestamp == LSTM end_time")

    val = align(if_val, lstm_val)
    normal = align(if_normal, lstm_normal)
    anomaly = align(if_anomaly, lstm_anomaly)

    print(f"\nAligned rows:")
    print(f"  validation : {len(val):,}")
    print(f"  normal test: {len(normal):,}")
    print(f"  anomaly test: {len(anomaly):,}")

    if len(val) == 0:
        raise RuntimeError(
            "No validation rows aligned between IF and LSTM."
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print("\nAlignment sanity checks:")

    print(
        f"  Validation labels: "
        f"{val['label'].value_counts(dropna=False).to_dict()}"
    )

    print(
        f"  Anomaly labels: "
        f"{anomaly['label'].value_counts(dropna=False).to_dict()}"
    )

    # --------------------------------------------------------
    # Score correlations
    # --------------------------------------------------------

    pearson_window = val[
        ["if_score", "window_mse"]
    ].corr(method="pearson").iloc[0, 1]

    spearman_window = val[
        ["if_score", "window_mse"]
    ].corr(method="spearman").iloc[0, 1]

    pearson_last = val[
        ["if_score", "last_step_mse"]
    ].corr(method="pearson").iloc[0, 1]

    spearman_last = val[
        ["if_score", "last_step_mse"]
    ].corr(method="spearman").iloc[0, 1]

    correlation = pd.DataFrame([
        {
            "comparison": "IF vs LSTM Window MSE",
            "pearson": pearson_window,
            "spearman": spearman_window,
        },
        {
            "comparison": "IF vs LSTM Last-step MSE",
            "pearson": pearson_last,
            "spearman": spearman_last,
        },
    ])

    correlation.to_csv(
        OUT_DIR / "step4_score_correlations.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Standalone thresholds
    #
    # Recompute from the actual validation outputs so Step 4
    # remains self-contained.
    # --------------------------------------------------------

    if_threshold = float(
        np.quantile(
            if_val["if_score"],
            0.99,
        )
    )

    window_threshold = float(
        np.quantile(
            lstm_val["window_mse"],
            0.99,
        )
    )

    last_threshold = float(
        np.quantile(
            lstm_val["last_step_mse"],
            0.99,
        )
    )

    print("\nValidation-derived thresholds:")
    print(f"  IF:              {if_threshold:.8f}")
    print(f"  LSTM Window MSE: {window_threshold:.8f}")
    print(f"  LSTM Last-step:  {last_threshold:.8f}")

    # --------------------------------------------------------
    # Standalone alarms
    # --------------------------------------------------------

    for df in [normal, anomaly]:

        df["if_alarm"] = (
            df["if_score"] >= if_threshold
        )

        df["window_alarm"] = (
            df["window_mse"] >= window_threshold
        )

        df["last_alarm"] = (
            df["last_step_mse"] >= last_threshold
        )

    # --------------------------------------------------------
    # Detection overlap
    # --------------------------------------------------------

    def make_overlap(df, lstm_column):

        both = (
            df["if_alarm"]
            & df[lstm_column]
        )

        if_only = (
            df["if_alarm"]
            & ~df[lstm_column]
        )

        lstm_only = (
            ~df["if_alarm"]
            & df[lstm_column]
        )

        neither = (
            ~df["if_alarm"]
            & ~df[lstm_column]
        )

        return {
            "both": int(both.sum()),
            "IF_only": int(if_only.sum()),
            "LSTM_only": int(lstm_only.sum()),
            "neither": int(neither.sum()),
        }

    overlap_rows = []

    for dataset_name, df in [
        ("normal_test", normal),
        ("anomaly_test", anomaly),
    ]:

        for score_name, alarm_column in [
            ("window_mse", "window_alarm"),
            ("last_step_mse", "last_alarm"),
        ]:

            result = make_overlap(
                df,
                alarm_column,
            )

            for category, count in result.items():

                overlap_rows.append({
                    "dataset": dataset_name,
                    "lstm_score": score_name,
                    "category": category,
                    "count": count,
                })

    overlap = pd.DataFrame(overlap_rows)

    overlap.to_csv(
        OUT_DIR / "step4_detection_overlap.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Unique detection analysis
    # --------------------------------------------------------

    anomaly["window_unique_to_if"] = (
        anomaly["if_alarm"]
        & ~anomaly["window_alarm"]
    )

    anomaly["window_unique_to_lstm"] = (
        ~anomaly["if_alarm"]
        & anomaly["window_alarm"]
    )

    anomaly["last_unique_to_if"] = (
        anomaly["if_alarm"]
        & ~anomaly["last_alarm"]
    )

    anomaly["last_unique_to_lstm"] = (
        ~anomaly["if_alarm"]
        & anomaly["last_alarm"]
    )

    unique_summary = pd.DataFrame([
        {
            "comparison": "IF vs Window-MSE",
            "IF_only_rows": int(
                anomaly["window_unique_to_if"].sum()
            ),
            "LSTM_only_rows": int(
                anomaly["window_unique_to_lstm"].sum()
            ),
        },
        {
            "comparison": "IF vs Last-step-MSE",
            "IF_only_rows": int(
                anomaly["last_unique_to_if"].sum()
            ),
            "LSTM_only_rows": int(
                anomaly["last_unique_to_lstm"].sum()
            ),
        },
    ])

    unique_summary.to_csv(
        OUT_DIR / "step4_unique_detection.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Validation-normalized fusion
    #
    # We convert each detector score to an empirical percentile
    # relative to its NORMAL validation distribution.
    #
    # Fusion = 0.5 * IF percentile
    #        + 0.5 * LSTM percentile
    #
    # Threshold is the 99th percentile on validation.
    # --------------------------------------------------------

    if_reference = np.sort(
        val["if_score"].to_numpy()
    )

    window_reference = np.sort(
        val["window_mse"].to_numpy()
    )

    last_reference = np.sort(
        val["last_step_mse"].to_numpy()
    )

    val["if_percentile"] = percentile_against_normal(
        val["if_score"],
        if_reference,
    )

    val["window_percentile"] = percentile_against_normal(
        val["window_mse"],
        window_reference,
    )

    val["last_percentile"] = percentile_against_normal(
        val["last_step_mse"],
        last_reference,
    )

    val["fusion_window"] = (
        0.5 * val["if_percentile"]
        + 0.5 * val["window_percentile"]
    )

    val["fusion_last"] = (
        0.5 * val["if_percentile"]
        + 0.5 * val["last_percentile"]
    )

    fusion_window_threshold = float(
        np.quantile(
            val["fusion_window"],
            0.99,
        )
    )

    fusion_last_threshold = float(
        np.quantile(
            val["fusion_last"],
            0.99,
        )
    )

    def add_fusion_scores(df):

        df["if_percentile"] = percentile_against_normal(
            df["if_score"],
            if_reference,
        )

        df["window_percentile"] = percentile_against_normal(
            df["window_mse"],
            window_reference,
        )

        df["last_percentile"] = percentile_against_normal(
            df["last_step_mse"],
            last_reference,
        )

        df["fusion_window"] = (
            0.5 * df["if_percentile"]
            + 0.5 * df["window_percentile"]
        )

        df["fusion_last"] = (
            0.5 * df["if_percentile"]
            + 0.5 * df["last_percentile"]
        )

        df["fusion_window_alarm"] = (
            df["fusion_window"]
            >= fusion_window_threshold
        )

        df["fusion_last_alarm"] = (
            df["fusion_last"]
            >= fusion_last_threshold
        )

    add_fusion_scores(normal)
    add_fusion_scores(anomaly)

    # --------------------------------------------------------
    # Evaluate fusion
    # --------------------------------------------------------

    results = []

    def evaluate(
        name,
        normal_prediction,
        anomaly_prediction,
    ):

        y_normal = np.zeros(
            len(normal_prediction),
            dtype=int,
        )

        y_anomaly = np.ones(
            len(anomaly_prediction),
            dtype=int,
        )

        y_true = np.concatenate(
            [y_normal, y_anomaly]
        )

        pred = np.concatenate(
            [
                normal_prediction,
                anomaly_prediction,
            ]
        )

        metrics = binary_metrics(
            y_true,
            pred,
        )

        scores = np.concatenate(
            [
                normal_prediction.astype(float),
                anomaly_prediction.astype(float),
            ]
        )

        results.append({
            "detector": name,
            **metrics,
        })

    # The prediction arrays above are booleans; calculate metrics
    # using normal and anomaly labels separately.
    for name, normal_col, anomaly_col in [
        (
            "IF + LSTM Window-MSE Fusion",
            "fusion_window_alarm",
            "fusion_window_alarm",
        ),
        (
            "IF + LSTM Last-step Fusion",
            "fusion_last_alarm",
            "fusion_last_alarm",
        ),
    ]:

        normal_pred = normal[normal_col].to_numpy()
        anomaly_pred = anomaly[anomaly_col].to_numpy()

        y_true = np.concatenate([
            np.zeros(len(normal_pred), dtype=int),
            np.ones(len(anomaly_pred), dtype=int),
        ])

        pred = np.concatenate([
            normal_pred.astype(int),
            anomaly_pred.astype(int),
        ])

        metrics = binary_metrics(
            y_true,
            pred,
        )

        # PR-AUC requires the continuous fusion score.
        if name.endswith("Window-MSE Fusion"):
            scores = np.concatenate([
                normal["fusion_window"].to_numpy(),
                anomaly["fusion_window"].to_numpy(),
            ])
        else:
            scores = np.concatenate([
                normal["fusion_last"].to_numpy(),
                anomaly["fusion_last"].to_numpy(),
            ])

        pr_auc = average_precision_score(
            y_true,
            scores,
        )

        metrics["PR_AUC"] = pr_auc
        metrics["detector"] = name

        results.append(metrics)

    # --------------------------------------------------------
    # Event evaluation
    # --------------------------------------------------------

    print("\nEvaluating event-level detection...")

    # Use anomaly test only for event definitions.
    anomaly_for_events = anomaly.copy()

    window_event_info, window_events = event_recall(
        anomaly_for_events.rename(
            columns={"fusion_window": "score"}
        ),
        "score",
        fusion_window_threshold,
    )

    last_event_info, last_events = event_recall(
        anomaly_for_events.rename(
            columns={"fusion_last": "score"}
        ),
        "score",
        fusion_last_threshold,
    )

    event_results = pd.DataFrame([
        {
            "detector": "IF + LSTM Window-MSE Fusion",
            **window_event_info,
        },
        {
            "detector": "IF + LSTM Last-step Fusion",
            **last_event_info,
        },
    ])

    # --------------------------------------------------------
    # Add standalone PR-AUC for comparison
    # --------------------------------------------------------

    comparison_rows = []

    for name, normal_df, anomaly_df, score_col in [
        (
            "Isolation Forest",
            normal,
            anomaly,
            "if_score",
        ),
        (
            "LSTM Window-MSE",
            normal,
            anomaly,
            "window_mse",
        ),
        (
            "LSTM Last-step-MSE",
            normal,
            anomaly,
            "last_step_mse",
        ),
        (
            "IF + LSTM Window-MSE Fusion",
            normal,
            anomaly,
            "fusion_window",
        ),
        (
            "IF + LSTM Last-step Fusion",
            normal,
            anomaly,
            "fusion_last",
        ),
    ]:

        y_true = np.concatenate([
            np.zeros(len(normal_df), dtype=int),
            np.ones(len(anomaly_df), dtype=int),
        ])

        scores = np.concatenate([
            normal_df[score_col].to_numpy(),
            anomaly_df[score_col].to_numpy(),
        ])

        pr_auc = average_precision_score(
            y_true,
            scores,
        )

        comparison_rows.append({
            "detector": name,
            "PR_AUC": pr_auc,
        })

    pr_auc_df = pd.DataFrame(comparison_rows)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pd.DataFrame(results).to_csv(
        OUT_DIR / "step4_fusion_metrics.csv",
        index=False,
    )

    event_results.to_csv(
        OUT_DIR / "step4_event_metrics.csv",
        index=False,
    )

    pr_auc_df.to_csv(
        OUT_DIR / "step4_pr_auc_comparison.csv",
        index=False,
    )

    correlation.to_csv(
        OUT_DIR / "step4_score_correlations.csv",
        index=False,
    )

    overlap.to_csv(
        OUT_DIR / "step4_detection_overlap.csv",
        index=False,
    )

    config = {
        "fusion_method": "equal_weight_empirical_percentile_fusion",
        "if_weight": 0.5,
        "lstm_weight": 0.5,
        "validation_percentile": 0.99,
        "fusion_window_threshold": fusion_window_threshold,
        "fusion_last_threshold": fusion_last_threshold,
        "alignment": "hostname + IF timestamp == LSTM end_time",
        "anomaly_test_used_for_threshold": False,
    }

    with open(
        OUT_DIR / "step4_fusion_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    anomaly.to_csv(
        OUT_DIR / "step4_anomaly_aligned_scores.csv",
        index=False,
    )

    normal.to_csv(
        OUT_DIR / "step4_normal_aligned_scores.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 4 - SCORE CORRELATION")
    print("=" * 70)

    print(
        correlation.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("STEP 4 - DETECTION OVERLAP")
    print("=" * 70)

    print(
        overlap.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("STEP 4 - PR-AUC COMPARISON")
    print("=" * 70)

    print(
        pr_auc_df.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("STEP 4 - FUSION METRICS")
    print("=" * 70)

    print(
        pd.DataFrame(results).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("STEP 4 - EVENT METRICS")
    print("=" * 70)

    print(
        event_results.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE")
    print("=" * 70)

    print(f"Reports saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()