import os
import json
import numpy as np
import pandas as pd


# ============================================================
# SENTINEL - STEP 5B RISK CALIBRATION / EVALUATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RISK_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "step5",
    "step5b_risk_scores.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "step5",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Candidate thresholds are evaluated empirically.
# They are NOT automatically declared as final risk levels.

CANDIDATE_THRESHOLDS = [
    1,
    2,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
]

# Threshold used for event-level analysis.
# We will evaluate several thresholds rather than assuming one.
EVENT_THRESHOLDS = [
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
]


# ============================================================
# HELPERS
# ============================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return a / b


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SENTINEL - STEP 5B: RISK SCORE CALIBRATION")
print("=" * 70)

print(f"Project root: {PROJECT_ROOT}")
print(f"Risk file:    {RISK_FILE}")


# ============================================================
# LOAD
# ============================================================

print("\nLoading Step 5B risk scores...")

df = pd.read_csv(RISK_FILE)

print(f"Loaded rows: {len(df):,}")

required = [
    "hostname",
    "timestamp",
    "risk_score",
    "failure_label",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["risk_score"] = pd.to_numeric(
    df["risk_score"],
    errors="coerce"
)

df["failure_label"] = pd.to_numeric(
    df["failure_label"],
    errors="coerce"
).fillna(0).astype(int)


df = df.dropna(
    subset=[
        "timestamp",
        "risk_score",
    ]
)

df = df.sort_values(
    ["hostname", "timestamp"]
).reset_index(drop=True)


# ============================================================
# SPLIT NORMAL / FAILURE
# ============================================================

normal = df["failure_label"] == 0
failure = df["failure_label"] == 1

normal_scores = df.loc[
    normal,
    "risk_score"
]

failure_scores = df.loc[
    failure,
    "risk_score"
]

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"Total rows:    {len(df):,}"
)

print(
    f"Normal rows:   {normal.sum():,}"
)

print(
    f"Failure rows:  {failure.sum():,}"
)

print(
    f"Hosts:         {df['hostname'].nunique()}"
)


# ============================================================
# DISTRIBUTION STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("RISK SCORE DISTRIBUTIONS")
print("=" * 70)


percentiles = [
    50,
    75,
    90,
    95,
    99,
    99.5,
    99.9,
    99.95,
    99.99,
    100,
]


distribution_rows = []

for p in percentiles:

    normal_value = np.percentile(
        normal_scores,
        p
    )

    failure_value = np.percentile(
        failure_scores,
        p
    )

    distribution_rows.append({
        "percentile": p,
        "normal_risk": normal_value,
        "failure_risk": failure_value,
    })

    print(
        f"P{p:<6} "
        f"Normal={normal_value:>10.4f} "
        f"Failure={failure_value:>10.4f}"
    )


distribution_df = pd.DataFrame(
    distribution_rows
)


# ============================================================
# CANDIDATE THRESHOLD EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("ROW-LEVEL THRESHOLD EVALUATION")
print("=" * 70)

threshold_rows = []

for threshold in CANDIDATE_THRESHOLDS:

    predicted = (
        df["risk_score"] >= threshold
    )

    tp = int(
        (predicted & failure).sum()
    )

    fn = int(
        (~predicted & failure).sum()
    )

    fp = int(
        (predicted & normal).sum()
    )

    tn = int(
        (~predicted & normal).sum()
    )

    recall = safe_divide(
        tp,
        tp + fn
    )

    precision = safe_divide(
        tp,
        tp + fp
    )

    fpr = safe_divide(
        fp,
        fp + tn
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    threshold_rows.append({
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    })

    print(
        f"\nThreshold >= {threshold}"
    )

    print(
        f"  TP={tp:,}  "
        f"FP={fp:,}  "
        f"TN={tn:,}  "
        f"FN={fn:,}"
    )

    print(
        f"  Precision={precision:.6f}"
    )

    print(
        f"  Recall={recall:.6f}"
    )

    print(
        f"  F1={f1:.6f}"
    )

    print(
        f"  FPR={fpr:.6f}"
    )


threshold_df = pd.DataFrame(
    threshold_rows
)


# ============================================================
# NORMAL FALSE-ALARM RATE BY THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("NORMAL FALSE-ALARM ANALYSIS")
print("=" * 70)

for threshold in CANDIDATE_THRESHOLDS:

    normal_alarm_count = int(
        (
            normal_scores >= threshold
        ).sum()
    )

    normal_fpr = safe_divide(
        normal_alarm_count,
        len(normal_scores)
    )

    print(
        f"Risk >= {threshold:<3} : "
        f"{normal_alarm_count:>8,} normal rows "
        f"({normal_fpr:.6%})"
    )


# ============================================================
# EVENT RECONSTRUCTION
# ============================================================

print("\n" + "=" * 70)
print("EVENT-LEVEL CALIBRATION")
print("=" * 70)

#
# An XID-31 event is a contiguous sequence of failure rows
# for one host.
#
# A gap greater than 15 seconds separates events.
#

failure_df = df[
    failure
].copy()

failure_df = failure_df.sort_values(
    ["hostname", "timestamp"]
)

failure_df["time_gap"] = (
    failure_df
    .groupby("hostname")["timestamp"]
    .diff()
    .dt.total_seconds()
)

failure_df["new_event"] = (
    failure_df["time_gap"].isna()
    | (failure_df["time_gap"] > 15)
)

failure_df["event_number"] = (
    failure_df
    .groupby("hostname")["new_event"]
    .cumsum()
)

failure_df["event_id"] = (
    failure_df["hostname"].astype(str)
    + "_event_"
    + failure_df["event_number"].astype(str)
)

event_ids = (
    failure_df["event_id"]
    .unique()
)

print(
    f"Detected failure events: "
    f"{len(event_ids)}"
)


# ============================================================
# EVENT-LEVEL THRESHOLD METRICS
# ============================================================

event_rows = []

for threshold in EVENT_THRESHOLDS:

    detected_events = 0

    event_max_scores = []

    for event_id, event_group in failure_df.groupby(
        "event_id"
    ):

        max_score = (
            event_group["risk_score"]
            .max()
        )

        event_max_scores.append(
            max_score
        )

        if max_score >= threshold:
            detected_events += 1

    event_recall = safe_divide(
        detected_events,
        len(event_ids)
    )

    event_rows.append({
        "threshold": threshold,
        "events_detected": detected_events,
        "total_events": len(event_ids),
        "event_recall": event_recall,
        "mean_event_max_risk": np.mean(
            event_max_scores
        ),
        "median_event_max_risk": np.median(
            event_max_scores
        ),
    })

    print(
        f"\nRisk >= {threshold}"
    )

    print(
        f"  Events detected: "
        f"{detected_events}/{len(event_ids)}"
    )

    print(
        f"  Event recall: "
        f"{event_recall:.6f}"
    )

    print(
        f"  Mean event max risk: "
        f"{np.mean(event_max_scores):.4f}"
    )

    print(
        f"  Median event max risk: "
        f"{np.median(event_max_scores):.4f}"
    )


event_df = pd.DataFrame(
    event_rows
)


# ============================================================
# HOST-LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HOST-LEVEL ANALYSIS")
print("=" * 70)

host_summary = (
    df.groupby("hostname")
    .agg(
        max_risk=("risk_score", "max"),
        mean_risk=("risk_score", "mean"),
        failure_rows=("failure_label", "sum"),
    )
    .reset_index()
)

host_summary["has_failure"] = (
    host_summary["failure_rows"] > 0
)


failure_hosts = host_summary[
    host_summary["has_failure"]
]

normal_hosts = host_summary[
    ~host_summary["has_failure"]
]

print(
    f"Failure-associated hosts: "
    f"{len(failure_hosts)}"
)

print(
    f"Hosts without XID-31: "
    f"{len(normal_hosts)}"
)

print(
    f"\nFailure-host max-risk median: "
    f"{failure_hosts['max_risk'].median():.4f}"
)

print(
    f"Failure-host max-risk mean: "
    f"{failure_hosts['max_risk'].mean():.4f}"
)

if len(normal_hosts) > 0:

    print(
        f"Normal-host max-risk median: "
        f"{normal_hosts['max_risk'].median():.4f}"
    )

    print(
        f"Normal-host max-risk mean: "
        f"{normal_hosts['max_risk'].mean():.4f}"
    )


# ============================================================
# TOP FAILURE-HOST RISK
# ============================================================

print("\n" + "=" * 70)
print("FAILURE-HOST MAXIMUM RISK")
print("=" * 70)

top_failure_hosts = (
    failure_hosts
    .sort_values(
        "max_risk",
        ascending=False
    )
    .head(20)
)

print(
    top_failure_hosts[
        [
            "hostname",
            "max_risk",
            "mean_risk",
            "failure_rows",
        ]
    ].to_string(index=False)
)


# ============================================================
# PRE-EVENT RISK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PRE-EVENT RISK ANALYSIS")
print("=" * 70)

PRE_EVENT_WINDOWS = [
    60,
    120,
    300,
    600,
]

pre_event_rows = []

for event_id, event_group in failure_df.groupby(
    "event_id"
):

    event_start = event_group[
        "timestamp"
    ].min()

    hostname = event_group[
        "hostname"
    ].iloc[0]

    host_rows = df[
        df["hostname"] == hostname
    ]

    for window_seconds in PRE_EVENT_WINDOWS:

        window_start = (
            event_start
            - pd.Timedelta(
                seconds=window_seconds
            )
        )

        pre_rows = host_rows[
            (host_rows["timestamp"] >= window_start)
            & (host_rows["timestamp"] < event_start)
        ]

        if len(pre_rows) == 0:

            max_risk = 0.0
            mean_risk = 0.0

        else:

            max_risk = (
                pre_rows["risk_score"]
                .max()
            )

            mean_risk = (
                pre_rows["risk_score"]
                .mean()
            )

        pre_event_rows.append({
            "event_id": event_id,
            "hostname": hostname,
            "event_start": event_start,
            "window_seconds": window_seconds,
            "max_pre_event_risk": max_risk,
            "mean_pre_event_risk": mean_risk,
        })


pre_event_df = pd.DataFrame(
    pre_event_rows
)


# ============================================================
# PRE-EVENT SUMMARY
# ============================================================

print("\nPre-event maximum risk by window:")

for window_seconds in PRE_EVENT_WINDOWS:

    subset = pre_event_df[
        pre_event_df["window_seconds"]
        == window_seconds
    ]

    print(
        f"\n{window_seconds}s before event:"
    )

    print(
        f"  Mean max risk: "
        f"{subset['max_pre_event_risk'].mean():.4f}"
    )

    print(
        f"  Median max risk: "
        f"{subset['max_pre_event_risk'].median():.4f}"
    )

    for threshold in [
        5,
        10,
        15,
        20,
        25,
        30,
    ]:

        count = (
            subset["max_pre_event_risk"]
            >= threshold
        ).sum()

        print(
            f"  Events with risk >= "
            f"{threshold}: "
            f"{count}/{len(subset)}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

distribution_path = os.path.join(
    OUTPUT_DIR,
    "step5b_risk_distribution.csv",
)

threshold_path = os.path.join(
    OUTPUT_DIR,
    "step5b_threshold_evaluation.csv",
)

event_path = os.path.join(
    OUTPUT_DIR,
    "step5b_event_threshold_evaluation.csv",
)

host_path = os.path.join(
    OUTPUT_DIR,
    "step5b_host_calibration.csv",
)

pre_event_path = os.path.join(
    OUTPUT_DIR,
    "step5b_pre_event_risk.csv",
)


distribution_df.to_csv(
    distribution_path,
    index=False
)

threshold_df.to_csv(
    threshold_path,
    index=False
)

event_df.to_csv(
    event_path,
    index=False
)

host_summary.to_csv(
    host_path,
    index=False
)

pre_event_df.to_csv(
    pre_event_path,
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "step": "5B_calibration",

    "rows": int(len(df)),

    "normal_rows": int(normal.sum()),

    "failure_rows": int(failure.sum()),

    "failure_events": int(len(event_ids)),

    "failure_hosts": int(len(failure_hosts)),

    "normal_hosts": int(len(normal_hosts)),

    "normal_risk": {
        "mean": float(
            normal_scores.mean()
        ),
        "median": float(
            normal_scores.median()
        ),
        "max": float(
            normal_scores.max()
        ),
    },

    "failure_risk": {
        "mean": float(
            failure_scores.mean()
        ),
        "median": float(
            failure_scores.median()
        ),
        "max": float(
            failure_scores.max()
        ),
    },

    "failure_host_max_risk": {
        "mean": float(
            failure_hosts["max_risk"].mean()
        ),
        "median": float(
            failure_hosts["max_risk"].median()
        ),
    },

    "normal_host_max_risk": {
        "mean": float(
            normal_hosts["max_risk"].mean()
        )
        if len(normal_hosts)
        else None,
        "median": float(
            normal_hosts["max_risk"].median()
        )
        if len(normal_hosts)
        else None,
    },
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "step5b_calibration_summary.json",
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5B CALIBRATION COMPLETE")
print("=" * 70)

print("\nReports saved to:")

print(distribution_path)
print(threshold_path)
print(event_path)
print(host_path)
print(pre_event_path)
print(summary_path)

print("\nNext step:")
print(
    "Use the empirical threshold and event-level results "
    "to finalize the SENTINEL risk-state design."
)

print("=" * 70)