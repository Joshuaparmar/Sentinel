import os
import json
import numpy as np
import pandas as pd


# ============================================================
# SENTINEL - STEP 5B
# FAILURE RISK SCORING
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

STEP4_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "step4",
    "step4_anomaly_aligned_scores.csv",
)

STEP5A_EVENT_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "step5",
    "step5a_event_level_results.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "step5",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PERSISTENCE_WINDOW = 6       # 6 samples = 60 seconds
MIN_HISTORY = 1

# Risk score weights.
#
# These are deliberately transparent heuristic weights.
#
# 60% = current anomaly evidence
# 25% = persistence
# 15% = detector agreement
#
EVIDENCE_WEIGHT = 0.60
PERSISTENCE_WEIGHT = 0.25
AGREEMENT_WEIGHT = 0.15


# ============================================================
# HELPERS
# ============================================================

def clamp_0_100(value):
    return np.clip(value, 0.0, 100.0)


def classify_risk(score):
    if score < 25:
        return "LOW"
    elif score < 50:
        return "MODERATE"
    elif score < 75:
        return "HIGH"
    else:
        return "CRITICAL"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SENTINEL - STEP 5B: FAILURE RISK SCORING")
print("=" * 70)

print(f"Project root: {PROJECT_ROOT}")
print(f"Output dir:   {OUTPUT_DIR}")


# ============================================================
# LOAD STEP 4 SCORES
# ============================================================

print("\nLoading Step 4 detector scores...")

df = pd.read_csv(STEP4_FILE)

print(f"Loaded rows: {len(df):,}")

print("\nAvailable columns:")
print(list(df.columns))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "hostname",
    "time",
    "if_score",
    "window_mse",
    "last_step_mse",
    "if_percentile",
    "window_percentile",
    "last_percentile",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)

if df["timestamp"].isna().any():
    bad = df["timestamp"].isna().sum()
    raise ValueError(
        f"Could not parse {bad} timestamps."
    )


# ============================================================
# SORT BY HOST AND TIME
# ============================================================

df = df.sort_values(
    ["hostname", "timestamp"]
).reset_index(drop=True)


# ============================================================
# CLEAN PERCENTILES
# ============================================================

percentile_columns = [
    "if_percentile",
    "window_percentile",
    "last_percentile",
]

for col in percentile_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    if df[col].isna().any():
        print(
            f"Warning: {col} contains "
            f"{df[col].isna().sum()} NaN values."
        )

        df[col] = df[col].fillna(0.0)

    df[col] = clamp_0_100(df[col])


# ============================================================
# 1. CURRENT ANOMALY EVIDENCE
# ============================================================

print("\nCalculating current anomaly evidence...")

df["evidence_mean"] = (
    df["if_percentile"]
    + df["window_percentile"]
    + df["last_percentile"]
) / 3.0

df["evidence_mean"] = clamp_0_100(
    df["evidence_mean"]
)


# Also calculate the strongest detector evidence.

df["evidence_max"] = df[
    [
        "if_percentile",
        "window_percentile",
        "last_percentile",
    ]
].max(axis=1)

df["evidence_max"] = clamp_0_100(
    df["evidence_max"]
)


# ============================================================
# 2. DETECTOR AGREEMENT
# ============================================================

print("Calculating detector agreement...")


# Use the validation-derived alarm thresholds already present
# in the Step 4 output.

if "if_alarm" in df.columns:
    df["if_alarm_used"] = (
        df["if_alarm"]
        .fillna(0)
        .astype(int)
    )
else:
    raise ValueError(
        "if_alarm column not found."
    )


if "window_alarm" in df.columns:
    df["window_alarm_used"] = (
        df["window_alarm"]
        .fillna(0)
        .astype(int)
    )
else:
    raise ValueError(
        "window_alarm column not found."
    )


if "last_alarm" in df.columns:
    df["last_alarm_used"] = (
        df["last_alarm"]
        .fillna(0)
        .astype(int)
    )
else:
    raise ValueError(
        "last_alarm column not found."
    )


df["detector_count"] = (
    df["if_alarm_used"]
    + df["window_alarm_used"]
    + df["last_alarm_used"]
)


# Convert:
#
# 0 detectors -> 0
# 1 detector  -> 33.33
# 2 detectors -> 66.67
# 3 detectors -> 100
#
df["agreement_score"] = (
    df["detector_count"] / 3.0
) * 100.0


# ============================================================
# 3. PERSISTENCE
# ============================================================

print(
    f"Calculating causal persistence "
    f"over {PERSISTENCE_WINDOW} samples..."
)


# A point is considered anomalous if at least one detector
# raises an alarm.

df["any_detector_alarm"] = (
    (
        df["if_alarm_used"]
        + df["window_alarm_used"]
        + df["last_alarm_used"]
    ) > 0
).astype(int)


# IMPORTANT:
#
# We use a causal rolling window.
#
# shift(1) means the current sample is NOT included
# when calculating prior persistence.
#
# Therefore this quantity represents:
#
# "How much anomaly activity existed immediately BEFORE
#  the current timestamp?"
#

df["prior_alarm_count"] = (
    df.groupby("hostname")["any_detector_alarm"]
    .transform(
        lambda s: s.shift(1)
        .rolling(
            window=PERSISTENCE_WINDOW,
            min_periods=MIN_HISTORY
        )
        .sum()
    )
    .fillna(0)
)


df["persistence_score"] = (
    df["prior_alarm_count"]
    / PERSISTENCE_WINDOW
) * 100.0

df["persistence_score"] = clamp_0_100(
    df["persistence_score"]
)


# ============================================================
# 4. FINAL RISK SCORE
# ============================================================

print("Calculating final SENTINEL risk score...")


df["risk_score"] = (
    EVIDENCE_WEIGHT * df["evidence_mean"]
    + PERSISTENCE_WEIGHT * df["persistence_score"]
    + AGREEMENT_WEIGHT * df["agreement_score"]
)

df["risk_score"] = clamp_0_100(
    df["risk_score"]
)


# ============================================================
# 5. RISK CATEGORY
# ============================================================

df["risk_level"] = (
    df["risk_score"]
    .apply(classify_risk)
)


# ============================================================
# 6. ADD FAILURE LABEL FOR EVALUATION ONLY
# ============================================================

# IMPORTANT:
#
# label is NOT used to calculate risk_score.
#
# It is retained only so that we can evaluate whether
# the risk score rises around the known XID-31 events.

if "label" in df.columns:
    df["failure_label"] = (
        pd.to_numeric(
            df["label"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )
else:
    df["failure_label"] = 0


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("RISK SCORE SUMMARY")
print("=" * 70)

print(
    f"Minimum risk score: "
    f"{df['risk_score'].min():.3f}"
)

print(
    f"Mean risk score: "
    f"{df['risk_score'].mean():.3f}"
)

print(
    f"Median risk score: "
    f"{df['risk_score'].median():.3f}"
)

print(
    f"Maximum risk score: "
    f"{df['risk_score'].max():.3f}"
)


# ============================================================
# RISK LEVEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RISK LEVEL DISTRIBUTION")
print("=" * 70)

risk_distribution = (
    df["risk_level"]
    .value_counts()
    .reindex(
        ["LOW", "MODERATE", "HIGH", "CRITICAL"],
        fill_value=0
    )
)

for level, count in risk_distribution.items():

    percentage = (
        count / len(df)
    ) * 100.0

    print(
        f"{level:<10} "
        f"{count:>10,} "
        f"({percentage:>7.3f}%)"
    )


# ============================================================
# FAILURE VS NORMAL RISK
# ============================================================

print("\n" + "=" * 70)
print("FAILURE VS NORMAL RISK")
print("=" * 70)

normal_rows = df["failure_label"] == 0
failure_rows = df["failure_label"] == 1

print(
    f"Normal rows:  {normal_rows.sum():,}"
)

print(
    f"Failure rows: {failure_rows.sum():,}"
)

if normal_rows.any():
    print(
        f"\nNormal mean risk: "
        f"{df.loc[normal_rows, 'risk_score'].mean():.3f}"
    )

    print(
        f"Normal median risk: "
        f"{df.loc[normal_rows, 'risk_score'].median():.3f}"
    )

if failure_rows.any():
    print(
        f"\nFailure mean risk: "
        f"{df.loc[failure_rows, 'risk_score'].mean():.3f}"
    )

    print(
        f"Failure median risk: "
        f"{df.loc[failure_rows, 'risk_score'].median():.3f}"
    )


# ============================================================
# FAILURE RISK-LEVEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RISK LEVELS DURING KNOWN FAILURE ROWS")
print("=" * 70)

if failure_rows.any():

    failure_risk_distribution = (
        df.loc[failure_rows, "risk_level"]
        .value_counts()
        .reindex(
            ["LOW", "MODERATE", "HIGH", "CRITICAL"],
            fill_value=0
        )
    )

    for level, count in failure_risk_distribution.items():

        percentage = (
            count / failure_rows.sum()
        ) * 100.0

        print(
            f"{level:<10} "
            f"{count:>8,} "
            f"({percentage:>7.3f}%)"
        )


# ============================================================
# HIGH / CRITICAL FAILURE COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("HIGH / CRITICAL FAILURE COVERAGE")
print("=" * 70)

high_or_critical = (
    df["risk_score"] >= 50
)

if failure_rows.any():

    detected_failure_rows = (
        high_or_critical
        & failure_rows
    ).sum()

    failure_coverage = (
        detected_failure_rows
        / failure_rows.sum()
    )

    print(
        f"Failure rows with risk >= 50: "
        f"{detected_failure_rows:,}/"
        f"{failure_rows.sum():,}"
    )

    print(
        f"Row-level coverage: "
        f"{failure_coverage:.6f}"
    )


# ============================================================
# HOST-LEVEL MAX RISK
# ============================================================

print("\n" + "=" * 70)
print("HOST-LEVEL RISK SUMMARY")
print("=" * 70)


host_summary = (
    df.groupby("hostname")
    .agg(
        max_risk_score=("risk_score", "max"),
        mean_risk_score=("risk_score", "mean"),
        max_evidence=("evidence_max", "max"),
        max_persistence=("persistence_score", "max"),
        max_agreement=("agreement_score", "max"),
        failure_rows=("failure_label", "sum"),
    )
    .reset_index()
)


host_summary["risk_level"] = (
    host_summary["max_risk_score"]
    .apply(classify_risk)
)


print(
    f"Total hosts evaluated: "
    f"{len(host_summary)}"
)

print(
    f"Hosts with max risk >= 75: "
    f"{(host_summary['max_risk_score'] >= 75).sum()}"
)

print(
    f"Hosts with max risk >= 50: "
    f"{(host_summary['max_risk_score'] >= 50).sum()}"
)


# ============================================================
# SAVE ROW-LEVEL RESULTS
# ============================================================

output_columns = [
    "hostname",
    "timestamp",
    "if_score",
    "window_mse",
    "last_step_mse",
    "if_percentile",
    "window_percentile",
    "last_percentile",
    "evidence_mean",
    "evidence_max",
    "if_alarm_used",
    "window_alarm_used",
    "last_alarm_used",
    "detector_count",
    "agreement_score",
    "any_detector_alarm",
    "prior_alarm_count",
    "persistence_score",
    "risk_score",
    "risk_level",
    "failure_label",
]

output_columns = [
    col for col in output_columns
    if col in df.columns
]

risk_output = df[output_columns].copy()

risk_output_path = os.path.join(
    OUTPUT_DIR,
    "step5b_risk_scores.csv"
)

risk_output.to_csv(
    risk_output_path,
    index=False
)


# ============================================================
# SAVE HOST SUMMARY
# ============================================================

host_output_path = os.path.join(
    OUTPUT_DIR,
    "step5b_host_risk_summary.csv"
)

host_summary.to_csv(
    host_output_path,
    index=False
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "step": "5B",
    "description": "SENTINEL failure risk scoring",

    "rows_evaluated": int(len(df)),
    "hosts_evaluated": int(df["hostname"].nunique()),

    "weights": {
        "current_evidence": EVIDENCE_WEIGHT,
        "persistence": PERSISTENCE_WEIGHT,
        "detector_agreement": AGREEMENT_WEIGHT,
    },

    "persistence_window_samples": PERSISTENCE_WINDOW,
    "persistence_window_seconds": PERSISTENCE_WINDOW * 10,

    "risk_levels": {
        "LOW": "0-24.999",
        "MODERATE": "25-49.999",
        "HIGH": "50-74.999",
        "CRITICAL": "75-100",
    },

    "risk_statistics": {
        "min": float(df["risk_score"].min()),
        "mean": float(df["risk_score"].mean()),
        "median": float(df["risk_score"].median()),
        "max": float(df["risk_score"].max()),
    },

    "normal_mean_risk": (
        float(
            df.loc[
                normal_rows,
                "risk_score"
            ].mean()
        )
        if normal_rows.any()
        else None
    ),

    "failure_mean_risk": (
        float(
            df.loc[
                failure_rows,
                "risk_score"
            ].mean()
        )
        if failure_rows.any()
        else None
    ),

    "failure_rows": int(
        failure_rows.sum()
    ),

    "failure_rows_risk_ge_50": int(
        (
            high_or_critical
            & failure_rows
        ).sum()
    ),
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "step5b_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
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
print("STEP 5B COMPLETE")
print("=" * 70)

print("\nReports saved to:")

print(risk_output_path)
print(host_output_path)
print(summary_path)

print("\nNext step:")
print(
    "Review the risk-score distributions and failure coverage "
    "before finalizing the health-state logic."
)

print("=" * 70)