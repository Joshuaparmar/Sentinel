from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "dataset"
    / "ClusterWise_mini"
    / "olcf_task"
    / "test_anomaly.csv"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SENTINEL - STEP 1 DATASET FORENSICS")
print("=" * 70)

print(f"\nLoading:\n{DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ============================================================
# BASIC DATASET CHECK
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print("\n--- BASIC DATASET ---")

print(f"Hosts: {df['hostname'].nunique():,}")
print(f"Invalid timestamps: {df['timestamp'].isna().sum():,}")

print(
    f"Time range: "
    f"{df['timestamp'].min()} -> {df['timestamp'].max()}"
)

print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False))

print("\nXID distribution:")
print(df["xid"].value_counts(dropna=False))

# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["hostname", "timestamp"]
).reset_index(drop=True)

# ============================================================
# IDENTIFY XID 31 EVENTS
# ============================================================

events = df[df["xid"] == 31].copy()

print("\n" + "=" * 70)
print("XID-31 EVENT ANALYSIS")
print("=" * 70)

print(f"\nTotal XID-31 records: {len(events):,}")
print(f"Hosts affected: {events['hostname'].nunique():,}")

print("\nXID-31 events by host:")
print(events["hostname"].value_counts().to_string())

print("\nFirst 10 XID-31 events:")
print(
    events[
        ["timestamp", "hostname", "label", "xid"]
    ].head(10).to_string(index=False)
)

# Save event list
events[
    ["timestamp", "hostname", "label", "xid"]
].to_csv(
    REPORT_DIR / "xid31_events.csv",
    index=False
)

# ============================================================
# TELEMETRY FEATURES
# ============================================================

EXCLUDE = {
    "timestamp",
    "hostname",
    "node_state",
    "yymmdd",
    "job_begin_time",
    "job_end_time",
    "allocation_id",
    "label",
    "xid",
}

numeric_features = []

for col in df.columns:

    if col in EXCLUDE:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_features.append(col)

print("\nNumeric telemetry features:")
print(f"Count: {len(numeric_features)}")

for feature in numeric_features:
    print(f"  - {feature}")

# ============================================================
# PRE-EVENT ANALYSIS
# ============================================================

WINDOWS = [
    10,
    30,
    60,
    120,
    300
]

results = []

print("\n" + "=" * 70)
print("PRE-EVENT TELEMETRY ANALYSIS")
print("=" * 70)

print(
    "\nFor each XID-31 event we compare:"
    "\n  PRE-EVENT = telemetry immediately before XID-31"
    "\n  BASELINE  = an earlier normal window of the same duration"
)

# ------------------------------------------------------------
# For each failure event
# ------------------------------------------------------------

for event_idx, event in events.iterrows():

    host = event["hostname"]
    event_time = event["timestamp"]

    host_df = df[df["hostname"] == host].copy()

    for window_seconds in WINDOWS:

        # ----------------------------------------------------
        # Pre-event window
        # ----------------------------------------------------

        pre_start = (
            event_time -
            pd.Timedelta(seconds=window_seconds)
        )

        pre_end = event_time

        pre = host_df[
            (host_df["timestamp"] >= pre_start) &
            (host_df["timestamp"] < pre_end)
        ]

        # ----------------------------------------------------
        # Baseline window
        #
        # Same duration, immediately before the pre-event
        # window, with a gap of equal size.
        #
        # Example for 60s:
        #
        # baseline | gap | pre-event | XID31
        # <---60---><--60--><---60--->
        # ----------------------------------------------------

        baseline_end = pre_start

        baseline_start = (
            baseline_end -
            pd.Timedelta(seconds=window_seconds)
        )

        baseline = host_df[
            (host_df["timestamp"] >= baseline_start) &
            (host_df["timestamp"] < baseline_end)
        ]

        if len(pre) == 0 or len(baseline) == 0:
            continue

        # ----------------------------------------------------
        # Feature comparison
        # ----------------------------------------------------

        for feature in numeric_features:

            pre_values = pd.to_numeric(
                pre[feature],
                errors="coerce"
            ).dropna()

            base_values = pd.to_numeric(
                baseline[feature],
                errors="coerce"
            ).dropna()

            if len(pre_values) == 0 or len(base_values) == 0:
                continue

            pre_mean = pre_values.mean()
            base_mean = base_values.mean()

            pre_std = pre_values.std()
            base_std = base_values.std()

            delta = pre_mean - base_mean

            # Pooled standard deviation
            pooled_std = np.sqrt(
                (
                    pre_values.var() +
                    base_values.var()
                ) / 2
            )

            if pooled_std > 0:
                standardized_shift = delta / pooled_std
            else:
                standardized_shift = 0.0

            # Relative change
            if abs(base_mean) > 1e-9:
                relative_change = (
                    delta / abs(base_mean)
                )
            else:
                relative_change = np.nan

            results.append({
                "event_timestamp": event_time,
                "hostname": host,
                "window_seconds": window_seconds,
                "feature": feature,

                "baseline_mean": base_mean,
                "pre_event_mean": pre_mean,

                "delta": delta,
                "relative_change": relative_change,

                "baseline_std": base_std,
                "pre_event_std": pre_std,

                "standardized_shift":
                    standardized_shift,

                "baseline_samples":
                    len(base_values),

                "pre_event_samples":
                    len(pre_values),
            })


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df = pd.DataFrame(results)

raw_output = REPORT_DIR / "pre_event_feature_shifts.csv"

results_df.to_csv(
    raw_output,
    index=False
)

print(f"\nRaw pre-event results saved:")
print(raw_output)

# ============================================================
# AGGREGATE ACROSS EVENTS
# ============================================================

if len(results_df) > 0:

    summary = (
        results_df
        .groupby(
            ["window_seconds", "feature"],
            as_index=False
        )
        .agg(
            mean_delta=("delta", "mean"),

            median_delta=("delta", "median"),

            mean_relative_change=(
                "relative_change",
                "mean"
            ),

            median_relative_change=(
                "relative_change",
                "median"
            ),

            mean_standardized_shift=(
                "standardized_shift",
                "mean"
            ),

            median_standardized_shift=(
                "standardized_shift",
                "median"
            ),

            abs_mean_standardized_shift=(
                "standardized_shift",
                lambda x: np.mean(np.abs(x))
            ),

            event_count=(
                "hostname",
                "count"
            )
        )
    )

    summary = summary.sort_values(
        [
            "window_seconds",
            "abs_mean_standardized_shift"
        ],
        ascending=[True, False]
    )

    summary_output = (
        REPORT_DIR /
        "pre_event_feature_summary.csv"
    )

    summary.to_csv(
        summary_output,
        index=False
    )

    print(
        f"\nAggregated feature analysis saved:"
    )

    print(summary_output)

    # --------------------------------------------------------
    # Print strongest signals for each window
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRONGEST PRE-EVENT SIGNALS")
    print("=" * 70)

    for window in WINDOWS:

        subset = summary[
            summary["window_seconds"] == window
        ].copy()

        subset = subset.sort_values(
            "abs_mean_standardized_shift",
            ascending=False
        )

        print(
            f"\n--- {window} second window ---"
        )

        print(
            subset[
                [
                    "feature",
                    "mean_delta",
                    "mean_relative_change",
                    "mean_standardized_shift",
                    "abs_mean_standardized_shift"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

# ============================================================
# SUMMARY FILE
# ============================================================

summary_text = []

summary_text.append(
    "SENTINEL STEP 1 DATASET FORENSICS\n"
)

summary_text.append(
    f"Rows: {len(df):,}\n"
)

summary_text.append(
    f"Columns: {len(df.columns)}\n"
)

summary_text.append(
    f"Hosts: {df['hostname'].nunique():,}\n"
)

summary_text.append(
    f"XID-31 records: {len(events):,}\n"
)

summary_text.append(
    f"Anomalous labels: "
    f"{int((df['label'] == 1).sum()):,}\n"
)

summary_text.append(
    f"Normal labels: "
    f"{int((df['label'] == 0).sum()):,}\n"
)

summary_text.append(
    "\nIMPORTANT:\n"
)

summary_text.append(
    "This analysis does NOT use label or XID as telemetry "
    "features. They are used only to identify known events "
    "for retrospective forensic analysis.\n"
)

with open(
    REPORT_DIR / "step1_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(summary_text))

print("\n" + "=" * 70)
print("STEP 1 FORENSICS COMPLETE")
print("=" * 70)

print(
    f"\nReports created in:\n{REPORT_DIR}"
)