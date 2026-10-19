from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "ClusterWise_mini" / "olcf_task"
REPORT_DIR = BASE_DIR / "reports"

DATA_FILE = DATA_DIR / "test_anomaly.csv"
EPISODE_FILE = REPORT_DIR / "xid31_episodes.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_TIME = pd.Timestamp("2020-01-19 02:07:00")

WINDOWS = [30, 60, 120, 300]

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_relative_change(pre_mean, baseline_mean):
    """Calculate percentage change safely."""
    if pd.isna(baseline_mean) or baseline_mean == 0:
        return np.nan

    return ((pre_mean - baseline_mean) / abs(baseline_mean)) * 100.0


def calculate_host_shift(
    host_df,
    reference_time,
    feature,
    window_seconds
):
    """
    Calculate baseline -> pre-event shift for one host.

    Baseline:
        [T - 2W, T - W)

    Pre-event:
        [T - W, T)

    Only historical observations are used.
    """

    baseline_start = reference_time - pd.Timedelta(
        seconds=2 * window_seconds
    )

    baseline_end = reference_time - pd.Timedelta(
        seconds=window_seconds
    )

    pre_start = baseline_end
    pre_end = reference_time

    baseline = host_df[
        (host_df["timestamp"] >= baseline_start)
        & (host_df["timestamp"] < baseline_end)
    ][feature]

    pre_event = host_df[
        (host_df["timestamp"] >= pre_start)
        & (host_df["timestamp"] < pre_end)
    ][feature]

    baseline_mean = baseline.mean()
    pre_mean = pre_event.mean()

    baseline_std = baseline.std()

    delta = pre_mean - baseline_mean

    relative_change = safe_relative_change(
        pre_mean,
        baseline_mean
    )

    if pd.isna(baseline_std) or baseline_std == 0:
        standardized_shift = np.nan
    else:
        standardized_shift = delta / baseline_std

    return {
        "baseline_mean": baseline_mean,
        "pre_mean": pre_mean,
        "delta": delta,
        "relative_change_pct": relative_change,
        "baseline_std": baseline_std,
        "standardized_shift": standardized_shift,
        "baseline_samples": len(baseline),
        "pre_samples": len(pre_event),
    }


def calculate_effect_size(affected, unaffected):
    """
    Descriptive standardized difference between affected
    and unaffected host-level shifts.

    Uses pooled standard deviation.
    """

    affected = pd.Series(affected).dropna()
    unaffected = pd.Series(unaffected).dropna()

    if len(affected) < 2 or len(unaffected) < 2:
        return np.nan

    var_a = affected.var(ddof=1)
    var_u = unaffected.var(ddof=1)

    pooled_var = (
        ((len(affected) - 1) * var_a)
        + ((len(unaffected) - 1) * var_u)
    ) / (
        len(affected) + len(unaffected) - 2
    )

    if pooled_var <= 0:
        return np.nan

    pooled_std = np.sqrt(pooled_var)

    return (
        affected.mean() - unaffected.mean()
    ) / pooled_std


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 1C — AFFECTED VS UNAFFECTED MATCHED-TIME ANALYSIS")
print("=" * 70)

print("\nLoading telemetry data...")

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Hosts: {df['hostname'].nunique()}")

print("\nLoading XID-31 episodes...")

episodes = pd.read_csv(EPISODE_FILE)

print("Episode file columns:")
print(list(episodes.columns))

# We only need hostname here.
# Step 1B already established that all 85 episodes
# start at the same reference timestamp:
#
# 2020-01-19 02:07:00

if "hostname" not in episodes.columns:
    raise RuntimeError(
        "The episode file does not contain a 'hostname' column."
    )

affected_hosts = sorted(
    episodes["hostname"].dropna().unique()
)

all_hosts = sorted(
    df["hostname"].dropna().unique()
)

unaffected_hosts = sorted(
    set(all_hosts) - set(affected_hosts)
)

print(f"Affected hosts:    {len(affected_hosts)}")
print(f"Unaffected hosts:  {len(unaffected_hosts)}")

print("\nReference time:")
print(f"  {REFERENCE_TIME}")


# ============================================================
# SANITY CHECK
# ============================================================

if len(affected_hosts) != 85:
    raise RuntimeError(
        f"Expected 85 affected hosts, "
        f"but found {len(affected_hosts)}."
    )

if len(unaffected_hosts) != 15:
    raise RuntimeError(
        f"Expected 15 unaffected hosts, "
        f"but found {len(unaffected_hosts)}."
    )

print(
    "\nPASS: 85 affected hosts and "
    "15 unaffected hosts identified."
)

print(
    "Using synchronized reference time: "
    f"{REFERENCE_TIME}"
)

print(
    "\nUsing synchronized episode reference time:"
)

print(
    f"  {REFERENCE_TIME}"
)

# ============================================================
# HOST-LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING HOST-LEVEL PRE-EVENT SHIFTS")
print("=" * 70)

results = []

grouped = {
    hostname: host_df.sort_values("timestamp").copy()
    for hostname, host_df in df.groupby("hostname")
}

total_combinations = (
    len(all_hosts)
    * len(BASE_FEATURES)
    * len(WINDOWS)
)

completed = 0

for hostname in all_hosts:

    host_df = grouped[hostname]

    if hostname in affected_hosts:
        group = "affected"
    else:
        group = "unaffected"

    for window in WINDOWS:

        for feature in BASE_FEATURES:

            if feature not in host_df.columns:
                continue

            stats = calculate_host_shift(
                host_df=host_df,
                reference_time=REFERENCE_TIME,
                feature=feature,
                window_seconds=window,
            )

            results.append({
                "hostname": hostname,
                "group": group,
                "reference_time": REFERENCE_TIME,
                "window_seconds": window,
                "feature": feature,
                **stats,
            })

            completed += 1

print(
    f"\nCompleted {completed:,} "
    f"host-feature-window measurements."
)


# ============================================================
# SAVE HOST-LEVEL RESULTS
# ============================================================

host_results = pd.DataFrame(results)

host_output = (
    REPORT_DIR /
    "step1c_host_level_shifts.csv"
)

host_results.to_csv(
    host_output,
    index=False
)

print(f"\nSaved:")
print(f"  {host_output}")


# ============================================================
# GROUP COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("AFFECTED VS UNAFFECTED COMPARISON")
print("=" * 70)

summary_rows = []

for window in WINDOWS:

    window_df = host_results[
        host_results["window_seconds"] == window
    ]

    for feature in BASE_FEATURES:

        feature_df = window_df[
            window_df["feature"] == feature
        ]

        affected = feature_df[
            feature_df["group"] == "affected"
        ]

        unaffected = feature_df[
            feature_df["group"] == "unaffected"
        ]

        affected_delta = affected["delta"].dropna()
        unaffected_delta = unaffected["delta"].dropna()

        affected_std_shift = (
            affected["standardized_shift"]
            .dropna()
        )

        unaffected_std_shift = (
            unaffected["standardized_shift"]
            .dropna()
        )

        row = {
            "window_seconds": window,
            "feature": feature,

            "affected_hosts": len(affected_delta),
            "unaffected_hosts": len(unaffected_delta),

            "affected_mean_delta":
                affected_delta.mean(),

            "unaffected_mean_delta":
                unaffected_delta.mean(),

            "delta_difference":
                affected_delta.mean()
                - unaffected_delta.mean(),

            "affected_median_delta":
                affected_delta.median(),

            "unaffected_median_delta":
                unaffected_delta.median(),

            "affected_mean_relative_pct":
                affected["relative_change_pct"].mean(),

            "unaffected_mean_relative_pct":
                unaffected["relative_change_pct"].mean(),

            "affected_mean_std_shift":
                affected_std_shift.mean(),

            "unaffected_mean_std_shift":
                unaffected_std_shift.mean(),

            "std_shift_difference":
                affected_std_shift.mean()
                - unaffected_std_shift.mean(),

            "effect_size":
                calculate_effect_size(
                    affected_delta,
                    unaffected_delta
                ),

            "affected_pct_positive_delta":
                (
                    (affected_delta > 0).mean()
                    * 100
                ),

            "unaffected_pct_positive_delta":
                (
                    (unaffected_delta > 0).mean()
                    * 100
                ),

            "affected_pct_abs_std_gt_1":
                (
                    (affected_std_shift.abs() > 1).mean()
                    * 100
                ),

            "unaffected_pct_abs_std_gt_1":
                (
                    (unaffected_std_shift.abs() > 1).mean()
                    * 100
                ),

            "affected_pct_abs_std_gt_2":
                (
                    (affected_std_shift.abs() > 2).mean()
                    * 100
                ),

            "unaffected_pct_abs_std_gt_2":
                (
                    (unaffected_std_shift.abs() > 2).mean()
                    * 100
                ),
        }

        summary_rows.append(row)


summary = pd.DataFrame(summary_rows)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_output = (
    REPORT_DIR /
    "step1c_affected_vs_unaffected_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

print(f"\nSaved:")
print(f"  {summary_output}")


# ============================================================
# TOP DIFFERENCES
# ============================================================

print("\n" + "=" * 70)
print("TOP AFFECTED VS UNAFFECTED DIFFERENCES")
print("=" * 70)

for window in WINDOWS:

    print(
        f"\n--- {window}-second window ---"
    )

    window_summary = summary[
        summary["window_seconds"] == window
    ].copy()

    window_summary["abs_effect"] = (
        window_summary["effect_size"].abs()
    )

    top = (
        window_summary
        .sort_values(
            "abs_effect",
            ascending=False
        )
        .head(10)
    )

    for _, row in top.iterrows():

        print(
            f"{row['feature']:20s} "
            f"| affected Δ={row['affected_mean_delta']:+.4f} "
            f"| unaffected Δ={row['unaffected_mean_delta']:+.4f} "
            f"| difference={row['delta_difference']:+.4f} "
            f"| effect={row['effect_size']:+.3f}"
        )


# ============================================================
# STANDARDIZED SHIFT COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TOP STANDARDIZED-SHIFT DIFFERENCES")
print("=" * 70)

for window in WINDOWS:

    print(
        f"\n--- {window}-second window ---"
    )

    window_summary = summary[
        summary["window_seconds"] == window
    ].copy()

    window_summary["abs_std_difference"] = (
        window_summary["std_shift_difference"].abs()
    )

    top = (
        window_summary
        .sort_values(
            "abs_std_difference",
            ascending=False
        )
        .head(10)
    )

    for _, row in top.iterrows():

        print(
            f"{row['feature']:20s} "
            f"| affected z={row['affected_mean_std_shift']:+.3f} "
            f"| unaffected z={row['unaffected_mean_std_shift']:+.3f} "
            f"| difference={row['std_shift_difference']:+.3f}"
        )


# ============================================================
# CONSISTENCY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CONSISTENCY: |STANDARDIZED SHIFT| > 1")
print("=" * 70)

for window in WINDOWS:

    window_summary = summary[
        summary["window_seconds"] == window
    ]

    window_summary = window_summary.copy()

    window_summary["consistency_gap"] = (
        window_summary["affected_pct_abs_std_gt_1"]
        - window_summary["unaffected_pct_abs_std_gt_1"]
    )

    top = (
        window_summary
        .sort_values(
            "consistency_gap",
            ascending=False
        )
        .head(10)
    )

    print(
        f"\n--- {window}-second window ---"
    )

    for _, row in top.iterrows():

        print(
            f"{row['feature']:20s} "
            f"| affected="
            f"{row['affected_pct_abs_std_gt_1']:.1f}% "
            f"| unaffected="
            f"{row['unaffected_pct_abs_std_gt_1']:.1f}% "
            f"| gap="
            f"{row['consistency_gap']:+.1f} pp"
        )


# ============================================================
# FINAL SANITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("SANITY CHECKS")
print("=" * 70)

print(
    f"Expected affected hosts:    85"
)
print(
    f"Observed affected hosts:    "
    f"{len(affected_hosts)}"
)

print(
    f"Expected unaffected hosts:  15"
)
print(
    f"Observed unaffected hosts:  "
    f"{len(unaffected_hosts)}"
)

print(
    f"Reference timestamp:        "
    f"{REFERENCE_TIME}"
)

print(
    "\nNo XID or label values were used "
    "in the telemetry shift calculations."
)

print("\nStep 1C complete.")