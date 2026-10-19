from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# STEP 1B — EPISODE-LEVEL PRE-EVENT FORENSICS
# ============================================================
#
# Purpose:
#   1. Treat each XID-31 episode as ONE event.
#   2. Analyze ONLY telemetry before the first XID-31 timestamp.
#   3. Compare pre-event telemetry against an earlier normal baseline.
#   4. Measure consistency of precursor behaviour across 85 episodes.
#   5. Analyze whether XID-31 episodes are synchronized across hosts.
#
# IMPORTANT:
#   - xid and label are NEVER used as telemetry features.
#   - No XID-31 row is allowed inside a pre-event window.
#   - This is forensic analysis, NOT model training.
# ============================================================


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

EPISODE_FILE = (
    BASE_DIR
    / "reports"
    / "xid31_episodes.csv"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Analysis windows, in seconds.
WINDOWS = [30, 60, 120, 300]

# The baseline is the immediately preceding window before
# the pre-event window.
#
# Example for W = 60:
#
# baseline:   T-120 -> T-60
# pre-event:  T-60  -> T
# episode:    T     -> ...
#
# This keeps the baseline reasonably close to the event while
# ensuring that it occurs before the precursor window.
#
# We use half-open intervals:
#   baseline  = [T-2W, T-W)
#   pre-event = [T-W, T)
#
# Therefore the episode itself is NEVER included.
#
# Minimum observations required for a reliable window.
# At 10-second sampling:
#   30 sec  -> ~3 observations
#   60 sec  -> ~6 observations
#   120 sec -> ~12 observations
#   300 sec -> ~30 observations
#
# We require at least 2 observations because standard deviation
# cannot be meaningfully computed with fewer than 2.
MIN_OBSERVATIONS = 2

# A small tolerance when identifying timestamps that belong to
# the episode. This is only used for validation.
XID_VALUE = 31


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "p0_gpu0_power",
    "p0_gpu1_power",
    "p0_gpu2_power",
    "p1_gpu0_power",
    "p1_gpu1_power",
    "p1_gpu2_power",
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

def safe_relative_change(baseline_mean, pre_mean):
    """
    Relative change:

        (pre - baseline) / |baseline| * 100

    Returns NaN when baseline is zero or unavailable.
    """
    if pd.isna(baseline_mean) or pd.isna(pre_mean):
        return np.nan

    if abs(baseline_mean) < 1e-12:
        return np.nan

    return (
        (pre_mean - baseline_mean)
        / abs(baseline_mean)
        * 100.0
    )


def safe_standardized_shift(
    baseline_mean,
    pre_mean,
    baseline_std
):
    """
    Standardized shift:

        (pre_mean - baseline_mean) / baseline_std

    This tells us how large the pre-event change is relative
    to normal variation in the preceding baseline.

    We intentionally use BASELINE standard deviation only.
    """
    if (
        pd.isna(baseline_mean)
        or pd.isna(pre_mean)
        or pd.isna(baseline_std)
    ):
        return np.nan

    if baseline_std < 1e-12:
        return np.nan

    return (
        (pre_mean - baseline_mean)
        / baseline_std
    )


def summarize_window(
    host_df,
    event_time,
    feature,
    window_seconds
):
    """
    Calculate baseline and pre-event statistics for ONE
    host, ONE event, ONE feature and ONE window.

    Intervals:

        baseline:
            [event - 2W, event - W)

        pre-event:
            [event - W, event)

    This guarantees that the event itself and all XID-31
    observations are excluded.
    """

    baseline_start = (
        event_time
        - pd.Timedelta(seconds=2 * window_seconds)
    )

    baseline_end = (
        event_time
        - pd.Timedelta(seconds=window_seconds)
    )

    pre_start = baseline_end
    pre_end = event_time

    baseline_mask = (
        (host_df["timestamp"] >= baseline_start)
        & (host_df["timestamp"] < baseline_end)
    )

    pre_mask = (
        (host_df["timestamp"] >= pre_start)
        & (host_df["timestamp"] < pre_end)
    )

    baseline_values = pd.to_numeric(
        host_df.loc[baseline_mask, feature],
        errors="coerce"
    ).dropna()

    pre_values = pd.to_numeric(
        host_df.loc[pre_mask, feature],
        errors="coerce"
    ).dropna()

    baseline_count = len(baseline_values)
    pre_count = len(pre_values)

    if baseline_count < MIN_OBSERVATIONS:
        return None

    if pre_count < MIN_OBSERVATIONS:
        return None

    baseline_mean = baseline_values.mean()
    baseline_std = baseline_values.std(ddof=1)

    pre_mean = pre_values.mean()
    pre_std = pre_values.std(ddof=1)

    delta = pre_mean - baseline_mean

    relative_change = safe_relative_change(
        baseline_mean,
        pre_mean
    )

    standardized_shift = safe_standardized_shift(
        baseline_mean,
        pre_mean,
        baseline_std
    )

    return {
        "baseline_count": baseline_count,
        "pre_count": pre_count,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "pre_mean": pre_mean,
        "pre_std": pre_std,
        "delta": delta,
        "relative_change_pct": relative_change,
        "standardized_shift": standardized_shift,
    }


# ============================================================
# START
# ============================================================

print("=" * 75)
print("STEP 1B — EPISODE-LEVEL PRE-EVENT FORENSICS")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading telemetry dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamps = df["timestamp"].isna().sum()

if invalid_timestamps > 0:
    print(
        f"\nWARNING: dropping "
        f"{invalid_timestamps:,} invalid timestamps."
    )

    df = df.dropna(
        subset=["timestamp"]
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for feature in NUMERIC_FEATURES:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["hostname", "timestamp"]
).reset_index(drop=True)


# ============================================================
# LOAD EPISODES
# ============================================================

print("\nLoading episode definitions...")

episodes = pd.read_csv(
    EPISODE_FILE
)

episodes["episode_start"] = pd.to_datetime(
    episodes["episode_start"],
    errors="coerce"
)

episodes["episode_end"] = pd.to_datetime(
    episodes["episode_end"],
    errors="coerce"
)

print(f"Episodes loaded: {len(episodes):,}")


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "-" * 75)
print("EPISODE VALIDATION")
print("-" * 75)

print(
    f"Unique episode IDs: "
    f"{episodes['episode_id'].nunique():,}"
)

print(
    f"Unique hosts: "
    f"{episodes['hostname'].nunique():,}"
)

print(
    f"Unique episode start times: "
    f"{episodes['episode_start'].nunique():,}"
)


# ============================================================
# CHECK XID STATUS BEFORE EVERY EPISODE
# ============================================================
#
# This is a critical leakage check.
#
# For every episode, the entire pre-event analysis period must
# contain NO XID-31 records.
#
# We check the largest analysis window:
#
#     baseline + pre-event = 10 minutes
#
#     [T - 600 sec, T)
#
# ============================================================

print("\nChecking for XID-31 leakage into pre-event periods...")

leakage_records = []

for _, episode in episodes.iterrows():

    host = episode["hostname"]
    event_time = episode["episode_start"]

    host_df = df[
        df["hostname"] == host
    ]

    check_start = (
        event_time
        - pd.Timedelta(seconds=600)
    )

    check_end = event_time

    mask = (
        (host_df["timestamp"] >= check_start)
        & (host_df["timestamp"] < check_end)
        & (host_df["xid"] == XID_VALUE)
    )

    leakage_count = mask.sum()

    if leakage_count > 0:

        leakage_records.append({
            "episode_id": episode["episode_id"],
            "hostname": host,
            "episode_start": event_time,
            "xid_records_in_pre_event": leakage_count,
        })


if leakage_records:

    leakage_df = pd.DataFrame(
        leakage_records
    )

    print(
        f"WARNING: {len(leakage_df)} episodes contain "
        f"XID-31 records in the 10-minute pre-event period."
    )

    leakage_df.to_csv(
        REPORT_DIR / "step1b_pre_event_leakage.csv",
        index=False
    )

else:

    print(
        "PASS: No XID-31 records detected in the "
        "10-minute pre-event analysis period."
    )


# ============================================================
# SYNCHRONIZATION ANALYSIS
# ============================================================

print("\n" + "-" * 75)
print("EPISODE SYNCHRONIZATION")
print("-" * 75)

start_counts = (
    episodes["episode_start"]
    .value_counts()
    .sort_index()
)

print("\nEpisode starts by timestamp:")

print(
    start_counts
    .to_string()
)


# Percentage of all episodes occurring at each timestamp
sync_summary = (
    start_counts
    .rename("episode_count")
    .reset_index()
    .rename(columns={"index": "episode_start"})
)

sync_summary["percentage_of_episodes"] = (
    sync_summary["episode_count"]
    / len(episodes)
    * 100
)

sync_summary.to_csv(
    REPORT_DIR / "xid31_episode_synchronization.csv",
    index=False
)


# ============================================================
# SYNCHRONIZATION WINDOW ANALYSIS
# ============================================================
#
# Episodes can be considered synchronized if they begin within
# a small time window of each other.
#
# We calculate how many episodes begin within:
#
#   10 sec
#   30 sec
#   60 sec
#   120 sec
#
# of the earliest episode.
#
# ============================================================

earliest_start = episodes["episode_start"].min()

print(
    f"\nEarliest episode start: {earliest_start}"
)

for seconds in [10, 30, 60, 120, 300]:

    cutoff = (
        earliest_start
        + pd.Timedelta(seconds=seconds)
    )

    count = (
        episodes["episode_start"] <= cutoff
    ).sum()

    percentage = (
        count / len(episodes) * 100
    )

    print(
        f"Episodes beginning within "
        f"{seconds:>3} sec of earliest: "
        f"{count:>3} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# PREPARE HOST GROUPS
# ============================================================

affected_hosts = set(
    episodes["hostname"]
)

all_hosts = set(
    df["hostname"].unique()
)

unaffected_hosts = sorted(
    all_hosts - affected_hosts
)

print("\n" + "-" * 75)
print("HOST GROUPS")
print("-" * 75)

print(
    f"Total hosts:       {len(all_hosts)}"
)

print(
    f"Affected hosts:    {len(affected_hosts)}"
)

print(
    f"Unaffected hosts:  {len(unaffected_hosts)}"
)


# ============================================================
# EVENT-LEVEL FEATURE ANALYSIS
# ============================================================

print("\n" + "-" * 75)
print("PRE-EVENT FEATURE ANALYSIS")
print("-" * 75)

print(
    "\nAnalyzing "
    f"{len(episodes)} episodes × "
    f"{len(NUMERIC_FEATURES)} features × "
    f"{len(WINDOWS)} windows..."
)

results = []

# Create a lookup so we do not repeatedly filter the complete
# dataset for every feature.

host_data = {
    host: host_df.reset_index(drop=True)
    for host, host_df
    in df.groupby("hostname", sort=False)
}


for episode_index, episode in episodes.iterrows():

    host = episode["hostname"]
    event_time = episode["episode_start"]
    episode_id = episode["episode_id"]

    host_df = host_data[host]

    for window_seconds in WINDOWS:

        for feature in NUMERIC_FEATURES:

            summary = summarize_window(
                host_df=host_df,
                event_time=event_time,
                feature=feature,
                window_seconds=window_seconds
            )

            if summary is None:
                continue

            results.append({
                "episode_id": episode_id,
                "hostname": host,
                "episode_start": event_time,
                "window_seconds": window_seconds,
                "feature": feature,
                **summary,
            })

    if (episode_index + 1) % 10 == 0:
        print(
            f"Processed "
            f"{episode_index + 1}/{len(episodes)} episodes..."
        )


results_df = pd.DataFrame(results)


# ============================================================
# SAVE RAW EVENT-LEVEL RESULTS
# ============================================================

raw_output = (
    REPORT_DIR
    / "pre_event_episode_feature_shifts.csv"
)

results_df.to_csv(
    raw_output,
    index=False
)

print(
    f"\nSaved event-level results:\n"
    f"{raw_output}"
)


# ============================================================
# AGGREGATED FEATURE SUMMARY
# ============================================================

print("\nCreating aggregate feature summary...")


def fraction_positive(series):
    series = series.dropna()

    if len(series) == 0:
        return np.nan

    return (
        (series > 0).mean() * 100
    )


def fraction_negative(series):
    series = series.dropna()

    if len(series) == 0:
        return np.nan

    return (
        (series < 0).mean() * 100
    )


def fraction_abs_gt_1(series):
    series = series.dropna()

    if len(series) == 0:
        return np.nan

    return (
        (series.abs() > 1).mean() * 100
    )


def fraction_abs_gt_2(series):
    series = series.dropna()

    if len(series) == 0:
        return np.nan

    return (
        (series.abs() > 2).mean() * 100
    )


summary = (
    results_df
    .groupby(
        ["window_seconds", "feature"],
        as_index=False
    )
    .agg(
        episodes_analyzed=(
            "episode_id",
            "nunique"
        ),

        mean_delta=(
            "delta",
            "mean"
        ),

        median_delta=(
            "delta",
            "median"
        ),

        mean_relative_change_pct=(
            "relative_change_pct",
            "mean"
        ),

        median_relative_change_pct=(
            "relative_change_pct",
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

        mean_abs_standardized_shift=(
            "standardized_shift",
            lambda x: x.abs().mean()
        ),

        median_abs_standardized_shift=(
            "standardized_shift",
            lambda x: x.abs().median()
        ),

        positive_change_pct=(
            "delta",
            fraction_positive
        ),

        negative_change_pct=(
            "delta",
            fraction_negative
        ),

        abs_shift_gt_1_pct=(
            "standardized_shift",
            fraction_abs_gt_1
        ),

        abs_shift_gt_2_pct=(
            "standardized_shift",
            fraction_abs_gt_2
        ),
    )
)


summary_output = (
    REPORT_DIR
    / "pre_event_episode_feature_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

print(
    f"Saved aggregate summary:\n"
    f"{summary_output}"
)


# ============================================================
# TOP FEATURES BY WINDOW
# ============================================================

print("\n" + "-" * 75)
print("TOP PRE-EVENT FEATURES")
print("-" * 75)

for window_seconds in WINDOWS:

    window_summary = summary[
        summary["window_seconds"] == window_seconds
    ].copy()

    window_summary = window_summary.sort_values(
        "mean_abs_standardized_shift",
        ascending=False
    )

    print(
        f"\nWINDOW: {window_seconds} seconds"
    )

    print(
        window_summary[
            [
                "feature",
                "episodes_analyzed",
                "mean_delta",
                "mean_relative_change_pct",
                "mean_standardized_shift",
                "mean_abs_standardized_shift",
                "abs_shift_gt_1_pct",
                "abs_shift_gt_2_pct",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# CONSISTENCY SUMMARY
# ============================================================
#
# This identifies features that:
#
#   1. have a large average standardized shift
#   2. behave consistently across episodes
#
# A large mean shift alone is not sufficient.
#
# ============================================================

consistency_rows = []

for window_seconds in WINDOWS:

    window_df = results_df[
        results_df["window_seconds"]
        == window_seconds
    ]

    grouped = (
        window_df
        .groupby("feature")
    )

    for feature, feature_df in grouped:

        shifts = (
            feature_df["standardized_shift"]
            .dropna()
        )

        if len(shifts) == 0:
            continue

        consistency_rows.append({
            "window_seconds": window_seconds,
            "feature": feature,
            "episodes": len(shifts),
            "mean_abs_shift": shifts.abs().mean(),
            "median_abs_shift": shifts.abs().median(),
            "fraction_abs_shift_gt_1": (
                (shifts.abs() > 1).mean()
            ),
            "fraction_abs_shift_gt_2": (
                (shifts.abs() > 2).mean()
            ),
            "positive_fraction": (
                (shifts > 0).mean()
            ),
            "negative_fraction": (
                (shifts < 0).mean()
            ),
        })


consistency_df = pd.DataFrame(
    consistency_rows
)

consistency_output = (
    REPORT_DIR
    / "pre_event_feature_consistency.csv"
)

consistency_df.to_csv(
    consistency_output,
    index=False
)

print(
    f"\nSaved consistency analysis:\n"
    f"{consistency_output}"
)


# ============================================================
# CHECK DATA COVERAGE
# ============================================================

print("\n" + "-" * 75)
print("WINDOW DATA COVERAGE")
print("-" * 75)

coverage = (
    results_df
    .groupby("window_seconds")
    .agg(
        feature_measurements=("feature", "count"),
        unique_episodes=("episode_id", "nunique"),
        unique_features=("feature", "nunique"),
    )
    .reset_index()
)

print(
    coverage.to_string(index=False)
)

coverage.to_csv(
    REPORT_DIR / "step1b_window_coverage.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 1B COMPLETE")
print("=" * 75)

print(
    "\nGenerated reports:"
)

print(
    f"  1. {REPORT_DIR / 'xid31_episode_synchronization.csv'}"
)

print(
    f"  2. {REPORT_DIR / 'pre_event_episode_feature_shifts.csv'}"
)

print(
    f"  3. {REPORT_DIR / 'pre_event_episode_feature_summary.csv'}"
)

print(
    f"  4. {REPORT_DIR / 'pre_event_feature_consistency.csv'}"
)

print(
    f"  5. {REPORT_DIR / 'step1b_window_coverage.csv'}"
)

if leakage_records:
    print(
        f"  6. {REPORT_DIR / 'step1b_pre_event_leakage.csv'}"
    )

print("\nNext step:")
print(
    "Review synchronization + precursor consistency "
    "before making any ML architecture decisions."
)