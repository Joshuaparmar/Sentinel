from pathlib import Path
import pandas as pd


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
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = REPORT_DIR / "xid31_episodes.csv"


# ============================================================
# CONFIGURATION
# ============================================================

XID_VALUE = 31

# Expected telemetry sampling interval is 10 seconds.
# We allow a small tolerance so that minor timestamp irregularities
# do not incorrectly split one episode.
MAX_GAP_SECONDS = 15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 1A — XID-31 EPISODE DETECTION")
print("=" * 70)

print(f"\nLoading:\n{DATA_FILE}")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# PREPARE TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_ts = df["timestamp"].isna().sum()

if invalid_ts > 0:
    print(f"\nWARNING: {invalid_ts:,} invalid timestamps found.")
    df = df.dropna(subset=["timestamp"])

df = df.sort_values(
    ["hostname", "timestamp"]
).reset_index(drop=True)


# ============================================================
# IDENTIFY XID-31 ROWS
# ============================================================

xid_mask = df["xid"] == XID_VALUE

xid_df = df.loc[
    xid_mask,
    ["timestamp", "hostname", "xid", "label"]
].copy()

print("\n" + "-" * 70)
print("XID-31 OVERVIEW")
print("-" * 70)

print(f"XID-31 rows:       {len(xid_df):,}")
print(f"Affected hosts:    {xid_df['hostname'].nunique():,}")


# ============================================================
# BUILD EPISODES
# ============================================================

# Only XID-31 rows are considered here.
#
# Within each host:
#   - sort chronologically
#   - calculate time gap between consecutive XID-31 rows
#   - start a new episode if the gap exceeds MAX_GAP_SECONDS

xid_df = xid_df.sort_values(
    ["hostname", "timestamp"]
).reset_index(drop=True)

xid_df["previous_timestamp"] = (
    xid_df
    .groupby("hostname")["timestamp"]
    .shift(1)
)

xid_df["gap_seconds"] = (
    xid_df["timestamp"]
    - xid_df["previous_timestamp"]
).dt.total_seconds()


# A new episode occurs when:
#   1. It is the first XID row for that host, OR
#   2. The gap from the previous XID row exceeds the allowed gap.

xid_df["new_episode"] = (
    xid_df["previous_timestamp"].isna()
    | (xid_df["gap_seconds"] > MAX_GAP_SECONDS)
)


# Cumulative episode number within each host
xid_df["episode_number"] = (
    xid_df
    .groupby("hostname")["new_episode"]
    .cumsum()
)


# Create a globally unique episode ID
xid_df["episode_id"] = (
    xid_df["hostname"].astype(str)
    + "_episode_"
    + xid_df["episode_number"].astype(str)
)


# ============================================================
# SUMMARIZE EPISODES
# ============================================================

episodes = (
    xid_df
    .groupby(
        ["hostname", "episode_number", "episode_id"],
        as_index=False
    )
    .agg(
        episode_start=("timestamp", "min"),
        episode_end=("timestamp", "max"),
        sample_count=("timestamp", "size"),
        max_gap_within_episode=("gap_seconds", "max"),
        xid_values=("xid", "nunique"),
        label_values=("label", "nunique"),
    )
)


episodes["duration_seconds"] = (
    episodes["episode_end"]
    - episodes["episode_start"]
).dt.total_seconds()


episodes["duration_minutes"] = (
    episodes["duration_seconds"] / 60.0
)


# ============================================================
# HOST-LEVEL SUMMARY
# ============================================================

episodes_per_host = (
    episodes
    .groupby("hostname")
    .size()
    .reset_index(name="episode_count")
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "-" * 70)
print("EPISODE RESULTS")
print("-" * 70)

print(f"Distinct XID-31 episodes: {len(episodes):,}")
print(f"Hosts with episodes:       {episodes['hostname'].nunique():,}")

print("\nEpisodes per host:")
print(
    episodes_per_host["episode_count"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nEpisode sample-count distribution:")
print(
    episodes["sample_count"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nEpisode duration distribution (seconds):")
print(
    episodes["duration_seconds"]
    .describe()
    .to_string()
)


print("\nMaximum gap inside episodes (seconds):")
print(
    episodes["max_gap_within_episode"]
    .describe()
    .to_string()
)


# ============================================================
# CHECK FOR MULTIPLE EPISODES
# ============================================================

multi_episode_hosts = episodes_per_host[
    episodes_per_host["episode_count"] > 1
]

print("\n" + "-" * 70)
print("MULTIPLE-EPISODE HOST CHECK")
print("-" * 70)

if len(multi_episode_hosts) == 0:
    print("No host has multiple XID-31 episodes.")
else:
    print(
        f"{len(multi_episode_hosts)} hosts have multiple episodes."
    )
    print(
        multi_episode_hosts
        .sort_values("episode_count", ascending=False)
        .to_string(index=False)
    )


# ============================================================
# CHECK EPISODE STRUCTURE
# ============================================================

print("\n" + "-" * 70)
print("EPISODE STRUCTURE CHECK")
print("-" * 70)

print(
    f"All episodes have exactly 13 samples: "
    f"{(episodes['sample_count'] == 13).all()}"
)

print(
    f"All episodes have duration 120 seconds: "
    f"{(episodes['duration_seconds'] == 120).all()}"
)

print(
    f"All episodes contain only one XID value: "
    f"{(episodes['xid_values'] == 1).all()}"
)

print(
    f"All episodes contain only one label value: "
    f"{(episodes['label_values'] == 1).all()}"
)


# ============================================================
# SAVE EPISODES
# ============================================================

episodes = episodes.sort_values(
    ["episode_start", "hostname"]
)

episodes.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "-" * 70)
print("OUTPUT")
print("-" * 70)

print(f"Saved episode report to:")
print(OUTPUT_FILE)


# ============================================================
# SHOW FIRST EPISODES
# ============================================================

print("\nFirst 15 episodes:")
print(
    episodes[
        [
            "episode_id",
            "hostname",
            "episode_start",
            "episode_end",
            "sample_count",
            "duration_seconds",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("STEP 1A COMPLETE")
print("=" * 70)