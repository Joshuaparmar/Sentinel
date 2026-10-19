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

PRE_SECONDS = 300
POST_SECONDS = 300

CONTEXT_COLUMNS = [
    "allocation_id",
    "node_in_rack",
    "rack_id",
    "cabinet_id",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 1D — SYNCHRONIZED EVENT CONTEXT ANALYSIS")
print("=" * 70)

print("\nLoading telemetry data...")

# low_memory=False avoids the harmless mixed-type warning
df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Hosts: {df['hostname'].nunique()}")

print("\nLoading XID-31 episodes...")

episodes = pd.read_csv(
    EPISODE_FILE,
    low_memory=False
)

if "hostname" not in episodes.columns:
    raise RuntimeError(
        "xid31_episodes.csv does not contain hostname."
    )

affected_hosts = sorted(
    episodes["hostname"]
    .dropna()
    .unique()
)

all_hosts = sorted(
    df["hostname"]
    .dropna()
    .unique()
)

unaffected_hosts = sorted(
    set(all_hosts) - set(affected_hosts)
)

print(f"Affected hosts:    {len(affected_hosts)}")
print(f"Unaffected hosts:  {len(unaffected_hosts)}")

if len(affected_hosts) != 85:
    raise RuntimeError(
        f"Expected 85 affected hosts, "
        f"found {len(affected_hosts)}."
    )

if len(unaffected_hosts) != 15:
    raise RuntimeError(
        f"Expected 15 unaffected hosts, "
        f"found {len(unaffected_hosts)}."
    )


# ============================================================
# EVENT WINDOWS
# ============================================================

pre_start = (
    REFERENCE_TIME
    - pd.Timedelta(seconds=PRE_SECONDS)
)

post_end = (
    REFERENCE_TIME
    + pd.Timedelta(seconds=POST_SECONDS)
)

event_start = REFERENCE_TIME
event_end = (
    REFERENCE_TIME
    + pd.Timedelta(seconds=120)
)

print("\nEvent context window:")
print(f"  Pre-event:  {pre_start} -> {REFERENCE_TIME}")
print(f"  Event:      {event_start} -> {event_end}")
print(f"  Post-event: {REFERENCE_TIME} -> {post_end}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print("\nChecking context columns...")

for column in CONTEXT_COLUMNS:
    if column not in df.columns:
        print(
            f"WARNING: '{column}' is not present "
            f"in the dataset."
        )
    else:
        print(
            f"  {column}: available"
        )


# ============================================================
# HOST-LEVEL CONTEXT
# ============================================================

print("\n" + "=" * 70)
print("BUILDING HOST-LEVEL CONTEXT TABLE")
print("=" * 70)

# Select one representative row per host.
#
# We prefer a row exactly at the synchronized event time.
# If unavailable, use the closest row within the pre-event
# context window.

host_context_rows = []

for hostname in all_hosts:

    host_df = df[
        df["hostname"] == hostname
    ].copy()

    exact = host_df[
        host_df["timestamp"] == REFERENCE_TIME
    ]

    if len(exact) > 0:
        row = exact.iloc[0].copy()
        reference_source = "exact_event_timestamp"

    else:
        candidates = host_df[
            (host_df["timestamp"] >= pre_start)
            & (host_df["timestamp"] < REFERENCE_TIME)
        ].copy()

        if len(candidates) == 0:
            continue

        candidates["time_distance"] = (
            candidates["timestamp"]
            - REFERENCE_TIME
        ).abs()

        row = (
            candidates
            .sort_values("time_distance")
            .iloc[0]
            .copy()
        )

        reference_source = "nearest_pre_event"

    record = {
        "hostname": hostname,
        "group": (
            "affected"
            if hostname in affected_hosts
            else "unaffected"
        ),
        "context_reference_timestamp":
            row["timestamp"],
        "context_reference_source":
            reference_source,
    }

    for column in CONTEXT_COLUMNS:
        if column in df.columns:
            record[column] = row[column]

    host_context_rows.append(record)


host_context = pd.DataFrame(
    host_context_rows
)

host_context_output = (
    REPORT_DIR /
    "step1d_host_context.csv"
)

host_context.to_csv(
    host_context_output,
    index=False
)

print(
    f"Host context rows: {len(host_context):,}"
)

print(
    f"Saved:\n  {host_context_output}"
)


# ============================================================
# HOST CONTEXT DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("AFFECTED VS UNAFFECTED CONTEXT DISTRIBUTIONS")
print("=" * 70)

distribution_rows = []

for column in CONTEXT_COLUMNS:

    if column not in host_context.columns:
        continue

    temp = host_context[
        ["hostname", "group", column]
    ].copy()

    temp["context_column"] = column
    temp["context_value"] = (
        temp[column].astype(str)
    )

    distribution_rows.append(
        temp[
            [
                "hostname",
                "group",
                "context_column",
                "context_value",
            ]
        ]
    )


if distribution_rows:

    context_distribution = pd.concat(
        distribution_rows,
        ignore_index=True
    )

else:

    context_distribution = pd.DataFrame(
        columns=[
            "hostname",
            "group",
            "context_column",
            "context_value",
        ]
    )


context_distribution_output = (
    REPORT_DIR /
    "step1d_context_distribution.csv"
)

context_distribution.to_csv(
    context_distribution_output,
    index=False
)

print(
    f"Saved:\n  {context_distribution_output}"
)


# ============================================================
# CONTEXT GROUP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CONTEXT GROUP ANALYSIS")
print("=" * 70)

group_rows = []

for column in CONTEXT_COLUMNS:

    if column not in host_context.columns:
        continue

    temp = host_context[
        ["hostname", "group", column]
    ].copy()

    # Drop missing context values
    temp = temp[
        temp[column].notna()
    ]

    # Convert to string so mixed-type columns are handled safely
    temp["context_value"] = (
        temp[column].astype(str)
    )

    grouped = (
        temp
        .groupby("context_value", dropna=False)
    )

    for value, group_df in grouped:

        total_hosts = len(group_df)

        affected_count = (
            group_df["group"] == "affected"
        ).sum()

        unaffected_count = (
            group_df["group"] == "unaffected"
        ).sum()

        affected_fraction = (
            affected_count / total_hosts
            if total_hosts > 0
            else np.nan
        )

        group_rows.append({
            "context_column": column,
            "context_value": value,
            "total_hosts": total_hosts,
            "affected_hosts": affected_count,
            "unaffected_hosts": unaffected_count,
            "affected_fraction":
                affected_fraction,
            "affected_percent":
                affected_fraction * 100
                if not pd.isna(affected_fraction)
                else np.nan,
        })


context_groups = pd.DataFrame(
    group_rows
)

context_groups_output = (
    REPORT_DIR /
    "step1d_context_group_analysis.csv"
)

context_groups.to_csv(
    context_groups_output,
    index=False
)

print(
    f"Saved:\n  {context_groups_output}"
)


# ============================================================
# PRINT CONTEXT RESULTS
# ============================================================

for column in CONTEXT_COLUMNS:

    if column not in context_groups.columns:
        continue

    print(
        f"\n--- {column} ---"
    )

    temp = context_groups[
        context_groups["context_column"] == column
    ].copy()

    temp = temp.sort_values(
        [
            "affected_hosts",
            "total_hosts"
        ],
        ascending=False
    )

    if len(temp) == 0:
        print("No usable values.")
        continue

    for _, row in temp.head(20).iterrows():

        print(
            f"{str(row['context_value']):30s} "
            f"| total={int(row['total_hosts']):3d} "
            f"| affected={int(row['affected_hosts']):3d} "
            f"| unaffected={int(row['unaffected_hosts']):3d} "
            f"| affected%="
            f"{row['affected_percent']:.1f}%"
        )


# ============================================================
# JOB / ALLOCATION OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("AFFECTED / UNAFFECTED ALLOCATION OVERLAP")
print("=" * 70)

if "allocation_id" in host_context.columns:

    allocation_table = (
        host_context[
            ["hostname", "group", "allocation_id"]
        ]
        .copy()
    )

    allocation_table = allocation_table[
        allocation_table["allocation_id"].notna()
    ]

    allocation_table[
        "allocation_value"
    ] = allocation_table[
        "allocation_id"
    ].astype(str)

    allocation_groups = (
        allocation_table
        .groupby("allocation_value")
        .agg(
            total_hosts=("hostname", "count"),
            affected_hosts=(
                "group",
                lambda x: (x == "affected").sum()
            ),
            unaffected_hosts=(
                "group",
                lambda x: (x == "unaffected").sum()
            ),
        )
        .reset_index()
    )

    allocation_groups["affected_percent"] = (
        allocation_groups["affected_hosts"]
        / allocation_groups["total_hosts"]
        * 100
    )

    allocation_groups = (
        allocation_groups
        .sort_values(
            "total_hosts",
            ascending=False
        )
    )

    allocation_output = (
        REPORT_DIR /
        "step1d_allocation_analysis.csv"
    )

    allocation_groups.to_csv(
        allocation_output,
        index=False
    )

    print(
        f"Saved:\n  {allocation_output}"
    )

    print("\nAllocations containing multiple hosts:")

    multi = allocation_groups[
        allocation_groups["total_hosts"] > 1
    ]

    for _, row in multi.head(20).iterrows():

        print(
            f"{str(row['allocation_value']):30s} "
            f"| hosts={int(row['total_hosts']):3d} "
            f"| affected={int(row['affected_hosts']):3d} "
            f"| unaffected={int(row['unaffected_hosts']):3d} "
            f"| affected%="
            f"{row['affected_percent']:.1f}%"
        )

else:

    print(
        "allocation_id unavailable."
    )


# ============================================================
# TOPOLOGY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TOPOLOGY SUMMARY")
print("=" * 70)

topology_columns = [
    column
    for column in [
        "node_in_rack",
        "rack_id",
        "cabinet_id",
    ]
    if column in host_context.columns
]

for column in topology_columns:

    temp = host_context[
        ["hostname", "group", column]
    ].copy()

    temp = temp[
        temp[column].notna()
    ]

    temp["value"] = (
        temp[column].astype(str)
    )

    topology = (
        temp
        .groupby("value")
        .agg(
            total_hosts=("hostname", "count"),
            affected_hosts=(
                "group",
                lambda x: (x == "affected").sum()
            ),
            unaffected_hosts=(
                "group",
                lambda x: (x == "unaffected").sum()
            ),
        )
        .reset_index()
    )

    topology["affected_percent"] = (
        topology["affected_hosts"]
        / topology["total_hosts"]
        * 100
    )

    print(f"\n--- {column} ---")

    for _, row in topology.iterrows():

        print(
            f"{str(row['value']):20s} "
            f"| total={int(row['total_hosts']):3d} "
            f"| affected={int(row['affected_hosts']):3d} "
            f"| unaffected={int(row['unaffected_hosts']):3d} "
            f"| affected%="
            f"{row['affected_percent']:.1f}%"
        )


# ============================================================
# TEMPORAL COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL COVERAGE CHECK")
print("=" * 70)

pre_df = df[
    (df["timestamp"] >= pre_start)
    & (df["timestamp"] < REFERENCE_TIME)
]

event_df = df[
    (df["timestamp"] >= event_start)
    & (df["timestamp"] <= event_end)
]

post_df = df[
    (df["timestamp"] > event_end)
    & (df["timestamp"] <= post_end)
]

print(
    f"Pre-event rows:  {len(pre_df):,}"
)

print(
    f"Event rows:      {len(event_df):,}"
)

print(
    f"Post-event rows: {len(post_df):,}"
)

print(
    f"Pre-event hosts: "
    f"{pre_df['hostname'].nunique()}"
)

print(
    f"Event hosts:     "
    f"{event_df['hostname'].nunique()}"
)

print(
    f"Post-event hosts: "
    f"{post_df['hostname'].nunique()}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 1D COMPLETE")
print("=" * 70)

print(
    "\nReports generated:"
)

print(
    "  reports/step1d_host_context.csv"
)

print(
    "  reports/step1d_context_distribution.csv"
)

print(
    "  reports/step1d_context_group_analysis.csv"
)

if "allocation_id" in host_context.columns:
    print(
        "  reports/step1d_allocation_analysis.csv"
    )