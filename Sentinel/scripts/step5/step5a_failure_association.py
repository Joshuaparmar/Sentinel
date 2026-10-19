import os
import json
import numpy as np
import pandas as pd


# ============================================================
# SENTINEL - STEP 5A
# FAILURE ASSOCIATION ANALYSIS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
STEP4_DIR = os.path.join(REPORTS_DIR, "step4")
OUTPUT_DIR = os.path.join(REPORTS_DIR, "step5")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Event definition
FAILURE_XID = 31

# The observed XID-31 event in this dataset contains
# 13 samples at 10-second intervals.
EXPECTED_EVENT_SAMPLES = 13

# Pre-event windows
PRE_EVENT_WINDOWS = {
    "60s": 60,
    "120s": 120,
    "300s": 300,
    "600s": 600,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(filename, directory):
    path = os.path.join(directory, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    df = pd.read_csv(path)
    return df


def parse_datetime(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def safe_mean(series):
    if len(series) == 0:
        return np.nan
    return float(series.mean())


def safe_max(series):
    if len(series) == 0:
        return np.nan
    return float(series.max())


def safe_min(series):
    if len(series) == 0:
        return np.nan
    return float(series.min())


def first_timestamp(df, time_column):
    if len(df) == 0:
        return pd.NaT
    return df[time_column].min()


def last_timestamp(df, time_column):
    if len(df) == 0:
        return pd.NaT
    return df[time_column].max()


# ============================================================
# LOAD STEP 4 ALIGNED ANOMALY SCORES
# ============================================================

print("=" * 70)
print("SENTINEL - STEP 5A: FAILURE ASSOCIATION ANALYSIS")
print("=" * 70)

print(f"Project root: {PROJECT_ROOT}")
print(f"Output dir:   {OUTPUT_DIR}")
print()


anomaly_path = os.path.join(
    STEP4_DIR,
    "step4_anomaly_aligned_scores.csv"
)

if not os.path.isfile(anomaly_path):
    raise FileNotFoundError(
        "\nStep 4 anomaly-aligned file was not found:\n"
        f"{anomaly_path}\n\n"
        "Please make sure Step 4 completed successfully."
    )


anomaly = pd.read_csv(anomaly_path)

print(
    f"Loaded Step 4 anomaly-aligned scores: "
    f"{len(anomaly):,} rows"
)


# ============================================================
# INSPECT COLUMNS
# ============================================================

print()
print("Available columns:")
print(list(anomaly.columns))


# ============================================================
# PARSE TIMESTAMPS
# ============================================================

if "time" not in anomaly.columns:
    raise ValueError(
        "Expected 'time' column was not found in "
        "step4_anomaly_aligned_scores.csv"
    )

if "hostname" not in anomaly.columns:
    raise ValueError(
        "Expected 'hostname' column was not found."
    )

anomaly["time"] = pd.to_datetime(
    anomaly["time"],
    errors="coerce"
)

# Use the actual Step 4 column name consistently
anomaly["timestamp"] = anomaly["time"]

anomaly = anomaly.dropna(
    subset=["timestamp", "hostname"]
).copy()

anomaly = anomaly.sort_values(
    ["hostname", "timestamp"]
).reset_index(drop=True)


# ============================================================
# IDENTIFY IMPORTANT SCORE COLUMNS
# ============================================================

required_columns = [
    "hostname",
    "timestamp",
    "label",
]

for col in required_columns:
    if col not in anomaly.columns:
        raise ValueError(
            f"Required column '{col}' was not found."
        )


# Step 4 creates fusion scores and detector evidence.
# We use the actual columns if available.

possible_columns = [
    "if_score",
    "anomaly_score",
    "lstm_window_mse",
    "window_mse",
    "lstm_last_step_mse",
    "last_step_mse",
    "fusion_window",
    "fusion_last",
]

print()
print("Detected score columns:")

for col in possible_columns:
    if col in anomaly.columns:
        print(f"  {col}")


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

# Isolation Forest score
if "if_score" not in anomaly.columns:
    raise ValueError(
        "Could not find Isolation Forest score column."
    )

anomaly["if_score_used"] = anomaly["if_score"]


# LSTM Window MSE
if "window_mse" not in anomaly.columns:
    raise ValueError(
        "Could not find LSTM Window-MSE column."
    )

anomaly["lstm_window_used"] = anomaly["window_mse"]


# LSTM Last-step MSE
if "last_step_mse" not in anomaly.columns:
    raise ValueError(
        "Could not find LSTM Last-step-MSE column."
    )

anomaly["lstm_last_step_used"] = anomaly[
    "last_step_mse"
]


# Fusion scores if available
if "fusion_window" in anomaly.columns:
    anomaly["fusion_window_used"] = anomaly["fusion_window"]
else:
    anomaly["fusion_window_used"] = np.nan

if "fusion_last" in anomaly.columns:
    anomaly["fusion_last_used"] = anomaly["fusion_last"]
else:
    anomaly["fusion_last_used"] = np.nan


# ============================================================
# LOAD ORIGINAL TEST DATA
# ============================================================

dataset_dir = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "ClusterWise_mini",
    "olcf_task"
)

test_anomaly_path = os.path.join(
    dataset_dir,
    "test_anomaly.parquet"
)

if not os.path.isfile(test_anomaly_path):
    raise FileNotFoundError(
        f"Could not find:\n{test_anomaly_path}"
    )


print()
print("Loading original anomaly test dataset...")

raw = pd.read_parquet(test_anomaly_path)

print(
    f"Loaded anomaly test dataset: "
    f"{len(raw):,} rows"
)


raw["timestamp"] = pd.to_datetime(
    raw["timestamp"],
    errors="coerce"
)

raw = raw.dropna(
    subset=["timestamp", "hostname"]
).copy()

raw = raw.sort_values(
    ["hostname", "timestamp"]
).reset_index(drop=True)


# ============================================================
# IDENTIFY XID-31 EVENTS
# ============================================================

failure_rows = raw[
    (raw["xid"] == FAILURE_XID)
    & (raw["label"] == 1)
].copy()

print()
print(
    f"XID-{FAILURE_XID} anomaly rows: "
    f"{len(failure_rows):,}"
)


# Group consecutive XID-31 rows by hostname.
# A new event begins when:
#   - hostname changes
#   - time gap is greater than 15 seconds

failure_rows = failure_rows.sort_values(
    ["hostname", "timestamp"]
).reset_index(drop=True)

failure_rows["time_diff"] = (
    failure_rows
    .groupby("hostname")["timestamp"]
    .diff()
    .dt.total_seconds()
)

failure_rows["new_event"] = (
    failure_rows["hostname"].ne(
        failure_rows["hostname"].shift()
    )
    |
    failure_rows["time_diff"].gt(15)
    |
    failure_rows["time_diff"].isna()
)

failure_rows["event_id"] = (
    failure_rows["new_event"]
    .cumsum()
)


# ============================================================
# BUILD EVENT TABLE
# ============================================================

events = []

for event_id, group in failure_rows.groupby("event_id"):

    group = group.sort_values("timestamp")

    host = group["hostname"].iloc[0]

    event_start = group["timestamp"].min()
    event_end = group["timestamp"].max()

    duration_seconds = (
        event_end - event_start
    ).total_seconds()

    events.append(
        {
            "event_id": int(event_id),
            "hostname": host,
            "event_start": event_start,
            "event_end": event_end,
            "event_duration_seconds": duration_seconds,
            "event_samples": len(group),
        }
    )


events_df = pd.DataFrame(events)

print()
print("=" * 70)
print("FAILURE EVENT SUMMARY")
print("=" * 70)

print(
    f"Total XID-{FAILURE_XID} events: "
    f"{len(events_df)}"
)

print(
    f"Total failure rows: "
    f"{len(failure_rows):,}"
)

if len(events_df) > 0:
    print(
        f"Event samples - min: "
        f"{events_df['event_samples'].min()}"
    )

    print(
        f"Event samples - max: "
        f"{events_df['event_samples'].max()}"
    )

    print(
        f"Event duration - min: "
        f"{events_df['event_duration_seconds'].min():.0f}s"
    )

    print(
        f"Event duration - max: "
        f"{events_df['event_duration_seconds'].max():.0f}s"
    )


# ============================================================
# SCORE THRESHOLDS
# ============================================================

# Use the same thresholds established during Step 4.
# These values came from NORMAL validation data.

IF_THRESHOLD = 0.01788996
LSTM_WINDOW_THRESHOLD = 0.15553104
LSTM_LAST_THRESHOLD = 0.26312965

# Fusion thresholds are not hardcoded because Step 5A
# is primarily about detector association.
# We use detector-level thresholds here.


# ============================================================
# ANALYZE EACH FAILURE EVENT
# ============================================================

print()
print("=" * 70)
print("ANALYZING INDIVIDUAL FAILURE EVENTS")
print("=" * 70)

event_results = []


for _, event in events_df.iterrows():

    event_id = int(event["event_id"])
    hostname = event["hostname"]

    event_start = event["event_start"]
    event_end = event["event_end"]

    # --------------------------------------------------------
    # Detector scores DURING the failure event
    # --------------------------------------------------------

    event_scores = anomaly[
        (anomaly["hostname"] == hostname)
        &
        (anomaly["timestamp"] >= event_start)
        &
        (anomaly["timestamp"] <= event_end)
    ].copy()

    # --------------------------------------------------------
    # IF detection
    # --------------------------------------------------------

    if len(event_scores) > 0:

        if_scores = event_scores["if_score_used"].dropna()

        if_anomalous = (
            if_scores >= IF_THRESHOLD
        )

        if_detected = bool(if_anomalous.any())

        if_count = int(if_anomalous.sum())

        if_first = (
            event_scores.loc[
                event_scores["if_score_used"]
                >= IF_THRESHOLD,
                "timestamp"
            ].min()
            if if_detected
            else pd.NaT
        )

        if_max = safe_max(if_scores)
        if_mean = safe_mean(if_scores)

    else:

        if_detected = False
        if_count = 0
        if_first = pd.NaT
        if_max = np.nan
        if_mean = np.nan

    # --------------------------------------------------------
    # LSTM Window detection
    # --------------------------------------------------------

    if len(event_scores) > 0:

        window_scores = (
            event_scores["lstm_window_used"]
            .dropna()
        )

        window_anomalous = (
            window_scores
            >= LSTM_WINDOW_THRESHOLD
        )

        lstm_window_detected = bool(
            window_anomalous.any()
        )

        lstm_window_count = int(
            window_anomalous.sum()
        )

        lstm_window_first = (
            event_scores.loc[
                event_scores["lstm_window_used"]
                >= LSTM_WINDOW_THRESHOLD,
                "timestamp"
            ].min()
            if lstm_window_detected
            else pd.NaT
        )

        lstm_window_max = safe_max(
            window_scores
        )

        lstm_window_mean = safe_mean(
            window_scores
        )

    else:

        lstm_window_detected = False
        lstm_window_count = 0
        lstm_window_first = pd.NaT
        lstm_window_max = np.nan
        lstm_window_mean = np.nan

    # --------------------------------------------------------
    # LSTM Last-step detection
    # --------------------------------------------------------

    if len(event_scores) > 0:

        last_scores = (
            event_scores["lstm_last_step_used"]
            .dropna()
        )

        last_anomalous = (
            last_scores
            >= LSTM_LAST_THRESHOLD
        )

        lstm_last_detected = bool(
            last_anomalous.any()
        )

        lstm_last_count = int(
            last_anomalous.sum()
        )

        lstm_last_first = (
            event_scores.loc[
                event_scores["lstm_last_step_used"]
                >= LSTM_LAST_THRESHOLD,
                "timestamp"
            ].min()
            if lstm_last_detected
            else pd.NaT
        )

        lstm_last_max = safe_max(
            last_scores
        )

        lstm_last_mean = safe_mean(
            last_scores
        )

    else:

        lstm_last_detected = False
        lstm_last_count = 0
        lstm_last_first = pd.NaT
        lstm_last_max = np.nan
        lstm_last_mean = np.nan

    # --------------------------------------------------------
    # Detector agreement
    # --------------------------------------------------------

    both_window = (
        if_detected
        and lstm_window_detected
    )

    both_last = (
        if_detected
        and lstm_last_detected
    )

    if if_detected and lstm_window_detected:
        agreement_window = "both"

    elif if_detected:
        agreement_window = "IF_only"

    elif lstm_window_detected:
        agreement_window = "LSTM_only"

    else:
        agreement_window = "neither"

    if if_detected and lstm_last_detected:
        agreement_last = "both"

    elif if_detected:
        agreement_last = "IF_only"

    elif lstm_last_detected:
        agreement_last = "LSTM_only"

    else:
        agreement_last = "neither"

    # --------------------------------------------------------
    # Detection timing
    # --------------------------------------------------------

    if if_detected:
        if_delay = (
            if_first - event_start
        ).total_seconds()
    else:
        if_delay = np.nan

    if lstm_window_detected:
        lstm_window_delay = (
            lstm_window_first - event_start
        ).total_seconds()
    else:
        lstm_window_delay = np.nan

    if lstm_last_detected:
        lstm_last_delay = (
            lstm_last_first - event_start
        ).total_seconds()
    else:
        lstm_last_delay = np.nan

    # --------------------------------------------------------
    # Event-level record
    # --------------------------------------------------------

    event_results.append(
        {
            "event_id": event_id,
            "hostname": hostname,
            "event_start": event_start,
            "event_end": event_end,
            "event_duration_seconds":
                event["event_duration_seconds"],
            "event_samples":
                event["event_samples"],

            "scored_samples":
                len(event_scores),

            # Isolation Forest
            "if_detected":
                if_detected,
            "if_anomalous_samples":
                if_count,
            "if_detection_fraction":
                (
                    if_count / len(event_scores)
                    if len(event_scores) > 0
                    else np.nan
                ),
            "if_first_detection":
                if_first,
            "if_detection_delay_seconds":
                if_delay,
            "if_max_score":
                if_max,
            "if_mean_score":
                if_mean,

            # LSTM Window
            "lstm_window_detected":
                lstm_window_detected,
            "lstm_window_anomalous_samples":
                lstm_window_count,
            "lstm_window_detection_fraction":
                (
                    lstm_window_count
                    / len(event_scores)
                    if len(event_scores) > 0
                    else np.nan
                ),
            "lstm_window_first_detection":
                lstm_window_first,
            "lstm_window_detection_delay_seconds":
                lstm_window_delay,
            "lstm_window_max_score":
                lstm_window_max,
            "lstm_window_mean_score":
                lstm_window_mean,

            # LSTM Last-step
            "lstm_last_detected":
                lstm_last_detected,
            "lstm_last_anomalous_samples":
                lstm_last_count,
            "lstm_last_detection_fraction":
                (
                    lstm_last_count
                    / len(event_scores)
                    if len(event_scores) > 0
                    else np.nan
                ),
            "lstm_last_first_detection":
                lstm_last_first,
            "lstm_last_detection_delay_seconds":
                lstm_last_delay,
            "lstm_last_max_score":
                lstm_last_max,
            "lstm_last_mean_score":
                lstm_last_mean,

            # Agreement
            "if_lstm_window_both":
                both_window,
            "if_lstm_last_both":
                both_last,
            "window_agreement":
                agreement_window,
            "last_step_agreement":
                agreement_last,
        }
    )


event_results_df = pd.DataFrame(
    event_results
)


# ============================================================
# PRE-EVENT ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PRE-EVENT ANOMALY ANALYSIS")
print("=" * 70)


pre_event_results = []


for _, event in events_df.iterrows():

    event_id = int(event["event_id"])
    hostname = event["hostname"]
    event_start = event["event_start"]

    host_scores = anomaly[
        anomaly["hostname"] == hostname
    ].copy()

    for window_name, seconds in PRE_EVENT_WINDOWS.items():

        window_start = (
            event_start
            - pd.Timedelta(seconds=seconds)
        )

        pre = host_scores[
            (host_scores["timestamp"] >= window_start)
            &
            (host_scores["timestamp"] < event_start)
        ].copy()

        # IF
        if_scores = pre["if_score_used"].dropna()

        if_anomalous = (
            if_scores >= IF_THRESHOLD
        )

        # LSTM window
        window_scores = (
            pre["lstm_window_used"]
            .dropna()
        )

        window_anomalous = (
            window_scores
            >= LSTM_WINDOW_THRESHOLD
        )

        # LSTM last-step
        last_scores = (
            pre["lstm_last_step_used"]
            .dropna()
        )

        last_anomalous = (
            last_scores
            >= LSTM_LAST_THRESHOLD
        )

        pre_event_results.append(
            {
                "event_id": event_id,
                "hostname": hostname,
                "event_start": event_start,
                "pre_event_window": window_name,
                "pre_event_seconds": seconds,
                "scored_samples": len(pre),

                "if_anomalous_samples":
                    int(if_anomalous.sum()),
                "if_anomaly_fraction":
                    (
                        float(if_anomalous.mean())
                        if len(if_anomalous) > 0
                        else 0.0
                    ),
                "if_max_score":
                    safe_max(if_scores),
                "if_mean_score":
                    safe_mean(if_scores),

                "lstm_window_anomalous_samples":
                    int(window_anomalous.sum()),
                "lstm_window_anomaly_fraction":
                    (
                        float(window_anomalous.mean())
                        if len(window_anomalous) > 0
                        else 0.0
                    ),
                "lstm_window_max_score":
                    safe_max(window_scores),
                "lstm_window_mean_score":
                    safe_mean(window_scores),

                "lstm_last_anomalous_samples":
                    int(last_anomalous.sum()),
                "lstm_last_anomaly_fraction":
                    (
                        float(last_anomalous.mean())
                        if len(last_anomalous) > 0
                        else 0.0
                    ),
                "lstm_last_max_score":
                    safe_max(last_scores),
                "lstm_last_mean_score":
                    safe_mean(last_scores),
            }
        )


pre_event_df = pd.DataFrame(
    pre_event_results
)


# ============================================================
# FIRST PRE-EVENT DETECTION
# ============================================================

print()
print("=" * 70)
print("FIRST PRE-EVENT DETECTION ANALYSIS")
print("=" * 70)


early_detection_results = []


for _, event in events_df.iterrows():

    event_id = int(event["event_id"])
    hostname = event["hostname"]
    event_start = event["event_start"]

    host_scores = anomaly[
        anomaly["hostname"] == hostname
    ].copy()

    # Look up to 10 minutes before the event.
    pre = host_scores[
        (host_scores["timestamp"] < event_start)
        &
        (
            host_scores["timestamp"]
            >=
            event_start
            - pd.Timedelta(seconds=600)
        )
    ].copy()

    # --------------------------------------------------------
    # IF first pre-event anomaly
    # --------------------------------------------------------

    if_candidates = pre[
        pre["if_score_used"] >= IF_THRESHOLD
    ]

    if len(if_candidates) > 0:

        first_if = if_candidates["timestamp"].min()

        if_lead = (
            event_start - first_if
        ).total_seconds()

    else:

        first_if = pd.NaT
        if_lead = np.nan

    # --------------------------------------------------------
    # LSTM Window first pre-event anomaly
    # --------------------------------------------------------

    window_candidates = pre[
        pre["lstm_window_used"]
        >= LSTM_WINDOW_THRESHOLD
    ]

    if len(window_candidates) > 0:

        first_window = (
            window_candidates["timestamp"].min()
        )

        window_lead = (
            event_start - first_window
        ).total_seconds()

    else:

        first_window = pd.NaT
        window_lead = np.nan

    # --------------------------------------------------------
    # LSTM Last-step first pre-event anomaly
    # --------------------------------------------------------

    last_candidates = pre[
        pre["lstm_last_step_used"]
        >= LSTM_LAST_THRESHOLD
    ]

    if len(last_candidates) > 0:

        first_last = (
            last_candidates["timestamp"].min()
        )

        last_lead = (
            event_start - first_last
        ).total_seconds()

    else:

        first_last = pd.NaT
        last_lead = np.nan

    early_detection_results.append(
        {
            "event_id": event_id,
            "hostname": hostname,
            "event_start": event_start,

            "if_first_pre_event_detection":
                first_if,
            "if_lead_time_seconds":
                if_lead,

            "lstm_window_first_pre_event_detection":
                first_window,
            "lstm_window_lead_time_seconds":
                window_lead,

            "lstm_last_first_pre_event_detection":
                first_last,
            "lstm_last_lead_time_seconds":
                last_lead,
        }
    )


early_detection_df = pd.DataFrame(
    early_detection_results
)


# ============================================================
# SUMMARY METRICS
# ============================================================

print()
print("=" * 70)
print("STEP 5A - EVENT-LEVEL SUMMARY")
print("=" * 70)


total_events = len(event_results_df)


if_events = int(
    event_results_df["if_detected"].sum()
)

window_events = int(
    event_results_df["lstm_window_detected"].sum()
)

last_events = int(
    event_results_df["lstm_last_detected"].sum()
)

both_window_events = int(
    event_results_df["if_lstm_window_both"].sum()
)

both_last_events = int(
    event_results_df["if_lstm_last_both"].sum()
)


print(
    f"Total XID-{FAILURE_XID} events: "
    f"{total_events}"
)

print()

print(
    f"Isolation Forest detected: "
    f"{if_events}/{total_events} "
    f"({if_events / total_events:.2%})"
)

print(
    f"LSTM Window-MSE detected: "
    f"{window_events}/{total_events} "
    f"({window_events / total_events:.2%})"
)

print(
    f"LSTM Last-step-MSE detected: "
    f"{last_events}/{total_events} "
    f"({last_events / total_events:.2%})"
)

print()

print(
    f"IF + LSTM Window both detected: "
    f"{both_window_events}/{total_events} "
    f"({both_window_events / total_events:.2%})"
)

print(
    f"IF + LSTM Last-step both detected: "
    f"{both_last_events}/{total_events} "
    f"({both_last_events / total_events:.2%})"
)


# ============================================================
# PRE-EVENT SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 5A - PRE-EVENT SUMMARY")
print("=" * 70)


for window_name in PRE_EVENT_WINDOWS.keys():

    subset = pre_event_df[
        pre_event_df["pre_event_window"]
        == window_name
    ]

    print()
    print(f"--- {window_name} before event ---")

    if_count = int(
        (
            subset["if_anomalous_samples"]
            > 0
        ).sum()
    )

    lstm_window_count = int(
        (
            subset[
                "lstm_window_anomalous_samples"
            ]
            > 0
        ).sum()
    )

    lstm_last_count = int(
        (
            subset[
                "lstm_last_anomalous_samples"
            ]
            > 0
        ).sum()
    )

    print(
        f"Events with IF anomaly before event: "
        f"{if_count}/{total_events}"
    )

    print(
        f"Events with LSTM Window anomaly before event: "
        f"{lstm_window_count}/{total_events}"
    )

    print(
        f"Events with LSTM Last-step anomaly before event: "
        f"{lstm_last_count}/{total_events}"
    )


# ============================================================
# LEAD TIME SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 5A - LEAD TIME SUMMARY")
print("=" * 70)


def print_lead_summary(label, series):

    valid = series.dropna()

    print()
    print(label)

    print(
        f"  Events with pre-event detection: "
        f"{len(valid)}/{total_events}"
    )

    if len(valid) > 0:

        print(
            f"  Minimum lead time: "
            f"{valid.min():.0f} sec"
        )

        print(
            f"  Median lead time: "
            f"{valid.median():.0f} sec"
        )

        print(
            f"  Mean lead time: "
            f"{valid.mean():.0f} sec"
        )

        print(
            f"  Maximum lead time: "
            f"{valid.max():.0f} sec"
        )


print_lead_summary(
    "Isolation Forest",
    early_detection_df[
        "if_lead_time_seconds"
    ]
)

print_lead_summary(
    "LSTM Window-MSE",
    early_detection_df[
        "lstm_window_lead_time_seconds"
    ]
)

print_lead_summary(
    "LSTM Last-step-MSE",
    early_detection_df[
        "lstm_last_lead_time_seconds"
    ]
)


# ============================================================
# SAVE EVENT-LEVEL RESULTS
# ============================================================

event_results_path = os.path.join(
    OUTPUT_DIR,
    "step5a_event_level_results.csv"
)

event_results_df.to_csv(
    event_results_path,
    index=False
)


# ============================================================
# SAVE PRE-EVENT RESULTS
# ============================================================

pre_event_path = os.path.join(
    OUTPUT_DIR,
    "step5a_pre_event_analysis.csv"
)

pre_event_df.to_csv(
    pre_event_path,
    index=False
)


# ============================================================
# SAVE EARLY DETECTION RESULTS
# ============================================================

early_detection_path = os.path.join(
    OUTPUT_DIR,
    "step5a_early_detection.csv"
)

early_detection_df.to_csv(
    early_detection_path,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "failure_xid": FAILURE_XID,
    "total_events": total_events,

    "isolation_forest": {
        "events_detected": if_events,
        "event_recall":
            if_events / total_events
            if total_events > 0
            else 0,
    },

    "lstm_window": {
        "events_detected": window_events,
        "event_recall":
            window_events / total_events
            if total_events > 0
            else 0,
    },

    "lstm_last_step": {
        "events_detected": last_events,
        "event_recall":
            last_events / total_events
            if total_events > 0
            else 0,
    },

    "both_if_lstm_window": {
        "events_detected": both_window_events,
        "event_recall":
            both_window_events / total_events
            if total_events > 0
            else 0,
    },

    "both_if_lstm_last_step": {
        "events_detected": both_last_events,
        "event_recall":
            both_last_events / total_events
            if total_events > 0
            else 0,
    },

    "thresholds": {
        "isolation_forest":
            IF_THRESHOLD,
        "lstm_window_mse":
            LSTM_WINDOW_THRESHOLD,
        "lstm_last_step_mse":
            LSTM_LAST_THRESHOLD,
    },

    "pre_event_windows_seconds":
        PRE_EVENT_WINDOWS,
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "step5a_summary.json"
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
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STEP 5A COMPLETE")
print("=" * 70)

print()
print("Reports saved to:")

print(event_results_path)
print(pre_event_path)
print(early_detection_path)
print(summary_path)

print()
print("Next step:")
print(
    "Review the Step 5A event-level and pre-event "
    "results before designing the risk score."
)

print("=" * 70)