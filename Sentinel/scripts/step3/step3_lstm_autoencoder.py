"""
SENTINEL - Step 3: LSTM Autoencoder anomaly detector (ML/DL Anomaly Detection stage)
====================================================================================

Question answered by this step
    "Can a temporal LSTM Autoencoder identify unusual GPU telemetry patterns that
     correspond to known anomalous episodes?"
This is ANOMALY DETECTION, not failure prediction. An alarm does not mean a GPU will fail.

Design summary
--------------
* The scaled telemetry matrix (N x 22, float32) is the only large array held in RAM.
* A host- and gap-aware index stores only window END positions (int64). Windows are
  gathered lazily per batch by SequenceDataset; no (N, 60, 22) tensor is ever built.
* Causality: a window ending at row t contains rows t-59..t only. Preprocessing uses
  per-host forward fill (past only) and train-only statistics. No bfill, no future stats.
* Label alignment: the evaluation label of a window is the label of its FINAL row
  (the state at time t). The anomaly score of a window is stamped with the same time t.
* Score direction: HIGHER SCORE = MORE ANOMALOUS. Alarm rule: score >= threshold.
* Scores
    window_mse     (PRIMARY)   mean squared reconstruction error over all 60 steps x 22
                               features. Matches the training objective. An anomaly that
                               only occupies the last few steps is diluted by the window.
    last_step_mse  (SECONDARY) squared error of the final timestep only. More sensitive to
                               the current state, but the decoder reconstructs it from a
                               single latent vector, so it is noisier. Own val threshold.
* Threshold: 99th percentile of NORMAL VALIDATION scores (dense stride 1), per variant.
  Nothing from test.parquet / test_anomaly.parquet influences any model decision.
* label and xid are evaluation-only fields and are never model inputs.

Run (from the Sentinel/ project root):
    python scripts\\step3\\step3_lstm_autoencoder.py
Smoke test (small host subset, 2 epochs, writes to models\\_smoke and reports\\_smoke):
    python scripts\\step3\\step3_lstm_autoencoder.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parents[2]  # Sentinel/ (script is in scripts/step3/)
DATA_DIR = BASE_DIR / "dataset" / "ClusterWise_mini" / "olcf_task"

TRAIN_FILE = DATA_DIR / "train.parquet"
VAL_FILE = DATA_DIR / "val.parquet"
TEST_FILE = DATA_DIR / "test.parquet"
ANOMALY_FILE = DATA_DIR / "test_anomaly.parquet"

SEED = 42

SEQUENCE_LENGTH = 60          # timesteps (10 s each -> 10 minutes)
SAMPLING_SECONDS = 10
MAX_STEP_SECONDS = 15         # a step larger than this (or <= 0) breaks a sequence segment

TRAIN_STRIDE = 10             # reduces redundant overlapping training windows
EARLY_STOP_VAL_STRIDE = 10    # sub-sampled validation used for per-epoch early stopping only
EVAL_STRIDE = 1               # dense evaluation (final val / normal test / anomaly test)

HIDDEN_SIZE = 64
LATENT_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.0
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 5
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0               # Windows: avoid worker processes / memory duplication

VALIDATION_FALSE_POSITIVE_RATE = 0.01   # threshold = 99th percentile of normal val scores

SCORE_VARIANTS = ("window_mse", "last_step_mse")
PRIMARY_SCORE = "window_mse"

BASE_FEATURES = [
    "p0_power", "p1_power",
    "gpu0_core_temp", "gpu0_mem_temp",
    "gpu1_core_temp", "gpu1_mem_temp",
    "gpu2_core_temp", "gpu2_mem_temp",
    "gpu3_core_temp", "gpu3_mem_temp",
    "gpu4_core_temp", "gpu4_mem_temp",
    "gpu5_core_temp", "gpu5_mem_temp",
    "ps0_input_power", "ps1_input_power",
    "p0_temp_mean", "p0_temp_min", "p0_temp_max",
    "p1_temp_mean", "p1_temp_min", "p1_temp_max",
]
FORBIDDEN_INPUT_COLUMNS = {"label", "xid"}
assert not FORBIDDEN_INPUT_COLUMNS & set(BASE_FEATURES), "label/xid must never be features"
assert len(BASE_FEATURES) == 22

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. CPU training is deterministic for a fixed thread
    count; on GPU, cuDNN determinism is requested (may cost some speed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ================================================================
# DATA LOADING AND CLEANING
# ================================================================

def load_dataset(path: Path, hosts=None) -> pd.DataFrame:
    """Load one parquet split, sorted by (hostname, timestamp). Features stay float64 here."""
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    print(f"\nLoading: {path.name}")
    columns = ["timestamp", "hostname", *BASE_FEATURES, "label", "xid"]
    df = pd.read_parquet(path, columns=columns)
    if hosts is not None:
        df = df[df["hostname"].isin(hosts)]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["hostname", "timestamp"]).reset_index(drop=True)
    print(f"  rows: {len(df):,}   hosts: {df['hostname'].nunique()}")
    return df


def clean_features(df: pd.DataFrame, fill_values: pd.Series | None = None):
    """Causal missing-value handling.

    1. +/-inf -> NaN.
    2. Per-host FORWARD fill (uses only earlier rows of the same host; never bfill).
    3. Remaining NaN (leading rows of a host / all-NaN stretches) -> per-feature median
       fitted on the TRAIN split only. For train, fill_values=None -> computed here;
       val/test/anomaly receive the train-fitted values.
    """
    feats = df[BASE_FEATURES]
    n_missing_raw = int(feats.isna().sum().sum())
    df[BASE_FEATURES] = feats.replace([np.inf, -np.inf], np.nan)
    n_missing = int(df[BASE_FEATURES].isna().sum().sum())
    df[BASE_FEATURES] = df.groupby("hostname", sort=False)[BASE_FEATURES].ffill()
    n_after_ffill = int(df[BASE_FEATURES].isna().sum().sum())

    if fill_values is None:
        fill_values = df[BASE_FEATURES].median(skipna=True)
        if fill_values.isna().any():
            raise ValueError(f"Features with no valid train values: "
                             f"{list(fill_values[fill_values.isna()].index)}")
    df[BASE_FEATURES] = df[BASE_FEATURES].fillna(fill_values)

    assert np.isfinite(df[BASE_FEATURES].to_numpy()).all(), "non-finite values remain"
    print(f"  missing/inf cells: {n_missing:,} (raw NaN {n_missing_raw:,}) -> "
          f"after causal ffill {n_after_ffill:,} -> after train-median fill 0")
    return df, fill_values


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """StandardScaler fitted on TRAIN features only (float64 for precision)."""
    scaler = StandardScaler()
    scaler.fit(train_df[BASE_FEATURES].to_numpy(dtype=np.float64))
    print(f"\nScaler fitted on TRAIN only ({int(scaler.n_samples_seen_):,} rows).")
    return scaler


@dataclass
class TelemetryTable:
    """One split as flat arrays. This is the only large object kept in memory."""
    name: str
    features: np.ndarray        # (N, F) float32, scaled, C-contiguous
    timestamps: np.ndarray      # (N,) datetime64[ns]
    host_codes: np.ndarray      # (N,) int32 -> index into host_names
    host_names: np.ndarray      # (H,)
    labels: np.ndarray          # (N,) int8   - evaluation only
    xids: np.ndarray            # (N,) int16  - evaluation only
    segment_ids: np.ndarray     # (N,) int64  - contiguous same-host runs with valid steps


def compute_segments(host_codes: np.ndarray, timestamps: np.ndarray, name: str) -> np.ndarray:
    """Segment = maximal run of rows from one host with 0 < dt <= MAX_STEP_SECONDS.
    Windows are only allowed inside a segment, so they can never cross hosts or time gaps."""
    n = len(host_codes)
    brk = np.ones(n, dtype=bool)
    if n > 1:
        dt = np.diff(timestamps) / np.timedelta64(1, "s")
        same_host = host_codes[1:] == host_codes[:-1]
        step_ok = (dt > 0) & (dt <= MAX_STEP_SECONDS)
        brk[1:] = ~(same_host & step_ok)
        gaps = int((same_host & ~step_ok).sum())
        if same_host.any():
            d = dt[same_host]
            print(f"  [{name}] step seconds (same host): min={d.min():.1f} "
                  f"median={np.median(d):.1f} max={d.max():.1f}; "
                  f"time-gap/duplicate breaks: {gaps}")
    seg = np.cumsum(brk) - 1
    print(f"  [{name}] segments: {int(seg[-1]) + 1}")
    return seg.astype(np.int64)


def to_table(df: pd.DataFrame, scaler: StandardScaler, name: str) -> TelemetryTable:
    """Scale (transform only) and flatten a cleaned DataFrame into a TelemetryTable."""
    codes, names = pd.factorize(df["hostname"])
    features = np.ascontiguousarray(
        scaler.transform(df[BASE_FEATURES].to_numpy(dtype=np.float64)).astype(np.float32))
    timestamps = df["timestamp"].to_numpy(dtype="datetime64[ns]")
    host_codes = codes.astype(np.int32)
    return TelemetryTable(
        name=name,
        features=features,
        timestamps=timestamps,
        host_codes=host_codes,
        host_names=np.asarray(names),
        labels=df["label"].to_numpy(dtype=np.int8),
        xids=df["xid"].to_numpy(dtype=np.int16),
        segment_ids=compute_segments(host_codes, timestamps, name),
    )


# ================================================================
# HOST/GAP-AWARE SEQUENCE INDEX + LAZY DATASET
# ================================================================

def build_sequence_index(table: TelemetryTable, seq_len: int, stride: int) -> np.ndarray:
    """Return int64 array of window END row positions (window = rows end-seq_len+1 .. end).

    Built from the actual data per segment (no assumption of equal host lengths). The first
    window of each segment ends at segment_start + seq_len - 1; later ends advance by `stride`.
    """
    seg = table.segment_ids
    n = len(seg)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    stops = np.r_[starts[1:], n]
    parts = [np.arange(s + seq_len - 1, e, stride, dtype=np.int64)
             for s, e in zip(starts, stops) if e - s >= seq_len]
    if not parts:
        raise RuntimeError(f"No sequences could be built for {table.name}.")
    return np.concatenate(parts)


def check_sequence_index(table: TelemetryTable, ends: np.ndarray, seq_len: int) -> None:
    """Always-on integrity checks (vectorised, cheap)."""
    starts = ends - (seq_len - 1)
    assert starts.min() >= 0 and ends.max() < len(table.features)
    assert (table.host_codes[starts] == table.host_codes[ends]).all(), "window crosses hosts"
    assert (table.segment_ids[starts] == table.segment_ids[ends]).all(), "window crosses a gap"
    span = (table.timestamps[ends] - table.timestamps[starts]) / np.timedelta64(1, "s")
    assert (span > 0).all() and (span <= (seq_len - 1) * MAX_STEP_SECONDS).all(), "bad time span"


class SequenceDataset(torch.utils.data.Dataset):
    """Lazily gathers windows from the shared (N, F) float32 matrix.

    __getitem__ accepts an int or a list/array of ints. Combined with a BatchSampler and
    DataLoader(batch_size=None) only the current batch (B, T, F) is ever materialised.
    """

    def __init__(self, features: np.ndarray, ends: np.ndarray, seq_len: int):
        assert features.dtype == np.float32 and features.flags["C_CONTIGUOUS"]
        self.features = features                       # reference, not a copy
        self.ends = ends
        self.offsets = np.arange(-(seq_len - 1), 1, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, idx):
        e = self.ends[np.asarray(idx)]
        rows = e[..., None] + self.offsets             # (T,) or (B, T)
        return torch.from_numpy(self.features[rows])   # (T, F) or (B, T, F)


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int = SEED):
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
        base = torch.utils.data.RandomSampler(dataset, generator=gen)
    else:
        base = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(base, batch_size=batch_size, drop_last=False)
    return torch.utils.data.DataLoader(
        dataset, sampler=batch_sampler, batch_size=None, num_workers=NUM_WORKERS)


# ================================================================
# MODEL
# ================================================================

class LSTMAutoencoder(nn.Module):
    """Encoder LSTM(F->H) -> final hidden -> Linear(H->L) latent -> Linear(L->H) ->
    repeated across T steps -> decoder LSTM(H->H) -> Linear(H->F)."""

    def __init__(self, input_size, hidden_size, latent_size, num_layers=1, dropout=0.0):
        super().__init__()
        inter_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=inter_dropout)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=inter_dropout)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):                               # x: (B, T, F)
        enc_out, _ = self.encoder(x)
        latent = self.to_latent(enc_out[:, -1, :])      # (B, L)
        dec_in = self.from_latent(latent).unsqueeze(1).expand(-1, x.size(1), -1)
        dec_out, _ = self.decoder(dec_in)
        return self.output_layer(dec_out)               # (B, T, F)


# ================================================================
# TRAINING / SCORING
# ================================================================

def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total += loss.item() * batch.size(0)
        count += batch.size(0)
    return total / count


@torch.inference_mode()
def evaluate_reconstruction(model, loader) -> dict[str, np.ndarray]:
    """Per-window reconstruction errors (higher = more anomalous), in loader order."""
    model.eval()
    window, last = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        err = (model(batch) - batch).pow(2)             # (B, T, F), freed each iteration
        window.append(err.mean(dim=(1, 2)).cpu().numpy())
        last.append(err[:, -1, :].mean(dim=1).cpu().numpy())
    return {"window_mse": np.concatenate(window).astype(np.float64),
            "last_step_mse": np.concatenate(last).astype(np.float64)}


def train_model(model, train_loader, val_loader):
    """Adam + MSE. Early stopping on VALIDATION loss (normal data, stride-10 subset);
    best checkpoint restored. The final threshold uses a separate dense validation pass."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []
    print("\nTraining LSTM Autoencoder...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = float(evaluate_reconstruction(model, val_loader)["window_mse"].mean())
        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "is_best": improved, "seconds": round(time.time() - t0, 1)})
        print(f"Epoch {epoch:02d}/{EPOCHS} - train {train_loss:.8f} - val {val_loss:.8f}"
              f"{' *' if improved else ''} ({history[-1]['seconds']}s)")
        if bad >= PATIENCE:
            print(f"Early stopping (no val improvement for {PATIENCE} epochs).")
            break
    model.load_state_dict(best_state)
    print(f"Restored best checkpoint: epoch {best_epoch}, val loss {best_val:.8f}")
    return model, pd.DataFrame(history), best_epoch, best_val


# ================================================================
# SCORING FRAMES, THRESHOLD, METRICS
# ================================================================

def score_table(model, table: TelemetryTable, stride: int):
    """Dense/strided scoring of a split. Returns (frame, ends). Each row = one window,
    stamped with the END timestamp; label = label of the final row."""
    ends = build_sequence_index(table, SEQUENCE_LENGTH, stride)
    check_sequence_index(table, ends, SEQUENCE_LENGTH)
    print(f"  [{table.name}] windows: {len(ends):,} (stride {stride}); "
          f"per-batch tensor {EVAL_BATCH_SIZE}x{SEQUENCE_LENGTH}x{len(BASE_FEATURES)} float32 "
          f"= {EVAL_BATCH_SIZE * SEQUENCE_LENGTH * len(BASE_FEATURES) * 4 / 2**20:.1f} MiB")
    ds = SequenceDataset(table.features, ends, SEQUENCE_LENGTH)
    scores = evaluate_reconstruction(model, make_loader(ds, EVAL_BATCH_SIZE, shuffle=False))

    cs = np.r_[0, np.cumsum(table.labels, dtype=np.int64)]
    frame = pd.DataFrame({
        "hostname": pd.Categorical.from_codes(table.host_codes[ends], categories=table.host_names),
        "start_time": table.timestamps[ends - (SEQUENCE_LENGTH - 1)],
        "end_time": table.timestamps[ends],
        "label": table.labels[ends].astype(np.int8),            # label at the endpoint
        "xid": table.xids[ends],                                 # evaluation/report only
        # diagnostic: window contains an anomalous row anywhere (NOT the label)
        "window_contains_anomaly":
            ((cs[ends + 1] - cs[ends - SEQUENCE_LENGTH + 1]) > 0).astype(np.int8),
        **scores,
    })
    return frame, ends


def select_threshold(normal_validation_scores: np.ndarray, fpr: float) -> float:
    """99th percentile of NORMAL validation scores. Alarm rule: score >= threshold."""
    return float(np.quantile(normal_validation_scores, 1.0 - fpr))


def evaluate_normal_test(frame: pd.DataFrame, col: str, threshold: float) -> dict:
    """Normal-only split: only FP / TN / FPR are meaningful (no positive labels)."""
    assert (frame["label"] == 0).all(), "normal test must contain no positive labels"
    alarms = frame[col].to_numpy() >= threshold
    fp = int(alarms.sum())
    n = len(frame)
    return {"normal_test_samples": n, "normal_test_fp": fp, "normal_test_tn": n - fp,
            "normal_test_fpr": fp / n}


def evaluate_predictions(frame: pd.DataFrame, col: str, threshold: float) -> dict:
    y = frame["label"].to_numpy().astype(int)
    s = frame[col].to_numpy()
    pred = s >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pr_auc = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    tail_fp = int((pred & (y == 0) & (frame["window_contains_anomaly"].to_numpy() == 1)).sum())
    return {"anomaly_samples_scored": len(y), "anomaly_positives_scored": int(y.sum()),
            "anomaly_prevalence_scored": float(y.mean()),
            "anomaly_tp": tp, "anomaly_fp": fp, "anomaly_tn": tn, "anomaly_fn": fn,
            "anomaly_precision": precision, "anomaly_recall": recall, "anomaly_f1": f1,
            "anomaly_pr_auc": pr_auc,
            "anomaly_fp_window_overlaps_event": tail_fp,
            "anomaly_fp_window_clean": fp - tail_fp}


def build_events(table: TelemetryTable) -> tuple[pd.DataFrame, np.ndarray]:
    """Events = maximal runs of consecutive label==1 rows within one segment (one host)."""
    pos = table.labels == 1
    prev_pos = np.r_[False, pos[:-1]]
    prev_same_seg = np.r_[False, table.segment_ids[1:] == table.segment_ids[:-1]]
    start = pos & ~(prev_pos & prev_same_seg)
    event_of_row = np.where(pos, np.cumsum(start) - 1, -1).astype(np.int64)
    if not pos.any():
        return pd.DataFrame(), event_of_row
    rows = np.flatnonzero(pos)
    g = pd.DataFrame({"event_id": event_of_row[rows], "row": rows})
    ev = g.groupby("event_id")["row"].agg(first_row="min", last_row="max", n_rows="size")
    ev["hostname"] = table.host_names[table.host_codes[ev["first_row"].to_numpy()]]
    ev["start_time"] = table.timestamps[ev["first_row"].to_numpy()]
    ev["end_time"] = table.timestamps[ev["last_row"].to_numpy()]
    return ev.reset_index(), event_of_row


def evaluate_events(table, ends, frame, col, threshold, events, event_of_row) -> tuple[dict, pd.DataFrame]:
    """An event is detected if at least one window whose ENDPOINT lies inside the event's
    labelled rows scores >= threshold (causal: alarm raised at a time within the event).
    Windows ending before onset or after the episode do not count."""
    ev_id = event_of_row[ends]
    inside = ev_id >= 0
    alarm = frame[col].to_numpy() >= threshold
    det = pd.Series(alarm[inside]).groupby(ev_id[inside]).any()
    scored = pd.Series(1, index=np.arange(inside.sum())).groupby(ev_id[inside]).size()
    peak = pd.Series(frame[col].to_numpy()[inside]).groupby(ev_id[inside]).max()
    out = events.copy()
    out["n_scored_endpoints"] = out["event_id"].map(scored).fillna(0).astype(int)
    out["detected"] = out["event_id"].map(det).fillna(False).astype(bool)
    out["peak_score"] = out["event_id"].map(peak)
    total = len(out)
    detected = int(out["detected"].sum())
    return ({"total_events": total, "detected_events": detected,
             "missed_events": total - detected,
             "event_recall": detected / total if total else float("nan"),
             "events_with_unscorable_rows": int((out["n_scored_endpoints"] < out["n_rows"]).sum())},
            out)


# ================================================================
# ARTIFACTS
# ================================================================

def save_artifacts(model_dir, model, scaler, fill_values, thresholds, run_info) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "sentinel_lstm_autoencoder.pt")
    joblib.dump(scaler, model_dir / "sentinel_lstm_scaler.joblib")
    joblib.dump(float(thresholds[PRIMARY_SCORE]), model_dir / "sentinel_lstm_threshold.joblib")
    config = {
        "model": "LSTM_Autoencoder",
        "feature_list": BASE_FEATURES,
        "input_size": len(BASE_FEATURES),
        "sequence_length": SEQUENCE_LENGTH,
        "sampling_seconds": SAMPLING_SECONDS,
        "max_step_seconds": MAX_STEP_SECONDS,
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "loss": "MSE",
        "train_stride": TRAIN_STRIDE,
        "early_stop_val_stride": EARLY_STOP_VAL_STRIDE,
        "evaluation_stride": EVAL_STRIDE,
        "seed": SEED,
        "primary_score": PRIMARY_SCORE,
        "score_direction": "higher = more anomalous",
        "alarm_rule": "score >= threshold",
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "threshold_rule": f"quantile {1 - VALIDATION_FALSE_POSITIVE_RATE:.2f} of normal "
                          f"validation scores (dense stride {EVAL_STRIDE}), per score variant",
        "score_definitions": {
            "window_mse": "mean squared error over all timesteps and features of the window "
                          "ending at t; stamped with t",
            "last_step_mse": "mean squared error of the final timestep (t) only; stamped with t"},
        "label_alignment": "window label = label of the final row (time t)",
        "preprocessing": {
            "inf_to_nan": True,
            "missing_values": "per-host causal forward fill, then train-split median",
            "train_median_fill_values": {k: float(v) for k, v in fill_values.items()},
            "scaler": "StandardScaler fitted on train.parquet only (sentinel_lstm_scaler.joblib)",
            "windows": "never cross hostnames or time gaps (> max_step_seconds or dt <= 0)",
        },
        "torch_version": torch.__version__,
        **run_info,
    }
    with open(model_dir / "sentinel_lstm_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ================================================================
# MAIN
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SENTINEL Step 3 - LSTM Autoencoder")
    p.add_argument("--smoke-test", action="store_true",
                   help="small host subset, 2 epochs, outputs to models/_smoke and reports/_smoke")
    p.add_argument("--max-hosts", type=int, default=None, help="use only the first N hosts")
    p.add_argument("--epochs", type=int, default=None, help="override max epochs")
    p.add_argument("--data-dir", type=Path, default=None, help="override dataset directory")
    return p.parse_args()


def main():
    global EPOCHS, TRAIN_FILE, VAL_FILE, TEST_FILE, ANOMALY_FILE
    args = parse_args()
    max_hosts = args.max_hosts or (4 if args.smoke_test else None)
    if args.epochs or args.smoke_test:
        EPOCHS = args.epochs or 2
    if args.data_dir:
        TRAIN_FILE, VAL_FILE = args.data_dir / "train.parquet", args.data_dir / "val.parquet"
        TEST_FILE, ANOMALY_FILE = args.data_dir / "test.parquet", args.data_dir / "test_anomaly.parquet"
    model_dir = BASE_DIR / "models" / ("_smoke" if args.smoke_test else "")
    report_dir = BASE_DIR / "reports" / ("_smoke" if args.smoke_test else "")
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    print("=" * 70)
    print("SENTINEL - STEP 3: LSTM AUTOENCODER (anomaly detection)")
    print("=" * 70)
    print(f"PyTorch {torch.__version__} | device {DEVICE} | threads {torch.get_num_threads()}")
    print(f"Project root: {BASE_DIR}\nData dir:     {DATA_DIR if not args.data_dir else args.data_dir}")
    print(f"Sequence {SEQUENCE_LENGTH} x {len(BASE_FEATURES)} | train stride {TRAIN_STRIDE} | "
          f"eval stride {EVAL_STRIDE} | max epochs {EPOCHS}")
    print("Score direction: HIGHER = MORE ANOMALOUS; alarm if score >= threshold.")
    if args.smoke_test:
        print(f"*** SMOKE TEST: hosts={max_hosts}, outputs -> {model_dir}, {report_dir} ***")

    # ---- 1. TRAIN: load, causal clean, fit scaler (train only) ----------------------
    train_df = load_dataset(TRAIN_FILE)
    hosts = None
    if max_hosts:
        hosts = sorted(train_df["hostname"].unique())[:max_hosts]
        train_df = train_df[train_df["hostname"].isin(hosts)].reset_index(drop=True)
        print(f"  restricted to {len(hosts)} hosts")
    assert (train_df["label"] == 0).all(), "train must be all-normal"
    train_df, fill_values = clean_features(train_df)
    scaler = fit_scaler(train_df)
    train_table = to_table(train_df, scaler, "train")
    del train_df

    # ---- 2. VALIDATION table (transform only) --------------------------------------
    val_df = load_dataset(VAL_FILE, hosts)
    val_df, _ = clean_features(val_df, fill_values)
    val_table = to_table(val_df, scaler, "val")
    del val_df

    # ---- 3. Index + loaders ---------------------------------------------------------
    train_ends = build_sequence_index(train_table, SEQUENCE_LENGTH, TRAIN_STRIDE)
    check_sequence_index(train_table, train_ends, SEQUENCE_LENGTH)
    val_sub_ends = build_sequence_index(val_table, SEQUENCE_LENGTH, EARLY_STOP_VAL_STRIDE)
    check_sequence_index(val_table, val_sub_ends, SEQUENCE_LENGTH)
    print(f"\nTrain windows (stride {TRAIN_STRIDE}): {len(train_ends):,} | "
          f"early-stop val windows (stride {EARLY_STOP_VAL_STRIDE}): {len(val_sub_ends):,}")
    print(f"Resident feature matrices: train {train_table.features.nbytes / 2**20:.1f} MiB, "
          f"val {val_table.features.nbytes / 2**20:.1f} MiB "
          f"(a full stride-1 train tensor would be "
          f"{(len(train_table.features) - 59 * len(train_table.host_names)) * SEQUENCE_LENGTH * len(BASE_FEATURES) * 4 / 2**30:.2f} GiB)")
    train_loader = make_loader(
        SequenceDataset(train_table.features, train_ends, SEQUENCE_LENGTH),
        BATCH_SIZE, shuffle=True)
    val_sub_loader = make_loader(
        SequenceDataset(val_table.features, val_sub_ends, SEQUENCE_LENGTH),
        EVAL_BATCH_SIZE, shuffle=False)

    # ---- 4. Model + training --------------------------------------------------------
    model = LSTMAutoencoder(len(BASE_FEATURES), HIDDEN_SIZE, LATENT_SIZE,
                            NUM_LAYERS, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    model, history, best_epoch, best_val = train_model(model, train_loader, val_sub_loader)
    history.to_csv(report_dir / "step3_lstm_training_history.csv", index=False)
    torch.save(model.state_dict(), model_dir / "sentinel_lstm_autoencoder.pt")  # crash-safe

    # ---- 5. Dense validation -> thresholds (normal validation only) ----------------
    print("\nDense validation scoring...")
    val_frame, _ = score_table(model, val_table, EVAL_STRIDE)
    thresholds = {v: select_threshold(val_frame[v].to_numpy(), VALIDATION_FALSE_POSITIVE_RATE)
                  for v in SCORE_VARIANTS}
    val_alarm = {v: float((val_frame[v] >= thresholds[v]).mean()) for v in SCORE_VARIANTS}
    for v in SCORE_VARIANTS:
        print(f"  threshold[{v}] = {thresholds[v]:.8f}   validation alarm rate = {val_alarm[v]:.6f}")
    val_frame.drop(columns=["xid", "window_contains_anomaly"]).to_csv(
        report_dir / "step3_lstm_validation_scores.csv", index=False)
    del val_table, val_frame

    # ---- 6. Normal test --------------------------------------------------------------
    test_df = load_dataset(TEST_FILE, hosts)
    test_df, _ = clean_features(test_df, fill_values)
    test_table = to_table(test_df, scaler, "normal_test")
    del test_df
    print("\nNormal test scoring...")
    test_frame, _ = score_table(model, test_table, EVAL_STRIDE)
    normal_metrics = {v: evaluate_normal_test(test_frame, v, thresholds[v]) for v in SCORE_VARIANTS}
    test_frame.drop(columns=["xid", "window_contains_anomaly"]).to_csv(
        report_dir / "step3_lstm_normal_test_scores.csv", index=False)
    del test_table, test_frame

    # ---- 7. Anomaly test ---------------------------------------------------------------
    anom_df = load_dataset(ANOMALY_FILE, hosts)
    print(f"  xid values on labelled rows: {sorted(anom_df.loc[anom_df['label'] == 1, 'xid'].unique())}")
    anom_df, _ = clean_features(anom_df, fill_values)
    anom_table = to_table(anom_df, scaler, "anomaly_test")
    del anom_df
    print("\nAnomaly test scoring...")
    anom_frame, anom_ends = score_table(model, anom_table, EVAL_STRIDE)
    events, event_of_row = build_events(anom_table)
    print(f"  events found: {len(events)} (expected 85 for the full dataset); "
          f"rows/event: {events['n_rows'].value_counts().to_dict() if len(events) else {}}")
    print(f"  anomalous rows total: {int(anom_table.labels.sum())}, "
          f"scorable as window endpoints: {int(anom_frame['label'].sum())}")

    metric_rows, event_tables = [], {}
    for v in SCORE_VARIANTS:
        pm = evaluate_predictions(anom_frame, v, thresholds[v])
        em, per_event = evaluate_events(anom_table, anom_ends, anom_frame, v, thresholds[v],
                                        events, event_of_row) if len(events) else ({}, None)
        event_tables[v] = per_event
        metric_rows.append({"model": "LSTM_Autoencoder", "score_variant": v,
                            "is_primary": v == PRIMARY_SCORE, "threshold": thresholds[v],
                            "validation_alarm_rate": val_alarm[v],
                            **normal_metrics[v], **pm, **em})
        tag = "PRIMARY" if v == PRIMARY_SCORE else "secondary"
        print("\n" + "-" * 70 + f"\nRESULTS [{v}] ({tag})  threshold={thresholds[v]:.8f}\n" + "-" * 70)
        nm = normal_metrics[v]
        print(f"Normal test : n={nm['normal_test_samples']:,}  FP={nm['normal_test_fp']:,}  "
              f"TN={nm['normal_test_tn']:,}  FPR={nm['normal_test_fpr']:.6f}")
        print(f"Anomaly test: TP={pm['anomaly_tp']:,} FP={pm['anomaly_fp']:,} "
              f"TN={pm['anomaly_tn']:,} FN={pm['anomaly_fn']:,}")
        print(f"              precision={pm['anomaly_precision']:.6f} recall={pm['anomaly_recall']:.6f} "
              f"F1={pm['anomaly_f1']:.6f} PR-AUC={pm['anomaly_pr_auc']:.6f} "
              f"(prevalence {pm['anomaly_prevalence_scored']:.6f})")
        print(f"              FPs whose window overlaps an event: {pm['anomaly_fp_window_overlaps_event']:,} "
              f"| FPs from clean windows: {pm['anomaly_fp_window_clean']:,}")
        if em:
            print(f"Events      : detected {em['detected_events']}/{em['total_events']} "
                  f"missed {em['missed_events']} event recall={em['event_recall']:.6f} "
                  f"(events with unscorable rows: {em['events_with_unscorable_rows']})")

    anom_frame.drop(columns=["xid"]).to_csv(
        report_dir / "step3_lstm_anomaly_test_scores.csv", index=False)
    if event_tables.get(PRIMARY_SCORE) is not None:
        pd.concat([t.assign(score_variant=v) for v, t in event_tables.items()]).to_csv(
            report_dir / "step3_lstm_event_detection.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(report_dir / "step3_lstm_metrics.csv", index=False)

    # ---- 8. Artifacts ----------------------------------------------------------------
    save_artifacts(model_dir, model, scaler, fill_values, thresholds, {
        "best_epoch": best_epoch, "best_val_loss_stride10": best_val,
        "epochs_run": len(history), "parameter_count": n_params,
        "hosts_used": None if hosts is None else list(hosts)})

    print("\n" + "=" * 70 + "\nSTEP 3 COMPLETE\n" + "=" * 70)
    for p in sorted(model_dir.glob("sentinel_lstm_*")) + sorted(report_dir.glob("step3_lstm_*")):
        print(p)


if __name__ == "__main__":
    main()
