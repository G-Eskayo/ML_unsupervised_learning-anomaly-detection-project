"""Loading, features and evaluation helpers for the CTU-IoT-Malware captures (anomaly detection v2).

Features deliberately exclude the labels, the IP addresses (an address identifies a device and, across captures, the scenario),
and the timestamp. The only temporal feature looks at the past: how many connections the same source opened in the last 10 s.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROTOS = ("tcp", "udp", "icmp")
STATES = ("S0", "S1", "S2", "S3", "SF", "REJ", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "OTH")
NUMERIC = ("duration", "orig_bytes", "resp_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "missed_bytes")
WINDOW_SECONDS = 10.0


def load_capture(path) -> pd.DataFrame:
    """One pipe-delimited Zeek conn.log capture. Adds `capture` (id from the file name), `malicious` (0/1) and `attack`."""
    path = Path(path)
    df = pd.read_csv(path, sep="|", na_values=["-", "(empty)"], low_memory=False)
    df = df.rename(columns={"id.orig_h": "orig_h", "id.orig_p": "orig_p", "id.resp_h": "resp_h", "id.resp_p": "resp_p",
                            "detailed-label": "attack"})
    df["capture"] = re.search(r"Capture-(\d+)-", path.name).group(1)
    df["malicious"] = df["label"].astype(str).str.strip().str.lower().str.startswith("malicious").astype(int)
    df["attack"] = df["attack"].fillna("")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def recent_connection_counts(df: pd.DataFrame) -> np.ndarray:
    """For each row, connections from the same source in (ts - 10 s, ts], itself included. `df` must be sorted by ts."""
    out = np.ones(len(df), dtype=float)
    ts = df["ts"].to_numpy(dtype=float)
    for _, idx in df.groupby("orig_h", sort=False).indices.items():
        t = ts[idx]
        out[idx] = np.arange(1, len(idx) + 1) - np.searchsorted(t, t - WINDOW_SECONDS, side="right")
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """A fixed set of columns, identical for every capture."""
    order = np.argsort(df["ts"].to_numpy(), kind="stable")
    sorted_df = df.iloc[order]
    recent = pd.Series(recent_connection_counts(sorted_df), index=sorted_df.index).reindex(df.index)
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC:
        X[f"log_{col}"] = np.log1p(df[col].fillna(0).clip(lower=0))
        X[f"missing_{col}"] = df[col].isna().astype(int)
    X["recent_conns"] = np.log1p(recent)
    X["history_len"] = df["history"].fillna("").astype(str).str.len()
    port = df["resp_p"].fillna(-1)
    X["port_well_known"] = ((port >= 0) & (port < 1024)).astype(int)
    X["port_registered"] = ((port >= 1024) & (port < 49152)).astype(int)
    X["port_ephemeral"] = (port >= 49152).astype(int)
    for p in PROTOS:
        X[f"proto_{p}"] = (df["proto"] == p).astype(int)
    for s in STATES:
        X[f"state_{s}"] = (df["conn_state"] == s).astype(int)
    return X


def leave_one_capture_out(df: pd.DataFrame):
    """Yields (held_out_capture, train_index, test_index). The detector never sees the capture it is scored on."""
    for cap in sorted(df["capture"].unique(), key=int):
        yield cap, df.index[df["capture"] != cap], df.index[df["capture"] == cap]


def recall_at_fpr(y_true, score, fpr: float) -> float:
    """Share of malicious connections caught while raising false alarms on at most `fpr` of the benign ones."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    neg = np.sort(score[y_true == 0])[::-1]
    allowed = int(np.floor(fpr * len(neg)))
    threshold = neg[min(allowed, len(neg) - 1)]
    pos = score[y_true == 1]
    return float((pos > threshold).mean()) if len(pos) else float("nan")


def time_blocks(n: int, k: int):
    """k contiguous, ordered blocks covering rows 0..n-1. Cross-validating on time blocks (not random rows) keeps neighbouring
    connections of one burst out of both the training and the test side."""
    return np.array_split(np.arange(n), k)


def cluster_majority_balanced_accuracy(y_true, clusters) -> float:
    """Label each cluster with its majority class, then score that labelling with balanced accuracy.
    1.0 = every cluster is pure; 0.5 = the clusters tell you nothing beyond the majority class."""
    y_true, clusters = np.asarray(y_true), np.asarray(clusters)
    pred = np.zeros_like(y_true)
    for c in np.unique(clusters):
        m = clusters == c
        pred[m] = int(y_true[m].mean() > 0.5)
    recalls = [(pred[y_true == cls] == cls).mean() for cls in np.unique(y_true)]
    return float(np.mean(recalls))
