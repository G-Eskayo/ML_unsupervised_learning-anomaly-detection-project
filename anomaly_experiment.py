"""The v2 experiment: leave one mostly-benign capture out, fit detectors on the others, score the held-out one.

Used by v2/run.py (writes the results) and by the notebook, so both always agree.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import anomaly_features as af

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "Developer" / "data" / "anomaly-detection-v2"))
SEED = 0
BENIGN_DOMINATED = ["20", "21", "42", "44"]    # >= 89% benign: the setting anomaly detection assumes
LABELLED_EXTRA = ["34", "8"]                    # mostly malicious: only the supervised baseline may learn from them
MAX_TRAIN_PER_CAPTURE = 20000
METHODS = ["isolation_forest", "local_outlier_factor", "supervised_baseline"]


def load_all(data_dir=DATA):
    frames = {c: af.load_capture(Path(data_dir) / f"CTU-IoT-Malware-Capture-{c}-1conn.log.labeled.csv") for c in BENIGN_DOMINATED + LABELLED_EXTRA}
    return frames, {c: af.build_features(f) for c, f in frames.items()}


def v1_setting(frames, feats):
    """What v1's method does on capture 34 (92% malicious), with the v2 features."""
    X, y = feats["34"], frames["34"]["malicious"].to_numpy()
    pred = (IsolationForest(contamination=0.01, random_state=SEED).fit(X).predict(X) == -1).astype(int)
    return {"connections": int(len(y)), "malicious": int(y.sum()), "share_malicious": float(y.mean()),
            "flagged": int(pred.sum()), "malicious_caught": int(((pred == 1) & (y == 1)).sum())}


def run_experiment(frames, feats, keep_scores=False):
    rng = np.random.default_rng(SEED)

    def sample(c, n=MAX_TRAIN_PER_CAPTURE):
        idx = np.arange(len(frames[c]))
        if len(idx) > n:
            idx = np.sort(rng.choice(idx, n, replace=False))
        return idx

    def stack(caps):
        X = pd.concat([feats[c].iloc[sample(c)] for c in caps]); y = pd.concat([frames[c]["malicious"].iloc[sample(c)] for c in caps])
        return X, y

    results, all_scores = {}, {}
    for held in BENIGN_DOMINATED:
        others = [c for c in BENIGN_DOMINATED if c != held]
        Xte, yte = feats[held], frames[held]["malicious"].to_numpy()
        Xtr, _ = stack(others)                                    # unsupervised: no labels used
        iso = IsolationForest(n_estimators=300, random_state=SEED).fit(Xtr)
        scaler = StandardScaler().fit(Xtr)
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(scaler.transform(Xtr.sample(min(5000, len(Xtr)), random_state=SEED)))
        Xs, ys = stack(others + LABELLED_EXTRA)                   # supervised baseline: labels, plus the mostly-malicious captures
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED).fit(Xs, ys)
        scores = {"isolation_forest": -iso.score_samples(Xte), "local_outlier_factor": -lof.score_samples(scaler.transform(Xte)),
                  "supervised_baseline": clf.predict_proba(Xte)[:, 1]}
        row = {"connections": int(len(yte)), "malicious": int(yte.sum()), "prevalence": float(yte.mean()),
               "attack_types": frames[held].loc[frames[held]["malicious"] == 1, "attack"].value_counts().to_dict()}
        for name, s in scores.items():
            row[name] = {"average_precision": float(average_precision_score(yte, s)),
                         "recall_at_1pct_fpr": af.recall_at_fpr(yte, s, 0.01), "recall_at_5pct_fpr": af.recall_at_fpr(yte, s, 0.05)}
        results[held] = row
        all_scores[held] = scores
    summary = {m: {k: float(np.mean([results[h][m][k] for h in BENIGN_DOMINATED])) for k in ("average_precision", "recall_at_1pct_fpr", "recall_at_5pct_fpr")} for m in METHODS}
    out = {"v1_setting_capture_34": v1_setting(frames, feats), "held_out_captures": results, "mean_over_captures": summary,
           "mean_prevalence": float(np.mean([results[h]["prevalence"] for h in BENIGN_DOMINATED]))}
    return (out, all_scores) if keep_scores else out
