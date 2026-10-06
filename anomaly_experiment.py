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


# ── Part A: v1's own question, on v1's own data (capture 34) ──────────────────────────────────────────────────────────
# v1's hypothesis: malicious connections can be identified from a connection log, without labels, using features
# engineered from the raw columns. Capture 34 is 92% malicious, so "anomaly" and "malicious" are not the same thing here.

def _scores(y, s):
    from sklearn.metrics import roc_auc_score
    return {"auroc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s)),
            "recall_at_1pct_fpr": af.recall_at_fpr(y, s, 0.01), "recall_at_5pct_fpr": af.recall_at_fpr(y, s, 0.05)}


def study_capture34(frames, feats, n_blocks=5):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, roc_auc_score
    from sklearn.svm import OneClassSVM

    order = frames["34"]["ts"].argsort(kind="stable").to_numpy()
    df = frames["34"].iloc[order].reset_index(drop=True)
    X = feats["34"].iloc[order].reset_index(drop=True)
    y = df["malicious"].to_numpy()
    attack = df["label"].str.replace("Malicious   ", "", regex=False).str.strip().replace({"Malicious": "Malicious"}).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    blocks = af.time_blocks(len(y), n_blocks)
    out = {"connections": int(len(y)), "malicious": int(y.sum()), "share_malicious": float(y.mean()),
           "labels": {str(k): int(v) for k, v in df["label"].value_counts().items()},
           "block_share_malicious": [float(y[b].mean()) for b in blocks]}

    # 1. v1's detector on the whole capture, scored both ways round
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(X)
    s = -iso.score_samples(X)
    out["isolation_forest_unsupervised"] = {**_scores(y, s), "auroc_if_flipped": float(roc_auc_score(y, -s))}

    # 2. clustering, labelled afterwards by each cluster's majority class
    out["kmeans"] = {}
    for k in (2, 3, 4, 6, 8):
        lab = KMeans(k, n_init=5, random_state=SEED).fit_predict(Xs)
        out["kmeans"][str(k)] = {"adjusted_rand": float(adjusted_rand_score(y, lab)),
                                 "balanced_accuracy": af.cluster_majority_balanced_accuracy(y, lab)}
    lab4 = KMeans(4, n_init=5, random_state=SEED).fit_predict(Xs)
    out["kmeans4_composition"] = {int(c): {str(a): int(((lab4 == c) & (attack == a)).sum()) for a in np.unique(attack)} for c in np.unique(lab4)}

    # 3. cross-validated on time blocks: detectors that learn "normal" from benign rows only, and a supervised reference
    def cv(fit_score):
        sc = np.zeros(len(y))
        for i, te in enumerate(blocks):
            tr = np.concatenate([b for j, b in enumerate(blocks) if j != i])
            sc[te] = fit_score(tr, te)
        return sc

    def ocsvm(tr, te):
        ben = tr[y[tr] == 0][:3000]
        return -OneClassSVM(nu=0.05, gamma="scale").fit(Xs[ben]).decision_function(Xs[te])

    def iso_benign(tr, te):
        return -IsolationForest(n_estimators=300, random_state=SEED).fit(X.iloc[tr[y[tr] == 0]]).score_samples(X.iloc[te])

    def supervised(cols=None):
        cols = list(X.columns) if cols is None else cols
        return lambda tr, te: HistGradientBoostingClassifier(random_state=SEED).fit(X.iloc[tr][cols], y[tr]).predict_proba(X.iloc[te][cols])[:, 1]

    sc_ocsvm = cv(ocsvm)
    out["novelty_one_class_svm"] = _scores(y, sc_ocsvm)
    out["novelty_isolation_forest"] = _scores(y, cv(iso_benign))
    out["supervised_reference"] = _scores(y, cv(supervised()))
    out["novelty_one_class_svm_by_attack"] = {
        a: float(roc_auc_score((attack[(attack == a) | (y == 0)] != "Benign").astype(int), sc_ocsvm[(attack == a) | (y == 0)]))
        for a in np.unique(attack) if a != "Benign"}

    # 4. which of v1's engineered features matter? drop a group, re-score the supervised reference
    groups = {"the rolling 10-second connection count (v1's feature)": ["recent_conns"], "byte and packet counts": ["log_", "missing_"],
              "destination port class": ["port_"], "protocol and connection state": ["proto_", "state_"]}
    out["ablation"] = {}
    for name, prefixes in groups.items():
        cols = [c for c in X.columns if not any(c.startswith(p) for p in prefixes)]
        out["ablation"][name] = _scores(y, cv(supervised(cols)))

    # 5. how easy is it? the best single feature, and what each class talks to
    singles = sorted(((float(max(roc_auc_score(y, X[c]), 1 - roc_auc_score(y, X[c]))), c) for c in X.columns if X[c].nunique() > 1), reverse=True)[:5]
    out["best_single_features"] = [{"feature": c, "auroc": a} for a, c in singles]
    out["top_destination_ports"] = {str(k): {str(p): int(n) for p, n in df[attack == k]["resp_p"].value_counts().head(2).items()} for k in np.unique(attack)}
    return out


def transfer_from_capture34(frames, feats):
    """Train on capture 34 only, score every other capture: does what it learned travel?"""
    clf = HistGradientBoostingClassifier(random_state=SEED).fit(feats["34"], frames["34"]["malicious"])
    res = {}
    for c in [c for c in frames if c != "34"]:
        y = frames[c]["malicious"].to_numpy()
        res[c] = {"connections": int(len(y)), "malicious": int(y.sum()), **_scores(y, clf.predict_proba(feats[c])[:, 1])}
    return res
