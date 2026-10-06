"""Regenerates v2/results.json and the figures. Data: $DATA_DIR (default ~/Developer/data/anomaly-detection-v2)."""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import anomaly_features as af  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "Developer" / "data" / "anomaly-detection-v2"))
SEED = 0
BENIGN_DOMINATED = ["20", "21", "42", "44"]    # >= 89% benign: the setting anomaly detection assumes
LABELLED_EXTRA = ["34", "8"]                    # mostly malicious: only the supervised baseline may learn from them
MAX_TRAIN_PER_CAPTURE = 20000
rng = np.random.default_rng(SEED)

frames = {c: af.load_capture(DATA / f"CTU-IoT-Malware-Capture-{c}-1conn.log.labeled.csv") for c in BENIGN_DOMINATED + LABELLED_EXTRA}
feats = {c: af.build_features(f) for c, f in frames.items()}


def sample(c, n=MAX_TRAIN_PER_CAPTURE):
    idx = np.arange(len(frames[c]))
    if len(idx) > n:
        idx = np.sort(rng.choice(idx, n, replace=False))
    return idx


def stack(caps):
    X = pd.concat([feats[c].iloc[sample(c)] for c in caps]); y = pd.concat([frames[c]["malicious"].iloc[sample(c)] for c in caps])
    return X, y


# --- v1's setting, for the record: capture 34 with Isolation Forest flagging the rarest 1%
X34 = feats["34"]
v1 = IsolationForest(contamination=0.01, random_state=SEED).fit(X34)
pred34 = (v1.predict(X34) == -1).astype(int)
y34 = frames["34"]["malicious"].to_numpy()
v1_setting = {"connections": int(len(y34)), "malicious": int(y34.sum()), "share_malicious": float(y34.mean()),
              "flagged": int(pred34.sum()), "malicious_caught": int(((pred34 == 1) & (y34 == 1)).sum())}

# --- v2: leave one benign-dominated capture out; detectors see only the OTHER captures
results = {}
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
    print(held, json.dumps(row)[:300])

methods = ["isolation_forest", "local_outlier_factor", "supervised_baseline"]
summary = {m: {k: float(np.mean([results[h][m][k] for h in BENIGN_DOMINATED])) for k in ("average_precision", "recall_at_1pct_fpr", "recall_at_5pct_fpr")} for m in methods}
out = {"v1_setting_capture_34": v1_setting, "held_out_captures": results, "mean_over_captures": summary,
       "mean_prevalence": float(np.mean([results[h]["prevalence"] for h in BENIGN_DOMINATED]))}
(ROOT / "v2" / "results.json").write_text(json.dumps(out, indent=2))

# figure: average precision per held-out capture, with the "guess at random" level
labels = {"isolation_forest": "Isolation Forest", "local_outlier_factor": "Local Outlier Factor", "supervised_baseline": "Supervised baseline (uses labels)"}
fig, ax = plt.subplots(figsize=(8, 3.8)); w = 0.26; x = np.arange(len(BENIGN_DOMINATED))
for i, m in enumerate(methods):
    ax.bar(x + (i - 1) * w, [results[h][m]["average_precision"] for h in BENIGN_DOMINATED], w, label=labels[m],
           color=["#4c78a8", "#59a14f", "#999"][i])
for j, h in enumerate(BENIGN_DOMINATED):
    ax.hlines(results[h]["prevalence"], j - 0.45, j + 0.45, colors="#c0392b", linestyles="--", lw=1.2, label="random guessing" if j == 0 else None)
ax.set_xticks(x); ax.set_xticklabels([f"capture {h}\n{results[h]['malicious']} malicious of {results[h]['connections']:,}" for h in BENIGN_DOMINATED], fontsize=8)
ax.set_ylabel("average precision (higher is better)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
ax.set_title("v2: each capture held out in turn; the detector never saw it", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT / "v2" / "v2-held-out.png", dpi=150); plt.close(fig)
print(json.dumps(out["mean_over_captures"], indent=1)); print(v1_setting)
