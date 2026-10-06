"""Regenerates v2/results.json and the figures. Data: $DATA_DIR (default ~/Developer/data/anomaly-detection-v2)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import anomaly_experiment as ex  # noqa: E402


frames, feats = ex.load_all()
out = ex.run_experiment(frames, feats)
out["same_data_capture_34"] = ex.study_capture34(frames, feats)
out["transfer_from_capture_34"] = ex.transfer_from_capture34(frames, feats)
results, BENIGN_DOMINATED, methods = out["held_out_captures"], ex.BENIGN_DOMINATED, ex.METHODS
(ROOT / "v2" / "results.json").write_text(json.dumps(out, indent=2))

# figure A: v1's question on v1's data
A = out["same_data_capture_34"]
bars = [("v1's detector as used\n(flag the rarest 1%)", A["isolation_forest_unsupervised"]["auroc"], "#c0392b"),
        ("same scores,\nread the other way round", A["isolation_forest_unsupervised"]["auroc_if_flipped"], "#e08e8e"),
        ("Isolation Forest\nlearning from benign only", A["novelty_isolation_forest"]["auroc"], "#4c78a8"),
        ("One-class SVM\nlearning from benign only", A["novelty_one_class_svm"]["auroc"], "#2f5d8c"),
        ("Supervised reference\n(uses labels)", A["supervised_reference"]["auroc"], "#999")]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
ax.axhline(0.5, color="k", ls="--", lw=1); ax.text(4.45, 0.51, "coin flip", ha="right", fontsize=8)
for i, b in enumerate(bars):
    ax.text(i, b[1] + 0.015, f"{b[1]:.2f}", ha="center", fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("AUROC (1.0 = perfect, 0.5 = chance)"); ax.tick_params(axis="x", labelsize=8)
ax.set_title("Capture 34 (v1's data): can the detectors tell malicious from benign?", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT / "v2" / "v2-capture34.png", dpi=150); plt.close(fig)

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
print(json.dumps(out["mean_over_captures"], indent=1)); print(out["v1_setting_capture_34"])
