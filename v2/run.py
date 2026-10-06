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
results, BENIGN_DOMINATED, methods = out["held_out_captures"], ex.BENIGN_DOMINATED, ex.METHODS
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
print(json.dumps(out["mean_over_captures"], indent=1)); print(out["v1_setting_capture_34"])
