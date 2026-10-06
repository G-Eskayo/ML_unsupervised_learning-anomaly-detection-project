# Anomaly detection v2: plan

## What v1 was

A course assignment on one capture of the CTU-IoT-Malware dataset (capture 34-1: 23,145 connections). An `IsolationForest` with `contamination=0.01`, scored against the labels with a confusion matrix: it found 16 of 21,222 malicious connections (recall 0.00, accuracy 0.07). Preserved as the git tag `v1-coursework`.

## What was wrong

**The data does not fit the method.** Isolation Forest assumes anomalies are rare. Capture 34 is **92% malicious**, so "flag the rarest 1%" selects the wrong rows by construction. The same dataset contains captures that *are* the textbook setting:

| Capture | Connections | Malicious |
|---|---|---|
| 20-1 | 3,209 | 16 (0.5%) |
| 21-1 | 3,286 | 14 (0.4%) |
| 42-1 | 4,426 | 6 (0.1%) |
| 44-1 | 237 | 26 (11%) |
| 8-1 | 10,403 | 8,222 (79%) |
| 3-1 | 156,103 | 151,567 (97%) |
| 34-1 (v1) | 23,145 | 21,222 (92%) |

Other problems: the whole file was used for fitting and for scoring (no held-out data); one threshold, no precision/recall trade-off curve; no comparison with a simple supervised baseline; accuracy reported on a 92/8 imbalance.

## v2 goals

1. State the problem correctly: anomaly detection on traffic that is mostly benign (captures 20, 21, 42, 44), and say plainly why capture 34 is a different problem.
2. Train on benign-looking traffic and **hold out whole captures**: fit on some captures, evaluate on a capture the detector never saw.
3. Judge with the right metrics: precision-recall curve and average precision, recall at a fixed false-alarm rate. Not accuracy.
4. Compare Isolation Forest against at least one other unsupervised detector and a supervised baseline (to show what labels buy).
5. Error analysis: which malicious connections are caught, which are missed, and what they look like (scan, C&C, DDoS).

## Non-goals

No deep models, no live traffic, no claim about real-world networks beyond these captures. The malicious count in the mostly-benign captures is tiny (6 to 26), so every number will carry a wide uncertainty and the README will say so.

## Layout

- `anomaly_features.py`: loading the pipe-delimited captures, features (no label leakage), hold-out splits (tested)
- `tests/`
- `v2/run.py`: regenerates `v2/results.json` and the figures
- README: "Version history" table, v1 tag vs v2

## Data

Not committed (3.4 GB). `scripts/fetch-data.sh` downloads the Kaggle archive and checks its size; the captures are expected in `$DATA_DIR` (default `~/Developer/data/anomaly-detection-v2`).

## Success criteria

The README shows v1's number next to v2's held-out numbers with their uncertainty, a plain statement of what changed, and `python v2/run.py` reproduces every figure.
