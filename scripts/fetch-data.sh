#!/bin/sh
# Downloads the Kaggle archive (about 755 MB) and unpacks the captures v2 uses into $DATA_DIR.
# Keep DATA_DIR outside iCloud-synced folders (~/Documents, ~/Desktop): the full archive is 3.4 GB.
set -eu
DATA_DIR="${DATA_DIR:-$HOME/Developer/data/anomaly-detection-v2}"
mkdir -p "$DATA_DIR"
[ -s "$DATA_DIR/dataset.zip" ] || curl -L -o "$DATA_DIR/dataset.zip" \
  https://www.kaggle.com/api/v1/datasets/download/agungpambudi/network-malware-detection-connection-analysis
for c in 8 20 21 34 42 44; do
  unzip -q -n "$DATA_DIR/dataset.zip" "CTU-IoT-Malware-Capture-$c-1conn.log.labeled.csv" -d "$DATA_DIR"
done
echo "captures in $DATA_DIR"
