import numpy as np
import pandas as pd
import pytest

import anomaly_features as af

HEADER = "ts|uid|id.orig_h|id.orig_p|id.resp_h|id.resp_p|proto|service|duration|orig_bytes|resp_bytes|conn_state|local_orig|local_resp|missed_bytes|history|orig_pkts|orig_ip_bytes|resp_pkts|resp_ip_bytes|tunnel_parents|label|detailed-label"


def write_capture(tmp_path, rows, name="CTU-IoT-Malware-Capture-9-1conn.log.labeled.csv"):
    p = tmp_path / name
    p.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return p


def row(ts, src="1.1.1.1", dport=80, proto="tcp", state="SF", dur="0.5", ob="10", rb="20", label="Benign", detail="-"):
    return f"{ts}|U{ts}|{src}|1234|2.2.2.2|{dport}|{proto}|-|{dur}|{ob}|{rb}|{state}|-|-|0|S|3|180|2|100|-|{label}|{detail}"


def test_load_capture_reads_dashes_as_missing_and_makes_a_binary_label(tmp_path):
    p = write_capture(tmp_path, [row(1, dur="-", ob="-"), row(2, label="Malicious   C&C", detail="C&C"), row(3, label="Malicious", detail="Attack")])
    df = af.load_capture(p)
    assert df["malicious"].tolist() == [0, 1, 1]
    assert df["capture"].tolist() == ["9"] * 3
    assert np.isnan(df.loc[0, "duration"]) and np.isnan(df.loc[0, "orig_bytes"])
    assert df.loc[1, "attack"] == "C&C"


def test_features_never_include_labels_addresses_or_time(tmp_path):
    df = af.load_capture(write_capture(tmp_path, [row(1), row(2)]))
    X = af.build_features(df)
    banned = ("label", "malicious", "attack", "uid", "orig_h", "resp_h", "ts", "capture")
    assert not [c for c in X.columns if any(c == b or c.startswith(b + "_") for b in banned)]


def test_feature_columns_are_the_same_for_every_capture(tmp_path):
    a = af.build_features(af.load_capture(write_capture(tmp_path, [row(1, proto="tcp", state="SF")], "CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv")))
    b = af.build_features(af.load_capture(write_capture(tmp_path, [row(1, proto="udp", state="S0")], "CTU-IoT-Malware-Capture-2-1conn.log.labeled.csv")))
    assert list(a.columns) == list(b.columns)


def test_recent_connection_count_counts_the_past_ten_seconds_only(tmp_path):
    rows = [row(100.0), row(103.0), row(105.0), row(120.0), row(100.0, src="9.9.9.9")]
    df = af.load_capture(write_capture(tmp_path, rows)).sort_values("ts").reset_index(drop=True)
    counts = af.recent_connection_counts(df)
    by_ts = {(r.ts, r.orig_h): c for r, c in zip(df.itertuples(), counts)}
    assert by_ts[(100.0, "1.1.1.1")] == 1 and by_ts[(103.0, "1.1.1.1")] == 2
    assert by_ts[(105.0, "1.1.1.1")] == 3 and by_ts[(120.0, "1.1.1.1")] == 1
    assert by_ts[(100.0, "9.9.9.9")] == 1


def test_leave_one_capture_out_never_trains_on_the_held_out_capture():
    df = pd.DataFrame({"capture": ["20"] * 3 + ["21"] * 3 + ["42"] * 3})
    folds = list(af.leave_one_capture_out(df))
    assert [f[0] for f in folds] == ["20", "21", "42"]
    for held, tr, te in folds:
        assert set(df.loc[tr, "capture"]) == set(df["capture"]) - {held}
        assert set(df.loc[te, "capture"]) == {held}


def test_recall_at_fpr_picks_the_threshold_that_allows_that_false_alarm_rate():
    y = np.array([0] * 98 + [1, 1])
    score = np.concatenate([np.linspace(0, 0.5, 98), [0.9, 0.3]])
    assert af.recall_at_fpr(y, score, fpr=0.01) == pytest.approx(0.5)   # threshold above all but the top benign: only the 0.9 positive
    assert af.recall_at_fpr(y, score, fpr=0.5) == pytest.approx(1.0)


def test_time_blocks_are_contiguous_cover_every_row_once_and_keep_time_order():
    blocks = af.time_blocks(23, 5)
    flat = np.concatenate(blocks)
    assert len(blocks) == 5 and sorted(flat.tolist()) == list(range(23))
    assert all(b[-1] + 1 == nxt[0] for b, nxt in zip(blocks, blocks[1:]))   # block k ends where block k+1 starts


def test_cluster_majority_score_is_perfect_for_pure_clusters_and_chance_for_mixed_ones():
    y = np.array([0, 0, 1, 1, 1, 1])
    assert af.cluster_majority_balanced_accuracy(y, np.array([0, 0, 1, 1, 1, 1])) == pytest.approx(1.0)
    assert af.cluster_majority_balanced_accuracy(y, np.array([0, 0, 0, 0, 0, 0])) == pytest.approx(0.5)
