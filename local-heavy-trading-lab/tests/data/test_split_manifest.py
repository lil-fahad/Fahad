from __future__ import annotations

import pandas as pd


def test_split_manifest_records_boundaries_and_stable_hashes():
    from heavy_lab.data.splits import build_split_manifest, build_walk_forward_splits

    index = pd.date_range("2026-01-02 14:30", periods=30, freq="5min", tz="UTC")
    folds = build_walk_forward_splits(
        index=index,
        train_size=10,
        val_size=4,
        test_size=4,
        horizon=2,
        embargo=1,
    )

    first = build_split_manifest(index=index, folds=folds, horizon=2, embargo=1)
    second = build_split_manifest(index=index, folds=folds, horizon=2, embargo=1)

    assert first == second
    assert first["horizon"] == 2
    assert first["embargo"] == 1
    assert first["folds"][0]["train"]["start_utc"] == "2026-01-02T14:30:00+00:00"
    assert first["folds"][0]["train"]["end_utc"] == "2026-01-02T15:05:00+00:00"
    assert first["folds"][0]["validation"]["start_utc"] == "2026-01-02T15:20:00+00:00"
    assert first["folds"][0]["test"]["start_utc"] == "2026-01-02T15:45:00+00:00"

    for split_name in ("train", "validation", "test"):
        digest = first["folds"][0][split_name]["row_hash"]
        assert len(digest) == 64
        int(digest, 16)


def test_split_manifest_hash_changes_when_timestamp_changes():
    from heavy_lab.data.splits import build_split_manifest, build_walk_forward_splits

    index = pd.date_range("2026-01-02 14:30", periods=30, freq="5min", tz="UTC")
    folds = build_walk_forward_splits(
        index=index,
        train_size=10,
        val_size=4,
        test_size=4,
        horizon=2,
        embargo=1,
    )

    original = build_split_manifest(index=index, folds=folds, horizon=2, embargo=1)
    changed_index = index.copy()
    changed_index = changed_index.where(changed_index != changed_index[0], changed_index[0] + pd.Timedelta(seconds=1))
    changed = build_split_manifest(index=changed_index, folds=folds, horizon=2, embargo=1)

    assert original["folds"][0]["train"]["row_hash"] != changed["folds"][0]["train"]["row_hash"]
