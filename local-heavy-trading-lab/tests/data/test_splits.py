from __future__ import annotations

import pandas as pd


def test_walk_forward_purges_horizon_and_embargoes_before_test():
    from heavy_lab.data.splits import build_walk_forward_splits

    index = pd.date_range("2026-01-02 14:30", periods=30, freq="5min", tz="UTC")
    folds = build_walk_forward_splits(
        index=index,
        train_size=10,
        val_size=4,
        test_size=4,
        horizon=2,
        embargo=1,
    )

    assert folds
    first = folds[0]
    assert first.train_indices == tuple(range(0, 8))
    assert first.validation_indices == tuple(range(10, 14))
    assert first.test_indices == tuple(range(15, 19))
    assert set(first.train_indices).isdisjoint(first.validation_indices)
    assert set(first.validation_indices).isdisjoint(first.test_indices)
    assert 8 not in first.train_indices
    assert 9 not in first.train_indices
    assert 14 not in first.validation_indices
    assert 14 not in first.test_indices


def test_walk_forward_is_deterministic_and_chronological():
    from heavy_lab.data.splits import build_walk_forward_splits

    index = pd.date_range("2026-01-02", periods=50, freq="5min", tz="UTC")
    kwargs = dict(
        index=index,
        train_size=12,
        val_size=5,
        test_size=5,
        horizon=3,
        embargo=2,
    )

    a = build_walk_forward_splits(**kwargs)
    b = build_walk_forward_splits(**kwargs)

    assert a == b
    for fold in a:
        assert max(fold.train_indices) < min(fold.validation_indices)
        assert max(fold.validation_indices) < min(fold.test_indices)
