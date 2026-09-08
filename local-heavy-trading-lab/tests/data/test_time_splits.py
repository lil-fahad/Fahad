def test_walk_forward_split_keeps_gap_before_test():
    from heavy_lab.data.walkforward import make_walk_forward_splits

    splits = make_walk_forward_splits(length=100, train_size=50, test_size=10, gap=3, step=10)
    assert splits
    for train_rows, test_rows in splits:
        assert len(test_rows) == 10
        assert train_rows[-1] + 3 < test_rows[0]
        assert not set(train_rows).intersection(test_rows)
