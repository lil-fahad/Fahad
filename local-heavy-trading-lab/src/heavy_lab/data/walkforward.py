from __future__ import annotations


def make_walk_forward_splits(
    *,
    length: int,
    train_size: int,
    test_size: int,
    gap: int,
    step: int,
) -> list[tuple[list[int], list[int]]]:
    if min(length, train_size, test_size, step) <= 0:
        raise ValueError("length, train_size, test_size, and step must be positive")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if train_size + gap + test_size > length:
        return []

    splits: list[tuple[list[int], list[int]]] = []
    test_start = train_size + gap
    while test_start + test_size <= length:
        train_end = test_start - gap
        train_start = max(0, train_end - train_size)
        train_rows = list(range(train_start, train_end))
        test_rows = list(range(test_start, test_start + test_size))
        if train_rows and test_rows:
            splits.append((train_rows, test_rows))
        test_start += step
    return splits
