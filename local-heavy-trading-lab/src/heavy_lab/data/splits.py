from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Fold:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def build_walk_forward_splits(
    index: Sequence,
    train_size: int,
    val_size: int,
    test_size: int,
    horizon: int,
    embargo: int,
) -> list[Fold]:
    """Build deterministic chronological folds with purge and embargo.

    The final horizon rows of each raw training window are purged because
    their labels can overlap the validation period. Embargo rows are left
    unused between validation and test, and again before the next fold starts.
    """
    n = len(index)
    if n <= 0:
        raise ValueError("index must not be empty")
    if min(train_size, val_size, test_size, horizon) <= 0:
        raise ValueError("train_size, val_size, test_size, and horizon must be positive")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    if horizon >= train_size:
        raise ValueError("horizon must be smaller than train_size")

    folds: list[Fold] = []
    cursor = int(train_size)

    while True:
        validation_start = cursor
        validation_end = validation_start + int(val_size)
        test_start = validation_end + int(embargo)
        test_end = test_start + int(test_size)
        if test_end > n:
            break

        raw_train_start = max(0, validation_start - int(train_size))
        purged_train_end = validation_start - int(horizon)
        if purged_train_end <= raw_train_start:
            break

        train_indices = tuple(range(raw_train_start, purged_train_end))
        validation_indices = tuple(range(validation_start, validation_end))
        test_indices = tuple(range(test_start, test_end))

        folds.append(
            Fold(
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
            )
        )

        # Walk forward beyond this test window and its post-test embargo.
        cursor = test_end + int(embargo)

    return folds
