from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def make_supervised_windows(
    values: Sequence[T],
    *,
    context_length: int,
    prediction_length: int,
    stride: int = 1,
) -> list[tuple[list[T], list[T]]]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    total = context_length + prediction_length
    if len(values) < total:
        return []

    windows: list[tuple[list[T], list[T]]] = []
    for start in range(0, len(values) - total + 1, stride):
        origin = start + context_length
        context = list(values[start:origin])
        target = list(values[origin : origin + prediction_length])
        windows.append((context, target))
    return windows
