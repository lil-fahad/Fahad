from __future__ import annotations

from typing import Any

import pandas as pd

from heavy_lab.data.schema import validate_canonical_frame


def build_ttm_examples(
    frame: pd.DataFrame,
    *,
    context_length: int,
    prediction_length: int,
    stride: int = 1,
    value_column: str = "close",
) -> list[dict[str, Any]]:
    """Convert canonical bars into chronological univariate TTM examples.

    Windows are constructed independently per symbol. The forecast origin is
    always the final context timestamp and targets begin strictly after it.
    Metadata is retained for auditing; Hugging Face Trainer removes columns not
    accepted by the model forward signature.
    """
    validate_canonical_frame(frame)
    if context_length <= 0 or prediction_length <= 0 or stride <= 0:
        raise ValueError("context_length, prediction_length, and stride must be positive")
    if value_column not in frame.columns:
        raise ValueError(f"missing value column: {value_column}")

    examples: list[dict[str, Any]] = []
    total = context_length + prediction_length
    for symbol, group in frame.groupby("symbol", sort=False):
        ordered = group.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
        if len(ordered) < total:
            continue
        for start in range(0, len(ordered) - total + 1, stride):
            origin = start + context_length
            target_end = origin + prediction_length
            context_slice = ordered.iloc[start:origin]
            target_slice = ordered.iloc[origin:target_end]
            past_values = [[float(value)] for value in context_slice[value_column].tolist()]
            future_values = [[float(value)] for value in target_slice[value_column].tolist()]
            examples.append(
                {
                    "past_values": past_values,
                    "future_values": future_values,
                    "past_observed_mask": [[True] for _ in past_values],
                    "future_observed_mask": [[True] for _ in future_values],
                    "symbol": str(symbol),
                    "forecast_origin_utc": context_slice["timestamp_utc"].iloc[-1],
                    "target_end_utc": target_slice["timestamp_utc"].iloc[-1],
                }
            )
    return examples
