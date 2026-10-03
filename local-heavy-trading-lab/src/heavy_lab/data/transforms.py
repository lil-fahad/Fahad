from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


class FoldTransformer:
    """Fit normalization on one training fold and reuse it unchanged elsewhere."""

    def __init__(self, *, columns: Iterable[str], normalization: str = "standard") -> None:
        self.columns = tuple(str(column) for column in columns)
        if not self.columns:
            raise ValueError("columns must not be empty")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must not contain duplicates")
        if normalization not in {"standard", "robust", "none"}:
            raise ValueError("normalization must be standard, robust, or none")
        self.normalization = normalization
        self.center: dict[str, float] = {}
        self.scale: dict[str, float] = {}
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "FoldTransformer":
        missing = [column for column in self.columns if column not in train_df.columns]
        if missing:
            raise ValueError(f"missing transform columns: {missing}")

        if self.normalization == "none":
            self.center = {}
            self.scale = {}
            self._fitted = True
            return self

        center: dict[str, float] = {}
        scale: dict[str, float] = {}
        for column in self.columns:
            values = pd.to_numeric(train_df[column], errors="raise")
            if values.isna().any():
                raise ValueError(f"{column} contains NaN in training fold")

            if self.normalization == "standard":
                column_center = float(values.mean())
                column_scale = float(values.std(ddof=0))
            else:
                column_center = float(values.median())
                q75 = float(values.quantile(0.75))
                q25 = float(values.quantile(0.25))
                column_scale = q75 - q25

            if column_scale == 0.0:
                column_scale = 1.0
            center[column] = column_center
            scale[column] = column_scale

        self.center = center
        self.scale = scale
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FoldTransformer must be fit before transform")
        missing = [column for column in self.columns if column not in df.columns]
        if missing:
            raise ValueError(f"missing transform columns: {missing}")

        out = df.copy()
        if self.normalization == "none":
            return out

        for column in self.columns:
            out[column] = (pd.to_numeric(out[column], errors="raise") - self.center[column]) / self.scale[column]
        return out

    def to_dict(self) -> dict[str, object]:
        if not self._fitted:
            raise RuntimeError("FoldTransformer must be fit before serialization")
        return {
            "normalization": self.normalization,
            "columns": list(self.columns),
            "center": dict(self.center),
            "scale": dict(self.scale),
        }

    def save(self, path: Path) -> Path:
        payload = self.to_dict()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination
