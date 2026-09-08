from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabPaths:
    root: Path
    models_base: Path
    models_trained: Path
    checkpoints: Path
    data_raw: Path
    data_canonical: Path
    data_features: Path
    data_splits: Path
    artifacts: Path
    runs: Path

    @classmethod
    def from_root(cls, root: Path) -> "LabPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            models_base=root / "models" / "base",
            models_trained=root / "models" / "trained",
            checkpoints=root / "checkpoints",
            data_raw=root / "data" / "raw",
            data_canonical=root / "data" / "canonical",
            data_features=root / "data" / "features",
            data_splits=root / "data" / "splits",
            artifacts=root / "artifacts",
            runs=root / "runs",
        )

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.models_base,
            self.models_trained,
            self.checkpoints,
            self.data_raw,
            self.data_canonical,
            self.data_features,
            self.data_splits,
            self.artifacts,
            self.runs,
        ):
            path.mkdir(parents=True, exist_ok=True)
