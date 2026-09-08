from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    source: str
    source_revision: str
    raw_sha256: str
    created_at_utc: str
    rows: int
    symbols: tuple[str, ...]
    interval: str
    start_utc: str
    end_utc: str

    def as_dict(self) -> dict:
        data = asdict(self)
        data["symbols"] = list(self.symbols)
        return data

    def write_atomic(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
        return target
