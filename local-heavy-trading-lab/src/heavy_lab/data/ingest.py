from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_raw_payload(
    root: Path,
    *,
    source: str,
    symbol: str,
    interval: str,
    payload: dict,
) -> Path:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    directory = Path(root) / "raw" / source / symbol / interval
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if not path.exists():
        path.write_bytes(canonical)
    return path
