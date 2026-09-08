from __future__ import annotations

from pathlib import Path


VALID_MARKERS = ("checkpoint.ok", "trainer_state.json")


def is_valid_checkpoint(path: Path | str | None) -> bool:
    if not path:
        return False
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return False
    return any((p / marker).is_file() for marker in VALID_MARKERS)
