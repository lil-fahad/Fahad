from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets

from heavy_lab.checkpoints import is_valid_checkpoint


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    kind: str
    status: str
    config: dict
    latest_checkpoint: str | None
    created_at_utc: str
    updated_at_utc: str


class RunRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _write_state(self, run_dir: Path, payload: dict) -> None:
        payload = dict(payload)
        payload["updated_at_utc"] = self._now()
        self._atomic_json(run_dir / "state.json", payload)

    def start(self, kind: str, config: dict) -> RunRecord:
        created = self._now()
        run_id = f"{created[:10].replace('-', '')}-{kind}-{secrets.token_hex(4)}"
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        self._atomic_json(run_dir / "config.json", dict(config))
        self._atomic_json(run_dir / "hardware.json", {})
        self._atomic_json(run_dir / "git.json", {})
        (run_dir / "events.jsonl").write_text("", encoding="utf-8")
        state = {
            "run_id": run_id,
            "kind": str(kind),
            "status": "RUNNING",
            "latest_checkpoint": None,
            "created_at_utc": created,
            "updated_at_utc": created,
        }
        self._atomic_json(run_dir / "state.json", state)
        return RunRecord(run_id, str(kind), "RUNNING", dict(config), None, created, created)

    def append_event(self, run_id: str, event: dict) -> None:
        run_dir = self._run_dir(run_id)
        event_path = run_dir / "events.jsonl"
        if not event_path.is_file():
            raise FileNotFoundError(f"Unknown run: {run_id}")
        payload = dict(event)
        payload.setdefault("at", self._now())
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def mark_checkpoint(self, run_id: str, checkpoint: Path | str) -> None:
        run_dir = self._run_dir(run_id)
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        state["latest_checkpoint"] = str(Path(checkpoint).expanduser().resolve())
        self._write_state(run_dir, state)
        self.append_event(run_id, {"event": "checkpoint", "path": state["latest_checkpoint"]})

    def set_status(self, run_id: str, status: str) -> None:
        if status not in {"RUNNING", "INTERRUPTED", "FAILED", "COMPLETED"}:
            raise ValueError(f"Invalid run status: {status}")
        run_dir = self._run_dir(run_id)
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        state["status"] = status
        self._write_state(run_dir, state)

    def resume(self, run_id: str) -> RunRecord:
        run_dir = self._run_dir(run_id)
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        checkpoint = state.get("latest_checkpoint")
        if not is_valid_checkpoint(checkpoint):
            checkpoint = None
        return RunRecord(
            run_id=state["run_id"],
            kind=state["kind"],
            status=state["status"],
            config=config,
            latest_checkpoint=checkpoint,
            created_at_utc=state["created_at_utc"],
            updated_at_utc=state["updated_at_utc"],
        )
