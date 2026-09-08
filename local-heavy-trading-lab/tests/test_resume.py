from pathlib import Path
import json


def test_resume_returns_latest_valid_checkpoint(tmp_path: Path):
    from heavy_lab.runs import RunRegistry

    reg = RunRegistry(tmp_path)
    run = reg.start("timesfm25", {"seed": 7})
    ckpt = tmp_path / "checkpoint-20"
    ckpt.mkdir()
    (ckpt / "checkpoint.ok").write_text("ok", encoding="utf-8")
    reg.mark_checkpoint(run.run_id, ckpt)

    resumed = reg.resume(run.run_id)
    assert resumed.latest_checkpoint == str(ckpt.resolve())
    assert resumed.kind == "timesfm25"


def test_resume_rejects_missing_checkpoint_marker(tmp_path: Path):
    from heavy_lab.runs import RunRegistry

    reg = RunRegistry(tmp_path)
    run = reg.start("chronos2", {"seed": 9})
    ckpt = tmp_path / "bad-checkpoint"
    ckpt.mkdir()
    reg.mark_checkpoint(run.run_id, ckpt)

    resumed = reg.resume(run.run_id)
    assert resumed.latest_checkpoint is None


def test_append_event_persists_failure_reason(tmp_path: Path):
    from heavy_lab.runs import RunRegistry

    reg = RunRegistry(tmp_path)
    run = reg.start("timesfm25", {"seed": 11})
    reg.append_event(run.run_id, {"event": "training_failed", "error": "cuda oom"})

    event_path = tmp_path / run.run_id / "events.jsonl"
    lines = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines[-1]["event"] == "training_failed"
    assert lines[-1]["error"] == "cuda oom"
    assert "at" in lines[-1]
