from pathlib import Path


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
