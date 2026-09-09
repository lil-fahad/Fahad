from pathlib import Path


def test_windows_training_launcher_checks_readiness_then_runs_sequentially():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "start_training_windows.ps1").read_text(encoding="utf-8")

    assert "training-ready" in script
    required = ["train-ttm", "train-timesfm25", "train-chronos2", "train-kronos", "train-finbert"]
    positions = [script.index(command) for command in required]
    assert positions == sorted(positions)
    assert script.index("training-ready") < positions[0]
    assert "if (-not $Readiness.ready)" in script
    assert "--micro-batch-size 1" in script
    assert "ValidateSet(\"smoke\", \"max\")" in script
