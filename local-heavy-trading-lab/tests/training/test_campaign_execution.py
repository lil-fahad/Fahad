from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_campaign_runs_all_models_sequentially_and_isolates_failure(monkeypatch, tmp_path: Path):
    import heavy_lab.training.campaign as campaign

    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "train_parquet": "train.parquet",
                "validation_parquet": "validation.parquet",
                "finbert_train_jsonl": "finbert-train.jsonl",
                "finbert_validation_jsonl": "finbert-validation.jsonl",
                "profile": "smoke",
            }
        ),
        encoding="utf-8",
    )

    events: list[str] = []
    fake_hardware = SimpleNamespace(cuda=True, vram_gb=8.0, device="cuda")

    monkeypatch.setattr(campaign, "detect_hardware", lambda root: fake_hardware, raising=False)
    monkeypatch.setattr(
        campaign,
        "training_readiness",
        lambda **kwargs: {"ready": True, "missing": [], "hardware": {"cuda": True, "vram_gb": 8.0}},
    )
    monkeypatch.setattr(campaign, "_read_market_splits", lambda train, validation: ("TRAIN", "VALIDATION"), raising=False)
    monkeypatch.setattr(campaign, "_read_jsonl", lambda path: [{"text": str(path), "label": 1}], raising=False)
    monkeypatch.setattr(campaign, "_cleanup_cuda", lambda: events.append("cleanup"), raising=False)

    def ok(name):
        def runner(**kwargs):
            events.append(name)
            return SimpleNamespace(
                run_id=f"run-{name}",
                status="COMPLETED",
                checkpoint=f"checkpoint-{name}",
                metrics={"name": name},
            )
        return runner

    def fail_timesfm(**kwargs):
        events.append("timesfm25")
        raise RuntimeError("simulated timesfm failure")

    monkeypatch.setattr(campaign, "run_ttm_training", ok("ttm"))
    monkeypatch.setattr(campaign, "run_timesfm_training", fail_timesfm)
    monkeypatch.setattr(campaign, "run_chronos2_training", ok("chronos2"))
    monkeypatch.setattr(campaign, "run_kronos_training", ok("kronos"))
    monkeypatch.setattr(campaign, "run_finbert_training", ok("finbert"))

    summary = campaign.run_training_campaign(
        root=tmp_path,
        manifest_path=manifest,
        profile="smoke",
    )

    assert events == [
        "ttm", "cleanup",
        "timesfm25", "cleanup",
        "chronos2", "cleanup",
        "kronos", "cleanup",
        "finbert", "cleanup",
    ]
    assert summary["profile"] == "smoke"
    assert summary["models"]["ttm"]["status"] == "COMPLETED"
    assert summary["models"]["timesfm25"]["status"] == "FAILED"
    assert "simulated timesfm failure" in summary["models"]["timesfm25"]["error"]
    assert summary["models"]["chronos2"]["status"] == "COMPLETED"
    assert summary["models"]["kronos"]["status"] == "COMPLETED"
    assert summary["models"]["finbert"]["status"] == "COMPLETED"
