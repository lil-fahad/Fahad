from __future__ import annotations

import json
from pathlib import Path


def _doctor():
    return {
        "hardware": {"device": "cuda", "gpu_name": "RTX 3070 Ti", "vram_gb": 8.0},
        "training_profile": "low-vram",
    }


def test_status_is_owner_only_and_live_trading_is_disabled(tmp_path: Path):
    from heavy_lab.bot.commands import BotCommandProcessor

    bot = BotCommandProcessor(
        root=tmp_path,
        owner_chat_id=123,
        doctor_provider=_doctor,
        readiness_provider=lambda: {"ready": False, "missing": ["train_parquet"]},
    )

    denied = bot.handle(chat_id=999, text="/status")
    assert denied.action is None
    assert "غير مصرح" in denied.text

    reply = bot.handle(chat_id=123, text="/status")
    assert "RTX 3070 Ti" in reply.text
    assert "8" in reply.text
    assert "LIVE trading: DISABLED" in reply.text
    assert "PAPER/RESEARCH" in reply.text


def test_trainingready_reports_missing_requirements(tmp_path: Path):
    from heavy_lab.bot.commands import BotCommandProcessor

    bot = BotCommandProcessor(
        root=tmp_path,
        owner_chat_id=123,
        doctor_provider=_doctor,
        readiness_provider=lambda: {
            "ready": False,
            "missing": ["model:chronos2", "train_parquet"],
        },
    )
    reply = bot.handle(chat_id=123, text="/trainingready")
    assert "NOT READY" in reply.text
    assert "model:chronos2" in reply.text
    assert "train_parquet" in reply.text
    assert reply.action is None


def test_runs_lists_latest_registry_states(tmp_path: Path):
    from heavy_lab.bot.commands import BotCommandProcessor

    runs = tmp_path / "runs"
    old = runs / "20261001-ttm-old"
    new = runs / "20261003-timesfm-new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "state.json").write_text(json.dumps({
        "run_id": old.name,
        "kind": "ttm",
        "status": "COMPLETED",
        "updated_at_utc": "2026-10-01T10:00:00+00:00",
    }), encoding="utf-8")
    (new / "state.json").write_text(json.dumps({
        "run_id": new.name,
        "kind": "timesfm25",
        "status": "RUNNING",
        "updated_at_utc": "2026-10-03T09:00:00+00:00",
    }), encoding="utf-8")

    bot = BotCommandProcessor(
        root=tmp_path,
        owner_chat_id=123,
        doctor_provider=_doctor,
        readiness_provider=lambda: {"ready": True, "missing": []},
    )
    reply = bot.handle(chat_id=123, text="/runs")
    assert reply.text.index(new.name) < reply.text.index(old.name)
    assert "RUNNING" in reply.text
    assert "COMPLETED" in reply.text


def test_trainall_requires_ready_state_and_existing_manifest(tmp_path: Path):
    from heavy_lab.bot.commands import BotCommandProcessor

    manifest = tmp_path / "campaign.json"
    manifest.write_text("{}", encoding="utf-8")

    not_ready = BotCommandProcessor(
        root=tmp_path,
        owner_chat_id=123,
        campaign_manifest=manifest,
        doctor_provider=_doctor,
        readiness_provider=lambda: {"ready": False, "missing": ["model:ttm"]},
    )
    blocked = not_ready.handle(chat_id=123, text="/trainall max")
    assert blocked.action is None
    assert "NOT READY" in blocked.text

    ready = BotCommandProcessor(
        root=tmp_path,
        owner_chat_id=123,
        campaign_manifest=manifest,
        doctor_provider=_doctor,
        readiness_provider=lambda: {"ready": True, "missing": []},
    )
    reply = ready.handle(chat_id=123, text="/trainall max")
    assert reply.action is not None
    assert reply.action.kind == "train-all"
    assert reply.action.profile == "max"
    assert reply.action.manifest == manifest.resolve()
    assert "MAX" in reply.text


def test_help_exposes_only_research_and_training_commands(tmp_path: Path):
    from heavy_lab.bot.commands import BotCommandProcessor

    bot = BotCommandProcessor(
        root=tmp_path,
        owner_chat_id=123,
        doctor_provider=_doctor,
        readiness_provider=lambda: {"ready": True, "missing": []},
    )
    text = bot.handle(chat_id=123, text="/help").text.lower()
    for command in ("/status", "/trainingready", "/runs", "/trainall", "/trainstatus"):
        assert command in text
    for forbidden in ("/buy", "/sell", "/order", "/broker"):
        assert forbidden not in text
