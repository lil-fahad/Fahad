import json
from pathlib import Path

import pytest


def test_model_votes_agree_on_call():
    from telegram_bridge.model_ensemble import ModelVote, combine_votes

    result = combine_votes(
        technical_kind="call",
        technical_strength=0.70,
        votes=[
            ModelVote("chronos2", 0.80, 0.75, "forecast up"),
            ModelVote("timesfm2_5", 0.65, 0.70, "forecast up"),
            ModelVote("finbert", 0.55, 0.80, "positive news"),
        ],
    )
    assert result.kind == "call"
    assert result.confidence >= 0.5
    assert {item.source for item in result.votes} == {"chronos2", "timesfm2_5", "finbert"}


def test_model_disagreement_can_hold():
    from telegram_bridge.model_ensemble import ModelVote, combine_votes

    result = combine_votes(
        technical_kind="call",
        technical_strength=0.30,
        votes=[
            ModelVote("chronos2", 0.60, 0.70, "up"),
            ModelVote("timesfm2_5", -0.65, 0.75, "down"),
            ModelVote("finbert", -0.35, 0.70, "negative"),
        ],
    )
    assert result.kind == "hold"


def test_no_ai_votes_preserves_existing_technical_signal():
    from telegram_bridge.model_ensemble import combine_votes

    call = combine_votes("call", 0.44, [])
    put = combine_votes("put", 0.61, [])
    hold = combine_votes("hold", 0.10, [])
    assert (call.kind, put.kind, hold.kind) == ("call", "put", "hold")
    assert call.degraded is True


def test_model_ensemble_isolates_adapter_failure():
    from telegram_bridge.model_ensemble import ModelVote, OptionsModelEnsemble

    class Good:
        source = "chronos2"
        def vote(self, snapshot):
            return ModelVote(self.source, 0.7, 0.8, "ok")

    class Broken:
        source = "timesfm2_5"
        def vote(self, snapshot):
            raise RuntimeError("boom")

    ensemble = OptionsModelEnsemble(adapters=[Good(), Broken()])
    result = ensemble.evaluate({"closes_5m": [1.0, 1.1], "headlines": []}, "call", 0.5)
    assert result.kind == "call"
    assert [vote.source for vote in result.votes] == ["chronos2"]
    assert "timesfm2_5" in ensemble.last_errors


def _engine_fixture(tmp_path, direction="up"):
    import asyncio
    import time
    from telegram_bridge.config import Config, hash_password
    from telegram_bridge.storage import Store
    from tests.test_options_paper import FakeBridge, FakeFeed, TOKEN

    cfg = Config(bot_token=TOKEN, allowed_chat_ids=(101,), password_hash=hash_password("password"),
                 data_dir=tmp_path / "state", allow_send=False)
    store = Store(cfg.data_dir / "bridge.sqlite3")
    feed = FakeFeed(direction)
    bridge = FakeBridge(cfg, store, feed)
    return cfg, store, feed, bridge


def test_options_engine_uses_ensemble_direction_and_persists_components(tmp_path):
    from telegram_bridge.model_ensemble import EnsembleResult, ModelVote
    from telegram_bridge.options_paper import OptionsPaperEngine

    class ForcedPut:
        def evaluate(self, snapshot, technical_kind, technical_strength):
            return EnsembleResult("put", 0.81, -0.81, (ModelVote("chronos2", -1, 0.9, "down"),), False, "forced")
        def status(self):
            return {"available": ["chronos2"], "errors": {}}

    async def run():
        cfg, store, feed, bridge = _engine_fixture(tmp_path, "up")
        try:
            engine = OptionsPaperEngine(bridge, feed=feed, model_ensemble=ForcedPut())
            engine.set_running(101, True)
            result = await engine.once(symbols=("SPY",))
            assert result["action"] == "opened"
            assert result["position"]["option_type"] == "put"
            with store.lock:
                row = store.db.execute("SELECT decision,confidence,components FROM option_model_signals WHERE symbol='SPY'").fetchone()
            assert row["decision"] == "put"
            assert row["confidence"] == pytest.approx(0.81)
            assert json.loads(row["components"])[0]["source"] == "chronos2"
            await engine.close()
        finally:
            store.close()
    import asyncio
    asyncio.run(run())


def test_options_engine_falls_back_when_ensemble_raises(tmp_path):
    from telegram_bridge.options_paper import OptionsPaperEngine

    class Broken:
        def evaluate(self, snapshot, technical_kind, technical_strength):
            raise RuntimeError("model crash")
        def status(self):
            return {"available": [], "errors": {"ensemble": "model crash"}}

    async def run():
        cfg, store, feed, bridge = _engine_fixture(tmp_path, "up")
        try:
            engine = OptionsPaperEngine(bridge, feed=feed, model_ensemble=Broken())
            engine.set_running(101, True)
            result = await engine.once(symbols=("SPY",))
            assert result["action"] == "opened"
            assert result["position"]["option_type"] == "call"
            assert "model crash" in (engine.last_error or "")
            await engine.close()
        finally:
            store.close()
    import asyncio
    asyncio.run(run())


def test_forecast_vote_maps_short_horizon_return():
    from telegram_bridge.model_ensemble import forecast_vote
    up = forecast_vote("chronos2", 100.0, 100.8)
    down = forecast_vote("timesfm2_5", 100.0, 99.0)
    flat = forecast_vote("chronos2", 100.0, 100.01)
    assert up.score > 0 and up.confidence > 0
    assert down.score < 0 and down.confidence > up.confidence
    assert flat.confidence < 0.05


def test_config_accepts_optional_model_settings_and_loads_old_files(tmp_path):
    import json as _json
    from telegram_bridge.config import Config, hash_password, load_config
    from tests.test_options_paper import TOKEN

    base = {
        "bot_token": TOKEN,
        "allowed_chat_ids": [101],
        "password_hash": hash_password("password"),
        "data_dir": "state",
    }
    path = tmp_path / "config.json"
    path.write_text(_json.dumps(base))
    old = load_config(path)
    assert old.options_models_enabled is False
    assert old.options_model_device == "auto"
    assert old.options_model_cache_dir == tmp_path / "models"

    path.write_text(_json.dumps({**base, "options_models_enabled": True, "options_model_device": "cpu",
                                 "options_model_cache_dir": "ai-cache"}))
    new = load_config(path)
    assert new.options_models_enabled is True
    assert new.options_model_device == "cpu"
    assert new.options_model_cache_dir == tmp_path / "ai-cache"


def test_optionsmodels_command_is_private_owner_only(tmp_path):
    from telegram_bridge.commands import parse_update
    from telegram_bridge.config import Config, hash_password
    from tests.test_commands import command_update
    from tests.test_options_paper import TOKEN

    cfg = Config(bot_token=TOKEN, allowed_chat_ids=(101,), password_hash=hash_password("password"),
                 data_dir=tmp_path / "state")
    parsed = parse_update(command_update(text="/optionsmodels"), cfg)
    assert parsed and parsed["verb"] == "optionsmodels"
    assert parse_update(command_update(chat_id=999, text="/optionsmodels"), cfg) is None


def test_default_model_ensemble_is_lazy_and_names_all_models(tmp_path):
    from telegram_bridge.model_ensemble import build_default_ensemble
    ensemble = build_default_ensemble(tmp_path / "models", "cpu")
    status = ensemble.status()
    assert set(status["available"]) == {"chronos2", "timesfm2_5", "finbert"}
    assert status["errors"] == {}


def test_configure_options_models_persists_without_downloading(tmp_path):
    import json as _json
    from telegram_bridge.cli import configure_options_models
    from telegram_bridge.config import hash_password, load_config
    from tests.test_options_paper import TOKEN

    path = tmp_path / "config.json"
    path.write_text(_json.dumps({
        "bot_token": TOKEN,
        "allowed_chat_ids": [101],
        "password_hash": hash_password("password"),
        "data_dir": "state",
    }))
    configure_options_models(path, True, "cpu", Path("model-cache"))
    cfg = load_config(path)
    assert cfg.options_models_enabled is True
    assert cfg.options_model_device == "cpu"
    assert cfg.options_model_cache_dir == tmp_path / "model-cache"


def test_keyless_news_parser_deduplicates_headlines():
    from telegram_bridge.options_paper import _extract_headlines
    payload = {"news": [
        {"title": "Markets rise on data"},
        {"title": "Markets rise on data"},
        {"title": "Fed signals patience"},
        {"title": ""},
    ]}
    assert _extract_headlines(payload) == ["Markets rise on data", "Fed signals patience"]


def test_optionsmodels_status_command_reports_loaded_components():
    import asyncio
    from types import SimpleNamespace
    from telegram_bridge.commands import options_command

    class Engine:
        def model_status(self):
            return {"enabled": True, "available": ["chronos2", "timesfm2_5", "finbert"],
                    "loaded": ["chronos2"], "errors": {}, "device": "cpu"}

    bridge = SimpleNamespace(options_paper=Engine())
    job = {"request": {"verb": "optionsmodels"}, "chat_id": 101}
    text = asyncio.run(options_command(bridge, job))
    assert "مفعلة" in text
    assert "chronos2" in text and "finbert" in text


def test_bridge_wires_lazy_models_when_enabled(tmp_path):
    from telegram_bridge.config import Config, hash_password
    from telegram_bridge.storage import Store
    from telegram_bridge.telegram import Bridge
    from tests.test_options_paper import TOKEN

    class DummyAPI:
        async def call(self, *args, **kwargs):
            raise AssertionError("network must not be called during construction")
        async def close(self):
            return None

    cfg = Config(bot_token=TOKEN, allowed_chat_ids=(101,), password_hash=hash_password("password"),
                 data_dir=tmp_path / "state", options_models_enabled=True,
                 options_model_device="cpu", options_model_cache_dir=tmp_path / "models")
    store = Store(cfg.data_dir / "bridge.sqlite3")
    try:
        bridge = Bridge(cfg, store, DummyAPI())
        assert bridge.options_paper.model_ensemble is not None
        assert set(bridge.options_paper.model_status()["available"]) == {"chronos2", "timesfm2_5", "finbert"}
    finally:
        store.close()


def test_download_options_models_uses_configured_cache_and_fails_closed_on_missing_component(tmp_path):
    import json as _json
    from telegram_bridge.cli import download_options_models
    from telegram_bridge.config import hash_password
    from tests.test_options_paper import TOKEN

    path = tmp_path / "config.json"
    path.write_text(_json.dumps({
        "bot_token": TOKEN,
        "allowed_chat_ids": [101],
        "password_hash": hash_password("password"),
        "data_dir": "state",
        "options_models_enabled": True,
        "options_model_device": "cpu",
        "options_model_cache_dir": "models",
    }))

    class FakeEnsemble:
        def warmup(self):
            return {"available": ["chronos2", "timesfm2_5", "finbert"], "loaded": ["chronos2"],
                    "errors": {"timesfm2_5": "missing package"}}

    with pytest.raises(ValueError, match="timesfm2_5"):
        download_options_models(path, builder=lambda cache, device: FakeEnsemble())


def test_cli_help_exposes_model_configuration_commands():
    import subprocess, sys
    result = subprocess.run([sys.executable, "-m", "telegram_bridge", "--help"],
                            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert result.returncode == 0
    assert "configure-options-models" in result.stdout
    assert "download-options-models" in result.stdout


def test_ensemble_releases_low_memory_adapter_after_vote():
    from telegram_bridge.model_ensemble import ModelVote, OptionsModelEnsemble

    class Releasable:
        source = "chronos2"
        release_after_vote = True
        def __init__(self):
            self.released = 0
        def vote(self, snapshot):
            return ModelVote(self.source, 1.0, 0.5, "up")
        def release(self):
            self.released += 1

    adapter = Releasable()
    ensemble = OptionsModelEnsemble([adapter])
    ensemble.evaluate({"closes_5m": [1.0] * 20}, "call", 0.5)
    assert adapter.released == 1
