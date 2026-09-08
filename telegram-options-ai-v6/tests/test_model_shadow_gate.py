import asyncio


def test_default_ensemble_runs_in_shadow_mode(tmp_path):
    from telegram_bridge.model_ensemble import build_default_ensemble

    ensemble = build_default_ensemble(tmp_path / "models", "cpu")
    assert ensemble.shadow_mode is True


def test_shadow_model_signal_is_persisted_but_does_not_replace_technical_trade(tmp_path):
    import json
    import pytest

    from telegram_bridge.model_ensemble import EnsembleResult, ModelVote
    from telegram_bridge.options_paper import OptionsPaperEngine
    from tests.test_model_ensemble import _engine_fixture

    class ForcedPutShadow:
        shadow_mode = True

        def evaluate(self, snapshot, technical_kind, technical_strength):
            assert technical_kind == "call"
            return EnsembleResult(
                "put",
                0.91,
                -0.91,
                (ModelVote("timesfm2_5", -1.0, 0.95, "shadow says down"),),
                False,
                "shadow forced put",
            )

        def status(self):
            return {"available": ["timesfm2_5"], "errors": {}}

    async def run():
        cfg, store, feed, bridge = _engine_fixture(tmp_path, "up")
        try:
            engine = OptionsPaperEngine(bridge, feed=feed, model_ensemble=ForcedPutShadow())
            engine.set_running(101, True)
            result = await engine.once(symbols=("SPY",))

            assert result["action"] == "opened"
            # Technical signal remains authoritative while the heavy model is shadow-only.
            assert result["position"]["option_type"] == "call"

            with store.lock:
                row = store.db.execute(
                    "SELECT decision,confidence,components FROM option_model_signals WHERE symbol='SPY'"
                ).fetchone()
            assert row["decision"] == "put"
            assert row["confidence"] == pytest.approx(0.91)
            assert json.loads(row["components"])[0]["source"] == "timesfm2_5"
            await engine.close()
        finally:
            store.close()

    asyncio.run(run())


def test_chronos_adapter_uses_tensor_native_quantile_api(tmp_path):
    import torch

    from telegram_bridge.model_ensemble import Chronos2Adapter

    class FakePipeline:
        def __init__(self):
            self.calls = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size):
            self.calls.append(
                {
                    "inputs": inputs,
                    "prediction_length": prediction_length,
                    "quantile_levels": quantile_levels,
                    "batch_size": batch_size,
                }
            )
            quantiles = torch.zeros((1, prediction_length, len(quantile_levels)), dtype=torch.float32)
            mean = torch.tensor([[100.1, 100.2, 100.3, 100.4, 100.5, 100.6]], dtype=torch.float32)
            return quantiles, mean

    adapter = Chronos2Adapter(tmp_path / "chronos", "cpu")
    adapter.pipeline = FakePipeline()
    adapter.torch = torch

    vote = adapter.vote({
        "symbol": "SPY",
        "closes_5m": [100.0] * 60,
        "timestamp": 1_700_000_000,
    })

    assert vote is not None
    assert vote.source == "chronos2"
    assert vote.score > 0
    assert adapter.pipeline.calls[0]["prediction_length"] == 6
    assert adapter.pipeline.calls[0]["batch_size"] == 1
    assert len(adapter.pipeline.calls[0]["inputs"][0]) == 60
