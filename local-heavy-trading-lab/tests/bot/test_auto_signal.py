from __future__ import annotations


class FakeSignalAPI:
    def __init__(self):
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, "accepted": True}


def test_auto_publisher_blocks_candidate_and_shadow_models():
    from heavy_lab.bot.auto_signal import AutoSignalPublisher, ModelSignal

    api = FakeSignalAPI()
    publisher = AutoSignalPublisher(api)

    for state in ("CANDIDATE", "SHADOW"):
        result = publisher.publish(ModelSignal(
            decision_id=f"d-{state}",
            model_id="timesfm25",
            promotion_state=state,
            side="BUY",
            symbol="SPY",
            quantity=1,
            confidence=0.8,
        ))
        assert result["sent"] is False

    assert api.calls == []


def test_auto_publisher_sends_only_paper_eligible_model_signal():
    from heavy_lab.bot.auto_signal import AutoSignalPublisher, ModelSignal

    api = FakeSignalAPI()
    publisher = AutoSignalPublisher(api)

    result = publisher.publish(ModelSignal(
        decision_id="d-123",
        model_id="ensemble-v1",
        promotion_state="PAPER_ELIGIBLE",
        side="SELL",
        symbol="SPY",
        quantity=1,
        confidence=0.67,
    ))

    assert result["sent"] is True
    assert len(api.calls) == 1
    call = api.calls[0]
    assert call["idempotency_key"] == "model-d-123"
    assert call["source"] == "model:ensemble-v1"
    assert call["side"] == "SELL"
