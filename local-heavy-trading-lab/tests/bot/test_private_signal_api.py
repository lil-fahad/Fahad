from __future__ import annotations

import json


def test_private_signal_api_posts_non_executing_trade_intent():
    from heavy_lab.bot.private_signal_api import PrivateSignalAPI

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return b'{"ok": true, "accepted": true}'

    def opener(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    api = PrivateSignalAPI(
        "https://example.invalid/signals",
        token="secret",
        opener=opener,
    )
    result = api.send(
        side="BUY",
        symbol="SPY",
        quantity=2,
        confidence=0.72,
        idempotency_key="telegram-42",
        source="telegram-owner",
    )

    assert result["accepted"] is True
    assert captured["url"] == "https://example.invalid/signals"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["headers"]["Idempotency-key"] == "telegram-42"
    assert captured["body"]["side"] == "BUY"
    assert captured["body"]["symbol"] == "SPY"
    assert captured["body"]["quantity"] == 2
    assert captured["body"]["confidence"] == 0.72
    assert captured["body"]["paper_only"] is True
    assert captured["body"]["execute"] is False
    assert captured["body"]["type"] == "trade_intent"
