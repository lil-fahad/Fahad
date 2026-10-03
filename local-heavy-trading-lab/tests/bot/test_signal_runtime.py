from __future__ import annotations

from pathlib import Path


class FakeTelegram:
    def __init__(self):
        self.messages = []

    def send_message(self, chat_id, text):
        self.messages.append((chat_id, text))


class FakeSignalAPI:
    def __init__(self):
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, "accepted": True}


def test_runtime_forwards_owner_signal_to_private_api_with_idempotency(tmp_path: Path):
    from heavy_lab.bot.runtime import BotConfig, LocalTelegramBot

    telegram = FakeTelegram()
    signal_api = FakeSignalAPI()
    config = BotConfig(
        token="token",
        owner_chat_id=123,
        root=tmp_path,
        campaign_manifest=tmp_path / "campaign.json",
        signal_api_url="https://example.invalid/signals",
        signal_api_token="secret",
    )
    bot = LocalTelegramBot(config, api=telegram, signal_api=signal_api)

    next_offset = bot.process_update({
        "update_id": 42,
        "message": {"chat": {"id": 123}, "text": "/signal buy SPY 2 0.72"},
    })

    assert next_offset == 43
    assert len(signal_api.calls) == 1
    call = signal_api.calls[0]
    assert call["side"] == "BUY"
    assert call["symbol"] == "SPY"
    assert call["quantity"] == 2
    assert call["confidence"] == 0.72
    assert call["idempotency_key"] == "telegram-42"
    assert call["source"] == "telegram-owner"
    assert "SIGNAL SENT" in telegram.messages[-1][1]


def test_runtime_does_not_send_when_private_signal_api_is_not_configured(tmp_path: Path):
    from heavy_lab.bot.runtime import BotConfig, LocalTelegramBot

    telegram = FakeTelegram()
    config = BotConfig(
        token="token",
        owner_chat_id=123,
        root=tmp_path,
        campaign_manifest=tmp_path / "campaign.json",
    )
    bot = LocalTelegramBot(config, api=telegram)

    bot.process_update({
        "update_id": 43,
        "message": {"chat": {"id": 123}, "text": "/signal sell SPY 1"},
    })

    assert "SIGNAL API NOT CONFIGURED" in telegram.messages[-1][1]


def test_config_reads_private_signal_api_without_exposing_token(tmp_path: Path):
    from heavy_lab.bot.runtime import BotConfig

    config = BotConfig.from_env({
        "TELEGRAM_BOT_TOKEN": "telegram-secret",
        "TELEGRAM_OWNER_CHAT_ID": "123",
        "HEAVY_LAB_ROOT": str(tmp_path),
        "PRIVATE_SIGNAL_API_URL": "http://127.0.0.1:8080/signals",
        "PRIVATE_SIGNAL_API_TOKEN": "api-secret",
    })

    assert config.signal_api_url == "http://127.0.0.1:8080/signals"
    assert config.signal_api_token == "api-secret"
    assert "api-secret" not in repr(config)
