from __future__ import annotations

from pathlib import Path


def _doctor():
    return {
        "hardware": {"device": "cuda", "gpu_name": "RTX 3070 Ti", "vram_gb": 8.0},
        "training_profile": "low-vram",
    }


def test_signal_command_builds_paper_only_buy_sell_intent(tmp_path: Path):
    from heavy_lab.bot.commands import BotCommandProcessor, SignalAction

    bot = BotCommandProcessor(
        root=tmp_path,
        owner_chat_id=123,
        doctor_provider=_doctor,
        readiness_provider=lambda: {"ready": True, "missing": []},
    )

    reply = bot.handle(chat_id=123, text="/signal buy SPY 2 0.72")
    assert isinstance(reply.action, SignalAction)
    assert reply.action.side == "BUY"
    assert reply.action.symbol == "SPY"
    assert reply.action.quantity == 2
    assert reply.action.confidence == 0.72
    assert reply.action.paper_only is True
    assert reply.action.execute is False

    sell = bot.handle(chat_id=123, text="/signal sell SPY 1")
    assert isinstance(sell.action, SignalAction)
    assert sell.action.side == "SELL"
    assert sell.action.paper_only is True
    assert sell.action.execute is False


def test_signal_command_rejects_invalid_side_quantity_and_confidence(tmp_path: Path):
    from heavy_lab.bot.commands import BotCommandProcessor

    bot = BotCommandProcessor(
        root=tmp_path,
        owner_chat_id=123,
        doctor_provider=_doctor,
        readiness_provider=lambda: {"ready": True, "missing": []},
    )

    for text in (
        "/signal hold SPY 1",
        "/signal buy SPY 0",
        "/signal buy SPY 1 1.5",
        "/signal buy",
    ):
        reply = bot.handle(chat_id=123, text=text)
        assert reply.action is None
        assert "signal" in reply.text.lower() or "invalid" in reply.text.lower()


def test_help_exposes_signal_command_but_no_direct_execution_commands(tmp_path: Path):
    from heavy_lab.bot.commands import BotCommandProcessor

    bot = BotCommandProcessor(
        root=tmp_path,
        owner_chat_id=123,
        doctor_provider=_doctor,
        readiness_provider=lambda: {"ready": True, "missing": []},
    )
    text = bot.handle(chat_id=123, text="/help").text.lower()
    assert "/signal" in text
    for forbidden in ("/buy", "/sell", "/order", "/broker"):
        assert forbidden not in text
