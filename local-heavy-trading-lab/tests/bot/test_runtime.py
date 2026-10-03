from __future__ import annotations

from pathlib import Path
import sys

import pytest


def test_config_requires_token_owner_and_resolves_local_paths(tmp_path: Path):
    from heavy_lab.bot.runtime import BotConfig

    manifest = tmp_path / "campaign.json"
    manifest.write_text("{}", encoding="utf-8")

    config = BotConfig.from_env({
        "TELEGRAM_BOT_TOKEN": "secret-token",
        "TELEGRAM_OWNER_CHAT_ID": "123",
        "HEAVY_LAB_ROOT": str(tmp_path),
        "HEAVY_LAB_CAMPAIGN_MANIFEST": str(manifest),
    })
    assert config.token == "secret-token"
    assert config.owner_chat_id == 123
    assert config.root == tmp_path.resolve()
    assert config.campaign_manifest == manifest.resolve()

    with pytest.raises(ValueError):
        BotConfig.from_env({"TELEGRAM_OWNER_CHAT_ID": "123"})


def test_extract_message_accepts_text_updates_only():
    from heavy_lab.bot.runtime import extract_message

    update = {
        "update_id": 7,
        "message": {"chat": {"id": 123}, "text": "/status"},
    }
    assert extract_message(update) == (7, 123, "/status")
    assert extract_message({"update_id": 8, "message": {"chat": {"id": 123}}}) is None


def test_training_command_targets_local_cli_and_manifest(tmp_path: Path):
    from heavy_lab.bot.commands import BotAction
    from heavy_lab.bot.runtime import build_training_command

    manifest = (tmp_path / "campaign.json").resolve()
    action = BotAction(kind="train-all", profile="max", manifest=manifest)
    command = build_training_command(root=tmp_path, action=action)

    assert command[:3] == [sys.executable, "-m", "heavy_lab.cli"]
    assert "train-all" in command
    assert "--root" in command
    assert str(tmp_path.resolve()) in command
    assert "--manifest" in command
    assert str(manifest) in command
    assert command[-2:] == ["--profile", "max"]


def test_training_process_manager_reports_running_and_exit(tmp_path: Path):
    from heavy_lab.bot.commands import BotAction
    from heavy_lab.bot.runtime import TrainingProcessManager

    class FakeProcess:
        pid = 4321

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

    process = FakeProcess()

    def fake_popen(*args, **kwargs):
        assert kwargs["cwd"] == str(tmp_path.resolve())
        return process

    manifest = tmp_path / "campaign.json"
    manifest.write_text("{}", encoding="utf-8")

    manager = TrainingProcessManager(root=tmp_path, popen_factory=fake_popen)
    action = BotAction(kind="train-all", profile="smoke", manifest=manifest.resolve())

    launched = manager.launch(action)
    assert "STARTED" in launched
    assert "4321" in manager.status()

    duplicate = manager.launch(action)
    assert "ALREADY RUNNING" in duplicate

    process.returncode = 0
    assert "EXITED code=0" in manager.status()
