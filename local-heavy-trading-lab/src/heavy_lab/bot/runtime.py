from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping
from urllib import request as urllib_request

from heavy_lab.bot.commands import BotAction, BotCommandProcessor, SignalAction
from heavy_lab.bot.private_signal_api import PrivateSignalAPI


@dataclass(frozen=True)
class BotConfig:
    token: str = field(repr=False)
    owner_chat_id: int = 0
    root: Path = Path.cwd()
    campaign_manifest: Path = Path("campaign.json")
    poll_timeout: int = 30
    signal_api_url: str = ""
    signal_api_token: str = field(default="", repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "BotConfig":
        source = dict(os.environ if env is None else env)
        token = str(source.get("TELEGRAM_BOT_TOKEN", "")).strip()
        owner_raw = str(source.get("TELEGRAM_OWNER_CHAT_ID", "")).strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if not owner_raw:
            raise ValueError("TELEGRAM_OWNER_CHAT_ID is required")
        try:
            owner_chat_id = int(owner_raw)
        except ValueError as exc:
            raise ValueError("TELEGRAM_OWNER_CHAT_ID must be an integer") from exc

        root = Path(source.get("HEAVY_LAB_ROOT", Path.cwd())).expanduser().resolve()
        manifest_raw = str(source.get("HEAVY_LAB_CAMPAIGN_MANIFEST", "")).strip()
        manifest = (
            Path(manifest_raw).expanduser().resolve()
            if manifest_raw
            else (root / "campaign.json").resolve()
        )
        timeout_raw = str(source.get("TELEGRAM_POLL_TIMEOUT", "30")).strip()
        try:
            poll_timeout = max(1, min(50, int(timeout_raw)))
        except ValueError as exc:
            raise ValueError("TELEGRAM_POLL_TIMEOUT must be an integer") from exc

        return cls(
            token=token,
            owner_chat_id=owner_chat_id,
            root=root,
            campaign_manifest=manifest,
            poll_timeout=poll_timeout,
            signal_api_url=str(source.get("PRIVATE_SIGNAL_API_URL", "")).strip(),
            signal_api_token=str(source.get("PRIVATE_SIGNAL_API_TOKEN", "")).strip(),
        )


def extract_message(update: Mapping[str, Any]) -> tuple[int, int, str] | None:
    try:
        update_id = int(update["update_id"])
        message = update["message"]
        chat_id = int(message["chat"]["id"])
        text = message["text"]
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return update_id, chat_id, text.strip()


def build_training_command(*, root: Path, action: BotAction) -> list[str]:
    if action.kind != "train-all":
        raise ValueError(f"Unsupported bot action: {action.kind}")
    return [
        sys.executable,
        "-m",
        "heavy_lab.cli",
        "train-all",
        "--root",
        str(Path(root).expanduser().resolve()),
        "--manifest",
        str(Path(action.manifest).expanduser().resolve()),
        "--profile",
        action.profile,
    ]


class TrainingProcessManager:
    def __init__(
        self,
        *,
        root: Path,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self._popen_factory = popen_factory
        self._process: Any | None = None
        self._log_handle: Any | None = None
        self._last_returncode: int | None = None

    def launch(self, action: BotAction) -> str:
        if self._process is not None and self._process.poll() is None:
            return f"ALREADY RUNNING pid={self._process.pid}"

        log_dir = self.root / "artifacts" / "bot"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "training.log"
        self._log_handle = log_path.open("a", encoding="utf-8")
        command = build_training_command(root=self.root, action=action)
        self._process = self._popen_factory(
            command,
            cwd=str(self.root),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._last_returncode = None
        return f"STARTED pid={self._process.pid} profile={action.profile} log={log_path}"

    def status(self) -> str:
        if self._process is None:
            if self._last_returncode is None:
                return "IDLE"
            return f"EXITED code={self._last_returncode}"

        returncode = self._process.poll()
        if returncode is None:
            return f"RUNNING pid={self._process.pid}"

        self._last_returncode = int(returncode)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._process = None
        return f"EXITED code={self._last_returncode}"


class TelegramAPI:
    def __init__(
        self,
        token: str,
        *,
        opener: Callable[..., Any] = urllib_request.urlopen,
    ) -> None:
        self._base = f"https://api.telegram.org/bot{token}"
        self._opener = opener

    def call(self, method: str, payload: Mapping[str, Any], *, timeout: int = 40) -> Any:
        body = json.dumps(dict(payload)).encode("utf-8")
        req = urllib_request.Request(
            f"{self._base}/{method}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API {method} failed")
        return data.get("result")

    def get_updates(self, *, offset: int | None, poll_timeout: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": int(poll_timeout),
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = int(offset)
        result = self.call("getUpdates", payload, timeout=int(poll_timeout) + 10)
        return list(result or [])

    def send_message(self, chat_id: int, text: str) -> None:
        self.call(
            "sendMessage",
            {
                "chat_id": int(chat_id),
                "text": str(text)[:4000],
                "disable_web_page_preview": True,
            },
        )


class LocalTelegramBot:
    def __init__(
        self,
        config: BotConfig,
        *,
        api: TelegramAPI | None = None,
        process_manager: TrainingProcessManager | None = None,
        signal_api: Any | None = None,
    ) -> None:
        self.config = config
        self.api = api or TelegramAPI(config.token)
        self.process_manager = process_manager or TrainingProcessManager(root=config.root)
        if signal_api is not None:
            self.signal_api = signal_api
        elif config.signal_api_url:
            self.signal_api = PrivateSignalAPI(
                config.signal_api_url,
                token=config.signal_api_token,
            )
        else:
            self.signal_api = None
        self.processor = BotCommandProcessor(
            root=config.root,
            owner_chat_id=config.owner_chat_id,
            campaign_manifest=config.campaign_manifest,
        )

    def process_update(self, update: Mapping[str, Any]) -> int | None:
        parsed = extract_message(update)
        if parsed is None:
            return None

        update_id, chat_id, text = parsed
        command = text.split()[0].split("@", 1)[0].lower()

        if int(chat_id) == self.config.owner_chat_id and command == "/trainstatus":
            reply_text = self.process_manager.status()
        else:
            reply = self.processor.handle(chat_id=chat_id, text=text)
            reply_text = reply.text
            if isinstance(reply.action, BotAction):
                launch_status = self.process_manager.launch(reply.action)
                reply_text = f"{reply_text}\n{launch_status}"
            elif isinstance(reply.action, SignalAction):
                if self.signal_api is None:
                    reply_text = f"{reply_text}\nSIGNAL API NOT CONFIGURED"
                else:
                    result = self.signal_api.send(
                        side=reply.action.side,
                        symbol=reply.action.symbol,
                        quantity=reply.action.quantity,
                        confidence=reply.action.confidence,
                        idempotency_key=f"telegram-{update_id}",
                        source="telegram-owner",
                    )
                    accepted = result.get("accepted", result.get("ok", True))
                    status = "SIGNAL SENT" if accepted is not False else "SIGNAL API REJECTED"
                    reply_text = f"{reply_text}\n{status}"

        self.api.send_message(chat_id, reply_text)
        return update_id + 1

    def run_forever(self) -> None:
        offset: int | None = None
        while True:
            try:
                updates = self.api.get_updates(
                    offset=offset,
                    poll_timeout=self.config.poll_timeout,
                )
                for update in updates:
                    next_offset = self.process_update(update)
                    if next_offset is not None:
                        offset = max(offset or 0, next_offset)
            except KeyboardInterrupt:
                raise
            except Exception:
                time.sleep(3)


def main() -> None:
    config = BotConfig.from_env()
    LocalTelegramBot(config).run_forever()


if __name__ == "__main__":
    main()
