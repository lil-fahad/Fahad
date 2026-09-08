from __future__ import annotations

import asyncio
import logging
import secrets
import time
from contextlib import asynccontextmanager

import httpx

from .config import Config
from .storage import Store, normalize_message

logger = logging.getLogger("telegram_bridge")
UPDATES = ["message", "edited_message", "channel_post", "edited_channel_post"]


class TelegramError(Exception):
    def __init__(self, code: int, ambiguous: bool = False, retry_after: int = 0):
        self.code, self.ambiguous, self.retry_after = code, ambiguous, retry_after
        descriptions = {400: "Telegram rejected the message or parameters.",
                        401: "Telegram token is invalid or has been revoked.",
                        403: "The bot cannot access this chat or has been blocked.",
                        409: "Another poller or a webhook is already receiving this bot's updates.",
                        429: "Telegram rate limit reached."}
        super().__init__(descriptions.get(code, "Telegram request failed; delivery may be uncertain for a send."))


class TelegramAPI:
    def __init__(self, token: str, client: httpx.AsyncClient | None = None):
        # Telegram includes tokens in URLs. Do not enable HTTP client debug logging.
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
        self._base = "https://api.telegram.org/bot" + token + "/"
        self.client = client or httpx.AsyncClient(timeout=35, follow_redirects=False, trust_env=False)

    async def call(self, method: str, **parameters):
        if method not in {"getMe", "getWebhookInfo", "getUpdates", "sendMessage"}:
            raise ValueError("Unsupported Telegram method.")
        try:
            response = await self.client.post(self._base + method, json=parameters)
        except httpx.RequestError:
            raise TelegramError(0, ambiguous=method == "sendMessage") from None
        try:
            data = response.json()
        except ValueError:
            raise TelegramError(response.status_code, ambiguous=method == "sendMessage") from None
        if not isinstance(data, dict) or data.get("ok") is not True:
            code = data.get("error_code", response.status_code) if isinstance(data, dict) else response.status_code
            retry = data.get("parameters", {}).get("retry_after", 0) if isinstance(data, dict) else 0
            raise TelegramError(code, ambiguous=method == "sendMessage" and code >= 500,
                                retry_after=min(max(int(retry), 0), 86400))
        if not 200 <= response.status_code < 300 or "result" not in data:
            raise TelegramError(response.status_code, ambiguous=method == "sendMessage")
        return data["result"]

    async def close(self):
        await self.client.aclose()


class Bridge:
    def __init__(self, config: Config, store: Store, api: TelegramAPI):
        self.config, self.store, self.api = config, store, api
        self.owner = secrets.token_hex(16)
        self.last_poll_ok: float | None = None
        self.poll_error: str | None = None
        self.stop = asyncio.Event()
        self.bot_info: dict | None = None
        from .commands import CommandWorker
        from .market import MarketData
        from .prediction import PredictionService
        self.predictions = PredictionService(MarketData(config.market_provider, config.market_api_key,
            config.data_dir / "market", config.market_csv_dir, config.market_csv_basis,
            alpaca_key_id=config.alpaca_key_id, alpaca_secret_key=config.alpaca_secret_key))
        from .trading import AlpacaAPI, TradingService
        self.trading = TradingService(config, store, AlpacaAPI(config.trading_mode, config.alpaca_key_id, config.alpaca_secret_key))
        from .autotrading import AutoTrader
        self.autotrading = AutoTrader(self)
        from .options_paper import OptionsPaperEngine
        model_ensemble = None
        if config.options_models_enabled:
            from .model_ensemble import build_default_ensemble
            model_ensemble = build_default_ensemble(config.options_model_cache_dir, config.options_model_device)
        self.options_paper = OptionsPaperEngine(self, model_ensemble=model_ensemble)
        self.commands = CommandWorker(self, self.predictions)

    async def verify(self):
        info = await self.api.call("getMe")
        if info.get("username", "").casefold() != self.config.expected_bot_username.casefold():
            raise ValueError("The token belongs to a different bot than expected_bot_username.")
        stored_bot = self.store.get("bot_id")
        if stored_bot and stored_bot != str(info["id"]):
            raise ValueError("This database belongs to another bot. Configure a different data_dir.")
        self.store.set("bot_id", str(info["id"]))
        webhook = await self.api.call("getWebhookInfo")
        if webhook.get("url"):
            raise TelegramError(409)
        self.bot_info = {"id": str(info["id"]), "username": info["username"], "first_name": info.get("first_name", "")}

    async def poll_once(self):
        if not self.store.acquire_lease(self.owner):
            self.poll_error = "Another instance is receiving updates for this database."
            return False
        offset = int(self.store.get("offset", "0"))
        updates = await self.api.call("getUpdates", offset=offset, timeout=25, limit=100, allowed_updates=UPDATES)
        from .commands import parse_update
        commands = {u["update_id"]: command for u in updates if (command := parse_update(u, self.config))}
        self.store.ingest(updates, self.config.allowed_chat_ids, self.config.expected_bot_username,
                          self.config.retention_days, self.config.max_messages, commands)
        self.last_poll_ok, self.poll_error = time.time(), None
        return True

    async def poll(self):
        backoff = 1
        while not self.stop.is_set():
            delay = 0
            try:
                acquired = await self.poll_once()
                delay = 0.1 if acquired else 5
                backoff = 1
            except TelegramError as exc:
                self.poll_error = str(exc)
                logger.warning("Receive paused: %s", exc)
                if exc.code in {401, 409}:
                    return
                delay = max(exc.retry_after, backoff)
                backoff = min(backoff * 2, 60)
            except Exception:
                self.poll_error = "Storage or update validation failed. The receive cursor was not advanced."
                logger.error("Receive paused because storage or update validation failed.")
                delay = backoff
                backoff = min(backoff * 2, 60)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    @asynccontextmanager
    async def running(self):
        await self.verify()
        task = asyncio.create_task(self.poll())
        command_task = asyncio.create_task(self.commands.run())
        auto_task = asyncio.create_task(self.autotrading.run()) if self.config.auto_enabled else None
        options_task = asyncio.create_task(self.options_paper.run())
        try:
            yield
        finally:
            self.stop.set()
            tasks = [task, command_task, options_task] + ([auto_task] if auto_task else [])
            for active in tasks:
                active.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.store.release_lease(self.owner)
            await self.api.close()
            await self.trading.api.close()
            await self.options_paper.close()

    async def send(self, chat_id: int, text: str, request_id: str, *, reply_to: int | None = None,
                   _command_reply: bool = False, _auto_notice: bool = False, _options_notice: bool = False) -> dict:
        import json
        import uuid
        command_allowed = chat_id == self.config.allowed_chat_ids[0] or self.config.prediction_enabled or (self.config.trading_enabled and chat_id == self.config.trading_chat_id)
        auto_allowed = (_auto_notice and self.config.auto_enabled and self.config.auto_notify
                        and self.config.trading_mode == "paper" and chat_id == self.config.trading_chat_id)
        options_allowed = _options_notice and chat_id == self.config.allowed_chat_ids[0]
        command_reply_allowed = _command_reply and command_allowed and chat_id > 0 and reply_to is not None
        if not self.config.allow_send and not (command_reply_allowed or auto_allowed or options_allowed):
            raise ValueError("Sending is disabled in the server configuration.")
        if chat_id not in self.config.allowed_chat_ids:
            raise ValueError("This chat is not in the configured allowlist.")
        if not text.strip() or len(text.encode("utf-16-le")) // 2 > 4096:
            raise ValueError("Message must contain 1 to 4096 UTF-16 code units.")
        try:
            request_id = str(uuid.UUID(request_id))
        except ValueError:
            raise ValueError("request_id must be a UUID reused for retries of the same message.") from None
        parameters = {"chat_id": chat_id, "text": text}
        if reply_to is not None:
            parameters["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": False}
        payload = json.dumps(parameters, sort_keys=True, ensure_ascii=False)
        previous = self.store.begin_send(request_id, payload)
        if previous:
            return previous
        try:
            sent = await self.api.call("sendMessage", **parameters,
                                       link_preview_options={"is_disabled": True})
        except TelegramError as exc:
            result = {"status": "unknown" if exc.ambiguous else "failed", "request_id": request_id,
                      "detail": str(exc), "retry_after_seconds": exc.retry_after,
                      "automatic_retry": False}
            self.store.finish_send(request_id, result)
            return result
        result = {"status": "sent", "request_id": request_id, "chat_id": str(chat_id), "message_id": sent["message_id"]}
        self.store.finish_send(request_id, result, normalize_message(sent, self.config.expected_bot_username))
        return result
