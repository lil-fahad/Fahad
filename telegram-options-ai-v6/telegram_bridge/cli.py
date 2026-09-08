from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import uvicorn

from .config import Config, hash_password, load_config, write_private_json
from .storage import Store
from .telegram import Bridge, TelegramAPI, TelegramError
from .market import MarketData, atomic_json, read_csv
from .prediction import HORIZONS, PredictionService, evaluate, render
from .trading import AlpacaAPI, TradingService, TradeError


async def setup(path: Path, base_url: str):
    if path.exists():
        raise ValueError("Configuration already exists. Edit it locally; setup will not overwrite secrets.")
    token = getpass.getpass("New Telegram bot token (hidden): ").strip()
    api = TelegramAPI(token)
    try:
        info = await api.call("getMe")
        if info.get("username", "").casefold() != "lil_fahad_bot":
            raise ValueError("This package expects @Lil_fahad_bot. The supplied token belongs to a different bot.")
        if (await api.call("getWebhookInfo")).get("url"):
            raise ValueError("This bot already uses a webhook. Resolve that existing integration before setup.")
        print("Verified @" + info["username"])
        print("Open https://t.me/Lil_fahad_bot in Telegram and press Start.")
        input("Press Enter here after sending /start to your bot: ")
        # No offset: discovers pending chats without confirming/removing updates.
        updates = await api.call("getUpdates", timeout=0, limit=100)
        chats = {}
        for update in updates:
            for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
                if key in update:
                    chat = update[key]["chat"]
                    chats[chat["id"]] = chat
        if not chats:
            raise ValueError("No chats found. Send /start to the bot, then run setup again.")
        print("Pending chats (choose only chats you own or are authorized to connect):")
        for chat_id, chat in chats.items():
            title = chat.get("title") or chat.get("first_name") or str(chat_id)
            # JSON escapes terminal control characters in remote names.
            print(json.dumps({"chat_id": chat_id, "title": title, "type": chat["type"]}, ensure_ascii=True))
        chosen = tuple(dict.fromkeys(int(part.strip()) for part in input("Enter allowed numeric chat IDs, separated by commas: ").split(",")))
        if not chosen or any(chat_id not in chats for chat_id in chosen):
            raise ValueError("Select IDs from the verified list. Nothing was saved.")
        password = getpass.getpass("Create a connection password (at least 16 characters): ")
        if len(password) < 16 or len(password) > 256 or password != getpass.getpass("Repeat connection password: "):
            raise ValueError("Passwords must match and contain 16 to 256 characters.")
        allow_send = input("Enable the tool for explicitly requested sends? [y/N]: ").strip().lower() == "y"
        cfg = Config(bot_token=token, allowed_chat_ids=chosen, password_hash=hash_password(password),
                     public_base_url=base_url.rstrip("/"), data_dir=path.resolve().parent / "state", allow_send=allow_send)
        raw = {"bot_token": token, "allowed_chat_ids": chosen, "password_hash": cfg.password_hash,
               "public_base_url": cfg.public_base_url, "expected_bot_username": cfg.expected_bot_username,
               "data_dir": "state", "bind_host": "127.0.0.1", "port": 8787,
               "allow_send": allow_send, "retention_days": 30, "max_messages": 10000}
        write_private_json(path, raw)
        store = Store(cfg.data_dir / "bridge.sqlite3")
        try:
            store.set("bot_id", str(info["id"]))
            store.ingest(updates, chosen, cfg.expected_bot_username, cfg.retention_days, cfg.max_messages)
        finally:
            store.close()
        print("Configuration saved. No message was sent. Keep the connection password for ChatGPT authorization.")
        print("For Telegram predictions, next run: python -m telegram_bridge configure-predictions")
    finally:
        await api.close()


def configure_predictions(path: Path, provider: str, csv_dir: Path | None, basis: str):
    load_config(path)  # Validate existing bot settings before changing only the market fields.
    raw = json.loads(path.read_text(encoding="utf-8"))
    if provider == "csv":
        if not csv_dir or not csv_dir.is_dir():
            raise ValueError("Use --csv-dir with an existing folder containing SYMBOL.csv files.")
        key = ""
    else:
        print("API access for the standalone bot is separate from the apps connected in ChatGPT.")
        print("Alpha Vantage Daily Adjusted/full history requires an eligible plan; Financial Datasets requires credits.")
        key = getpass.getpass("Market data API key (hidden, stored only in private config): ").strip()
        if not key:
            raise ValueError("No key entered. Configuration was not changed.")
    raw.update(prediction_enabled=True, market_provider=provider, market_api_key=key,
               market_csv_dir=str(csv_dir.resolve()) if csv_dir else None, market_csv_basis=basis)
    # Validate before atomically replacing the private file. Keep allow_send and OAuth scopes unchanged.
    Config(**{**raw, "allowed_chat_ids": tuple(raw["allowed_chat_ids"]), "data_dir": Path(raw.get("data_dir", "state")),
              "market_csv_dir": csv_dir})
    atomic_json(path, raw)
    print("Prediction replies enabled for allowlisted private chats. Restart serve, then send /predict AAPL 5.")


async def configure_trading(path: Path, mode: str, chat_id: int | None, order_cap: str, daily_cap: str):
    existing = load_config(path)
    candidates = [value for value in existing.allowed_chat_ids if value > 0]
    if chat_id is None and len(candidates) == 1:
        chat_id = candidates[0]
    if chat_id not in candidates:
        raise TradeError("حدد --chat-id لمحادثة خاصة مسموحة يملكها صاحب حساب الوسيط.")
    print("Account mode: " + ("PAPER — simulated funds" if mode == "paper" else "LIVE — real funds after Telegram confirmation"))
    key = getpass.getpass("Alpaca API key ID for this mode (hidden): ").strip()
    secret = getpass.getpass("Alpaca API secret for this mode (hidden): ").strip()
    if not key or not secret:
        raise TradeError("لم تُدخل المفاتيح؛ لم يتغير الإعداد.")
    # Validate limits and key formats BEFORE issuing even a read-only request.
    from dataclasses import replace
    replace(existing, trading_mode=mode, alpaca_key_id=key, alpaca_secret_key=secret,
            max_order_usd=order_cap, daily_buy_limit_usd=daily_cap)
    api = AlpacaAPI(mode, key, secret)
    try:
        account = await api.call("GET", "/v2/account")
        if not isinstance(account, dict):
            raise TradeError("تعذر التحقق من الحساب.")
        try:
            account_id = str(uuid.UUID(account.get("id", "")))
        except (ValueError, AttributeError):
            raise TradeError("هوية حساب الوسيط غير صالحة.") from None
        updated = replace(existing, trading_enabled=True, trading_mode=mode, trading_chat_id=chat_id,
            trading_account_id=account_id, alpaca_key_id=key, alpaca_secret_key=secret,
            max_order_usd=order_cap, daily_buy_limit_usd=daily_cap, prediction_enabled=True, market_provider="alpaca")
        # Account restriction checks match the execution path; no orders are submitted by setup.
        TradingService(updated, None, api)._check_account(account)
        raw = json.loads(path.read_text(encoding="utf-8"))
        for field in ("trading_enabled", "trading_mode", "trading_chat_id", "trading_account_id", "alpaca_key_id",
                      "alpaca_secret_key", "max_order_usd", "daily_buy_limit_usd", "prediction_enabled", "market_provider"):
            raw[field] = getattr(updated, field)
        atomic_json(path, raw)
        print("Verified account saved. Mode=" + mode + ". No order was submitted.")
        print("Restart serve. Use /account, then /buy or /sell to preview an order. Confirm it yourself in Telegram.")
    finally:
        await api.close()


def configure_options_models(path: Path, enabled: bool, device: str, cache_dir: Path):
    existing = load_config(path)
    from dataclasses import replace
    cache_dir = Path(cache_dir)
    resolved = cache_dir if cache_dir.is_absolute() else path.resolve().parent / cache_dir
    replace(existing, options_models_enabled=bool(enabled), options_model_device=device,
            options_model_cache_dir=resolved)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["options_models_enabled"] = bool(enabled)
    raw["options_model_device"] = device
    raw["options_model_cache_dir"] = str(cache_dir)
    atomic_json(path, raw)
    print("Options AI models " + ("enabled" if enabled else "disabled") + ". No model was downloaded by this command.")


def download_options_models(path: Path, builder=None) -> dict:
    config = load_config(path)
    if not config.options_models_enabled:
        raise ValueError("Enable options models first with configure-options-models.")
    if builder is None:
        from .model_ensemble import build_default_ensemble
        builder = build_default_ensemble
    ensemble = builder(config.options_model_cache_dir, config.options_model_device)
    status = ensemble.warmup()
    errors = status.get("errors") or {}
    if errors:
        details = "; ".join(f"{name}: {message}" for name, message in sorted(errors.items()))
        raise ValueError("Model warmup failed — " + details + ". Install requirements-models.txt on this host.")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


def configure_autotrading(path: Path, symbols: str, order_usd: str, max_position_usd: str, horizon: int,
                          interval_seconds: int, buy_probability: str, sell_probability: str, notify: bool):
    existing = load_config(path)
    if not existing.trading_enabled or existing.trading_mode != "paper" or not existing.prediction_enabled or existing.market_provider != "alpaca":
        raise TradeError("Configure an Alpaca paper trading account first; automatic live trading is not supported.")
    auto_symbols = tuple(dict.fromkeys(part.strip().upper() for part in symbols.split(",") if part.strip()))
    from dataclasses import replace
    updated = replace(existing, auto_enabled=True, auto_symbols=auto_symbols, auto_horizon=horizon,
                      auto_interval_seconds=interval_seconds, auto_order_usd=order_usd,
                      auto_max_position_usd=max_position_usd, auto_buy_probability=buy_probability,
                      auto_sell_probability=sell_probability, auto_notify=notify)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for field in ("auto_enabled", "auto_symbols", "auto_horizon", "auto_interval_seconds", "auto_order_usd",
                  "auto_max_position_usd", "auto_buy_probability", "auto_sell_probability", "auto_notify"):
        value = getattr(updated, field)
        raw[field] = list(value) if isinstance(value, tuple) else value
    atomic_json(path, raw)
    from .autotrading import auto_state_key, policy_fingerprint
    state_store = Store(updated.data_dir / "bridge.sqlite3")
    try:
        state_store.set(auto_state_key(updated), json.dumps({
            "running": False,
            "fingerprint": policy_fingerprint(updated),
            "changed_at": time.time(),
        }, sort_keys=True))
    finally:
        state_store.close()
    print("Automatic paper-trading policy saved in stopped state. Restart serve, then use /autoon in the owner chat.")



def main():
    os.umask(0o077)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Private Telegram predictions and ChatGPT connector")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "serve", "doctor", "revoke-connections", "trading-status", "halt-trading", "autotrading-status", "halt-autotrading"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("private/config.json"))
        if name == "setup":
            command.add_argument("--public-base-url", default="http://127.0.0.1:8787")
        if name == "serve":
            command.add_argument("--stdio", action="store_true")
    configure = commands.add_parser("configure-predictions", help="Securely configure a data source and enable private command replies")
    configure.add_argument("--config", type=Path, default=Path("private/config.json"))
    configure.add_argument("--provider", choices=("alpha_vantage", "financial_datasets", "csv"), default="alpha_vantage")
    configure.add_argument("--csv-dir", type=Path)
    configure.add_argument("--basis", choices=("unverified", "split_adjusted", "synthetic"), default="unverified")
    trading = commands.add_parser("configure-trading", help="Verify an Alpaca account and enable human-confirmed order commands")
    trading.add_argument("--config", type=Path, default=Path("private/config.json"))
    trading.add_argument("--mode", choices=("paper", "live"), default="paper")
    trading.add_argument("--chat-id", type=int)
    trading.add_argument("--max-order-usd", default="500")
    trading.add_argument("--daily-buy-limit-usd", default="1000")
    auto = commands.add_parser("configure-autotrading", help="Configure forecast-driven automatic PAPER trading")
    auto.add_argument("--config", type=Path, default=Path("private/config.json"))
    auto.add_argument("--symbols", required=True)
    auto.add_argument("--order-usd", default="100")
    auto.add_argument("--max-position-usd", default="500")
    auto.add_argument("--horizon", type=int, choices=HORIZONS, default=5)
    auto.add_argument("--interval-seconds", type=int, default=300)
    auto.add_argument("--buy-probability", default="0.60")
    auto.add_argument("--sell-probability", default="0.40")
    auto.add_argument("--no-notify", action="store_true")
    options_models = commands.add_parser("configure-options-models", help="Enable or disable optional Chronos/TimesFM/FinBERT ensemble")
    options_models.add_argument("--config", type=Path, default=Path("private/config.json"))
    options_models.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    options_models.add_argument("--cache-dir", type=Path, default=Path("models"))
    options_models.add_argument("--disable", action="store_true")
    download_models = commands.add_parser("download-options-models", help="Download and warm up optional options AI models")
    download_models.add_argument("--config", type=Path, default=Path("private/config.json"))
    for name in ("predict", "backtest", "demo"):
        command = commands.add_parser(name)
        command.add_argument("--horizon", type=int, choices=HORIZONS, default=5)
        command.add_argument("--json", action="store_true", help="Include reproducible evaluation records as JSON")
        if name != "demo":
            command.add_argument("--config", type=Path, default=Path("private/config.json"))
            command.add_argument("--symbol", required=True)
            command.add_argument("--csv", type=Path)
            command.add_argument("--basis", choices=("unverified", "split_adjusted", "synthetic"), default="unverified")
    args = parser.parse_args()
    store = None
    try:
        if args.command == "configure-predictions":
            configure_predictions(args.config, args.provider, args.csv_dir, args.basis)
            return
        if args.command == "configure-trading":
            asyncio.run(configure_trading(args.config, args.mode, args.chat_id, args.max_order_usd, args.daily_buy_limit_usd))
            return
        if args.command == "configure-autotrading":
            configure_autotrading(args.config, args.symbols, args.order_usd, args.max_position_usd, args.horizon,
                                  args.interval_seconds, args.buy_probability, args.sell_probability, not args.no_notify)
            return
        if args.command == "configure-options-models":
            configure_options_models(args.config, not args.disable, args.device, args.cache_dir)
            return
        if args.command == "download-options-models":
            download_options_models(args.config)
            return
        if args.command in {"predict", "backtest", "demo"}:
            if args.command == "demo":
                from .demo import demo_series
                result = evaluate(demo_series(), args.horizon)
            elif args.csv:
                result = evaluate(read_csv(args.csv, args.symbol, args.basis), args.horizon)
            else:
                cfg = load_config(args.config)
                service = PredictionService(MarketData(cfg.market_provider, cfg.market_api_key,
                    cfg.data_dir / "market", cfg.market_csv_dir, cfg.market_csv_basis,
                    alpaca_key_id=cfg.alpaca_key_id, alpaca_secret_key=cfg.alpaca_secret_key))
                result = asyncio.run(service.predict(args.symbol, args.horizon))
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) if args.json
                  else render(result, backtest_only=args.command == "backtest"))
            return
        if args.command == "setup":
            asyncio.run(setup(args.config, args.public_base_url))
            return
        config = load_config(args.config)
        store = Store(config.data_dir / "bridge.sqlite3")
        if args.command in {"autotrading-status", "halt-autotrading"}:
            from types import SimpleNamespace
            from .autotrading import AutoTrader, auto_state_key, policy_fingerprint
            if args.command == "halt-autotrading":
                store.set(auto_state_key(config), json.dumps({"running": False, "fingerprint": policy_fingerprint(config)}))
                print(json.dumps({"running": False, "paper_only": True}, indent=2))
            else:
                class LocalTrading:
                    def enabled(self, chat_id=None):
                        if not config.trading_enabled: raise TradeError("Trading is not configured.")
                bridge = SimpleNamespace(config=config, store=store, trading=LocalTrading())
                print(json.dumps(AutoTrader(bridge).status(), ensure_ascii=False, indent=2))
            return
        if args.command in {"trading-status", "halt-trading"}:
            async def trading_status():
                broker = AlpacaAPI(config.trading_mode, config.alpaca_key_id, config.alpaca_secret_key)
                try:
                    service = TradingService(config, store, broker)
                    result = service.halt(config.trading_chat_id, True) if args.command == "halt-trading" else await service.account()
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                finally:
                    await broker.close()
            asyncio.run(trading_status())
            return
        if args.command == "revoke-connections":
            with store.transaction() as db:
                db.execute("DELETE FROM oauth WHERE kind != 'client'")
            print("All connection tokens and pending authorizations have been revoked.")
            return
        api = TelegramAPI(config.bot_token)
        if args.command == "doctor":
            async def doctor():
                try:
                    bridge = Bridge(config, store, api)
                    await bridge.verify()
                    print(json.dumps({"bot": bridge.bot_info, "webhook_conflict": False,
                                      "mcp_url": config.resource, "sending_enabled": config.allow_send,
                                      "prediction_commands_enabled": config.prediction_enabled,
                                      "market_provider": config.market_provider,
                                      "trading_enabled": config.trading_enabled, "trading_mode": config.trading_mode}, ensure_ascii=False))
                finally:
                    await api.close()
            asyncio.run(doctor())
        elif args.stdio:
            from .server import create_server
            mcp, _, _ = create_server(config, store, api, stdio=True)
            mcp.run(transport="stdio")
        else:
            from .server import create_app
            app = create_app(config, store, api)
            # Never log query strings: OAuth authorization URLs carry sensitive values.
            uvicorn.run(app, host=config.bind_host, port=config.port, workers=1,
                        access_log=False, log_level="warning", proxy_headers=False)
    except (ValueError, KeyError, FileNotFoundError, TelegramError) as exc:
        # Only controlled errors, never HTTP exception URLs or configuration reprs.
        if isinstance(exc, FileNotFoundError):
            print("Required file is missing. Check the CSV path or run setup to create the bot configuration.", file=sys.stderr)
        elif isinstance(exc, KeyError):
            print("Configuration or upstream response is missing a required field.", file=sys.stderr)
        else:
            print(str(exc), file=sys.stderr)
        sys.exit(1)
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    main()
