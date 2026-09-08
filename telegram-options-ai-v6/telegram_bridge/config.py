from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import uuid
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
    return salt + ":" + digest.hex()


def check_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split(":", 1)
        return secrets.compare_digest(hash_password(password, salt), stored)
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class Config:
    bot_token: str = field(repr=False)
    allowed_chat_ids: tuple[int, ...]
    password_hash: str = field(repr=False)
    public_base_url: str = "http://127.0.0.1:8787"
    expected_bot_username: str = "Lil_fahad_bot"
    data_dir: Path = Path("state")
    bind_host: str = "127.0.0.1"
    port: int = 8787
    allow_send: bool = False
    retention_days: int = 30
    max_messages: int = 10000
    prediction_enabled: bool = False
    market_provider: str = "alpha_vantage"
    market_api_key: str = field(default="", repr=False)
    market_csv_dir: Path | None = None
    market_csv_basis: str = "unverified"
    trading_enabled: bool = False
    trading_mode: str = "paper"
    trading_chat_id: int = 0
    trading_account_id: str = ""
    alpaca_key_id: str = field(default="", repr=False)
    alpaca_secret_key: str = field(default="", repr=False)
    max_order_usd: str = "500"
    daily_buy_limit_usd: str = "1000"
    max_open_orders: int = 5
    auto_enabled: bool = False
    auto_symbols: tuple[str, ...] = ()
    auto_horizon: int = 5
    auto_interval_seconds: int = 300
    auto_order_usd: str = "100"
    auto_max_position_usd: str = "500"
    auto_buy_probability: str = "0.60"
    auto_sell_probability: str = "0.40"
    auto_notify: bool = True
    options_models_enabled: bool = False
    options_model_device: str = "auto"
    options_model_cache_dir: Path = Path("models")

    def __post_init__(self):
        if not re.fullmatch(r"[0-9]{5,16}:[A-Za-z0-9_-]{30,80}", self.bot_token):
            raise ValueError("Invalid Telegram token format; use a newly issued BotFather token.")
        if not self.allowed_chat_ids or len(self.allowed_chat_ids) > 100:
            raise ValueError("Configure 1 to 100 allowed chat IDs.")
        if any(type(x) is not int or x == 0 or abs(x) >= 2**53 for x in self.allowed_chat_ids):
            raise ValueError("Chat IDs must be nonzero Telegram integer IDs.")
        if not re.fullmatch(r"[0-9a-f]{32}:[0-9a-f]{128}", self.password_hash):
            raise ValueError("Use the setup command to generate the connection password hash.")
        url = urlsplit(self.public_base_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("public_base_url must be an HTTP(S) origin.")
        if url.username or url.password or url.path or url.query or url.fragment:
            raise ValueError("public_base_url must be an origin without a path or trailing slash.")
        if url.scheme != "https" and url.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("A public MCP endpoint requires HTTPS.")
        if type(self.allow_send) is not bool:
            raise ValueError("allow_send must be a JSON boolean.")
        if type(self.prediction_enabled) is not bool:
            raise ValueError("prediction_enabled must be a JSON boolean.")
        if self.market_provider not in {"alpha_vantage", "financial_datasets", "csv", "alpaca"}:
            raise ValueError("Unsupported market_provider.")
        if not isinstance(self.market_api_key, str) or len(self.market_api_key) > 512 or any(ord(c) < 33 for c in self.market_api_key):
            raise ValueError("Invalid market API key format.")
        if self.market_csv_basis not in {"split_adjusted", "unverified", "synthetic"}:
            raise ValueError("Unsupported market_csv_basis.")
        if type(self.trading_enabled) is not bool or self.trading_mode not in {"paper", "live"}:
            raise ValueError("Trading mode must be paper or live; trading_enabled must be a boolean.")
        for value in (self.alpaca_key_id, self.alpaca_secret_key):
            if not isinstance(value, str) or len(value) > 512 or any(ord(c) < 33 for c in value):
                raise ValueError("Invalid Alpaca credential format.")
        if self.trading_enabled:
            if (type(self.trading_chat_id) is not int or self.trading_chat_id <= 0
                    or self.trading_chat_id not in self.allowed_chat_ids):
                raise ValueError("Trading requires one explicitly configured allowlisted private chat.")
            if not self.alpaca_key_id or not self.alpaca_secret_key:
                raise ValueError("Configure Alpaca key and secret locally before enabling trading.")
            try:
                uuid.UUID(self.trading_account_id)
            except (ValueError, TypeError, AttributeError):
                raise ValueError("Configure the verified Alpaca account ID.") from None
        try:
            order_cap, daily_cap = Decimal(self.max_order_usd), Decimal(self.daily_buy_limit_usd)
            if (not order_cap.is_finite() or not daily_cap.is_finite()
                    or not 1 <= order_cap <= daily_cap <= 1000000):
                raise InvalidOperation()
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("USD limits must be finite: 1 <= max_order_usd <= daily_buy_limit_usd <= 1000000.") from None
        if type(self.max_open_orders) is not int or not 1 <= self.max_open_orders <= 50:
            raise ValueError("max_open_orders must be between 1 and 50.")
        if type(self.auto_enabled) is not bool or type(self.auto_notify) is not bool:
            raise ValueError("Automatic trading switches must be JSON booleans.")
        if type(self.options_models_enabled) is not bool:
            raise ValueError("options_models_enabled must be a JSON boolean.")
        if self.options_model_device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("options_model_device must be auto, cpu, cuda or mps.")
        if not isinstance(self.options_model_cache_dir, Path):
            raise ValueError("options_model_cache_dir must be a filesystem path.")
        if not isinstance(self.auto_symbols, tuple) or len(self.auto_symbols) > 20:
            raise ValueError("auto_symbols must contain at most 20 symbols.")
        normalized = tuple(dict.fromkeys(symbol.strip().upper() for symbol in self.auto_symbols if isinstance(symbol, str)))
        if normalized != self.auto_symbols or any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol) for symbol in normalized):
            raise ValueError("auto_symbols must be unique normalized US-equity symbols.")
        if self.auto_enabled and not normalized:
            raise ValueError("Automatic trading requires at least one symbol.")
        if type(self.auto_horizon) is not int or self.auto_horizon not in {1, 5, 20}:
            raise ValueError("auto_horizon must be 1, 5 or 20 sessions.")
        if type(self.auto_interval_seconds) is not int or not 60 <= self.auto_interval_seconds <= 3600:
            raise ValueError("auto_interval_seconds must be between 60 and 3600.")
        try:
            auto_order = Decimal(self.auto_order_usd)
            auto_position = Decimal(self.auto_max_position_usd)
            buy_p = Decimal(self.auto_buy_probability)
            sell_p = Decimal(self.auto_sell_probability)
            if (not all(v.is_finite() for v in (auto_order, auto_position, buy_p, sell_p))
                    or auto_order <= 0 or auto_position <= 0 or auto_order > Decimal(self.max_order_usd)
                    or auto_position > Decimal("1000000")
                    or not Decimal("0") < sell_p < Decimal("0.5") < buy_p < Decimal("1")):
                raise InvalidOperation()
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("Invalid automatic trading budgets or probabilities.") from None
        if self.auto_enabled:
            if not (self.trading_enabled and self.prediction_enabled and self.market_provider == "alpaca"):
                raise ValueError("Automatic trading requires enabled Alpaca predictions and trading.")
            if self.trading_mode != "paper":
                raise ValueError("Automatic trading is paper-only; live mode is not supported.")
        if not 1 <= self.port <= 65535 or not 1 <= self.retention_days <= 365:
            raise ValueError("Invalid port or retention_days.")
        if not 100 <= self.max_messages <= 100000:
            raise ValueError("max_messages must be between 100 and 100000.")
        if not re.fullmatch(r"[A-Za-z0-9_]{5,32}", self.expected_bot_username):
            raise ValueError("Invalid expected bot username.")

    @property
    def resource(self) -> str:
        return self.public_base_url + "/mcp"

    @property
    def scopes(self) -> list[str]:
        return (["telegram:read"] + (["telegram:send"] if self.allow_send else [])
                + (["trading:read", "trading:prepare"] if self.trading_enabled else []))


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["allowed_chat_ids"] = tuple(raw["allowed_chat_ids"])
    raw["auto_symbols"] = tuple(raw.get("auto_symbols", ()))
    data = Path(raw.get("data_dir", "state"))
    raw["data_dir"] = data if data.is_absolute() else path.resolve().parent / data
    if raw.get("market_csv_dir"):
        csv_dir = Path(raw["market_csv_dir"])
        raw["market_csv_dir"] = csv_dir if csv_dir.is_absolute() else path.resolve().parent / csv_dir
    model_dir = Path(raw.get("options_model_cache_dir", "models"))
    raw["options_model_cache_dir"] = model_dir if model_dir.is_absolute() else path.resolve().parent / model_dir
    return Config(**raw)


def write_private_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
