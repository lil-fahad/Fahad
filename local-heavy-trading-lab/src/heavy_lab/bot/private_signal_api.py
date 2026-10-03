from __future__ import annotations

import json
from typing import Any, Callable
from urllib import request as urllib_request


class PrivateSignalAPI:
    """HTTP transport for non-executing PAPER trade intents."""

    def __init__(
        self,
        url: str,
        *,
        token: str = "",
        opener: Callable[..., Any] = urllib_request.urlopen,
    ) -> None:
        self.url = str(url).strip()
        if not self.url:
            raise ValueError("private signal API URL is required")
        self.token = str(token or "").strip()
        self._opener = opener

    def send(
        self,
        *,
        side: str,
        symbol: str,
        quantity: int,
        confidence: float | None,
        idempotency_key: str,
        source: str,
        timeout: int = 15,
    ) -> dict[str, Any]:
        side_value = str(side).strip().upper()
        symbol_value = str(symbol).strip().upper()
        if side_value not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if not symbol_value:
            raise ValueError("symbol is required")
        if int(quantity) <= 0:
            raise ValueError("quantity must be positive")
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        idem = str(idempotency_key).strip()
        if not idem:
            raise ValueError("idempotency_key is required")

        payload = {
            "schema_version": "1.0",
            "type": "trade_intent",
            "side": side_value,
            "symbol": symbol_value,
            "quantity": int(quantity),
            "confidence": None if confidence is None else float(confidence),
            "source": str(source).strip() or "local-bot",
            "paper_only": True,
            "execute": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": idem,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib_request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with self._opener(req, timeout=int(timeout)) as response:
            raw = response.read().decode("utf-8").strip()

        if not raw:
            return {"ok": True}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError("private signal API returned non-object JSON")
        return data
