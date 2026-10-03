from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSignal:
    decision_id: str
    model_id: str
    promotion_state: str
    side: str
    symbol: str
    quantity: int
    confidence: float | None = None


class AutoSignalPublisher:
    """Publish model-generated PAPER intents only after promotion."""

    def __init__(self, api: Any) -> None:
        self.api = api

    def publish(self, signal: ModelSignal) -> dict[str, Any]:
        if str(signal.promotion_state).strip().upper() != "PAPER_ELIGIBLE":
            return {
                "sent": False,
                "reason": "model_not_paper_eligible",
                "promotion_state": signal.promotion_state,
            }

        side = str(signal.side).strip().upper()
        symbol = str(signal.symbol).strip().upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if not symbol:
            raise ValueError("symbol is required")
        if int(signal.quantity) <= 0:
            raise ValueError("quantity must be positive")
        if signal.confidence is not None and not 0.0 <= float(signal.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        decision_id = str(signal.decision_id).strip()
        model_id = str(signal.model_id).strip()
        if not decision_id:
            raise ValueError("decision_id is required")
        if not model_id:
            raise ValueError("model_id is required")

        response = self.api.send(
            side=side,
            symbol=symbol,
            quantity=int(signal.quantity),
            confidence=None if signal.confidence is None else float(signal.confidence),
            idempotency_key=f"model-{decision_id}",
            source=f"model:{model_id}",
        )
        accepted = response.get("accepted", response.get("ok", True))
        return {
            "sent": accepted is not False,
            "accepted": accepted,
            "response": response,
        }
