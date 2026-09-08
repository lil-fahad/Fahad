from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from heavy_lab.data.providers.base import MarketProvider
from heavy_lab.data.schema import REQUIRED_BAR_COLUMNS, validate_canonical_frame


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _session_for_local_timestamp(ts: pd.Timestamp) -> str:
    local_time = ts.timetz().replace(tzinfo=None)
    if local_time >= datetime.strptime("09:30", "%H:%M").time() and local_time <= datetime.strptime("16:00", "%H:%M").time():
        return "regular"
    if local_time < datetime.strptime("09:30", "%H:%M").time():
        return "pre"
    return "post"


def parse_bars(payload: dict, *, source_revision: str) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, bars in payload.items():
        for bar in bars or []:
            local_ts = pd.Timestamp(bar["t"])
            if local_ts.tzinfo is None:
                local_ts = local_ts.tz_localize(NY)
            else:
                local_ts = local_ts.tz_convert(NY)
            utc_ts = local_ts.tz_convert(UTC)
            rows.append(
                {
                    "timestamp_utc": utc_ts,
                    "symbol": str(symbol),
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar.get("v", 0.0)),
                    "session": _session_for_local_timestamp(local_ts),
                    "source": "nasdaq",
                    "source_revision": source_revision,
                }
            )
    df = pd.DataFrame(rows, columns=REQUIRED_BAR_COLUMNS)
    if not df.empty:
        df = df.sort_values(["symbol", "timestamp_utc"], kind="stable").reset_index(drop=True)
        validate_canonical_frame(df)
    return df


class NasdaqBarsProvider(MarketProvider):
    name = "nasdaq"

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        base_url: str,
        source_revision: str = "nasdaq-v2",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.source_revision = source_revision
        self.timeout_seconds = timeout_seconds

    def _token(self, client: httpx.Client) -> str:
        response = client.post(
            f"{self.base_url}/v1/auth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise RuntimeError("Nasdaq auth response did not include access_token")
        return str(token)

    def fetch(self, symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            token = self._token(client)
            response = client.get(
                f"{self.base_url}/v2/markets/equities/bars/{symbol}",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "start": start.astimezone(UTC).isoformat(),
                    "end": end.astimezone(UTC).isoformat(),
                    "interval": interval,
                },
            )
            response.raise_for_status()
            payload = response.json()
        return parse_bars(payload, source_revision=self.source_revision)
