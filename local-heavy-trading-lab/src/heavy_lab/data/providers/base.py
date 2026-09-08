from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class MarketProvider(ABC):
    name: str

    @abstractmethod
    def fetch(self, symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        """Return canonical bars for the requested symbol/window."""
        raise NotImplementedError
