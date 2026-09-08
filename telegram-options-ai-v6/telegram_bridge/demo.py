"""Reproducible synthetic data for a credential-free smoke test, never a market feed."""
import math
import random
from datetime import datetime, timedelta, timezone

from .market import Bar, Series, completed_day


def demo_series(count: int = 1200) -> Series:
    rng = random.Random(511)
    days, day = [], completed_day()
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day -= timedelta(days=1)
    price, previous, bars = 100.0, 0.0, []
    for i, day in enumerate(reversed(days)):
        move = 0.0002 + 0.12 * previous + 0.002 * math.sin(i / 37) + rng.gauss(0, 0.014)
        price *= math.exp(move)
        bars.append(Bar(day, price, price))
        previous = move
    return Series("DEMO", "Synthetic seeded demonstration; NOT market prices", "synthetic", tuple(bars),
                  datetime.now(timezone.utc).isoformat()).checked()
