from __future__ import annotations

import json
import math
import time
from urllib.parse import quote

import httpx
import torch
from chronos import BaseChronosPipeline

SYMBOLS = {"SPY": "SPY", "SPX": "^SPX"}
CONTEXT = 96
HORIZON = 6
ORIGINS = 8


def fetch_bars(ticker: str) -> list[tuple[int, float]]:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(ticker, safe="")
    response = httpx.get(
        url,
        params={"interval": "5m", "range": "5d"},
        headers={"User-Agent": "Mozilla/5.0 ChronosBenchmark/1.0"},
        timeout=20,
        follow_redirects=False,
    )
    response.raise_for_status()
    root = response.json()["chart"]["result"][0]
    timestamps = root.get("timestamp") or []
    closes = ((root.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    rows = [
        (int(ts), float(close))
        for ts, close in zip(timestamps, closes)
        if isinstance(close, (int, float)) and close > 0 and math.isfinite(close)
    ]
    if len(rows) < CONTEXT + HORIZON + ORIGINS * HORIZON:
        raise RuntimeError(f"insufficient bars for {ticker}: {len(rows)}")
    return rows


def forecast(pipeline, history: list[float]) -> list[float]:
    context = torch.tensor(history, dtype=torch.float32)
    _, mean = pipeline.predict_quantiles(
        [context],
        prediction_length=HORIZON,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=1,
        context_length=CONTEXT,
    )
    values = mean[0].reshape(-1).float().cpu().tolist()
    if len(values) < HORIZON:
        raise RuntimeError(f"forecast length {len(values)} < {HORIZON}")
    return [float(x) for x in values[:HORIZON]]


def compute_metrics(actual_paths, forecast_paths, currents):
    path_errors, naive_path_errors = [], []
    terminal_errors, naive_terminal_errors, hits = [], [], []
    for actual, predicted, current in zip(actual_paths, forecast_paths, currents):
        path_errors.extend(abs(p - a) for p, a in zip(predicted, actual))
        naive_path_errors.extend(abs(current - a) for a in actual)
        terminal_errors.append(abs(predicted[-1] - actual[-1]))
        naive_terminal_errors.append(abs(current - actual[-1]))
        hits.append(int((predicted[-1] - current) * (actual[-1] - current) > 0))
    path_mae = sum(path_errors) / len(path_errors)
    naive_path_mae = sum(naive_path_errors) / len(naive_path_errors)
    terminal_mae = sum(terminal_errors) / len(terminal_errors)
    naive_terminal_mae = sum(naive_terminal_errors) / len(naive_terminal_errors)
    return {
        "path_mae": path_mae,
        "naive_path_mae": naive_path_mae,
        "path_skill": 1.0 - path_mae / naive_path_mae if naive_path_mae else None,
        "terminal_mae": terminal_mae,
        "naive_terminal_mae": naive_terminal_mae,
        "terminal_skill": 1.0 - terminal_mae / naive_terminal_mae if naive_terminal_mae else None,
        "direction_accuracy": sum(hits) / len(hits),
        "direction_hits": sum(hits),
    }


def evaluate_symbol(pipeline, ticker: str) -> dict:
    rows = fetch_bars(ticker)
    origin = len(rows) - HORIZON
    origins = []
    while origin >= CONTEXT and len(origins) < ORIGINS:
        origins.append(origin)
        origin -= HORIZON
    origins.reverse()

    actual_paths, forecast_paths, currents = [], [], []
    details = []
    for idx in origins:
        history = [x[1] for x in rows[idx - CONTEXT : idx]]
        actual = [x[1] for x in rows[idx : idx + HORIZON]]
        current = history[-1]
        predicted = forecast(pipeline, history)
        actual_paths.append(actual)
        forecast_paths.append(predicted)
        currents.append(current)
        details.append(
            {
                "origin_ts": rows[idx][0],
                "current": current,
                "predicted_terminal": predicted[-1],
                "actual_terminal": actual[-1],
            }
        )
    return {
        "bars": len(rows),
        "origins": len(origins),
        **compute_metrics(actual_paths, forecast_paths, currents),
        "details": details,
    }


def main() -> None:
    started = time.time()
    print("MODEL_LOAD_START", flush=True)
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print("MODEL_LOAD_OK", flush=True)
    result = {
        "model": "amazon/chronos-2",
        "dtype": "bfloat16",
        "protocol": "strict past-only 5m walk-forward; 96-bar context; 6-bar/30m horizon; 8 non-overlapping recent origins",
        "results": {},
    }
    for name, ticker in SYMBOLS.items():
        print(f"EVAL_START={name}", flush=True)
        result["results"][name] = evaluate_symbol(pipeline, ticker)
        print(f"EVAL_OK={name}", flush=True)
    result["elapsed_seconds"] = round(time.time() - started, 3)
    print("RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
