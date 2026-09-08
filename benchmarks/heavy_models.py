import gc
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import numpy as np
import torch
from dateutil import parser as dtparser

CTX = 60
H = 6
ORIGINS_PER_SYMBOL = 12
SYMBOLS = ["SPY", "SPX"]
NASDAQ_URL = "https://charting.nasdaq.com/data/charting/intraday"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nasdaq.com/",
}


@dataclass
class Bar:
    ts: datetime
    close: float


def _float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    try:
        x = float(str(v).strip().replace("$", "").replace(",", ""))
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        if x > 1e12:
            x /= 1000.0
        try:
            return datetime.fromtimestamp(x)
        except Exception:
            return None
    try:
        return dtparser.parse(str(v))
    except Exception:
        return None


def _pick(d: dict, keys: list[str]):
    lower = {str(k).lower(): k for k in d}
    for key in keys:
        if key.lower() in lower:
            return d[lower[key.lower()]]
    return None


def fetch_intraday(symbol: str) -> tuple[list[Bar], dict]:
    with httpx.Client(timeout=30.0, headers=HEADERS, follow_redirects=True) as client:
        r = client.get(
            NASDAQ_URL,
            params={"symbol": symbol, "mostRecent": 5, "includeLatestIntradayData": 1},
        )
        r.raise_for_status()
        root = r.json()

    raw = root.get("marketData") or root.get("data", {}).get("marketData") or []
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"{symbol}: Nasdaq returned no marketData; keys={list(root)[:20]}")

    parsed: list[Bar] = []
    first_keys = []
    for item in raw:
        if isinstance(item, dict):
            if not first_keys:
                first_keys = list(item.keys())
            tv = _pick(item, ["time", "timestamp", "datetime", "date", "x"])
            cv = _pick(item, ["close", "value", "price", "last", "lastsaleprice", "y"])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            tv, cv = item[0], item[1]
        else:
            continue
        ts, close = _dt(tv), _float(cv)
        if ts is not None and close is not None and close > 0:
            parsed.append(Bar(ts=ts, close=close))

    if len(parsed) < 100:
        raise RuntimeError(
            f"{symbol}: parsed only {len(parsed)}/{len(raw)} points; first_item_keys={first_keys} first_item={raw[0] if raw else None}"
        )

    parsed.sort(key=lambda b: b.ts)
    unique: list[Bar] = []
    seen = set()
    for b in parsed:
        k = b.ts.isoformat()
        if k not in seen:
            unique.append(b)
            seen.add(k)

    buckets: dict[tuple, Bar] = {}
    for b in unique:
        key = (b.ts.date(), b.ts.hour, b.ts.minute // 5)
        old = buckets.get(key)
        if old is None or b.ts >= old.ts:
            buckets[key] = b
    bars = sorted(buckets.values(), key=lambda b: b.ts)

    meta = {
        "companyName": root.get("companyName"),
        "raw_points": len(raw),
        "parsed_points": len(unique),
        "bars_5m": len(bars),
        "first_item_keys": first_keys,
    }
    return bars, meta


def contiguous(seq: list[Bar]) -> bool:
    return all(240 <= (b.ts - a.ts).total_seconds() <= 420 for a, b in zip(seq, seq[1:]))


def build_windows(bars: list[Bar]) -> list[dict]:
    out = []
    i = len(bars) - H
    while i >= CTX and len(out) < ORIGINS_PER_SYMBOL:
        full = bars[i - CTX : i + H]
        if len(full) == CTX + H and contiguous(full):
            hist = bars[i - CTX : i]
            out.append(
                {
                    "origin": hist[-1].ts.isoformat(),
                    "history": [b.close for b in hist],
                    "current": hist[-1].close,
                    "actual": bars[i + H - 1].close,
                    "actual_time": bars[i + H - 1].ts.isoformat(),
                }
            )
            i -= H
        else:
            i -= 1
    out.reverse()
    if len(out) < 4:
        raise RuntimeError(f"only {len(out)} contiguous windows found")
    return out


def wilson_interval(hits: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = hits / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(samples: list[dict]) -> dict:
    errs = [abs(x["predicted"] - x["actual"]) for x in samples]
    naive = [abs(x["current"] - x["actual"]) for x in samples]
    hits = []
    for x in samples:
        pdirection = np.sign(x["predicted"] - x["current"])
        adirection = np.sign(x["actual"] - x["current"])
        if adirection != 0:
            hits.append(int(pdirection == adirection))
    mae = float(np.mean(errs))
    naive_mae = float(np.mean(naive))
    hit_count = int(sum(hits))
    return {
        "n": len(samples),
        "mae": mae,
        "naive_mae": naive_mae,
        "skill_vs_last": (1.0 - mae / naive_mae) if naive_mae else None,
        "direction_hits": hit_count,
        "direction_n": len(hits),
        "direction_accuracy": float(np.mean(hits)) if hits else None,
        "direction_wilson95": wilson_interval(hit_count, len(hits)),
        "rmse": float(np.sqrt(np.mean([(x["predicted"] - x["actual"]) ** 2 for x in samples]))),
    }


def run_timesfm(windows_by_symbol: dict[str, list[dict]]) -> dict:
    from transformers import TimesFm2_5ModelForPrediction

    print("MODEL_LOAD_START timesfm", flush=True)
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        "google/timesfm-2.5-200m-transformers",
        low_cpu_mem_usage=True,
    ).to(torch.float32).eval()
    print("MODEL_LOAD_OK timesfm", flush=True)
    result = {}
    with torch.no_grad():
        for symbol, windows in windows_by_symbol.items():
            samples = []
            for w in windows:
                series = torch.tensor(w["history"], dtype=torch.float32, device=model.device)
                outputs = model(past_values=[series], return_dict=True)
                arr = outputs.mean_predictions.detach().cpu().numpy().reshape(-1)
                if len(arr) < H:
                    raise RuntimeError(f"TimesFM output too short: {len(arr)}")
                pred = float(arr[H - 1])
                samples.append({k: v for k, v in w.items() if k != "history"} | {"predicted": pred})
            result[symbol] = {"metrics": summarize(samples), "samples": samples}
    del model
    gc.collect()
    return result


def run_chronos(windows_by_symbol: dict[str, list[dict]]) -> dict:
    from chronos import Chronos2Pipeline

    print("MODEL_LOAD_START chronos2", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    print("MODEL_LOAD_OK chronos2", flush=True)
    result = {}
    with torch.no_grad():
        for symbol, windows in windows_by_symbol.items():
            samples = []
            for w in windows:
                series = torch.tensor(w["history"], dtype=torch.float32)
                _, mean = pipeline.predict_quantiles(
                    inputs=[series],
                    prediction_length=H,
                    quantile_levels=[0.1, 0.5, 0.9],
                    batch_size=1,
                )
                arr = mean[0].detach().cpu().numpy().reshape(-1)
                if len(arr) < H:
                    raise RuntimeError(f"Chronos output too short: {len(arr)}")
                pred = float(arr[H - 1])
                samples.append({k: v for k, v in w.items() if k != "history"} | {"predicted": pred})
            result[symbol] = {"metrics": summarize(samples), "samples": samples}
    del pipeline
    gc.collect()
    return result


def verdict(metrics: dict) -> str:
    s = metrics.get("skill_vs_last")
    d = metrics.get("direction_accuracy")
    if s is not None and d is not None and s > 0 and d >= 0.58:
        return "PASS_SHADOW_CANDIDATE"
    return "REJECT_OR_KEEP_SHADOW"


def main():
    print(f"torch={torch.__version__} cpu_threads={torch.get_num_threads()}", flush=True)
    data_meta = {}
    windows_by_symbol = {}
    for symbol in SYMBOLS:
        bars, meta = fetch_intraday(symbol)
        windows = build_windows(bars)
        data_meta[symbol] = meta | {"windows": len(windows)}
        windows_by_symbol[symbol] = windows
        print(f"DATA_OK {symbol} {json.dumps(data_meta[symbol])}", flush=True)

    output = {
        "protocol": {
            "source": "Nasdaq Charting intraday",
            "symbols": SYMBOLS,
            "bar": "5m aggregated from Nasdaq intraday points",
            "context_bars": CTX,
            "horizon_bars": H,
            "horizon_minutes": 30,
            "walk_forward": True,
            "holdouts_non_overlapping_by_horizon": True,
            "synthetic_market_data": False,
        },
        "data": data_meta,
        "models": {},
    }

    for name, fn in [("timesfm-2.5", run_timesfm), ("chronos-2", run_chronos)]:
        try:
            res = fn(windows_by_symbol)
            for sym in res:
                res[sym]["verdict"] = verdict(res[sym]["metrics"])
            output["models"][name] = {"status": "ok", "results": res}
            print(f"MODEL_OK {name}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            output["models"][name] = {"status": "error", "error": repr(exc)}
            print(f"MODEL_ERROR {name}: {exc!r}", flush=True)
        gc.collect()

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    lines = ["# Heavy Model Benchmark", "", f"Protocol: {output['protocol']}", ""]
    for name, block in output["models"].items():
        lines.append(f"## {name}")
        if block["status"] != "ok":
            lines.append(f"ERROR: {block['error']}")
            continue
        for sym, r in block["results"].items():
            m = r["metrics"]
            lines.append(
                f"- {sym}: n={m['n']}, MAE={m['mae']:.6f}, baseline={m['naive_mae']:.6f}, "
                f"skill={m['skill_vs_last']:.3%}, direction={m['direction_accuracy']:.3%}, "
                f"hits={m['direction_hits']}/{m['direction_n']}, Wilson95={m['direction_wilson95']}, verdict={r['verdict']}"
            )
    with open("benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("RESULT_JSON=" + json.dumps(output, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
