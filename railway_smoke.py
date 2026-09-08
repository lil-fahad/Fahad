from __future__ import annotations

import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.getenv("PORT", "8080"))


def check_yahoo() -> dict:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=5m&range=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "FahadRailwaySmoke/1"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read(2048)
            ok = response.status == 200 and b'chart' in body
            return {"ok": ok, "status": response.status, "latency_ms": round((time.time()-started)*1000)}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "latency_ms": round((time.time()-started)*1000)}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in {"/", "/healthz"}:
            self.send_response(404); self.end_headers(); return
        payload = {"service": "fahad-options-ai-v6-smoke", "python_ok": True, "market_feed": check_yahoo()}
        data = json.dumps(payload).encode()
        self.send_response(200 if payload["market_feed"].get("ok") else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


if __name__ == "__main__":
    print(f"smoke server starting on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
