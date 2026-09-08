from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.getenv("PORT", "8080"))


def _json_url(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "FahadRailwaySmoke/2"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_yahoo() -> dict:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=5m&range=1d"
    started = time.time()
    try:
        data = _json_url(url)
        return {"ok": bool(data.get("chart", {}).get("result")), "latency_ms": round((time.time()-started)*1000)}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "latency_ms": round((time.time()-started)*1000)}


def check_telegram() -> dict:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return {"ok": False, "error": "token_not_configured", "chats": []}
    base = f"https://api.telegram.org/bot{token}"
    try:
        me = _json_url(base + "/getMe")
        bot = me.get("result") or {}
        updates = _json_url(base + "/getUpdates?timeout=0&limit=100&allowed_updates=%5B%22message%22%5D")
        chats: dict[int, dict] = {}
        for update in updates.get("result") or []:
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            if isinstance(chat_id, int) and chat.get("type") == "private":
                chats[chat_id] = {
                    "id": chat_id,
                    "type": "private",
                    "username": chat.get("username"),
                    "first_name": chat.get("first_name"),
                }
        return {
            "ok": bool(me.get("ok")) and bool(updates.get("ok")),
            "bot_username": bot.get("username"),
            "bot_id": bot.get("id"),
            "private_chat_count": len(chats),
            "chats": list(chats.values()),
        }
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "chats": []}


TELEGRAM = check_telegram()
print("telegram discovery:", json.dumps(TELEGRAM, ensure_ascii=True), flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in {"/", "/healthz"}:
            self.send_response(404)
            self.end_headers()
            return
        market = check_yahoo()
        payload = {
            "service": "fahad-options-ai-v6-smoke",
            "python_ok": True,
            "market_feed": market,
            "telegram": TELEGRAM,
        }
        data = json.dumps(payload).encode()
        ok = market.get("ok") and TELEGRAM.get("ok")
        self.send_response(200 if ok else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


if __name__ == "__main__":
    print(f"smoke server starting on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
