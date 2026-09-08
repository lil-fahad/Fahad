from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.getenv("PORT", "8080"))
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def _json_url(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "FahadRailwaySmoke/3"})
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


def extract_private_messages(updates: list[dict]) -> tuple[list[dict], int | None]:
    rows: list[dict] = []
    max_update_id: int | None = None
    for update in updates:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            max_update_id = update_id if max_update_id is None else max(max_update_id, update_id)
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if isinstance(chat_id, int) and chat.get("type") == "private":
            rows.append({
                "id": chat_id,
                "type": "private",
                "username": chat.get("username"),
                "first_name": chat.get("first_name"),
                "text": message.get("text"),
            })
    return rows, (max_update_id + 1 if max_update_id is not None else None)


def check_telegram() -> dict:
    if not TOKEN:
        return {"ok": False, "error": "token_not_configured", "chats": []}
    base = f"https://api.telegram.org/bot{TOKEN}"
    try:
        me = _json_url(base + "/getMe")
        bot = me.get("result") or {}
        updates = _json_url(base + "/getUpdates?timeout=0&limit=100&allowed_updates=%5B%22message%22%5D")
        chats, _ = extract_private_messages(updates.get("result") or [])
        unique = {row["id"]: row for row in chats}
        return {
            "ok": bool(me.get("ok")) and bool(updates.get("ok")),
            "bot_username": bot.get("username"),
            "bot_id": bot.get("id"),
            "private_chat_count": len(unique),
            "chats": list(unique.values()),
        }
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "chats": []}


def poll_telegram_forever() -> None:
    if not TOKEN:
        return
    base = f"https://api.telegram.org/bot{TOKEN}"
    offset: int | None = None
    while True:
        query = "timeout=20&limit=100&allowed_updates=%5B%22message%22%5D"
        if offset is not None:
            query += "&offset=" + str(offset)
        try:
            payload = _json_url(base + "/getUpdates?" + query, timeout=25)
            rows, next_offset = extract_private_messages(payload.get("result") or [])
            for row in rows:
                print("TELEGRAM_CHAT_DISCOVERED=" + json.dumps(row, ensure_ascii=True), flush=True)
            if next_offset is not None:
                offset = next_offset
        except Exception as exc:
            print("telegram listener error=" + type(exc).__name__, flush=True)
            time.sleep(2)


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
            "telegram": {k: v for k, v in TELEGRAM.items() if k != "chats"},
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
    threading.Thread(target=poll_telegram_forever, name="telegram-discovery", daemon=True).start()
    print(f"smoke server starting on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
