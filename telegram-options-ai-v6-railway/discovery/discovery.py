from __future__ import annotations

import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.getenv("PORT", "8080"))
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def api(method: str) -> dict:
    if not TOKEN:
        return {"ok": False, "error": "token_not_configured"}
    url = f"https://api.telegram.org/bot{TOKEN}/{method}"
    req = urllib.request.Request(url, headers={"User-Agent": "FahadRailwayDiscovery/1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def discover() -> dict:
    me = api("getMe")
    updates = api('getUpdates?timeout=0&limit=100&allowed_updates=%5B%22message%22%5D')
    chats: dict[int, dict] = {}
    for update in updates.get("result") or []:
        msg = update.get("message") or {}
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        if isinstance(cid, int):
            chats[cid] = {
                "id": cid,
                "type": chat.get("type"),
                "username": chat.get("username"),
                "first_name": chat.get("first_name"),
                "text": msg.get("text"),
            }
    bot = me.get("result") or {}
    return {
        "telegram_ok": bool(me.get("ok")) and bool(updates.get("ok")),
        "bot_id": bot.get("id"),
        "bot_username": bot.get("username"),
        "chats": list(chats.values()),
    }


DISCOVERY = discover()
print("TELEGRAM_DISCOVERY=" + json.dumps(DISCOVERY, ensure_ascii=True), flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/healthz":
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps({"ok": bool(DISCOVERY.get("telegram_ok")), "bot_username": DISCOVERY.get("bot_username")}).encode()
        self.send_response(200 if DISCOVERY.get("telegram_ok") else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


if __name__ == "__main__":
    print(f"discovery server listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
