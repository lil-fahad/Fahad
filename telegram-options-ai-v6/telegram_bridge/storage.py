from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class Store:
    """One SQLite database per dedicated bot. Secrets never form SQL or log messages."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        os.chmod(path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL, chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
                date INTEGER NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL,
                folded TEXT NOT NULL, url TEXT NOT NULL, metadata TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_chat_seq ON messages(chat_id, seq DESC);
            CREATE TABLE IF NOT EXISTS sends (
                request_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT, created INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS oauth (
                kind TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                expires REAL NOT NULL, family TEXT,
                PRIMARY KEY(kind, key)
            );
            CREATE TABLE IF NOT EXISTS leases (
                name TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prediction_jobs (
                update_id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL,
                request TEXT NOT NULL, response TEXT, status TEXT NOT NULL,
                created INTEGER NOT NULL, delivery TEXT
            );
            CREATE TABLE IF NOT EXISTS trades (
                ticket TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
                chat_id INTEGER NOT NULL, mode TEXT NOT NULL, account_id TEXT NOT NULL,
                payload TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL,
                state TEXT NOT NULL, submitted_day TEXT, remote_id TEXT,
                result TEXT, cancel_state TEXT
            );
            CREATE TABLE IF NOT EXISTS auto_decisions (
                account_id TEXT NOT NULL, session_date TEXT NOT NULL, symbol TEXT NOT NULL,
                request_id TEXT UNIQUE NOT NULL, decision TEXT NOT NULL, state TEXT NOT NULL,
                reason TEXT NOT NULL, report TEXT NOT NULL, created REAL NOT NULL,
                ticket TEXT, result TEXT, updated REAL,
                PRIMARY KEY(account_id, session_date, symbol)
            );
            CREATE INDEX IF NOT EXISTS auto_decisions_request ON auto_decisions(request_id);
            CREATE TABLE IF NOT EXISTS option_decisions (
                symbol TEXT NOT NULL, decision_slot INTEGER NOT NULL, action TEXT NOT NULL,
                reason TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY(symbol, decision_slot)
            );
            CREATE TABLE IF NOT EXISTS option_model_signals (
                symbol TEXT NOT NULL, decision_slot INTEGER NOT NULL, decision TEXT NOT NULL,
                confidence REAL NOT NULL, score REAL NOT NULL, components TEXT NOT NULL,
                degraded INTEGER NOT NULL, reason TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY(symbol, decision_slot)
            );
            CREATE TABLE IF NOT EXISTS option_positions (
                id TEXT PRIMARY KEY, symbol TEXT NOT NULL, option_type TEXT NOT NULL,
                strike REAL NOT NULL, expiry TEXT NOT NULL, contracts INTEGER NOT NULL,
                multiplier INTEGER NOT NULL, entry_premium REAL NOT NULL, entry_spot REAL NOT NULL,
                entry_time REAL NOT NULL, signal_strength REAL NOT NULL, delta REAL NOT NULL, iv REAL NOT NULL,
                settlement TEXT NOT NULL, style TEXT NOT NULL, status TEXT NOT NULL,
                last_premium REAL NOT NULL, last_spot REAL NOT NULL, last_mark_time REAL NOT NULL,
                exit_premium REAL, exit_spot REAL, exit_time REAL, exit_reason TEXT, pnl REAL
            );
            CREATE INDEX IF NOT EXISTS option_positions_status_symbol ON option_positions(status, symbol);
        """)

    @contextmanager
    def transaction(self):
        with self.lock:
            try:
                self.db.execute("BEGIN IMMEDIATE")
                yield self.db
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def close(self):
        with self.lock:
            self.db.close()

    def get(self, key: str, default: str = "") -> str:
        with self.lock:
            row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
            return row[0] if row else default

    def set(self, key: str, value: str):
        with self.transaction() as db:
            db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, value))

    def acquire_lease(self, owner: str) -> bool:
        with self.transaction() as db:
            now = time.time()
            row = db.execute("SELECT owner,expires FROM leases WHERE name='poller'").fetchone()
            if row and row[0] != owner and row[1] > now:
                return False
            db.execute("INSERT OR REPLACE INTO leases VALUES ('poller', ?, ?)", (owner, now + 90))
            return True

    def release_lease(self, owner: str):
        with self.transaction() as db:
            db.execute("DELETE FROM leases WHERE owner=?", (owner,))

    def owns_lease(self, owner: str) -> bool:
        with self.lock:
            row = self.db.execute("SELECT owner,expires FROM leases WHERE name='poller'").fetchone()
        return bool(row and row[0] == owner and row[1] > time.time())

    @staticmethod
    def _put_message(db, item: dict):
        db.execute("""INSERT INTO messages
            (id,chat_id,message_id,date,title,text,folded,url,metadata) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET date=excluded.date,title=excluded.title,
            text=excluded.text,folded=excluded.folded,url=excluded.url,metadata=excluded.metadata""",
            (item["id"], item["chat_id"], item["message_id"], item["date"], item["title"],
             item["text"], item["text"].casefold(), item["url"], json.dumps(item["metadata"], ensure_ascii=False)))

    def ingest(self, updates: list[dict], allowed: tuple[int, ...], bot_username: str,
               retention_days: int, max_messages: int, commands: dict | None = None):
        """Commit permitted messages and the receive cursor together, before Telegram ACK."""
        with self.transaction() as db:
            row = db.execute("SELECT value FROM kv WHERE key='offset'").fetchone()
            offset = int(row[0]) if row else 0
            for update in sorted(updates, key=lambda x: x["update_id"]):
                if update["update_id"] < offset:
                    continue
                message = next((update[k] for k in (
                    "message", "edited_message", "channel_post", "edited_channel_post") if k in update), None)
                if message and message.get("chat", {}).get("id") in allowed:
                    self._put_message(db, normalize_message(message, bot_username))
                    command = (commands or {}).get(update["update_id"])
                    if command:
                        chat_id, now = message["chat"]["id"], int(time.time())
                        recent = db.execute("SELECT count(*) FROM prediction_jobs WHERE chat_id=? AND created>?",
                                            (chat_id, now - 60)).fetchone()[0]
                        pending = db.execute("SELECT count(*) FROM prediction_jobs WHERE status='pending'").fetchone()[0]
                        urgent = command.get("verb") in {"pause", "cancel", "autooff", "optionsoff"}
                        if (recent < 5 and pending < 100) or (urgent and pending < 200):
                            db.execute("INSERT OR IGNORE INTO prediction_jobs VALUES (?,?,?,NULL,'pending',?,NULL)",
                                       (update["update_id"], chat_id, json.dumps(command), now))
                offset = update["update_id"] + 1
            db.execute("INSERT OR REPLACE INTO kv VALUES ('offset', ?)", (str(offset),))
            placeholders = ",".join("?" for _ in allowed)
            db.execute(f"DELETE FROM messages WHERE chat_id NOT IN ({placeholders})", allowed)
            db.execute("DELETE FROM messages WHERE date < ?", (int(time.time()) - retention_days * 86400,))
            db.execute("DELETE FROM messages WHERE seq NOT IN (SELECT seq FROM messages ORDER BY seq DESC LIMIT ?)",
                       (max_messages,))
            db.execute("DELETE FROM prediction_jobs WHERE created < ? OR chat_id NOT IN (" + placeholders + ")",
                       (int(time.time()) - retention_days * 86400, *allowed))

    def next_prediction_job(self) -> dict | None:
        with self.transaction() as db:
            db.execute("UPDATE prediction_jobs SET status='expired' WHERE status='pending' AND created < ?",
                       (int(time.time()) - 600,))
            row = db.execute("""SELECT * FROM prediction_jobs WHERE status='pending'
                ORDER BY CASE json_extract(request, '$.verb') WHEN 'pause' THEN 0 WHEN 'optionsoff' THEN 1 WHEN 'autooff' THEN 2 WHEN 'cancel' THEN 3 ELSE 4 END,
                update_id LIMIT 1""").fetchone()
            if not row:
                return None
            job = dict(row)
            job["request"] = json.loads(job["request"])
            return job

    def prepare_prediction_reply(self, update_id: int, response: str):
        with self.transaction() as db:
            db.execute("UPDATE prediction_jobs SET response=? WHERE update_id=? AND response IS NULL",
                       (response, update_id))

    def finish_prediction_job(self, update_id: int, delivery: dict):
        with self.transaction() as db:
            db.execute("UPDATE prediction_jobs SET status='done',delivery=? WHERE update_id=?",
                       (json.dumps(delivery), update_id))

    def messages(self, allowed: tuple[int, ...], query: str = "", limit: int = 20,
                 before: int | None = None, chat_id: int | None = None,
                 min_date: int | None = None) -> list[dict]:
        clauses = ["chat_id IN (" + ",".join("?" for _ in allowed) + ")"]
        values: list = list(allowed)
        if query:
            clauses.append("instr(folded, ?) > 0")
            values.append(query.casefold())
        if before is not None:
            clauses.append("seq < ?")
            values.append(before)
        if chat_id is not None:
            clauses.append("chat_id = ?")
            values.append(chat_id)
        if min_date is not None:
            clauses.append("date >= ?")
            values.append(min_date)
        with self.lock:
            rows = self.db.execute("SELECT * FROM messages WHERE " + " AND ".join(clauses)
                                   + " ORDER BY seq DESC LIMIT ?", [*values, limit]).fetchall()
        return [self._message(row) for row in rows]

    @staticmethod
    def _message(row) -> dict:
        item = dict(row)
        item.pop("folded", None)
        item["metadata"] = json.loads(item["metadata"])
        return item

    def fetch(self, id: str, allowed: tuple[int, ...]) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM messages WHERE id=?", (id,)).fetchone()
            return self._message(row) if row and row["chat_id"] in allowed else None

    def begin_send(self, request_id: str, payload: str) -> dict | None:
        with self.transaction() as db:
            row = db.execute("SELECT * FROM sends WHERE request_id=?", (request_id,)).fetchone()
            if row:
                if row["payload_hash"] != digest(payload):
                    raise ValueError("This request_id was already used for different message content.")
                return json.loads(row["result"]) if row["result"] else {
                    "status": "unknown", "request_id": request_id,
                    "detail": "Delivery may be pending or interrupted. Do not resend automatically."}
            db.execute("INSERT INTO sends VALUES (?,?,'pending',NULL,?)", (request_id, digest(payload), int(time.time())))
            return None

    def finish_send(self, request_id: str, result: dict, message: dict | None = None):
        with self.transaction() as db:
            db.execute("UPDATE sends SET status=?,result=? WHERE request_id=?",
                       (result["status"], json.dumps(result), request_id))
            if message:
                self._put_message(db, message)

    def oauth_get(self, kind: str, secret: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT value,expires FROM oauth WHERE kind=? AND key=?",
                                  (kind, digest(secret))).fetchone()
            return json.loads(row[0]) if row and (row[1] == 0 or row[1] > time.time()) else None

    def oauth_put(self, kind: str, secret: str, value: dict, expires: float = 0, family: str | None = None):
        with self.transaction() as db:
            db.execute("DELETE FROM oauth WHERE expires > 0 AND expires < ?", (time.time(),))
            db.execute("INSERT OR REPLACE INTO oauth VALUES (?,?,?,?,?)",
                       (kind, digest(secret), json.dumps(value), expires, family))

    def oauth_consume(self, kind: str, secret: str) -> dict | None:
        with self.transaction() as db:
            row = db.execute("SELECT value,expires FROM oauth WHERE kind=? AND key=?",
                             (kind, digest(secret))).fetchone()
            db.execute("DELETE FROM oauth WHERE kind=? AND key=?", (kind, digest(secret)))
            return json.loads(row[0]) if row and (row[1] == 0 or row[1] > time.time()) else None

    def oauth_count(self, kind: str) -> int:
        with self.lock:
            return self.db.execute("SELECT count(*) FROM oauth WHERE kind=?", (kind,)).fetchone()[0]

    def revoke_family(self, family: str):
        with self.transaction() as db:
            db.execute("DELETE FROM oauth WHERE family=?", (family,))


def normalize_message(message: dict, bot_username: str) -> dict:
    chat = message["chat"]
    chat_id, message_id = chat["id"], message["message_id"]
    title = chat.get("title") or " ".join(filter(None, (chat.get("first_name"), chat.get("last_name")))) or str(chat_id)
    text = message.get("text") or message.get("caption") or "[Non-text message]"
    # Bot DMs have no general web permalink. Use the real bot-chat URL and say so.
    url = "https://t.me/" + bot_username
    exact = False
    if chat.get("type") in {"supergroup", "channel"}:
        if chat.get("username"):
            url = f"https://t.me/{chat['username']}/{message_id}"
            exact = True
        elif str(chat_id).startswith("-100"):
            url = f"https://t.me/c/{str(chat_id)[4:]}/{message_id}"
            exact = True
    return {"id": f"{chat_id}:{message_id}", "chat_id": chat_id, "message_id": message_id,
            "date": message.get("edit_date", message["date"]), "title": title,
            "text": text, "url": url, "metadata": {
                "chat_type": chat["type"], "chat_id": str(chat_id), "message_id": message_id,
                "sender_id": str(message.get("from", {}).get("id", "")),
                "edited": "edit_date" in message, "exact_message_url": exact,
                "media": [kind for kind in ("photo", "document", "video", "voice", "audio", "sticker") if kind in message]}}
