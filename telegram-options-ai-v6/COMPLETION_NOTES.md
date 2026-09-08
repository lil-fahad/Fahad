# Version 4 completion notes

Added opt-in automatic buy/sell execution for Alpaca paper accounts while preserving the existing confirmed manual-trading path.

Key additions:

- `telegram_bridge/autotrading.py`: forecast gate, paper-only worker, sizing, durable decisions, strategy ownership and reconciliation.
- owner Telegram controls: `/autoon`, `/autooff`, `/autostatus`.
- local CLI controls: `configure-autotrading`, `autotrading-status`, `halt-autotrading`.
- automatic execution guard rechecks receiver health, policy state, global pause and poller lease before broker submission; `/autooff` and lease ownership are checked atomically at claim time.
- one durable decision per account/session/symbol and stable client IDs prevent duplicate broker POSTs after restarts or ambiguous responses.
- automatic live trading is rejected. Manual live trading remains separate and explicitly configured.
- MCP automatic status is read-only; no MCP automatic-start or broker-submit tool was added.
- bilingual operating guides: `AUTOTRADING_AR.md` and `AUTOTRADING.md`.

Development verification status is recorded in `VALIDATION.md`. No real Telegram or Alpaca account was connected while building or testing this release.
