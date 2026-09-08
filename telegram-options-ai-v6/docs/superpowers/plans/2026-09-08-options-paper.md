# SPY + SPX Options Paper Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Telegram-controlled automatic paper options for SPY and SPX without broker credentials.

**Architecture:** Create an isolated `options_paper.py` subsystem containing the keyless underlying feed, theoretical pricer, strategy, durable paper portfolio, and worker. Wire it into existing SQLite, Telegram commands, Bridge lifecycle, and narrowly-scoped notices while leaving stock/Alpaca paths unchanged.

**Tech Stack:** Python 3.12, asyncio, httpx, sqlite3, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-options-paper-design.md`

## Global Constraints
- Paper simulation only; never call broker order endpoints.
- Symbols are fixed to SPY and SPX for this release.
- No API key is required for the options subsystem.
- Poll interval is 300 seconds; decision slots are durable.
- SPX European/cash-settled model; SPY American/share-settled model.
- Whole contracts, multiplier 100, one open position per underlying.

---

### Task 1: Pricing, feed, strategy and ledger
**Files:** Create `telegram_bridge/options_paper.py`; Modify `telegram_bridge/storage.py`; Test `tests/test_options_paper.py`.

**Interfaces:** `OptionsPaperEngine(bridge, feed=None)`, `once()`, `run()`, `set_running(chat_id, bool)`, `status()`, `positions()`, `trades()`, `pnl()`.

- [ ] Write failing tests for call/put selection, theoretical pricing, entry, exit, duplicate slot and restart persistence.
- [ ] Run focused tests and verify RED.
- [ ] Add SQLite tables and minimal engine implementation.
- [ ] Run focused tests and verify GREEN.

### Task 2: Telegram controls and lifecycle
**Files:** Modify `telegram_bridge/commands.py`, `telegram_bridge/telegram.py`, `telegram_bridge/storage.py`; Test `tests/test_options_paper.py` and `tests/test_commands.py`.

**Interfaces:** owner-only `/optionson`, `/optionsoff`, `/optionsstatus`, `/optionpositions`, `/optiontrades`, `/optionpnl`.

- [ ] Write failing authorization/command tests.
- [ ] Run focused tests and verify RED.
- [ ] Wire parser, worker, lifecycle and narrow `_options_notice` sending.
- [ ] Run focused tests and verify GREEN.

### Task 3: Documentation, version and package verification
**Files:** Modify `README.md`, `README_AR.md`, `pyproject.toml`, `VALIDATION.md`; Create `OPTIONS_PAPER_AR.md`.

- [ ] Document no-key setup and exact Telegram commands.
- [ ] Bump project to v5.0.0.
- [ ] Run all runnable tests, compileall and archive-content checks.
- [ ] Build clean ZIP excluding `.git`, caches, state and secrets.
