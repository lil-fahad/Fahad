# Automatic Paper Trading

Version 4 adds an opt-in forecast-driven execution loop for **Alpaca paper accounts only**. Manual trading retains the existing Telegram `/confirm` workflow. Automatic live trading is intentionally rejected by configuration and again at the execution boundary.

## Quick start

After the normal bot setup and Alpaca paper-account configuration:

```bash
python -m telegram_bridge configure-autotrading \
  --symbols AAPL,MSFT --order-usd 100 --max-position-usd 500
python -m telegram_bridge doctor
python -m telegram_bridge serve
```

Then, in the configured private owner chat:

```text
/autostatus
/autoon
```

Use `/autooff` to stop new automatic submissions and `/pause` to halt all new trading submissions. Reconfiguring the automatic policy always writes a stopped state; restart the service and explicitly use `/autoon` again.

## Decision gate

A trade requires a current, non-synthetic, split-adjusted Alpaca IEX forecast for the latest completed exchange session. The report must contain at least 40 evaluation periods, beat both Brier and MAE baselines, achieve at least 70% interval coverage, and not abstain.

Defaults:

- buy: upward signal, `probability_up >= 0.60`, positive forecast return;
- sell: downward signal, `probability_up <= 0.40`, negative forecast return;
- otherwise: persist a hold decision.

This forecast validation is not proof of trading profitability.

## Execution safety

- paper mode only;
- regular session and fresh broker clock required;
- fresh non-crossed IEX quote; automatic spread cap 1%;
- whole shares only, no short selling or automatic pyramiding;
- buys bounded by configured automatic budget, existing order limit, cash and buying power;
- sells bounded by strategy-owned filled shares and current broker availability;
- one durable decision per account/session/symbol;
- durable request UUID and Alpaca `client_order_id` prevent duplicate POSTs across retries/restarts;
- unresolved submissions are reconciled before another automatic decision;
- owner start state is tied to a policy fingerprint;
- the Telegram poller lease, receiver health, global pause and automatic running state are checked before scanning **and immediately after awaited preflight, before broker POST**.

There is no automatic stop-loss, take-profit, or end-of-day liquidation in this release.

## Commands

`/autoon`, `/autooff`, and `/autostatus` are owner-only bot commands. CLI equivalents are `autotrading-status` and `halt-autotrading`. MCP exposes automatic status as read-only when the feature is configured; MCP has no automatic start or execution tool.

See `AUTOTRADING_AR.md` for the full Arabic/Windows guide and `VALIDATION.md` for verification notes.
