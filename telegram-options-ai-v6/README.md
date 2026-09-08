# Fahad Telegram Trading + Predictions + MCP Bridge v6

## Optional AI ensemble (v6)

Version 6 can add **Chronos-2 + TimesFM 2.5 + FinBERT** to SPY/SPX PAPER decisions. The heavyweight dependencies and model weights are optional and are never bundled in the core ZIP. Configure/download them on the deployment host; failures fall back to the existing technical signal. See [MODELS.md](MODELS.md).


Private, single-owner connector for `@Lil_fahad_bot`. Arabic setup: [README_AR.md](README_AR.md).

Version 5 adds an independent **keyless SPY + SPX options PAPER simulator**. It uses only underlying-price bars, builds a synthetic theoretical option chain, selects Call/Put, expiry and strike, and keeps durable paper P/L. It never calls a broker order endpoint and does not require Alpaca credentials. See [OPTIONS_PAPER_AR.md](OPTIONS_PAPER_AR.md).

Version 4 adds opt-in **automatic buy/sell for Alpaca paper accounts** with durable decision and order ledgers, restart deduplication, receiver-health guards and owner `/autoon`/`/autooff` controls. Manual orders retain Telegram confirmation. Start with [AUTOTRADING_AR.md](AUTOTRADING_AR.md) or [AUTOTRADING.md](AUTOTRADING.md). No real broker account was connected during development; execution tests use simulated HTTP responses. On-demand forecasts remain experimental; no profitability claim is made.

## Architecture and starting point

Archetype: **tool-only**. Uses the official Python MCP SDK's `FastMCP`, Streamable HTTP transport, auth discovery, dynamic registration, PKCE verification, token and revocation handlers. The tool-only server patterns in the [OpenAI MCP server guide](https://developers.openai.com/plugins/build/mcp-server) are the starting point; Telegram adapters and the persistent single-owner authorization provider are implemented in this repository. The official UI showcase was reviewed; no widget is needed for this workflow.

One continuously running Python process receives Telegram updates using long polling. SQLite commits accepted messages and the receive cursor together before the next request acknowledges the batch. Every read and send is checked against the configured chat allowlist. Only text/captions are indexed. The server exposes `/mcp`; OAuth exposes discovery, registration, authorization, token exchange and revocation routes. `/connect` is the owner's authorization form.

OAuth uses DCR + authorization code with S256 PKCE. Access tokens last one hour; refresh tokens last 30 days and rotate. Bearer tokens/codes are stored by SHA-256 hash, bound to the configured MCP resource and scopes. Owner passwords use salted scrypt. The SDK handles client authentication; registered client secrets consequently remain in the protected SQLite database. Only ChatGPT callback addresses are accepted. CIMD and a general-purpose multi-user identity provider are not implemented.

## Tool contract

| Tool | Input | Behavior |
|---|---|---|
| `get_status` | none | Bot identity, receiver health and coverage |
| `list_chats` | none | Allowed numeric chat IDs and observed titles |
| `search` | `query` | One JSON text content item with `results: [{id,title,url}]`; 20 latest matches |
| `fetch` | `id` | One JSON text content item with `id,title,text,url,metadata` |
| `list_messages` | optional `chat_id,before,limit` | Cursor pagination with `has_more,next_before` |
| `send_message` | `chat_id,text,request_id` | Optional write tool; `request_id` is a UUID used for replay protection |
| `predict_market` | `symbol,horizon_sessions` | Experimental daily forecast and chronological validation summary; 1/5/20 sessions |
| `backtest_market` | `symbol,horizon_sessions` | Same evaluation with dated period-level records, no sends |
| `get_trading_account` | none | Optional; account mode, cash, limits and positions; `trading:read` |
| `list_trade_orders` | none | Optional; reconcile up to ten bot orders, unresolved first; `trading:read` |
| `get_autotrading_status` | none | Optional when auto paper is configured; read-only policy/decision status; `trading:read` |
| `prepare_trade` | `side,symbol,quantity,limit_price,request_id` | Optional; local draft only; `trading:prepare`; human confirms in Telegram within two minutes |

`search` uses literal Unicode casefolded substring matching, not embeddings. Private bot DMs do not have a general web message permalink: their `url` is the real bot chat URL and metadata explicitly sets `exact_message_url:false`. Channel/supergroup message links are used when available. Telegram message content is untrusted data, never an instruction or send authorization.

The `send_message` tool requires both `allow_send:true` in local configuration and the `telegram:send` OAuth scope. The caller/ChatGPT must enforce the user's explicit send request. Separately, `prediction_enabled:true` enables replies only to fresh `/predict`, `/backtest`, `/status`, `/help`, and `/start` commands from the owner of an allowlisted private chat. This does not grant MCP arbitrary send permission. Enabling trading adds owner-only `/buy`, `/sell`, `/confirm`, `/cancel`, `/account`, `/orders`, `/order`, `/pause` and `/resume`. When automatic paper trading is configured, `/autoon`, `/autooff`, and `/autostatus` control its in-process worker. MCP has no broker submit/cancel or automatic-start tool. Manual submissions require Telegram confirmation; qualifying automatic paper submissions use the saved policy instead.

Prediction jobs are committed atomically with received messages and the offset. A separate bounded worker stores exact reply text before submitting its deterministic UUID to the send ledger. Restarts preserve prepared replies; unknown delivery never causes an automatic resend. Jobs expire after ten minutes; forwarded, edited, group/channel and bot-authored messages do not initiate predictions. At most five commands per chat per minute and 100 queued jobs are admitted. Excess commands are ignored; the owner can issue a fresh request later. Trading requests must also be at most two minutes old; `/pause`, `/autooff` and `/cancel` bypass the normal per-minute admission cap, receive queue priority and use a bounded 200-job queue. They do not interrupt an in-flight request.

## Layout

| Path | Purpose |
|---|---|
| `telegram_bridge/config.py` | Validated configuration and password hashing |
| `telegram_bridge/storage.py` | SQLite inbox, receive cursor, leases, send ledger and OAuth persistence |
| `telegram_bridge/telegram.py` | Narrow Telegram HTTP client and receiver |
| `telegram_bridge/auth.py` | Owner login/consent and SDK OAuth provider |
| `telegram_bridge/server.py` | MCP tools, authorization and HTTP guards |
| `telegram_bridge/cli.py` | Hidden-token setup, start, doctor and revoke commands |
| `telegram_bridge/trading.py` | Fixed-host Alpaca adapter, account checks, expiring drafts, durable order ledger, reconciliation and cancellation |
| `telegram_bridge/autotrading.py` | Paper-only forecast policy, durable session decisions, ownership sizing and automatic worker |
| `telegram_bridge/market.py` | Alpaca IEX, Alpha Vantage, Financial Datasets and local CSV adapters; redacted errors and JSON cache |
| `telegram_bridge/prediction.py` | Causal historical analogues, empirical probabilities, walk-forward evaluation and Arabic rendering |
| `telegram_bridge/commands.py` | Private command authorization and durable reply worker |
| `telegram_bridge/demo.py` | Fixed-seed synthetic demonstration, not market data |
| `tests/` | Mock Telegram and full in-process HTTP OAuth/MCP integration tests |
| `requirements*.lock` | Versions used during validation |
| `Dockerfile`, `compose.yaml` | Single-process persistent deployment |

## Run and test

Python 3.12+; run inside this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m telegram_bridge setup
.venv/bin/python -m telegram_bridge configure-predictions --provider alpha_vantage
.venv/bin/python -m telegram_bridge doctor
.venv/bin/python -m telegram_bridge serve
```

For trading, after setup run `.venv/bin/python -m telegram_bridge configure-trading --mode paper`, then `trading-status` and `serve`. Setup reads the account to verify its UUID and restrictions, saves credentials using hidden prompts, enables prediction commands and selects Alpaca IEX daily split-adjusted history. It preserves OAuth and arbitrary-send permissions. `--mode live` applies only to the manual trading module and requires explicit local configuration and that account's keys. Automatic trading is rejected in live mode. Default caps are USD 500 per order, USD 1,000 gross submitted bot buys per New York day and five account open orders. These are implementation defaults, not a recommended investment budget.

Trades are whole-share limit/day orders with extended hours disabled. Recheck account binding, market clock, IEX quote freshness (60 seconds), price deviation (5%), cash/buying power and long shares available before submission. Claims and stable client IDs persist before POST; ambiguous outcomes trigger reads only and block further submissions until reconciled. Canceled/partially filled buys retain their full daily cap usage conservatively. Cancellation acknowledgment is not final cancellation. Pause does not cancel existing or already-started submissions. Preserve the durable `trades` ledger, which is retained beyond message retention and includes order/account details. Follow [TRADING_AR.md](TRADING_AR.md) for the complete operating contract.

Setup prompts for a **new** Telegram token without echoing it, verifies the expected bot, shows pending chat identities and asks the owner to choose exact numeric IDs. No first-user auto-enrollment. It creates a separate connection password and stores only its hash. It refuses to overwrite existing configuration or take over a webhook.

The separate `configure-predictions` command securely prompts for a market API key, atomically updates only prediction fields, and keeps existing send/OAuth permissions. Alternatively use `--provider csv --csv-dir /absolute/path --basis split_adjusted`. CSV adjustment semantics are owner-declared. Never mark total-return-adjusted closes as split-only. A new configuration remains read-only with prediction replies disabled until configured. Restart the process after changes.

No-key demonstration: `.venv/bin/python -m telegram_bridge demo`. CLI `predict` and `backtest` also accept `--csv PATH --symbol SYMBOL --basis split_adjusted --horizon 5 --json`; `--json` includes audit records. All model outputs retain source, as-of date, basis, input checksum, stale/synthetic flags, abstention reasons and actual historical interval coverage. The up probability is a smoothed empirical estimate, not a calibrated confidence score.

Runtime data APIs use the explicit local key, not the connected ChatGPT app's account. Alpha Vantage Daily Adjusted/full history requires an eligible plan. Financial Datasets uses `X-API-KEY` and may require credits. TradingCursor was checked through its ChatGPT app; no undocumented TradingCursor HTTP endpoint or automatic integration is claimed.

Tests have no dependency on a Telegram account or any live credentials:

```bash
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pytest -q
```

Stdio is supported for local trusted MCP clients:

```bash
.venv/bin/python -m telegram_bridge serve --stdio
```

The local client inherits the configured permissions in stdio mode; no HTTP listener is opened. Do not share an unprotected stdio-to-HTTP forwarding proxy. Use the authenticated HTTP server for public HTTPS hosting. A private [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels) is another transport option when available and authorized; its control-plane setup is separate from this code and has not been provisioned here.

## Reliability and boundaries

- Deploy one process with a persistent local SQLite volume; this is not a stateless serverless function or a horizontally scaled shared service.
- Duplicate update delivery is harmless; received message edits update the existing row.
- Receive errors back off, respect Telegram `retry_after`, and expose a safe status. Token/webhook conflicts stop the receiver. No automatic webhook deletion.
- Send requests are recorded before the upstream call. After a timeout or interrupted process, replaying the same UUID never sends again. An unknown result requires a human delivery check; this is not a claim of exactly-once Telegram delivery.
- Successful send results persist for deduplication. The ledger keeps request hashes and delivery metadata, not an extra copy of message text. Preserve the database across restarts; losing it also loses send deduplication history.
- Prepared prediction reply text and parsed requests are retained in `prediction_jobs` until the configured retention period or chat removal; delivery records remain in the send ledger. Market JSON caches live in `data_dir/market`, retain the latest snapshot per requested symbol, and can be deleted with the service stopped. They contain public price data, never API keys.
- Retention defaults to 30 days and 10,000 messages. Read endpoints exclude expired records even if the receiver is paused. Physical deletion occurs during ingestion. SQLite/WAL/backups may contain prior bytes; this is not a secure erasure facility.
- Authorization callbacks, bodies, hosts, origins and request rates are bounded. HTTP query/body logging is disabled. TLS is supplied by the deployment proxy.
- Reverse-proxy IP headers are intentionally not trusted; with one proxy, rate limits are shared across it. This is suitable for this personal connector, not a high-volume multi-tenant deployment.
- No live Telegram, hosted TLS or ChatGPT account connection was tested. See [VALIDATION.md](VALIDATION.md).

## Sources consulted

- [OpenAI MCP server guide](https://developers.openai.com/plugins/build/mcp-server)
- [OpenAI authentication](https://developers.openai.com/plugins/build/auth)
- [Define tools](https://developers.openai.com/plugins/plan/tools)
- [Official examples](https://developers.openai.com/plugins/build/examples)
- [Plugin quickstart](https://developers.openai.com/plugins/quickstart)
- [Connect and test](https://developers.openai.com/plugins/deploy/connect-chatgpt)
- [Optional UI](https://developers.openai.com/plugins/build/chatgpt-ui)
- [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk), installed version 1.29.1
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Alpaca paper trading](https://docs.alpaca.markets/us/docs/paper-trading)
- [Alpaca order lifecycle](https://docs.alpaca.markets/us/docs/orders-at-alpaca)
- [Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars)
- [Alpaca IEX coverage](https://docs.alpaca.markets/us/docs/market-data-faq)
- [Alpha Vantage Daily Adjusted](https://www.alphavantage.co/documentation/#dailyadj)
- [Financial Datasets historical prices](https://docs.financialdatasets.ai/api/prices/historical)
- [Chronological splitting and gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) — conceptual reference; sklearn is not a runtime dependency
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) — explains why empirical scores must not be presented as calibrated probabilities