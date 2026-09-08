# Validation

## Version 4 — automatic paper trading

Development verification in this environment used synthetic credentials, generated market reports, and stateful `httpx.MockTransport` broker/Telegram simulations. **No real Telegram bot message and no real Alpaca order was sent.**

Verified executable subset on 8 September 2026:

```text
70 passed, 1 deselected
```

The deselected test is the MCP trading integration test. The separate MCP/auth test module also cannot be collected in this execution environment because the preinstalled environment does not contain the pinned `mcp==1.29.1` dependency and outbound package installation is blocked. The project still declares that dependency in its lock/requirements; run the full suite after installing `requirements-dev.lock` on the target host.

The passing tests include the original non-MCP bridge, command, prediction and trading behaviors plus new automatic-paper cases for:

- strict paper-only automatic configuration;
- forecast quality/action thresholds;
- automatic buy, simulated fill, later strategy-owned sell, and restart deduplication;
- prevention of manual `/confirm` on automatic drafts;
- owner-only `/autoon`, `/autooff`, `/autostatus` and urgent auto-off queueing;
- narrowly scoped automatic owner notices when generic MCP sending is disabled;
- CLI policy persistence with an explicit stopped state;
- multi-symbol progression without waiting a full polling interval between symbols;
- required poller lease and receiver health;
- exchange-calendar completed-session handling;
- unknown-order reconciliation blocking a new decision;
- final automatic runtime recheck after awaited preflight and before broker POST;
- atomic claim-time recheck of `/autooff`, global pause state, and the active poller lease.

Additional static verification performed:

- `python -m compileall telegram_bridge`
- `git diff --check`

The forecasting backtest remains a forecast-quality evaluation, not a trading-profit backtest. It does not establish future profitability or suitability for live trading.

## Version 5 — keyless SPY/SPX options paper simulation

Version 5 adds a broker-independent paper-options engine. Tests use fake underlying snapshots only; no option or stock order is sent externally.

Verified behaviors include theoretical SPX European and SPY American pricing, automatic Call/Put selection, durable paper entries/exits and P/L, restart persistence, five-minute duplicate suppression, owner-only Telegram controls, urgent `/optionsoff`, and processing SPY and SPX in the same cycle.

The full MCP suite still requires the pinned `mcp==1.29.1` package. In this build environment that package is not installed, so MCP collection cannot run here. The non-MCP executable suite is run separately and recorded during packaging.

Live keyless-feed connectivity was not verified from the build sandbox because outbound DNS/network access is unavailable there. The feed parser and trading engine are exercised with deterministic synthetic snapshots; the target server must have normal outbound HTTPS/DNS access for the keyless underlying feed.

## Version 6 — optional local AI ensemble for options PAPER

Version 6 adds an optional local ensemble layer for the SPY/SPX paper-options engine. It does not add a broker execution path. The layer combines the existing technical signal with adapters for `amazon/chronos-2`, `google/timesfm-2.5-200m-transformers`, and `ProsusAI/finbert`. Missing or failing model components are isolated and the engine falls back to the remaining votes or the prior technical signal.

Fresh build-environment verification before packaging on 8 September 2026:

```text
94 passed, 1 deselected
```

The deselected case is the MCP-specific integration test because the build environment does not contain the pinned `mcp==1.29.1` package and outbound package installation is unavailable. The core project still pins MCP in `requirements.lock`; the full MCP test must be run after installing project dependencies on the deployment host.

The passing v6 tests cover ensemble agreement/disagreement, technical fallback, per-adapter failure isolation, persistence of model components in SQLite, Bridge wiring, owner-only `/optionsmodels`, CLI configuration, model-download fail-closed behavior, lazy loading and release-after-vote behavior, and preservation of the existing options, prediction, automatic-paper and manual-trading paths.

Static verification also includes:

```text
python -m compileall -q telegram_bridge
```

Actual Chronos-2, TimesFM 2.5, and FinBERT weights were **not downloaded or executed in this build sandbox** because the optional model dependencies are not installed and external model downloads are unavailable. Adapter behavior is covered with injected test doubles, and the deployment command `download-options-models` is designed to fail closed if any configured model cannot be loaded. Live Yahoo/keyless market or news connectivity was likewise not verified from this sandbox. No real broker order or option trade was sent.
