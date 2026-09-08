# Options Model Ensemble Design

## Goal
Add an optional local AI ensemble to the existing SPY/SPX options-paper engine without changing its paper-only safety boundary or requiring broker credentials.

## Selected models
- `amazon/chronos-2` for probabilistic short-horizon time-series forecasting.
- `google/timesfm-2.5-200m-transformers` as an independent forecasting vote.
- `ProsusAI/finbert` for financial-news sentiment.

The model weights are not bundled in the project archive. They are downloaded once on the deployment host into a private cache after the user explicitly installs/enables the model extra.

## Architecture
`model_ensemble.py` owns all heavyweight optional integrations. It exposes a small `OptionsModelEnsemble.evaluate(snapshot, technical_kind, technical_strength)` interface and a pure `combine_votes` function. `options_paper.py` stays responsible for paper positions, contract pricing, risk limits and Telegram notices.

If optional dependencies/models are absent, fail to load, or a model fails during inference, the engine falls back to the existing technical signal and records the degraded state. A model failure must never stop Telegram polling or position marking/exits.

## Ensemble policy
Directional scores use -1 for put, 0 for hold, +1 for call. Available votes are normalized by active weights: technical 0.20, Chronos-2 0.35, TimesFM 2.5 0.30, FinBERT 0.15. Chronos and TimesFM confidence are derived from forecast return magnitude with a bounded scale; FinBERT uses the returned class probability. With at least one AI vote, require absolute normalized score >= 0.18; otherwise hold. With no AI votes, preserve the pre-existing technical result exactly.

Forecast horizon is six 5-minute bars (about 30 minutes) for the paper options decision. AI model inference only affects entries and signal-reversal exits; theoretical option valuation/risk limits are unchanged.

## News
The keyless underlying feed may attach a small set of recent headlines on a best-effort basis. Missing news means the FinBERT vote is unavailable, not neutral. News retrieval failures are isolated from price-data retrieval.

## Persistence and observability
Each model-assisted decision is persisted in a new `option_model_signals` table with symbol, decision slot, final decision, confidence, component JSON and timestamp. `/optionsmodels` shows whether models are enabled, which components are loaded/available, and the most recent model error without revealing host paths or secrets.

## Configuration
Add these safe fields to private config with backward-compatible defaults:
- `options_models_enabled=false`
- `options_model_device="auto"` where allowed values are `auto`, `cpu`, `cuda`, `mps`
- `options_model_cache_dir="models"` resolved relative to the private config directory

`configure-options-models` toggles the feature and validates the device. `download-options-models` verifies optional dependencies and downloads all three named model repositories to the configured cache. It performs no trading and no Telegram send.

## Dependencies
Core `requirements.lock` remains unchanged so the bot stays lightweight. `requirements-models.txt` contains the optional model stack. Model weights remain external to the ZIP because they are roughly gigabyte-scale collectively.

## Testing
TDD covers: weighted agreement, disagreement hold, technical fallback, engine consumption/persistence, command authorization/status, config compatibility, and lazy-load failure isolation. Tests inject fake adapters and never download model weights or call real market/broker endpoints.
