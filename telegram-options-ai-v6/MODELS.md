# Optional AI ensemble for options PAPER

The v6 options-paper engine can optionally combine three local Hugging Face models:

- `amazon/chronos-2` — probabilistic short-horizon time-series forecast.
- `google/timesfm-2.5-200m-transformers` — independent time-series forecast.
- `ProsusAI/finbert` — financial-news sentiment.

Install on the deployment host only:

```bash
python -m pip install -r requirements-models.txt
python -m telegram_bridge configure-options-models --device auto --cache-dir models
python -m telegram_bridge download-options-models
python -m telegram_bridge serve
```

Weights are not included in the ZIP. The bot loads one model at a time from the cache and releases it after its vote to reduce peak memory. Missing or failing models degrade safely to the existing technical signal. Use `/optionsmodels` in the owner chat to inspect availability/errors.


## Server resources

The default `compose.yaml` is capped at **512 MB RAM** and is intended for the core bot without the local AI ensemble. Do not enable the model ensemble under that limit. For Chronos-2 + TimesFM 2.5 + FinBERT, use a larger host, install `requirements-models.txt` in a writable Python environment, and provide enough disk for the model cache. Sequential model loading reduces peak RAM compared with loading all three together, but 512 MB is not sufficient.

The AI layer does not alter the paper-only boundary: no broker order endpoint is used and no Alpaca credentials are required for the options simulator.
