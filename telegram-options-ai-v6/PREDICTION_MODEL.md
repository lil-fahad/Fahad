# Experimental forecast model v1

This is a reproducible statistical prototype. It has not been validated for market accuracy, profitability or suitability for trading. Version 4 adds a separate opt-in automatic **paper-only** worker that can consume a report only after additional quality, freshness, market and risk gates. Manual Alpaca orders still require personal Telegram confirmation; an ordinary forecast command does not start automatic execution.

## Inputs and forecast target

Daily US equity closes, at least 600 valid observations, at most 10,000 source rows and the latest 2,000 observations for computation. Horizons are fixed at 1, 5 or 20 observed trading sessions. The response does not invent a future calendar date. Missing exchange sessions shorter than a ten-calendar-day gap are not detectable without a full exchange calendar; input completeness must be checked before real use.

Alpha Vantage uses raw closes and split coefficients from Daily Adjusted/full history. A reverse cumulative product adjusts past closes for later splits; the provider's dividend-adjusted close is deliberately not used. Forecasts concern price return, not total return. The last actual raw close anchors the projected USD price. Future corporate actions are not forecast. Return-ratio features are invariant to a uniform split scale factor.

Alpaca uses paginated IEX `1Day` bars requested with `adjustment=split`, bounded to eight years and converted from timezone-aware timestamps to New York session dates. These are prices from one exchange, not consolidated-market closes. The latest complete split-adjusted close anchors its price forecast; source adjustments are not a point-in-time corporate-action database.

The Financial Datasets endpoint documents OHLCV but no explicit adjustment guarantee. Its output is therefore tagged `unverified` and never receives an approved directional signal. CSV may be owner-declared `split_adjusted`, `unverified` or `synthetic`. Supplying the wrong basis can invalidate results. Source names, retrieval time, last bar date, basis and a SHA-256 checksum travel with results.

Current Eastern-date bars are withheld until 18:00 New York time. This is conservative around early closes; it is not a full exchange calendar. Weekend rows, duplicate dates, non-finite/non-positive closes, gaps over ten calendar days and consecutive price ratios outside 1/1.8–1.8 are rejected for review. Large real movements can also trigger this guard. No missing prices are silently filled. Data older than seven calendar days has no current forecast price or range; historical evaluation remains available. A six-hour source cache limits repeated paid requests; failed refreshes may use a cache with an explicit warning and unchanged as-of date.

## Historical analogues

At each origin the eight features use only the origin and earlier closes: 1/5/20/60-session log returns; 20-session population volatility; log(close/SMA20); log(SMA20/SMA60); and a 14-session up/down log-return ratio. This last ratio is RSI-like, not Wilder's recursively smoothed RSI.

Candidates begin after 60 observations and must have a complete horizon return ending **strictly before** the forecast origin. This purges overlapping training labels across the evaluation boundary. Feature scales are computed on that candidate history only. Euclidean distance in these standardized features selects up to 40 nearest historical cases, with selected target-return windows at least one horizon apart. At least eight cases are required. Their features and market regimes may remain statistically dependent despite non-overlapping return windows.

The up estimate is `(up_cases + 5 × prior_up) / (case_count + 5)`, where `prior_up` is a Laplace-smoothed historical up frequency from non-overlapping past windows. This is an empirical estimate, **not separately calibrated** and not a probability of profitable trading. Flat returns count as not-up. The point forecast uses the median case log return. The range uses the cases' interpolated 10th and 90th return percentiles, then converts to a positive USD price range. Nominal 80% does not promise future coverage.

There are no tuned hyperparameters, random train/test splits, LLM-generated training labels, downloaded model executables or unsafe model deserialization. Python standard library implements the numerical model.

## Chronological evaluation

The evaluation uses expanding historical candidate sets, disjoint forward return windows, and a fixed final span of `max(200, 40 × horizon)` observations, subject to a minimum origin of 260. Forecasts are recreated at each origin with the same strict label cutoff. Audit rows record origin, target date, latest eligible label date, model and baseline probabilities, predicted/actual returns and interval inclusion.

Reported metrics:

| Metric | Meaning and comparison |
|---|---|
| Direction accuracy | Fraction of correctly classified positive versus non-positive returns; compare with historical-majority direction |
| Brier score | Mean squared probability error; lower is better; compare with the historical-frequency prior available at each origin |
| Return MAE | Mean absolute return error, in percentage points; compare with zero-return/unchanged-price forecast |
| Interval coverage | Fraction of held-out actual returns inside the historical 10th–90th percentile range |

This is forecast evaluation, not a portfolio simulation. There are no entry fills, transaction costs, spreads, slippage, taxes, position sizing, risk limits or profit metrics. Single-ticker testing has selection bias; only subsequent paper observations can assess forward performance.

A directional output abstains if the source basis is synthetic/unverified, data are old, there are fewer than 40 evaluation periods, either Brier or MAE fails to improve on its baseline, empirical interval coverage is below 70%, or the up probability lies in 45–55%. These are fixed reporting gates, not statistical significance tests or an assurance of edge. The same evaluation set informs the gate, so a passing gate is still exploratory and needs untouched prospective validation. For a 20-session horizon, about 1,100 observations are needed to obtain 40 evaluation periods; a 600-row source can run but must report insufficient evidence.

## Reproduction and actual evidence

Run `python -m telegram_bridge demo --horizon 5 --json` after dependency installation. This produces seeded synthetic `DEMO` prices and dated evaluation records, clearly labelled in every response. The seed, model and computations are deterministic; demonstration dates move with the execution date. An example report is included in `examples/synthetic_demo_report.json`. Its scores are software test evidence only, not real-market accuracy.

Live checks on 2026-09-07: Financial Datasets rejected historical retrieval because the connected balance was zero; Alpha Vantage returned an internal tool error on adjusted and raw daily requests. TradingCursor returned a current AAPL technical analysis, which was not used as historical data or treated as a calibrated probability. Runtime API adapters were tested with mocked documented responses. No live Telegram messages were sent.

References: [Alpha Vantage](https://www.alphavantage.co/documentation/#dailyadj), [Financial Datasets](https://docs.financialdatasets.ai/api/prices/historical), [time-based evaluation gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
