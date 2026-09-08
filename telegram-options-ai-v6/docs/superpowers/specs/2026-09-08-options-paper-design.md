# SPY + SPX Options Paper Engine Design

## Goal
Add an independent Telegram-controlled paper-options subsystem to v4 that requires no broker credentials. It trades only simulated SPY and SPX options, never submits external orders.

## Data and pricing
A keyless HTTP market feed retrieves recent underlying bars only. Option chains are synthetic: SPX uses European Black-Scholes-style pricing; SPY uses an American binomial approximation. Implied volatility is estimated from recent underlying returns and clamped to conservative bounds. Synthetic premiums and Greeks are explicitly labeled theoretical and are not market executable quotes.

## Strategy
Every five minutes while enabled, evaluate SPY and SPX. Direction comes from aligned short/slow EMA momentum and recent return. Up signals create calls; down signals create puts; weak/mixed signals hold. The engine chooses 0DTE during the earlier regular session when signal strength is high, otherwise 1DTE, and selects a nearby strike whose absolute delta is closest to 0.35.

## Risk and lifecycle
Initial paper equity is $100,000. A new trade may spend at most 5% of current paper equity and uses whole contracts with multiplier 100. Only one open option position per underlying is allowed. Target is +35% premium, stop is -25%, and the engine exits on target, stop, signal reversal, near-expiry/end of session, or expiry. A 3% daily realized-loss limit blocks new entries. Decisions use durable five-minute slots to prevent duplicates across restarts.

## Product differences
SPX is modeled as cash-settled and European-style. SPY is modeled as share-settled and American-style, but because this is paper simulation no real exercise or share delivery occurs.

## Telegram controls
Owner-only private commands: `/optionson`, `/optionsoff`, `/optionsstatus`, `/optionpositions`, `/optiontrades`, `/optionpnl`. The command worker runs even when prediction and Alpaca trading are disabled. Optional automatic notices go only to the configured private owner chat through a narrow internal permission.

## Persistence and recovery
SQLite stores decisions and paper option positions/trades. Restart restores open positions and running state. A unique `(symbol, decision_slot)` key prevents repeated entries in the same five-minute slot.

## Safety boundary
No Alpaca key, no broker API, no real order endpoint, no live trading mode. All outputs call the premium theoretical/paper. A future real broker adapter must be a separate explicit subsystem.

## Tests
Unit-test pricing, contract selection, direction, duplicate prevention, target/stop exits, restart persistence, Telegram authorization, and no-broker configuration. HTTP feed tests use mocked responses only.
