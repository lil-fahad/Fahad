# Run the private connector continuously

This package is prepared for deployment. It has not been deployed, assigned a hostname, or connected to the user's live Telegram bot. The screenshot token was not copied into the package or used for verification.

## Host and domain

Use a persistent machine/container with Python 3.12+, outbound HTTPS access to `api.telegram.org`, and a local disk for SQLite. A public connection also needs a domain and HTTPS reverse proxy. Serverless function platforms with ephemeral storage and suspended processes do not fit this long-polling implementation.

1. Create DNS records for the chosen domain pointing to the host.
2. Run the setup command with `--public-base-url https://YOUR_DOMAIN`. Replace `YOUR_DOMAIN` with the actual domain; no trailing slash. Enter the rotated bot token in the terminal's hidden prompt.
3. Keep `bind_host` set to `127.0.0.1` for a directly hosted process. Run it as a dedicated OS user with restricted access to `private/`.
4. Run `python -m telegram_bridge doctor`, then `python -m telegram_bridge serve` using the prepared virtual environment.
5. Terminate TLS in front of localhost. For example, after installing Caddy, use this Caddyfile with the real domain:

```caddyfile
YOUR_DOMAIN {
    reverse_proxy 127.0.0.1:8787
}
```

Use the operating system's normal Caddy service to keep it running. DNS and inbound ports 80/443 must reach Caddy for its certificate setup. Preserve the original Host header and do not log query strings or request bodies. [Caddy reverse-proxy documentation](https://caddyserver.com/docs/quick-starts/reverse-proxy).

Only the proxy should be exposed; never expose an unauthenticated test listener. The HTTP service's public base URL drives callback URLs, token audience and host checks, so it must equal the externally used origin.

## Optional Docker Compose on Linux

After interactive setup creates `private/config.json`, change its `bind_host` to `0.0.0.0` so the service is reachable inside the container. Keep `public_base_url` equal to the HTTPS origin. Then:

```bash
sudo chown -R 10001:10001 private
docker compose up --build -d
docker compose logs --tail=50 bridge
```

The container uses UID 10001 and a persistent bind mount. Host port 8787 is bound only on 127.0.0.1. Configure Caddy on the host as above. The ownership command is specific to this package's `private` directory; do not apply it elsewhere.

```bash
docker compose stop
```

Stop only this service when changing configuration. Do not run a second receiver for the same bot. The included Dockerfile was reviewed but Docker image build/run was not verified in this environment.

## ChatGPT connection

Use [OpenAI's connection guide](https://developers.openai.com/plugins/deploy/connect-chatgpt): enable Developer mode if available, create the plugin with the actual `https://YOUR_DOMAIN/mcp` endpoint, choose OAuth/DCR, then authorize using the connection password created by setup.

The SDK exposes:

- `/.well-known/oauth-authorization-server`
- `/.well-known/oauth-protected-resource/mcp`
- `/register`, `/authorize`, `/token`, `/revoke`
- `/connect`, `/mcp`, `/healthz`

This server supports DCR and client-secret authentication. It does not advertise CIMD or stable callback issuer identification. It accepts ChatGPT callback-specific redirect URLs (`https://chatgpt.com/connector/oauth/{callback_id}`), and the documented legacy stable callback. Choose DCR in any client-registration selector. Do not choose an anonymous connection for the HTTP endpoint.

Test `get_status`, `list_chats`, `list_messages` and `fetch` first. A send test is a separate real external action: run it only when the owner has requested the exact recipient and content.

## Operational checks

- `/healthz` shows process health only. `get_status` reports Telegram receiver health after authentication.
- If `get_status` shows a revoked token or receiver conflict, fix that condition and restart; the service will not override another integration.
- Keep an OS/service restart policy enabled and preserve `private/state/bridge.sqlite3` together with its WAL while the server is running. Stop the process for a simple consistent full-directory backup.
- Revoke all connection tokens with `python -m telegram_bridge revoke-connections`. Registered OAuth client identities are retained so reauthorization can work.
- A domain change requires updating `public_base_url`, restarting and reconnecting ChatGPT. Existing audience-bound access tokens will be rejected.
- After enabling or disabling sending, refresh the tool metadata in ChatGPT and reauthorize with the appropriate scopes.
- The running host and private connection have not been selected/provisioned by this task. A live end-to-end connection remains pending.
# Prediction and trading commands in v4

For Telegram command replies, a continuously running host with outbound HTTPS and persistent storage is sufficient; public inbound HTTPS is needed only for ChatGPT's remote MCP connection. First run `setup`, then `configure-predictions --provider alpha_vantage` (or `financial_datasets`) locally to enter a separate market API key without echoing it. Restart the service after changing configuration. Prediction command replies are enabled independently of the general MCP send permission.

Keep `private/state` across upgrades: it now contains the durable command queue and send ledger. One poller and one command worker share the process and SQLite lease; do not run multiple replicas. API keys from apps installed in ChatGPT are not exported to this host. Use an API plan that provides full daily history, or configure local CSV files with a truthful adjustment basis.

The existing Docker volume `./private:/private` includes market cache files automatically. If using CSV in Docker, place CSVs inside that volume and set `market_csv_dir` to `/private/market-data` in the container configuration, rather than a host-only path. Prediction results carry data dates and may abstain. Human-confirmed manual execution remains available. An optional automatic paper worker runs in the same process; see `AUTOTRADING.md`/`AUTOTRADING_AR.md`. Do not run a second scheduler or replica against the same state database.

## Alpaca account setup

Stop the service before configuration changes. Run `configure-trading --mode paper` to enter that account's two keys in hidden prompts, bind the verified account UUID to one allowlisted private chat, and enable trading and Alpaca IEX predictions. Then run `trading-status` (read-only) and restart `serve`. Explicit `--mode live` accepts live-account keys; never infer live authorization from a forecast. The running code only submits after a fresh human `/confirm` command.

Allow outbound HTTPS to `paper-api.alpaca.markets` for paper or `api.alpaca.markets` for live, plus `data.alpaca.markets` and `api.telegram.org`. These hosts are fixed in code. Keep the machine clock synchronized; stale quotes or market clocks block orders. Only one process may poll the bot. Paper and live require different credentials, and a Paper account reset requires locally configuring its replacement keys/account identity.

Preserve the entire SQLite database when upgrading, including the new `trades` table. Startup adds the table without deleting old data. It retains order details, broker/account identifiers, claims and reconciliation state beyond inbox retention. Credentials in `private/config.json` are private but not encrypted at rest; protect the host and backups. Losing the ledger can lose deduplication and daily-limit history. Do not clear it to bypass an unknown submission.

After trading is enabled, refresh ChatGPT tool metadata and reconnect with `trading:read` and/or `trading:prepare` plus `telegram:read`. There is no MCP execution or cancellation tool. A draft prepared through MCP must be confirmed by the owner in Telegram within two minutes. `/pause` or local `halt-trading` blocks new submissions; it does not cancel open orders or interrupt a submission already started. Use the broker dashboard to verify persistent unknown states. The new broker adapter has only simulated transport validation in this package; a real paper-account trial remains pending.


## Automatic paper trading

Configure it only after Alpaca paper trading and predictions are working:

```bash
python -m telegram_bridge configure-autotrading --symbols AAPL,MSFT --order-usd 100 --max-position-usd 500
python -m telegram_bridge serve
```

Then use `/autoon` in the configured owner chat. Preserve the full SQLite database and private config across deployments because `auto_decisions`, `trades`, the send ledger and policy state provide ownership and duplicate-prevention history. `/autooff` stops new automatic submissions; `/pause` stops all new trading. Neither command retroactively cancels an already claimed broker order. Automatic execution is rejected in live mode.
