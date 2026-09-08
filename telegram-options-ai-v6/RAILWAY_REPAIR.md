# Railway v6 deployment repair

The deployment branch is `railway-options-ai-v6` in `lil-fahad/Fahad`.
Railway builds `/telegram-options-ai-v6/Dockerfile`.

The previous release contained eight Base64 fragments. The first three held
19,999 characters each instead of their intended 30,000 characters. The full
payload had 84,041 characters, failed strict Base64 decoding, and lacked the
beginning of the archive, including `telegram_bridge/railway_bootstrap.py`.
The old release checksum was
`f3ef6475d6a699da870be36cd992b2c87f8b6f666a0b26a2846b7d8a8c569344`.
That exact archive cannot be reconstructed from the surviving fragments.

The recovered source is `Fahad-Telegram-Bridge-Options-AI-v6.zip`, SHA-256
`7724e01e61736ecfbc2d2183dc50b82fae7edf847da400aeddd0f1e2206bd53e`.
The original application modules are unchanged. The Railway bootstrap is new,
visible source. The runtime now builds directly from source, checks the
`runtime.sha256` manifest, runs the original tests and the bootstrap integration
tests, and only then packages the application into its final runtime image.
No heavy model dependencies or test dependencies are installed in that image.

Required environment:

- `TELEGRAM_BOT_TOKEN`: the existing Railway secret.
- `TELEGRAM_OWNER_CHAT_ID`: the existing private owner chat, `8644335458`.
- `PORT`: provided by Railway, default `8080`.
- `RAILWAY_PUBLIC_DOMAIN`: provided by Railway.
- `OPTIONS_MODELS_ENABLED`: `false` for this deployment.

The bootstrap always disables brokerage and generic outbound MCP messages.
SPY/SPX options remain local paper simulations. A fresh database starts the
paper engine; a saved `/optionsoff` decision is respected on restart.
`OPTIONS_PAPER_AUTOSTART=false` can keep a new database stopped.
State resides at `/data/state` unless `BRIDGE_DATA_DIR` is set. Without a Railway
volume, state does not survive container replacement. Use `BRIDGE_PASSWORD_HASH`
for a stable private MCP login if that optional interface is configured later.

`/healthz` exposes only service health and operating mode. It accepts Railway's
internal healthcheck Host header; the original MCP host and OAuth checks are
retained. Startup logs exercise the actual `/optionsstatus` handler, and a
separate receiver log confirms Telegram polling. The integration test verifies
the real command queue sends the status reply only to the configured owner.

To regenerate the manifest after an intentional source change:

```sh
python - <<'PY'
from pathlib import Path
from hashlib import sha256
files = [Path('pyproject.toml'), Path('requirements.lock'), Path('requirements-dev.lock')]
files += sorted(Path('telegram_bridge').glob('*.py'))
files += sorted(Path('tests').glob('*.py'))
Path('runtime.sha256').write_text(''.join(
    f'{sha256(p.read_bytes()).hexdigest()}  {p.as_posix()}\n' for p in files
))
PY
docker build -t fahad-options-ai-v6 .
```
