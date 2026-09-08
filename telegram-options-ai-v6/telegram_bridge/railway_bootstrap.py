"""Run the original v6 bridge on Railway with environment-based configuration."""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import Mapping

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .commands import options_command
from .config import Config, hash_password
from .server import create_app
from .storage import Store
from .telegram import TelegramAPI


def _boolean(value: str) -> bool:
    value = value.strip().lower()
    if value not in {'true', 'false', '1', '0'}:
        raise ValueError('A Railway boolean must be true, false, 1 or 0.')
    return value in {'true', '1'}


def config_from_environment(environ: Mapping[str, str] | None = None) -> Config:
    env = os.environ if environ is None else environ
    token = env.get('TELEGRAM_BOT_TOKEN', '').strip()
    try:
        owner = int(env.get('TELEGRAM_OWNER_CHAT_ID', ''))
    except ValueError:
        raise ValueError('TELEGRAM_OWNER_CHAT_ID must contain the private owner chat ID.') from None
    if owner <= 0:
        raise ValueError('TELEGRAM_OWNER_CHAT_ID must be a positive private chat ID.')
    port = int(env.get('PORT', '8080'))
    domain = env.get('RAILWAY_PUBLIC_DOMAIN', '').strip()
    data_dir = Path(env.get('BRIDGE_DATA_DIR', '/data/state'))
    return Config(
        bot_token=token, allowed_chat_ids=(owner,),
        password_hash=env.get('BRIDGE_PASSWORD_HASH') or hash_password(secrets.token_urlsafe(32)),
        expected_bot_username='Lil_fahad_bot',
        public_base_url=('https://' + domain) if domain else f'http://127.0.0.1:{port}',
        data_dir=data_dir, bind_host='0.0.0.0', port=port,
        allow_send=False, trading_enabled=False, trading_mode='paper', auto_enabled=False,
        options_models_enabled=_boolean(env.get('OPTIONS_MODELS_ENABLED', 'false')),
        options_model_cache_dir=Path(env.get('OPTIONS_MODEL_CACHE_DIR', '/data/models')),
    )


def create_railway_app(config: Config, *, api: TelegramAPI | None = None,
                       paper_autostart: bool = True) -> Starlette:
    store = Store(config.data_dir / 'bridge.sqlite3')
    inner = create_app(config, store, api or TelegramAPI(config.bot_token))
    bridge = inner.state.bridge
    if not store.get('options_paper_state') and paper_autostart:
        bridge.options_paper.set_running(config.allowed_chat_ids[0], True)
    started_at = time.time()

    async def report_receiver():
        while bridge.last_poll_ok is None and bridge.poll_error is None:
            await asyncio.sleep(0.2)
        print('TELEGRAM_RECEIVER=' + json.dumps({
            'receiving': bridge.last_poll_ok is not None and bridge.poll_error is None,
            'error': bridge.poll_error,
        }), flush=True)

    @asynccontextmanager
    async def lifespan(app):
        reporter = None
        try:
            async with inner.router.lifespan_context(inner):
                status = await options_command(bridge, {
                    'chat_id': config.allowed_chat_ids[0], 'request': {'verb': 'optionsstatus'},
                })
                print('OPTIONS_STATUS_SELF_CHECK=' + json.dumps({
                    'version': '6.0.0', 'bot': bridge.bot_info['username'],
                    'paper_only': True, 'models_enabled': config.options_models_enabled,
                    'reply': status,
                }, ensure_ascii=False), flush=True)
                reporter = asyncio.create_task(report_receiver())
                yield
        finally:
            if reporter:
                reporter.cancel()
                await asyncio.gather(reporter, return_exceptions=True)
            store.close()

    async def healthz(request):
        receiving = (bridge.last_poll_ok is not None and not bridge.poll_error
                     and time.time() - bridge.last_poll_ok < 120)
        # The first long poll may take 25 seconds; allow Railway to complete cutover.
        starting = bridge.last_poll_ok is None and time.time() - started_at < 60
        healthy = bridge.bot_info is not None and (receiving or starting)
        return JSONResponse({
            'status': 'ok' if healthy else 'unavailable',
            'service': 'fahad-options-ai-v6', 'paper_only': True,
            'models_enabled': config.options_models_enabled,
            'telegram_receiving': bool(receiving),
        }, status_code=200 if healthy else 503, headers={'Cache-Control': 'no-store'})

    # Keep the original authenticated MCP app intact. Railway's internal Host
    # header is accepted only for the non-sensitive health endpoint.
    app = Starlette(routes=[Route('/healthz', healthz), Mount('/', app=inner)], lifespan=lifespan)
    app.state.bridge = bridge
    return app


def main():
    import uvicorn

    config = config_from_environment()
    app = create_railway_app(config, paper_autostart=_boolean(os.getenv('OPTIONS_PAPER_AUTOSTART', 'true')))
    uvicorn.run(app, host=config.bind_host, port=config.port, workers=1,
                access_log=False, log_level='warning', proxy_headers=False)


if __name__ == '__main__':
    main()
