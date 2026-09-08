import asyncio
import json
import time

import httpx
import pytest
from starlette.testclient import TestClient

from telegram_bridge.telegram import TelegramAPI


def environment(tmp_path, **extra):
    return {
        'TELEGRAM_BOT_TOKEN': '123456789:' + 'x' * 35,
        'TELEGRAM_OWNER_CHAT_ID': '8644335458',
        'RAILWAY_PUBLIC_DOMAIN': 'bot.example.test',
        'PORT': '8080',
        'BRIDGE_DATA_DIR': str(tmp_path / 'state'),
        'OPTIONS_MODELS_ENABLED': 'false',
        **extra,
    }


def test_environment_cannot_enable_brokerage_or_heavy_models_by_default(tmp_path):
    from telegram_bridge.railway_bootstrap import config_from_environment

    config = config_from_environment(environment(tmp_path, TRADING_MODE='live', TRADING_ENABLED='true'))
    assert config.allowed_chat_ids == (8644335458,)
    assert config.trading_mode == 'paper'
    assert not config.trading_enabled and not config.auto_enabled
    assert not config.options_models_enabled and not config.allow_send
    assert config.public_base_url == 'https://bot.example.test'


@pytest.mark.parametrize('owner', ['', 'not-a-chat', '-8644335458', '0'])
def test_bootstrap_requires_a_positive_private_owner_chat(tmp_path, owner):
    from telegram_bridge.railway_bootstrap import config_from_environment

    with pytest.raises(ValueError):
        config_from_environment(environment(tmp_path, TELEGRAM_OWNER_CHAT_ID=owner))


@pytest.mark.parametrize('cutover_conflict', [False, True])
def test_railway_health_and_real_command_worker_reply_to_owner_only(tmp_path, capsys, cutover_conflict):
    from telegram_bridge.railway_bootstrap import config_from_environment, create_railway_app

    now = int(time.time())
    updates = [
        {'update_id': i, 'message': {'message_id': i, 'date': now,
         'from': {'id': owner, 'is_bot': False, 'first_name': 'Owner'},
         'chat': {'id': owner, 'type': 'private', 'first_name': 'Owner'},
         'text': '/optionsstatus', 'entities': [{'offset': 0, 'length': 14, 'type': 'bot_command'}]}}
        for i, owner in [(1, 123456), (2, 8644335458)]
    ]
    sent = []
    polls = 0

    async def telegram(request):
        nonlocal polls
        method = request.url.path.rsplit('/', 1)[-1]
        payload = json.loads(request.content)
        if method == 'getMe':
            result = {'id': 123456789, 'is_bot': True, 'first_name': 'Fahad', 'username': 'Lil_fahad_bot'}
        elif method == 'getWebhookInfo':
            result = {'url': '', 'has_custom_certificate': False, 'pending_update_count': 0}
        elif method == 'getUpdates':
            polls += 1
            if cutover_conflict and polls == 1:
                return httpx.Response(409, json={'ok': False, 'error_code': 409,
                                               'description': 'Conflict: another getUpdates request'})
            await asyncio.sleep(0.01)
            result = [u for u in updates if u['update_id'] >= payload['offset']]
        elif method == 'sendMessage':
            sent.append(payload)
            result = {'message_id': 10, 'date': now, 'text': payload['text'],
                      'chat': {'id': payload['chat_id'], 'type': 'private'}}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={'ok': True, 'result': result})

    config = config_from_environment(environment(tmp_path))
    api = TelegramAPI(config.bot_token, httpx.AsyncClient(transport=httpx.MockTransport(telegram)))
    app = create_railway_app(config, api=api, paper_autostart=False)
    with TestClient(app, base_url='https://bot.example.test') as client:
        # Railway's internal healthcheck does not use the public Host header.
        assert client.get('/healthz', headers={'Host': 'healthcheck.railway.app'}).status_code == 200
        deadline = time.monotonic() + 5
        while not sent and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(sent) == 1
        assert sent[0]['chat_id'] == 8644335458
        assert 'Options PAPER:' in sent[0]['text'] and 'SPY + SPX' in sent[0]['text']
        assert sent[0]['reply_parameters']['message_id'] == 2
        response = client.get('/healthz')
        assert response.json()['telegram_receiving'] is True
        assert '8644335458' not in response.text
    assert 'OPTIONS_STATUS_SELF_CHECK=' in capsys.readouterr().out


def test_owner_stop_state_survives_application_restart(tmp_path):
    from telegram_bridge.railway_bootstrap import config_from_environment, create_railway_app
    from telegram_bridge.storage import Store

    config = config_from_environment(environment(tmp_path))
    store = Store(config.data_dir / 'bridge.sqlite3')
    store.set('options_paper_state', json.dumps({'running': False}))
    store.close()
    app = create_railway_app(config, paper_autostart=True)
    assert not app.state.bridge.options_paper.running()
    app.state.bridge.store.close()
