import asyncio
import time
from datetime import datetime, timezone

import pytest

from telegram_bridge.config import Config, hash_password
from telegram_bridge.storage import Store

TOKEN = '123456789:' + 'x' * 35


class FakeFeed:
    def __init__(self, direction='up', spot=100.0):
        self.direction = direction
        self.spot = spot
        self.calls = 0

    async def snapshot(self, symbol):
        self.calls += 1
        base = self.spot if symbol == 'SPY' else 5200.0
        if self.direction == 'up':
            closes = [base * (1 + i * 0.0008) for i in range(30)]
        elif self.direction == 'down':
            closes = [base * (1 - i * 0.0008) for i in range(30)]
        else:
            closes = [base for _ in range(30)]
        now = int(time.time())
        return {
            'symbol': symbol,
            'spot': closes[-1],
            'closes_5m': closes,
            'daily_closes': [base * (1 + 0.002 * ((i % 5) - 2)) for i in range(45)],
            'timestamp': now,
        }

    async def close(self):
        return None


class FakeBridge:
    def __init__(self, config, store, feed):
        self.config = config
        self.store = store
        self.owner = 'owner-test'
        self.stop = asyncio.Event()
        self.last_poll_ok = time.time()
        self.poll_error = None
        self.sent = []
        self.options_feed = feed
        store.acquire_lease(self.owner)

    async def send(self, chat_id, text, request_id, **kwargs):
        self.sent.append((chat_id, text, kwargs))
        return {'status': 'sent', 'request_id': request_id, 'message_id': len(self.sent)}


@pytest.fixture
def config(tmp_path):
    return Config(bot_token=TOKEN, allowed_chat_ids=(101,), password_hash=hash_password('password'),
                  data_dir=tmp_path / 'state', allow_send=False)


@pytest.fixture
def store(config):
    db = Store(config.data_dir / 'bridge.sqlite3')
    yield db
    db.close()


def test_option_pricers_return_positive_premium_and_sensible_delta():
    from telegram_bridge.options_paper import theoretical_option

    spx = theoretical_option('SPX', 'call', 5200, 5200, 1/365, 0.25)
    spy = theoretical_option('SPY', 'put', 500, 500, 1/365, 0.25)
    assert spx['premium'] > 0 and 0 < spx['delta'] < 1
    assert spy['premium'] > 0 and -1 < spy['delta'] < 0
    assert spx['style'] == 'european' and spy['style'] == 'american'


def test_uptrend_opens_spy_call_and_downtrend_opens_spx_put(config, store):
    from telegram_bridge.options_paper import OptionsPaperEngine

    async def run():
        bridge = FakeBridge(config, store, FakeFeed('up'))
        engine = OptionsPaperEngine(bridge, feed=bridge.options_feed)
        engine.set_running(101, True)
        result = await engine.once(symbols=('SPY',))
        assert result['action'] == 'opened'
        assert result['position']['symbol'] == 'SPY'
        assert result['position']['option_type'] == 'call'
        assert result['position']['contracts'] >= 1

        bridge.options_feed.direction = 'down'
        result2 = await engine.once(symbols=('SPX',), force_new_slot=True)
        assert result2['action'] == 'opened'
        assert result2['position']['symbol'] == 'SPX'
        assert result2['position']['option_type'] == 'put'
        await engine.close()
    asyncio.run(run())


def test_same_slot_cannot_duplicate_entry(config, store):
    from telegram_bridge.options_paper import OptionsPaperEngine

    async def run():
        feed = FakeFeed('up')
        bridge = FakeBridge(config, store, feed)
        engine = OptionsPaperEngine(bridge, feed=feed)
        engine.set_running(101, True)
        first = await engine.once(symbols=('SPY',))
        second = await engine.once(symbols=('SPY',))
        assert first['action'] == 'opened'
        assert second['action'] in {'already_decided', 'holding'}
        assert len(engine.positions()) == 1
        await engine.close()
    asyncio.run(run())


def test_target_exit_persists_and_pnl_survives_restart(config, store):
    from telegram_bridge.options_paper import OptionsPaperEngine

    async def run():
        feed = FakeFeed('up')
        bridge = FakeBridge(config, store, feed)
        engine = OptionsPaperEngine(bridge, feed=feed)
        engine.set_running(101, True)
        opened = await engine.once(symbols=('SPY',))
        assert opened['action'] == 'opened'
        with store.transaction() as db:
            db.execute("UPDATE option_positions SET entry_premium=0.10,last_premium=0.10 WHERE status='open' AND symbol='SPY'")
        feed.spot *= 1.15
        closed = await engine.once(symbols=('SPY',), force_new_slot=True)
        assert closed['action'] == 'closed'
        assert closed['position']['exit_reason'] in {'target', 'signal_reversal', 'expiry'}
        assert engine.pnl()['closed_trades'] == 1
        restarted = OptionsPaperEngine(bridge, feed=feed)
        assert restarted.pnl()['closed_trades'] == 1
        assert restarted.positions() == []
        await engine.close()
    asyncio.run(run())


def test_options_commands_are_private_owner_only_without_alpaca(config):
    from telegram_bridge.commands import parse_update
    from tests.test_commands import command_update

    for verb in ('optionson', 'optionsoff', 'optionsstatus', 'optionpositions', 'optiontrades', 'optionpnl'):
        parsed = parse_update(command_update(text=f'/{verb}'), config)
        assert parsed and parsed['verb'] == verb
        assert parse_update(command_update(chat_id=999, text=f'/{verb}'), config) is None


def test_optionsoff_is_urgent_in_command_queue(config, store):
    from tests.test_commands import command_update, enqueue

    enqueue(store, config, [command_update(uid=i, mid=i, text='/optionsstatus') for i in range(20, 25)])
    enqueue(store, config, [command_update(uid=25, mid=25, text='/optionsoff')])
    assert store.next_prediction_job()['request']['verb'] == 'optionsoff'


def test_one_cycle_processes_spy_and_spx_together(config, store):
    from telegram_bridge.options_paper import OptionsPaperEngine

    async def run():
        feed = FakeFeed('up')
        bridge = FakeBridge(config, store, feed)
        engine = OptionsPaperEngine(bridge, feed=feed)
        engine.set_running(101, True)
        result = await engine.once()
        assert result['action'] == 'cycle'
        assert {item.get('position', {}).get('symbol') or item.get('symbol') for item in result['results']} == {'SPY', 'SPX'}
        assert len(engine.positions()) == 2
        await engine.close()
    asyncio.run(run())
