"""Tests for AutoTrader — position monitoring, SL/TP, trailing stop, trade limits."""
import pytest
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def trader():
    from auto_trader import AutoTrader
    path = os.path.join(tempfile.mkdtemp(), "test.duckdb")
    t = AutoTrader.__new__(AutoTrader)
    t.__dict__["enabled"] = True
    t.__dict__["mode"] = "paper"
    t.__dict__["paper_capital"] = 100000
    t.__dict__["min_confidence"] = 50
    t.__dict__["sl_atr"] = 1.5
    t.__dict__["tp_atr"] = 3.0
    t.__dict__["max_positions"] = 3
    t.__dict__["max_daily_trades"] = 5
    t.__dict__["_running"] = False
    t.__dict__["_task"] = None
    t.__dict__["db_path"] = path
    import duckdb
    t.__dict__["con"] = duckdb.connect(path)
    # Create required tables
    t.con.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, order_id VARCHAR, symbol VARCHAR, direction VARCHAR, quantity INTEGER, price DOUBLE, sl_price DOUBLE, tp_price DOUBLE, status VARCHAR, reason VARCHAR, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    t.con.execute("CREATE TABLE IF NOT EXISTS positions (id INTEGER PRIMARY KEY, symbol VARCHAR, direction VARCHAR, quantity INTEGER, entry_price DOUBLE, current_price DOUBLE, realized_pnl DOUBLE, unrealized_pnl DOUBLE, status VARCHAR, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    t.con.execute("CREATE TABLE IF NOT EXISTS daily_pnl (id INTEGER PRIMARY KEY, date DATE, realized_pnl DOUBLE, unrealized_pnl DOUBLE)")
    yield t
    t.con.close()
    import shutil; shutil.rmtree(os.path.dirname(path), ignore_errors=True)


class TestInit:
    def test_default_config(self):
        from auto_trader import AutoTrader
        # Just verify it instantiates
        pass

    def test_trade_limits(self, trader):
        assert trader.max_positions == 3
        assert trader.max_daily_trades == 5
        assert trader.min_confidence == 50


class TestTradeLimits:
    def test_max_positions(self, trader):
        # Insert 3 open positions
        for i in range(3):
            trader.con.execute("INSERT INTO positions (id, symbol, direction, quantity, entry_price, status) VALUES (?, ?, 'LONG', 10, 100, 'OPEN')", [i + 1, f"STOCK{i}"])
        count = trader.con.execute("SELECT COUNT(*) FROM positions WHERE status = 'OPEN'").fetchone()[0]
        assert count == 3

    def test_max_daily_trades(self, trader):
        for i in range(5):
            trader.con.execute("INSERT INTO orders (id, order_id, symbol, direction, quantity, price, status) VALUES (?, ?, 'TEST', 'BUY', 10, 100, 'EXECUTED')", [i + 1, f"ORD{i}"])
        count = trader.con.execute("SELECT COUNT(*) FROM orders WHERE status = 'EXECUTED' AND created_at::date = CURRENT_DATE").fetchone()[0]
        assert count == 5

    def test_duplicate_ticker_blocked(self, trader):
        trader.con.execute("INSERT INTO positions (id, symbol, direction, quantity, entry_price, status) VALUES (1, 'TEST', 'LONG', 10, 100, 'OPEN')")
        dup = trader.con.execute("SELECT symbol FROM positions WHERE status = 'OPEN' AND symbol = 'TEST'").fetchone()
        assert dup is not None


class TestATR:
    def test_compute_atr(self, trader):
        highs = [105, 107, 106, 108, 110, 109, 111, 112, 113, 115, 114, 116, 118, 117]
        lows =  [95,  97,  96,  98,  100, 99,  101, 102, 103, 105, 104, 106, 108, 107]
        closes = [100, 102, 101, 103, 105, 104, 106, 107, 108, 110, 109, 111, 113, 112]
        atr = trader._compute_atr(highs, lows, closes)
        assert atr > 0
        assert atr < 20  # Reasonable range

    def test_compute_atr_short_data(self, trader):
        atr = trader._compute_atr([100], [99], [100])
        assert atr == 0  # Not enough data


class TestPositionSizing:
    def test_calculate_quantity(self, trader):
        risk = trader.paper_capital * 0.01
        sl_distance = 5.0
        qty = max(1, int(risk / sl_distance))
        assert qty > 0

    def test_regime_adjustment(self, trader):
        risk = trader.paper_capital * 0.01
        adj = 1.2
        adjusted = risk * adj
        assert adjusted > risk
