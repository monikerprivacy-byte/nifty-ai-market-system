"""Tests for OrderFlowAnalyzer — depth, delta trends, large trades."""
import pytest
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from order_flow import OrderFlowAnalyzer


@pytest.fixture
def of():
    path = os.path.join(tempfile.mkdtemp(), "test_of.duckdb")
    db = OrderFlowAnalyzer.__new__(OrderFlowAnalyzer)
    import duckdb
    db.__dict__["db_path"] = path
    db.__dict__["con"] = duckdb.connect(path)
    db.__dict__["con"].execute("""
        CREATE TABLE IF NOT EXISTS securities (
            security_id VARCHAR PRIMARY KEY, symbol VARCHAR, name VARCHAR,
            segment VARCHAR, instrument_type VARCHAR,
            lot_size INTEGER DEFAULT 1, is_index BOOLEAN DEFAULT FALSE
        )
    """)
    db.__dict__["con"].execute("INSERT OR IGNORE INTO securities VALUES ('13', 'NIFTY', 'Nifty 50', 'NSE', 'INDEX', 1, TRUE)")
    db.__dict__["con"].execute("""
        CREATE TABLE IF NOT EXISTS market_depth (
            id INTEGER PRIMARY KEY, security_id VARCHAR, symbol VARCHAR,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, ltp FLOAT,
            bid_price1 FLOAT, bid_qty1 INTEGER, ask_price1 FLOAT, ask_qty1 INTEGER,
            total_bid_qty INTEGER, total_ask_qty INTEGER,
            imbalance FLOAT, spread FLOAT, spread_pct FLOAT
        )
    """)
    db.__dict__["con"].execute("""
        CREATE TABLE IF NOT EXISTS order_flow_signals (
            id INTEGER PRIMARY KEY, symbol VARCHAR,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cumulative_delta FLOAT, delta_5bar FLOAT,
            avg_imbalance FLOAT, large_trade_detected BOOLEAN,
            large_trade_size FLOAT, signal VARCHAR
        )
    """)
    yield db
    db.con.close()
    import shutil; shutil.rmtree(os.path.dirname(path), ignore_errors=True)


class TestSchema:
    def test_tables_exist(self, of):
        tables = of.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchdf()["table_name"].tolist()
        assert "market_depth" in tables
        assert "order_flow_signals" in tables


class TestDepth:
    def test_insert_depth(self, of):
        of.con.execute("""
            INSERT INTO market_depth (id, security_id, symbol, ltp, total_bid_qty, total_ask_qty, imbalance)
            VALUES (1, '1', 'TEST', 100.0, 5000, 3000, 0.25)
        """)
        of.con.commit()
        rows = of.con.execute("SELECT * FROM market_depth").fetchall()
        assert len(rows) == 1

    def test_get_depth_summary(self, of):
        of.con.execute("""
            INSERT INTO market_depth (id, security_id, symbol, ltp, total_bid_qty, total_ask_qty, imbalance)
            VALUES (1, '1', 'TEST', 100.0, 5000, 3000, 0.25)
        """)
        of.con.commit()
        summary = of.get_depth_summary("TEST")
        assert len(summary) >= 1
        assert summary[0]["symbol"] == "TEST"
        assert summary[0]["bid_pressure"] == "buying"

    def test_get_depth_summary_missing(self, of):
        summary = of.get_depth_summary("ZZZZ")
        assert len(summary) == 0

    def test_get_depth_summary_all(self, of):
        of.con.execute("""
            INSERT INTO market_depth (id, security_id, symbol, ltp, total_bid_qty, total_ask_qty, imbalance)
            VALUES (1, '1', 'A', 100.0, 5000, 3000, 0.25)
        """)
        of.con.execute("""
            INSERT INTO market_depth (id, security_id, symbol, ltp, total_bid_qty, total_ask_qty, imbalance)
            VALUES (2, '2', 'B', 200.0, 1000, 5000, -0.66)
        """)
        of.con.commit()
        summary = of.get_depth_summary()
        assert len(summary) >= 2


class TestDelta:
    def test_compute_delta_insufficient_data(self, of):
        result = of.compute_delta_trend("TEST")
        assert result is None  # Not enough snapshots

    def test_insert_signal(self, of):
        of.con.execute("""
            INSERT INTO order_flow_signals (id, symbol, cumulative_delta, delta_5bar, avg_imbalance, signal)
            VALUES (1, 'TEST', 5000, 1200, 0.15, 'bullish')
        """)
        of.con.commit()
        signals = of.get_recent_signals("TEST")
        assert len(signals) >= 1
        assert signals[0]["signal"] == "bullish"

    def test_get_signals_empty(self, of):
        signals = of.get_recent_signals("TEST")
        assert len(signals) == 0

    def test_get_signals_all(self, of):
        of.con.execute("""
            INSERT INTO order_flow_signals (id, symbol, cumulative_delta, delta_5bar, avg_imbalance, signal)
            VALUES (1, 'TEST', 5000, 1200, 0.15, 'bullish')
        """)
        signals = of.get_recent_signals()
        assert len(signals) >= 1


class TestLookup:
    def test_lookup_id(self, of):
        of.con.execute("""
            INSERT INTO securities (security_id, symbol) VALUES ('99', 'TEST_STOCK')
        """)
        sid = of._lookup_id("TEST_STOCK")
        assert sid == "99"

    def test_lookup_nifty(self, of):
        sid = of._lookup_id("NIFTY")
        assert sid == "13"

    def test_lookup_banknifty(self, of):
        sid = of._lookup_id("BANKNIFTY")
        assert sid == "25"

    def test_lookup_unknown(self, of):
        sid = of._lookup_id("ZZZZ")
        assert sid is None
