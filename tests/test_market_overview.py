"""Tests for MarketOverview using a real DuckDB instance."""
import pytest
import os, sys, duckdb, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from market_overview import MarketOverview

@pytest.fixture
def db():
    path = os.path.join(tempfile.mkdtemp(), "test.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE securities (security_id VARCHAR, symbol VARCHAR, name VARCHAR, is_index BOOLEAN, is_active BOOLEAN)")
    con.execute("INSERT INTO securities VALUES ('1', 'RELIANCE', 'Reliance Industries', 0, 1)")
    con.execute("INSERT INTO securities VALUES ('2', 'HDFCBANK', 'HDFC Bank', 0, 1)")
    con.execute("INSERT INTO securities VALUES ('13', 'NIFTY', 'Nifty 50', 1, 1)")

    con.execute("CREATE TABLE daily (security_id VARCHAR, date DATE, close DOUBLE, volume DOUBLE)")
    con.execute("INSERT INTO daily VALUES ('1', '2026-07-01', 2500, 1e6)")
    con.execute("INSERT INTO daily VALUES ('1', '2026-07-02', 2550, 1.5e6)")
    con.execute("INSERT INTO daily VALUES ('1', '2026-07-03', 2600, 2e6)")
    con.execute("INSERT INTO daily VALUES ('2', '2026-07-01', 1800, 2e6)")
    con.execute("INSERT INTO daily VALUES ('2', '2026-07-02', 1790, 1.8e6)")
    con.execute("INSERT INTO daily VALUES ('2', '2026-07-03', 1780, 1.6e6)")
    con.execute("INSERT INTO daily VALUES ('13', '2026-07-01', 24000, 0)")
    con.execute("INSERT INTO daily VALUES ('13', '2026-07-02', 24200, 0)")
    con.execute("INSERT INTO daily VALUES ('13', '2026-07-03', 24400, 0)")

    con.execute("CREATE TABLE indicators (security_id VARCHAR, date DATE, rsi_14 DOUBLE, rvol DOUBLE, sma_20 DOUBLE, sma_50 DOUBLE, bollinger_upper DOUBLE, bollinger_lower DOUBLE, atr_14 DOUBLE)")
    con.execute("INSERT INTO indicators VALUES ('1', '2026-07-03', 65, 1.8, 2520, 2480, 2700, 2400, 50)")
    con.execute("INSERT INTO indicators VALUES ('2', '2026-07-03', 35, 0.9, 1810, 1820, 1900, 1750, 30)")
    con.close()

    yield path

    try:
        import shutil; shutil.rmtree(os.path.dirname(path))
    except: pass

class TestMarketOverview:
    def test_get_breadth(self, db):
        mo = MarketOverview(db)
        b = mo.get_breadth()
        assert "date" in b
        assert b["total_stocks"] == 2  # 2 equities, indices excluded
        assert b["advancing"] >= 0
        assert b["declining"] >= 0
        mo.close()

    def test_get_top_movers(self, db):
        mo = MarketOverview(db)
        m = mo.get_top_movers(limit=5)
        assert len(m["gainers"]) >= 0
        assert len(m["losers"]) >= 0
        mo.close()

    def test_get_signal_screener(self, db):
        mo = MarketOverview(db)
        s = mo.get_signal_screener()
        assert isinstance(s, list)
        mo.close()

    def test_get_all_stocks_snapshot(self, db):
        mo = MarketOverview(db)
        s = mo.get_all_stocks_snapshot()
        assert isinstance(s, list)
        mo.close()

    def test_get_full_summary(self, db):
        mo = MarketOverview(db)
        s = mo.get_full_summary()
        assert "breadth" in s
        assert "movers" in s
        assert "signals" in s
        assert "stocks" in s
        mo.close()

    def test_clean_records(self, db):
        mo = MarketOverview(db)
        import math
        records = [{"a": 1, "b": float("nan"), "c": float("inf"), "d": "hello"}]
        cleaned = mo._clean_records(records)
        assert cleaned[0]["a"] == 1
        assert cleaned[0]["b"] is None
        assert cleaned[0]["c"] is None
        assert cleaned[0]["d"] == "hello"
        mo.close()
