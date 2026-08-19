"""Tests for MarketRegime — ADX, classification, get_regime_df."""
import pytest
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np


@pytest.fixture
def mr():
    from market_regime import MarketRegime
    path = os.path.join(tempfile.mkdtemp(), "test_regime.duckdb")
    db = MarketRegime.__new__(MarketRegime)
    import duckdb
    db.__dict__["db_path"] = path
    db.__dict__["con"] = duckdb.connect(path)
    db.__dict__["con"].execute("""
        CREATE TABLE IF NOT EXISTS market_regime (
            date DATE PRIMARY KEY, regime VARCHAR, adx DOUBLE,
            bb_width_pct DOUBLE, atr_ratio DOUBLE, rsi_14 DOUBLE,
            vwap_position VARCHAR, sma_position VARCHAR, details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    yield db
    db.con.close()
    import shutil; shutil.rmtree(os.path.dirname(path), ignore_errors=True)


class TestADX:
    def test_compute_adx(self):
        from market_regime import MarketRegime
        n = 50
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.normal(0, 1, n))
        highs = closes + abs(np.random.normal(0, 0.5, n))
        lows = closes - abs(np.random.normal(0, 0.5, n))
        adx = MarketRegime._compute_adx(highs.tolist(), lows.tolist(), closes.tolist(), 14, n)
        assert len(adx) == 50
        valid = [a for a in adx[14:] if a is not None and not np.isnan(float(a))]
        assert len(valid) > 0

    def test_adx_zero_period(self):
        from market_regime import MarketRegime
        adx = MarketRegime._compute_adx([100], [99], [100], 14, 1)
        assert len(adx) == 1

    def test_adx_trending(self):
        from market_regime import MarketRegime
        # Strong uptrend
        n = 100
        closes = [100 + i * 0.5 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        adx = MarketRegime._compute_adx(highs, lows, closes, 14, n)
        # Last values should be > 20
        last_adx = [a for a in adx[-10:] if a is not None and not np.isnan(a)]
        assert len(last_adx) > 0
        assert np.mean(last_adx) > 20


class TestBBWidth:
    def test_compute_bb_width(self):
        from market_regime import MarketRegime
        # Tight range
        closes = [100 + np.random.normal(0, 0.3) for _ in range(50)]
        bb = MarketRegime._compute_bb_width(closes, 20, 50)
        valid = [b for b in bb if b is not None and not np.isnan(b)]
        assert len(valid) > 0

    def test_bb_width_wide(self):
        from market_regime import MarketRegime
        # Wide range
        closes = [100 + i for i in range(50)]
        bb = MarketRegime._compute_bb_width(closes, 20, 50)
        valid = [b for b in bb[-5:] if b is not None and not np.isnan(b)]
        assert len(valid) > 0


class TestRSI:
    def test_rsi_compute(self):
        from market_regime import MarketRegime
        n = 30
        closes = [100 + np.random.normal(0, 2) for _ in range(n)]
        rsi = MarketRegime._compute_rsi(closes, 14, n)
        valid = [r for r in rsi if r is not None and not np.isnan(r)]
        assert len(valid) > 0
        assert all(0 <= r <= 100 for r in valid if r is not None)

    def test_rsi_oversold(self):
        from market_regime import MarketRegime
        # Downward trend
        closes = [100 - i * 0.5 for i in range(30)]
        rsi = MarketRegime._compute_rsi(closes, 14, 30)
        final = [r for r in rsi if r is not None and not np.isnan(r)]
        if final:
            assert final[-1] < 50


class TestSMA:
    def test_sma_compute(self):
        from market_regime import MarketRegime
        sma = MarketRegime._sma([1, 2, 3, 4, 5], 3, 5)
        assert sma[2] == 2.0
        assert sma[3] == 3.0
        assert sma[4] == 4.0
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])


class TestVWAP:
    def test_vwap_compute(self):
        from market_regime import MarketRegime
        vwap = MarketRegime._compute_vwap([100, 101, 102], [1000, 2000, 3000], 3)
        expected = (100*1000 + 101*2000 + 102*3000) / (1000 + 2000 + 3000)
        assert abs(vwap[-1] - expected) < 0.01

    def test_vwap_zero_volume(self):
        from market_regime import MarketRegime
        vwap = MarketRegime._compute_vwap([100], [0], 1)
        assert np.isnan(vwap[0])


class TestCRUD:
    def test_get_current_no_data(self, mr):
        r = mr.get_current()
        assert r is None

    def test_insert_and_get(self, mr):
        mr.con.execute("""
            INSERT INTO market_regime (date, regime, adx, bb_width_pct, atr_ratio, rsi_14, vwap_position, sma_position)
            VALUES (CURRENT_DATE, 'ranging', 18, 12.5, 0.8, 55, 'at', 'above')
        """)
        mr.con.commit()
        r = mr.get_current()
        assert r is not None
        assert r["regime"] == "ranging"

    def test_get_history(self, mr):
        import datetime
        for i in range(5):
            d = datetime.date.today() - datetime.timedelta(days=i)
            mr.con.execute("""
                INSERT INTO market_regime (date, regime, adx, bb_width_pct, atr_ratio, rsi_14, vwap_position, sma_position)
                VALUES (?, 'ranging', 18, 12, 0.8, 55, 'at', 'above')
            """, [d])
        mr.con.commit()
        history = mr.get_history(days=10)
        assert len(history) >= 5
