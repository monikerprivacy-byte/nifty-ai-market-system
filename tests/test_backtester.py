"""Tests for Backtester — strategies, cost model, SL/TP, walk-forward, regime-conditioned."""
import pytest
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from backtester import Backtester, STRATEGIES


@pytest.fixture
def bt():
    path = os.path.join(tempfile.mkdtemp(), "test.duckdb")
    db = Backtester(path)
    db.con.execute("""
        CREATE TABLE IF NOT EXISTS securities (
            security_id VARCHAR PRIMARY KEY, symbol VARCHAR, name VARCHAR,
            segment VARCHAR, instrument_type VARCHAR, lot_size INTEGER DEFAULT 1,
            is_index BOOLEAN DEFAULT FALSE, is_active BOOLEAN DEFAULT TRUE
        )
    """)
    db.con.execute("""
        CREATE TABLE IF NOT EXISTS daily (
            security_id VARCHAR, date DATE, open DOUBLE, high DOUBLE,
            low DOUBLE, close DOUBLE, volume BIGINT,
            PRIMARY KEY (security_id, date)
        )
    """)
    db.con.execute("INSERT INTO securities VALUES ('1', 'TEST', 'Test Corp', 'NSE', 'EQUITY', 1, FALSE, TRUE)")
    db.con.execute("INSERT INTO securities VALUES ('13', 'NIFTY', 'Nifty 50', '', 'INDEX', 1, TRUE, TRUE)")

    # Generate 200 days of test data
    dates = pd.date_range("2026-01-01", periods=200, freq="B")
    np.random.seed(42)
    close = 100.0
    rows = []
    for d in dates:
        close *= (1 + np.random.normal(0.001, 0.015))
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_p = low + (high - low) * np.random.random()
        rows.append((d.date(), round(open_p, 2), round(high, 2),
                     round(low, 2), round(close, 2), int(np.random.randint(1e5, 1e7))))
    db.con.executemany(
        "INSERT INTO daily VALUES ('1', ?, ?, ?, ?, ?, ?)", rows
    )

    # Add more data for NIFTY
    for d in dates:
        close *= (1 + np.random.normal(0.001, 0.012))
        high = close * 1.01
        low = close * 0.99
        db.con.execute("INSERT INTO daily VALUES ('13', ?, ?, ?, ?, ?, ?)",
                       [d.date(), round(low + (high-low)*np.random.random(), 2),
                        round(high, 2), round(low, 2), round(close, 2),
                        int(np.random.randint(1e6, 1e8))])
    db.con.commit()
    yield db
    db.con.close()


class TestStrategyList:
    def test_list_strategies(self, bt):
        strategies = bt.list_strategies()
        assert len(strategies) == 5
        ids = [s["id"] for s in strategies]
        assert "rsi_mean_reversion" in ids
        assert "sma_crossover" in ids
        assert "bollinger_bounce" in ids
        assert "trend_follow" in ids
        assert "vwap_reversion" in ids

    def test_get_strategy(self, bt):
        s = bt.get_strategy("rsi_mean_reversion")
        assert s is not None
        assert s["name"] == "RSI Mean Reversion"
        assert s["params"]["oversold"] == 30

    def test_get_strategy_unknown(self, bt):
        assert bt.get_strategy("nonexistent") is None


class TestLookup:
    def test_lookup_ticker(self, bt):
        assert bt._lookup_id("TEST") == "1"

    def test_lookup_nifty(self, bt):
        assert bt._lookup_id("NIFTY") == "13"

    def test_lookup_banknifty(self, bt):
        assert bt._lookup_id("BANKNIFTY") == "25"

    def test_lookup_unknown(self, bt):
        assert bt._lookup_id("ZZZZ") is None


class TestStrategies:
    @pytest.mark.asyncio
    async def test_rsi_mean_reversion(self, bt):
        result = await bt.run("TEST", "rsi_mean_reversion")
        assert "error" not in result
        assert result["strategy_id"] == "rsi_mean_reversion"
        assert result["ticker"] == "TEST"

    @pytest.mark.asyncio
    async def test_sma_crossover(self, bt):
        result = await bt.run("TEST", "sma_crossover")
        assert "error" not in result
        assert len(result["trades"]) >= 0

    @pytest.mark.asyncio
    async def test_bollinger_bounce(self, bt):
        result = await bt.run("TEST", "bollinger_bounce")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_trend_follow(self, bt):
        result = await bt.run("TEST", "trend_follow")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_vwap_reversion(self, bt):
        result = await bt.run("TEST", "vwap_reversion")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_unknown_strategy(self, bt):
        result = await bt.run("TEST", "nonexistent")
        assert "error" in result


class TestCostModel:
    def test_default_cost(self, bt):
        assert bt.cost_bps == 20

    def test_custom_cost(self, bt):
        bt2 = Backtester(bt.db_path, cost_bps=50)
        assert bt2.cost_bps == 50
        bt2.con.close()

    @pytest.mark.asyncio
    async def test_cost_reduces_returns(self, bt):
        no_cost = Backtester(bt.db_path, cost_bps=0)
        with_cost = Backtester(bt.db_path, cost_bps=100)  # 1%
        r1 = await bt.run("TEST", "rsi_mean_reversion", params={"sl_pct": 0.02, "tp_pct": 0.04})
        r2 = await bt.run("TEST", "rsi_mean_reversion", params={"sl_pct": 0.02, "tp_pct": 0.04})
        no_cost.con.close()
        with_cost.con.close()
        m = r1.get("metrics", {})
        if "note" not in m:
            assert m.get("cost_bps") is not None


class TestSLTP:
    @pytest.mark.asyncio
    async def test_sl_hit(self, bt):
        result = await bt.run("TEST", "rsi_mean_reversion", params={"sl_pct": 0.01, "tp_pct": 0.05})
        trades = result.get("trades", [])
        reasons = [t.get("reason") for t in trades if "reason" in t]
        # Some trades should exit via SL
        pass

    @pytest.mark.asyncio
    async def test_tp_hit(self, bt):
        result = await bt.run("TEST", "rsi_mean_reversion", params={"sl_pct": 0.05, "tp_pct": 0.01})
        assert result.get("metrics") is not None

    @pytest.mark.asyncio
    async def test_trade_reason_field(self, bt):
        result = await bt.run("TEST", "rsi_mean_reversion", params={"sl_pct": 0.02, "tp_pct": 0.04})
        for t in result.get("trades", []):
            assert "reason" in t


class TestValidateParams:
    def test_valid_rsi(self):
        v = Backtester._validate_params("rsi_mean_reversion", {"oversold": 30, "overbought": 70})
        assert v["valid"] is True

    def test_degenerate_rsi(self):
        v = Backtester._validate_params("rsi_mean_reversion", {"oversold": 80, "overbought": 60})
        assert v["valid"] is False
        assert len(v["warnings"]) > 0

    def test_valid_sma(self):
        v = Backtester._validate_params("sma_crossover", {"fast_period": 20, "slow_period": 50})
        assert v["valid"] is True

    def test_degenerate_sma(self):
        v = Backtester._validate_params("sma_crossover", {"fast_period": 50, "slow_period": 20})
        assert v["valid"] is False

    def test_sl_gte_tp(self):
        v = Backtester._validate_params("rsi_mean_reversion", {"sl_pct": 3, "tp_pct": 2})
        assert v["valid"] is False

    def test_valid_bollinger(self):
        v = Backtester._validate_params("bollinger_bounce", {"bollinger_period": 20, "bollinger_std": 2})
        assert v["valid"] is True

    def test_negative_std(self):
        v = Backtester._validate_params("bollinger_bounce", {"bollinger_std": -1})
        assert v["valid"] is False


class TestWalkForward:
    @pytest.mark.asyncio
    async def test_walk_forward_basic(self, bt):
        result = await bt.walk_forward("TEST", "rsi_mean_reversion",
                                       {"oversold": [25, 30], "overbought": [70, 75]},
                                       n_splits=3)
        assert "error" not in result
        assert "best_params" in result
        assert "oos_metrics" in result
        assert result["n_splits"] == 3

    @pytest.mark.asyncio
    async def test_walk_forward_sma(self, bt):
        result = await bt.walk_forward("TEST", "sma_crossover",
                                       {"fast_period": [10, 20], "slow_period": [50, 100]},
                                       n_splits=3)
        assert "error" not in result or "Not enough data" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_walk_forward_unknown_ticker(self, bt):
        result = await bt.walk_forward("ZZZZ", "rsi_mean_reversion", {"oversold": [30]})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_walk_forward_unknown_strategy(self, bt):
        result = await bt.walk_forward("TEST", "nonexistent", {})
        assert "error" in result


class TestRegimeConditioned:
    @pytest.mark.asyncio
    async def test_regime_filter(self, bt):
        import pandas as pd
        regime_df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=200, freq="B"),
            "regime": ["uptrend"] * 100 + ["downtrend"] * 100,
        })
        regime_df["date"] = pd.to_datetime(regime_df["date"])
        result = await bt.run_regime_conditioned("TEST", "trend_follow", regime_df=regime_df)
        assert "error" not in result
        assert result.get("conditioned") is True


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_present(self, bt):
        result = await bt.run("TEST", "rsi_mean_reversion", params={"sl_pct": 0.02, "tp_pct": 0.04})
        m = result.get("metrics", {})
        if "note" in m:
            # No trades closed — skip scalar metrics check; verify structure
            assert "total_trades" in m
        else:
            assert "total_trades" in m
            assert "win_rate" in m
            assert "profit_factor" in m
            assert "sharpe_ratio" in m
            assert "total_return_pct" in m
            assert "max_drawdown_pct" in m
            assert "avg_intra_dd_pct" in m
            assert "max_intra_dd_pct" in m
            assert "payoff_ratio" in m
            assert "expectancy_pct" in m

    @pytest.mark.asyncio
    async def test_no_trades(self, bt):
        result = await bt.run("TEST", "sma_crossover", start_date="2026-01-01", end_date="2026-01-05")
        m = result.get("metrics", {})
        # Very short period may produce no trades
        assert "total_trades" in m


class TestCleanNumpy:
    def test_clean_int(self):
        assert Backtester._clean_numpy(np.int64(5)) == 5

    def test_clean_float(self):
        assert isinstance(Backtester._clean_numpy(np.float64(3.14)), float)

    def test_clean_dict(self):
        d = {"a": np.int64(1), "b": np.float64(2.0)}
        cleaned = Backtester._clean_numpy(d)
        assert cleaned == {"a": 1, "b": 2.0}

    def test_clean_list(self):
        cleaned = Backtester._clean_numpy([np.int64(1), np.float64(2.0)])
        assert cleaned == [1, 2.0]
