"""Tests for AutoLearner — state management, market hours check, param tracking, parameter adaptation."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from datetime import datetime, time


class TestMarketHours:
    def test_is_market_hours_weekday(self):
        from auto_learner import AutoLearner
        # 10 AM on Monday
        d = datetime(2026, 7, 6, 10, 0, 0)  # Monday
        assert AutoLearner._is_market_hours(d) is True

    def test_is_market_hours_before_open(self):
        from auto_learner import AutoLearner
        d = datetime(2026, 7, 6, 8, 0, 0)
        assert AutoLearner._is_market_hours(d) is False

    def test_is_market_hours_after_close(self):
        from auto_learner import AutoLearner
        d = datetime(2026, 7, 6, 16, 0, 0)
        assert AutoLearner._is_market_hours(d) is False

    def test_is_market_hours_weekend(self):
        from auto_learner import AutoLearner
        d = datetime(2026, 7, 5, 10, 0, 0)  # Sunday
        assert AutoLearner._is_market_hours(d) is False

    def test_is_market_hours_saturday(self):
        from auto_learner import AutoLearner
        d = datetime(2026, 7, 4, 10, 0, 0)  # Saturday
        assert AutoLearner._is_market_hours(d) is False


class TestTrackStrategyResult:
    def test_track_strategy_result(self):
        """Test using the param DB that auto_learner creates."""
        from auto_learner import AutoLearner
        import tempfile, duckdb
        path = os.path.join(tempfile.mkdtemp(), "test_al.duckdb")
        con = duckdb.connect(path)
        con.execute("""
            CREATE TABLE strategy_params (
                id INTEGER PRIMARY KEY,
                date DATE NOT NULL DEFAULT CURRENT_DATE,
                strategy_id VARCHAR(50),
                ticker VARCHAR(20),
                params_json VARCHAR(500),
                accuracy DOUBLE DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                win_rate DOUBLE DEFAULT 0,
                sharpe DOUBLE DEFAULT 0,
                profit_factor DOUBLE DEFAULT 0,
                max_dd DOUBLE DEFAULT 0,
                trades_reviewed INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_sp_strategy ON strategy_params(strategy_id, ticker)
        """)
        # Mock the _param_db property
        al = AutoLearner.__new__(AutoLearner)
        al.__dict__["_param_db_path"] = path
        al.__dict__["_param_con"] = con

        al._track_strategy_result(
            strategy_id="smc_signal", ticker="TEST",
            params={"min_confidence": 50},
            accuracy=80.0, total_trades=1, win_rate=100.0,
        )
        rows = con.execute("SELECT * FROM strategy_params").fetchall()
        assert len(rows) == 1
        assert rows[0][4] == '{"min_confidence": 50}'  # params_json
        assert rows[0][5] == 80.0  # accuracy

    def test_track_strategy_result_ewma(self):
        """Verify exponential weighted average works correctly (accuracy converges)."""
        from auto_learner import AutoLearner
        import tempfile, duckdb
        path = os.path.join(tempfile.mkdtemp(), "test_al2.duckdb")
        con = duckdb.connect(path)
        con.execute("""
            CREATE TABLE strategy_params (
                id INTEGER PRIMARY KEY,
                date DATE NOT NULL DEFAULT CURRENT_DATE,
                strategy_id VARCHAR(50),
                ticker VARCHAR(20),
                params_json VARCHAR(500),
                accuracy DOUBLE DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                win_rate DOUBLE DEFAULT 0,
                sharpe DOUBLE DEFAULT 0,
                profit_factor DOUBLE DEFAULT 0,
                max_dd DOUBLE DEFAULT 0,
                trades_reviewed INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        al = AutoLearner.__new__(AutoLearner)
        al.__dict__["_param_db_path"] = path
        al.__dict__["_param_con"] = con

        # First call: accuracy = 100
        al._track_strategy_result("test", "TEST", {}, 100.0, 1, 100.0)
        row = con.execute("SELECT accuracy FROM strategy_params").fetchone()
        assert row[0] == 100.0

        # Second call: accuracy = 50, alpha=0.3
        # new = 0.3 * 50 + 0.7 * 100 = 85
        al._track_strategy_result("test", "TEST", {}, 50.0, 1, 0.0)
        row = con.execute("SELECT accuracy FROM strategy_params").fetchone()
        assert row[0] == pytest.approx(85.0, abs=0.01)

    def test_select_best_params_exploit(self):
        """With no history, should return default (explore)."""
        from auto_learner import AutoLearner
        al = AutoLearner.__new__(AutoLearner)
        al.__dict__["_param_db_path"] = None
        al.__dict__["_param_con"] = None
        params = al._select_best_params("test", "TEST", {"oversold": 30, "overbought": 70})
        assert params == {"oversold": 30, "overbought": 70}


class TestLLM:
    def test_llm_suggest_no_data(self):
        """With < 3 results, should return empty dict."""
        from auto_learner import AutoLearner
        al = AutoLearner.__new__(AutoLearner)
        result = al._llm_suggest_param_adjustment("rsi_mean_reversion", [])
        assert result == {}

    def test_llm_suggest_not_enough(self):
        from auto_learner import AutoLearner
        al = AutoLearner.__new__(AutoLearner)
        result = al._llm_suggest_param_adjustment("rsi_mean_reversion",
            [{"params": "{}", "accuracy": 80, "trades": 5}])
        assert result == {}  # Only 1 result, needs 3+


class TestStateManagement:
    def test_initial_state(self):
        from auto_learner import AutoLearner
        al = AutoLearner.__new__(AutoLearner)
        al.__dict__["_current_state"] = "idle"
        assert al.state == "idle"

    def test_publish_state(self):
        from auto_learner import AutoLearner
        al = AutoLearner.__new__(AutoLearner)
        al.__dict__["_current_state"] = "sleeping"
        # Should not crash
        al._publish_state()
