"""Tests for MemoryManager — store/resolve predictions, knowledge search, expiry, correlations."""
import pytest
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from memory_manager import MemoryManager


@pytest.fixture
def mm():
    path = os.path.join(tempfile.mkdtemp(), "test_memory.duckdb")
    db = MemoryManager(path)
    yield db
    db.con.close()
    import shutil; shutil.rmtree(os.path.dirname(path), ignore_errors=True)


class TestSchema:
    def test_tables_exist(self, mm):
        tables = mm.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchdf()["table_name"].tolist()
        assert "knowledge" in tables
        assert "sessions" in tables
        assert "market_facts" in tables
        assert "prediction_history" in tables
        assert "ticker_correlations" in tables
        assert "knowledge_decay" in tables

    def test_prediction_columns(self, mm):
        cols = mm.con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='prediction_history'"
        ).fetchdf()["column_name"].tolist()
        assert "strategy_id" in cols
        assert "params_used" in cols
        assert "self_review" in cols


class TestPredictions:
    def test_store_prediction(self, mm):
        ok = mm.store_prediction("TEST", "BUY", 100, 105, 95, 80,
                                  "RSI oversold", {"rsi_14": 25}, strategy_id="smc_signal")
        assert ok is True

    def test_store_prediction_with_strategy(self, mm):
        mm.store_prediction("TEST", "BUY", 100, 105, 95, 80,
                            "test", strategy_id="rsi_mean_reversion",
                            params_used={"oversold": 30, "overbought": 70})
        rows = mm.con.execute("SELECT strategy_id, params_used FROM prediction_history").fetchall()
        assert rows[0][0] == "rsi_mean_reversion"
        assert "oversold" in rows[0][1]

    def test_resolve_prediction(self, mm):
        mm.store_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        pid = mm.con.execute("SELECT id FROM prediction_history").fetchone()[0]
        resolved = mm.resolve_prediction(pid, "correct", 100, self_review="Good call")
        assert resolved is not None
        assert resolved["ticker"] == "TEST"
        assert resolved["accuracy"] == 100
        # Check self_review was stored
        review = mm.con.execute("SELECT self_review FROM prediction_history WHERE id = ?", [pid]).fetchone()[0]
        assert review == "Good call"

    def test_get_predictions_for_review(self, mm):
        mm.store_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        preds = mm.get_predictions_for_review(days_back=30)
        assert len(preds) >= 1

    def test_get_unresolved(self, mm):
        mm.store_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        unresolved = mm.get_unresolved_predictions("TEST")
        assert len(unresolved) == 1

    def test_get_prediction_accuracy(self, mm):
        acc = mm.get_prediction_accuracy()
        assert acc[0] == 0  # No resolved predictions

    def test_resolve_unknown_id(self, mm):
        result = mm.resolve_prediction("nonexistent", "correct", 100)
        assert result is None


class TestExpiry:
    def test_expire_old_predictions(self, mm):
        import datetime
        mm.store_prediction("TEST", "BUY", 100, 105, 95, 80, "old")
        pid = mm.con.execute("SELECT id FROM prediction_history").fetchone()[0]
        # Manually set created_at to 60 days ago
        mm.con.execute(
            "UPDATE prediction_history SET created_at = created_at - INTERVAL 60 DAY WHERE id = ?",
            [pid]
        )
        expired = mm.expire_old_predictions(max_days=30)
        assert expired >= 1
        # Verify it's marked as expired
        row = mm.con.execute(
            "SELECT actual_outcome FROM prediction_history WHERE id = ?", [pid]
        ).fetchone()
        assert row[0] == "expired"


class TestKnowledge:
    def test_store_knowledge(self, mm):
        ok = mm.store_knowledge("Test Title", "Test content", category="analysis", source="test")
        assert ok is True

    def test_search_knowledge(self, mm):
        mm.store_knowledge("Test", "RSI divergence is a powerful signal", category="analysis")
        results = mm.search_knowledge("RSI divergence signal", top_k=5)
        assert len(results) >= 1
        assert any("RSI" in r["content"] for r in results)

    def test_search_empty(self, mm):
        results = mm.search_knowledge("nonexistent query")
        assert len(results) == 0


class TestFacts:
    def test_store_fact(self, mm):
        ok = mm.store_fact("TEST", "test_type", "Test fact content", confidence=0.8)
        assert ok is True

    def test_get_facts(self, mm):
        mm.store_fact("TEST", "test_type", "Test fact")
        facts = mm.get_facts(ticker="TEST")
        assert len(facts) >= 1

    def test_get_facts_by_type(self, mm):
        mm.store_fact("TEST", "mover_analysis", "Big mover")
        facts = mm.get_facts(fact_type="mover_analysis")
        assert len(facts) >= 1

    def test_fact_expiry(self, mm):
        mm.store_fact("TEST", "test_type", "Expiring fact", valid_days=0)
        import time; time.sleep(0.1)
        facts = mm.get_facts(ticker="TEST")
        assert len(facts) == 0  # expired immediately


class TestSessions:
    def test_store_session(self, mm):
        ok = mm.store_session("test_sid", "user query", "ai response")
        assert ok is True

    def test_search_sessions(self, mm):
        mm.store_session("test_sid", "hello", "world")
        sessions = mm.search_sessions(session_id="test_sid")
        assert len(sessions) >= 1

    def test_search_sessions_no_id(self, mm):
        mm.store_session("test_sid", "hello", "world")
        sessions = mm.search_sessions()
        assert len(sessions) >= 1


class TestStats:
    def test_get_stats(self, mm):
        stats = mm.get_stats()
        assert "total_knowledge" in stats
        assert "total_sessions" in stats
        assert "total_facts" in stats
        assert "total_predictions" in stats
        assert "accuracy" in stats or "error" not in stats

    def test_stats_after_adding_data(self, mm):
        mm.store_knowledge("A", "Content", category="analysis")
        mm.store_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        mm.store_fact("TEST", "type", "fact")
        stats = mm.get_stats()
        assert stats["total_knowledge"] >= 1
        assert stats["total_predictions"] >= 1
        assert stats["total_facts"] >= 1


class TestCorrelations:
    def test_compute_correlations_empty(self, mm):
        count = mm.compute_correlations()
        assert count == 0  # No securities table in memory DB

    def test_record_access(self, mm):
        mm.store_knowledge("Test", "Content", category="analysis")
        kid = mm.con.execute("SELECT id FROM knowledge LIMIT 1").fetchone()
        if kid:
            mm.record_knowledge_access(kid[0])
