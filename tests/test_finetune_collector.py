"""Tests for FineTuneCollector — store predictions, link outcomes, find predictions."""
import pytest
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from finetune_collector import FineTuneCollector


@pytest.fixture
def fc():
    path = os.path.join(tempfile.mkdtemp(), "finetune")
    c = FineTuneCollector(path)
    yield c
    import shutil; shutil.rmtree(path, ignore_errors=True)


class TestPredictions:
    def test_log_prediction(self, fc):
        idx = fc.log_prediction("TEST", "BUY", 100, 105, 95, 80, "RSI oversold")
        assert idx >= 0
        assert len(fc._examples) == 1

    def test_log_prediction_with_features(self, fc):
        idx = fc.log_prediction("TEST", "SELL", 200, 190, 210, 75, "Overbought",
                                features={"rsi_14": 78, "rvol": 2.5})
        assert idx >= 0
        assert "rsi_14" in fc._examples[0]["features"]

    def test_log_multiple_predictions(self, fc):
        fc.log_prediction("A", "BUY", 100, 105, 95, 80, "a")
        fc.log_prediction("B", "SELL", 200, 190, 210, 75, "b")
        assert len(fc._examples) == 2


class TestOutcome:
    def test_log_outcome(self, fc):
        idx = fc.log_prediction("TEST", "BUY", 100, 105, 95, 80, "RSI oversold")
        ok = fc.log_outcome(idx, 106, "correct", 100)
        assert ok is True
        assert fc._examples[0]["outcome"]["outcome"] == "correct"

    def test_log_outcome_invalid_index(self, fc):
        ok = fc.log_outcome(-1, 100, "correct", 100)
        assert ok is False

    def test_log_outcome_out_of_range(self, fc):
        ok = fc.log_outcome(999, 100, "correct", 100)
        assert ok is False


class TestFindPrediction:
    def test_find_exact_match(self, fc):
        fc.log_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        idx = fc.find_prediction("TEST", "BUY", 100)
        assert idx >= 0

    def test_find_no_match(self, fc):
        fc.log_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        idx = fc.find_prediction("OTHER", "SELL", 200)
        assert idx == -1

    def test_find_among_multiple(self, fc):
        fc.log_prediction("A", "BUY", 100, 105, 95, 80, "a")
        fc.log_prediction("B", "SELL", 200, 190, 210, 75, "b")
        fc.log_prediction("A", "BUY", 150, 160, 140, 90, "a2")
        idx = fc.find_prediction("B", "SELL", 200)
        assert idx == 1  # Second entry

    def test_find_and_outcome(self, fc):
        fc.log_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        idx = fc.find_prediction("TEST", "BUY", 100)
        ok = fc.log_outcome(idx, 106, "correct", 100)
        assert ok is True


class TestExport:
    def test_export_jsonl(self, fc):
        fc.log_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        fc.log_outcome(0, 106, "correct", 100)
        result = fc.export_jsonl("test_export.jsonl")
        assert result["count"] >= 1
        import os
        assert os.path.exists(result["path"])

    def test_export_jsonl_no_resolved(self, fc):
        fc.log_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        result = fc.export_jsonl("test_noout.jsonl")
        assert result["count"] == 0

    def test_export_chat_format(self, fc):
        fc.log_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        fc.log_outcome(0, 106, "correct", 100)
        result = fc.export_chat_format("test_chat.jsonl")
        assert result["count"] >= 1


class TestStats:
    def test_get_stats_empty(self, fc):
        stats = fc.get_stats()
        assert stats["total_predictions_logged"] == 0

    def test_get_stats_with_data(self, fc):
        fc.log_prediction("TEST", "BUY", 100, 105, 95, 80, "test")
        fc.log_outcome(0, 106, "correct", 100)
        stats = fc.get_stats()
        assert stats["total_predictions_logged"] == 1
        assert stats["resolved"] == 1
        assert stats["correct"] == 1
