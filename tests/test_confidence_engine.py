"""Tests for ConfidenceEngine signal scoring."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from confidence_engine import ConfidenceEngine

@pytest.fixture
def engine():
    return ConfidenceEngine()

class TestBuySignal:
    def test_strong_buy_oversold_rsi(self, engine):
        f = {"rsi_14": 25, "rsi_28": 30, "trend": "uptrend", "rvol": 2.0,
             "sma_20": 105, "sma_50": 100, "ob_high": 100, "ob_low": 90,
             "bb_lower": 95}
        r = engine.score_buy_signal(f, price=95)
        assert r["direction"] == "BUY"
        assert r["signal"] in ("strong", "moderate")
        assert r["confidence"] > 50
        assert len(r["confirmations"]) >= 3

    def test_neutral_no_signals(self, engine):
        f = {"rsi_14": 50, "sma_20": 100, "sma_50": 100, "rvol": 1.0,
             "ob_high": None, "ob_low": None, "bb_lower": None}
        r = engine.score_buy_signal(f, price=100)
        assert r["signal"] == "neutral"

    def test_overbought_conflict(self, engine):
        f = {"rsi_14": 75, "trend": "downtrend", "rvol": 0.3,
             "sma_20": 100, "sma_50": 105}
        r = engine.score_buy_signal(f)
        assert len(r["conflicts"]) >= 2

    def test_price_at_order_block(self, engine):
        f = {"rsi_14": 35, "ob_high": 102, "ob_low": 98, "bb_lower": 95}
        r = engine.score_buy_signal(f, price=100)
        assert "order block" in " ".join(r["confirmations"]).lower()
        assert r["confidence"] > 0

    def test_price_below_ob(self, engine):
        f = {"rsi_14": 30, "ob_high": 100, "ob_low": 95, "bb_lower": 90, "ob_mitigated": False}
        r = engine.score_buy_signal(f, price=93)
        assert "below order block" in " ".join(r["confirmations"]).lower()

    def test_ob_mitigated_skips(self, engine):
        """When OB is mitigated, it should not confirm."""
        f = {"rsi_14": 30, "ob_high": 100, "ob_low": 95, "bb_lower": 90, "ob_mitigated": True}
        r = engine.score_buy_signal(f, price=97)
        assert "order block" not in " ".join(r.get("confirmations", [])).lower()

    def test_ob_mitigated_sell_skips(self, engine):
        f = {"rsi_14": 72, "ob_high": 100, "ob_low": 95, "ob_mitigated": True}
        r = engine.score_sell_signal(f, price=105)
        # Should not crash with NameError
        assert r["signal"] != "error"

    def test_rsi_divergence_combined(self, engine):
        f = {"rsi_14": 25, "rsi_divergence": "bullish_regular,bullish_hidden"}
        r = engine.score_buy_signal(f)
        assert r["confidence"] > 0

class TestSellSignal:
    def test_strong_sell_overbought(self, engine):
        f = {"rsi_14": 78, "trend": "downtrend", "rvol": 2.5,
             "sma_20": 100, "sma_50": 105, "ob_high": 120, "ob_low": 110}
        r = engine.score_sell_signal(f, price=115)
        assert r["direction"] == "SELL"
        assert r["confidence"] > 50
        assert len(r["confirmations"]) >= 2

    def test_sell_above_ob(self, engine):
        f = {"rsi_14": 72, "ob_high": 100, "ob_low": 95, "bb_upper": 102}
        r = engine.score_sell_signal(f, price=105)
        assert any("Bollinger" in c for c in r["confirmations"])

class EdgeCases:
    def test_missing_features(self, engine):
        r = engine.score_buy_signal({})
        assert r["signal"] == "neutral"
        assert r["confidence"] == 0

    def test_none_values(self, engine):
        f = {"rsi_14": None, "rvol": None, "sma_20": 100, "sma_50": 100}
        r = engine.score_buy_signal(f)
        assert r["signal"] in ("neutral", "weak")

    def test_extreme_rsi_negative(self, engine):
        f = {"rsi_14": -5, "sma_20": 100, "sma_50": 100}
        r = engine.score_buy_signal(f)
        assert len(r["confirmations"]) >= 1 or r["signal"] != "error"
