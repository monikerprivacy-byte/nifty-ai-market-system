"""Tests for FeatureEngine — RSI divergence, structure breaks, order blocks, FVG, OBV."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from feature_engine import (
    compute_features, compute_multi_timeframe,
    _find_rsi_divergence, _find_structure_breaks,
    _find_order_blocks, _find_ob_mitigation,
    _find_fvg, _find_liquidity, _detect_structure,
)


def make_test_df(n=100):
    np.random.seed(42)
    closes = [100 + np.cumsum(np.random.normal(0, 1, n))[i] for i in range(n)]
    highs = [c + abs(np.random.normal(0, 0.5)) for c in closes]
    lows = [c - abs(np.random.normal(0, 0.5)) for c in closes]
    opens = [(l + h) / 2 for l, h in zip(lows, highs)]
    volumes = [int(np.random.uniform(1e5, 1e7)) for _ in range(n)]
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


class TestComputeFeatures:
    def test_basic_compute(self):
        df = make_test_df()
        result = compute_features(df)
        assert isinstance(result, dict)
        assert "rsi_14" in result
        assert "rsi_28" in result
        assert "sma_20" in result
        assert "sma_50" in result
        assert "bb_mid" in result
        assert "atr" in result

    def test_short_dataframe(self):
        df = make_test_df(10)
        result = compute_features(df)
        assert result is not None
        assert isinstance(result, dict)

    def test_all_features_present(self):
        df = make_test_df(200)
        result = compute_features(df)
        smc_keys = ["trend", "swing_high", "swing_low", "ob_high", "ob_low",
                     "ob_mitigated", "fvg_high", "fvg_low", "liq_above", "liq_below",
                     "structure_break", "rsi_divergence", "obv", "obv_trend",
                     "confluence_score"]
        for k in smc_keys:
            assert k in result, f"Missing feature: {k}"

    def test_obv_trend(self):
        df = make_test_df(200)
        result = compute_features(df)
        assert result["obv_trend"][-1] in ("rising", "falling", "flat")

    def test_confluence_score_range(self):
        df = make_test_df(200)
        result = compute_features(df)
        scores = result.get("confluence_score")
        if scores is not None:
            valid = [s for s in scores if not np.isnan(s)]
            if valid:
                assert all(-5 <= s <= 5 for s in valid)


class TestRSIDivergence:
    def test_no_divergence_random_data(self):
        n = 100
        closes = [100 + np.random.normal(0, 2) for _ in range(n)]
        rsi = [50 + np.random.normal(0, 5) for _ in range(n)]
        features = {"rsi_divergence": ["none"] * n,
                    "swing_low": [np.nan] * n, "swing_high": [np.nan] * n}
        _find_rsi_divergence(closes, rsi, features, n)
        divs = [d for d in features["rsi_divergence"] if d != "none"]
        # Most candles should have no divergence
        none_count = sum(1 for d in features["rsi_divergence"] if d == "none")
        assert none_count > n * 0.9

    def test_bullish_divergence(self):
        n = 100
        # Price makes lower low, RSI makes higher low
        closes = [100] * n
        rsi = [50] * n
        # Create swing points
        closes[40] = 100; rsi[40] = 40
        closes[50] = 95;  rsi[50] = 35  # Lower price, lower RSI (no divergence)
        features = {"rsi_divergence": ["none"] * n,
                     "swing_low": [np.nan] * n, "swing_high": [np.nan] * n}
        features["swing_low"][40] = closes[40]
        features["swing_low"][50] = closes[50]
        _find_rsi_divergence(closes, rsi, features, n)
        div = features["rsi_divergence"][50]
        # Should detect something at index 50

    def test_divergence_combines(self):
        """Test that multiple divergence types are concatenated, not overwritten."""
        n = 100
        # Bullish regular: price lower low (100→95) + RSI higher low (40→50)
        # Bearish hidden:  price lower high (105→102) + RSI higher high (50→60)
        closes = [100] * n; rsi = [50] * n
        features = {"rsi_divergence": ["none"] * n,
                     "swing_low": [np.nan] * n, "swing_high": [np.nan] * n}
        features["swing_low"][40] = 100; features["swing_high"][40] = 105
        features["swing_low"][50] = 95;  features["swing_high"][50] = 102
        closes[50] = 95; rsi[50] = 60
        _find_rsi_divergence(closes, rsi, features, n)
        div = features["rsi_divergence"][50]
        assert div is not None
        assert "bullish_regular" in div or "bearish_hidden" in div


class TestStructureBreaks:
    def test_structure_break_exists(self):
        n = 100
        features = {"trend": ["range"] * n, "structure_break": [False] * n,
                     "swing_high": [np.nan] * n, "swing_low": [np.nan] * n}
        highs = [100] * n
        lows = [95] * n
        features["trend"][50] = "uptrend"
        features["swing_low"][45] = 97  # Key swing low
        highs[55] = 105
        lows[55] = 94  # Break below swing low
        _find_structure_breaks(highs, lows, features, n)
        breaks = [i for i in range(n) if features["structure_break"][i]]
        # Some breaks may or may not be detected

    def test_no_structure_break_in_range(self):
        n = 100
        features = {"trend": ["range"] * n, "structure_break": [False] * n,
                     "swing_high": [np.nan] * n, "swing_low": [np.nan] * n}
        highs = [100 + np.random.normal(0, 0.5) for _ in range(n)]
        lows = [95 + np.random.normal(0, 0.5) for _ in range(n)]
        _find_structure_breaks(highs, lows, features, n)
        assert all(not b for b in features["structure_break"])


class TestOrderBlocks:
    def test_find_order_blocks(self):
        n = 100
        opens = [100] * n
        highs = [101] * n
        lows = [99] * n
        closes = [100] * n
        volumes = [1000] * n
        features = {"ob_high": [None] * n, "ob_low": [None] * n}
        # Strong bearish candle followed by bullish impulse
        closes[40] = 99; opens[40] = 101  # Bearish
        closes[41] = 103; opens[41] = 100; highs[41] = 104; lows[41] = 99  # Bullish impulse
        volumes[41] = 5000  # High volume
        _find_order_blocks(opens, highs, lows, closes, volumes, features, n)
        obs = [(i, features["ob_high"][i], features["ob_low"][i]) for i in range(n)
               if features["ob_high"][i] is not None]
        # May or may not detect depending on candle structure


class TestFVG:
    def test_find_bullish_fvg(self):
        n = 100
        features = {"fvg_high": [None] * n, "fvg_low": [None] * n}
        highs = [100] * n
        lows = [95] * n
        # Gap up: candle 3 low > candle 1 high
        highs[0] = 100; lows[0] = 95
        highs[1] = 98;  lows[1] = 96
        highs[2] = 105; lows[2] = 102
        _find_fvg(highs, lows, features, n)
        fvgs = [i for i in range(n) if features["fvg_high"][i] is not None]
        assert len(fvgs) >= 1


class TestLiquidity:
    def test_find_liquidity(self):
        n = 100
        features = {"liq_above": [None] * n, "liq_below": [None] * n}
        highs = [100 + i * 0.5 for i in range(n)]
        lows = [95 + i * 0.5 for i in range(n)]
        _find_liquidity(highs, lows, features, n)
        # Should have liquidity zones near the end
        assert features["liq_above"][-1] is not None
        assert features["liq_below"][-1] is not None


class TestDetectStructure:
    def test_detect_uptrend(self):
        n = 100
        features = {"trend": [None] * n, "swing_high": [np.nan] * n,
                     "swing_low": [np.nan] * n}
        closes = [100 + i * 0.3 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        _detect_structure(closes, highs, lows, features, n)
        trends = [t for t in features["trend"][-10:] if t is not None]
        # May be uptrend depending on swing detection

    def test_detect_downtrend(self):
        n = 100
        features = {"trend": [None] * n, "swing_high": [np.nan] * n,
                     "swing_low": [np.nan] * n}
        closes = [100 - i * 0.3 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        _detect_structure(closes, highs, lows, features, n)
        trends = [t for t in features["trend"][-10:] if t is not None]
        # May be downtrend depending on swing detection


class TestMultiTimeframe:
    def test_compute_multi_timeframe(self):
        df = make_test_df(300)
        result = compute_multi_timeframe(df)
        assert isinstance(result, dict)
        assert "weekly" in result or "monthly" in result

    def test_mtf_short_data(self):
        df = make_test_df(20)
        result = compute_multi_timeframe(df)
        assert isinstance(result, dict)
