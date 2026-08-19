"""Feature Engine — converts raw candles → structured features for AI consumption.
AI never gets raw OHLCV. Only gets computed features."""

import numpy as np
import pandas as pd
from config_manager import get_config

def compute_features(df):
    """Compute all features from daily OHLCV DataFrame. Returns dict of arrays."""
    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    opens = df["open"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    cfg = get_config()
    n = len(closes)

    features = {
        "rsi_14": _rsi(closes, 14, n),
        "rsi_28": _rsi(closes, 28, n),
        "obv": _obv(closes, volumes, n),
        "vwap": _vwap(closes, volumes, n),
        "sma_20": _sma(closes, 20, n),
        "sma_50": _sma(closes, 50, n),
        "sma_200": _sma(closes, 200, n),
        "ema_20": _ema(closes, 20, n),
        "ema_50": _ema(closes, 50, n),
        "bb_upper": np.full(n, np.nan),
        "bb_mid": np.full(n, np.nan),
        "bb_lower": np.full(n, np.nan),
        "volume_avg": _sma(volumes, 20, n),
        "rvol": np.full(n, np.nan),
        "atr": _atr(highs, lows, closes, 14, n),
        "trend": ["range"] * n,
        "swing_high": np.full(n, np.nan),
        "swing_low": np.full(n, np.nan),
        "ob_high": np.full(n, np.nan),
        "ob_low": np.full(n, np.nan),
        "ob_mitigated": [False] * n,
        "fvg_high": np.full(n, np.nan),
        "fvg_low": np.full(n, np.nan),
        "liq_above": np.full(n, np.nan),
        "liq_below": np.full(n, np.nan),
        "structure_break": [False] * n,
        "rsi_divergence": [""] * n,
        "confluence_score": np.full(n, np.nan),
    }

    # Bollinger
    mid = pd.Series(closes).rolling(window=20).mean().values
    std = pd.Series(closes).rolling(window=20).std().values
    features["bb_mid"] = mid
    features["bb_upper"] = mid + (std * 2)
    features["bb_lower"] = mid - (std * 2)

    # RVOL
    vol_avg = features["volume_avg"]
    for i in range(n):
        features["rvol"][i] = volumes[i] / vol_avg[i] if vol_avg[i] and vol_avg[i] > 0 else 1.0

    # Market Structure (SMC)
    _detect_structure(closes, highs, lows, features, n)

    # Order Blocks
    _find_order_blocks(closes, highs, lows, opens, volumes, features, n)

    # Order Block Mitigation — mark OBs that have been touched by price since formation
    _find_ob_mitigation(lows, highs, features, n)

    # Fair Value Gaps
    _find_fvg(highs, lows, features, n)

    # Liquidity zones
    _find_liquidity(highs, lows, features, n)

    # Market Structure Breaks (MSS/CHoCH)
    _find_structure_breaks(highs, lows, features, n)

    # OBV trend (rising/falling over last 5 bars for divergence detection)
    obv_arr = features["obv"]
    obv_trend_arr = ["flat"] * n
    for i in range(5, n):
        if obv_arr[i] > obv_arr[i - 5]:
            obv_trend_arr[i] = "rising"
        elif obv_arr[i] < obv_arr[i - 5]:
            obv_trend_arr[i] = "falling"
    features["obv_trend"] = obv_trend_arr

    # RSI Divergence (regular + hidden)
    _find_rsi_divergence(closes, features["rsi_14"], features, n)

    # Confluence score — how many trend indicators agree
    sma20 = features["sma_20"]
    sma50 = features["sma_50"]
    sma200 = features["sma_200"]
    ema20 = features["ema_20"]
    ema50 = features["ema_50"]
    rsi = features["rsi_14"]
    for i in range(50, n):
        score = 0
        # SMA alignment: price above SMA20, SMA20 above SMA50, SMA50 above SMA200
        if _v(sma20[i]) and _v(sma50[i]) and _v(sma200[i]):
            if sma20[i] > sma50[i] and sma50[i] > sma200[i]:
                score += 2
            elif sma20[i] < sma50[i] and sma50[i] < sma200[i]:
                score -= 2
        # Price relative to SMAs
        if _v(sma20[i]):
            if closes[i] > sma20[i]:
                score += 1
            else:
                score -= 1
        # EMA crossover
        if _v(ema20[i]) and _v(ema50[i]):
            if ema20[i] > ema50[i]:
                score += 1
            else:
                score -= 1
        # RSI direction
        if _v(rsi[i]):
            if rsi[i] > 50:
                score += 1
            else:
                score -= 1
        features["confluence_score"][i] = score

    return features


def get_feature_summary(features, idx=-1):
    """Get the last row of features as a readable summary string for AI"""
    i = idx if idx >= 0 else len(features["rsi_14"]) - 1
    if i < 0 or i >= len(features["rsi_14"]):
        return "No data"
    ma_cross = "above" if _v(features["sma_20"][i]) and _v(features["sma_50"][i]) and \
        features["sma_20"][i] > features["sma_50"][i] else "below"

    return (
        f"RSI(14)={_fmt(features['rsi_14'][i])}, RSI(28)={_fmt(features['rsi_28'][i])}, "
        f"Trend={features['trend'][i] if i < len(features['trend']) else 'N/A'}, "
        f"ATR={_fmt(features['atr'][i])}, "
        f"SMA20={_fmt(features['sma_20'][i])}, SMA50={_fmt(features['sma_50'][i])}, "
        f"Price is {ma_cross} SMA crossover, "
        f"RVOL={_fmt(features['rvol'][i])}x, "
        f"Bollinger Upper={_fmt(features['bb_upper'][i])}, "
        f"Bollinger Lower={_fmt(features['bb_lower'][i])}, "
        f"Swing High={_fmt(features['swing_high'][i])}, "
        f"Swing Low={_fmt(features['swing_low'][i])}"
    )


def get_feature_json(features, idx=-1):
    """Get features as JSON for AI tool response"""
    i = idx if idx >= 0 else len(features["rsi_14"]) - 1
    if i < 0 or i >= len(features["rsi_14"]):
        return {}
    return {
        "rsi_14": round(_v(features["rsi_14"][i]), 1) if not pd.isna(_v(features["rsi_14"][i])) else None,
        "rsi_28": round(_v(features["rsi_28"][i]), 1) if not pd.isna(_v(features["rsi_28"][i])) else None,
        "trend": features["trend"][i] if i < len(features["trend"]) else None,
        "atr": round(_v(features["atr"][i]), 2) if not pd.isna(_v(features["atr"][i])) else None,
        "sma_20": round(_v(features["sma_20"][i]), 2) if not pd.isna(_v(features["sma_20"][i])) else None,
        "sma_50": round(_v(features["sma_50"][i]), 2) if not pd.isna(_v(features["sma_50"][i])) else None,
        "sma_200": round(_v(features["sma_200"][i]), 2) if not pd.isna(_v(features["sma_200"][i])) else None,
        "rvol": round(_v(features["rvol"][i]), 2) if not pd.isna(_v(features["rvol"][i])) else None,
        "bb_upper": round(_v(features["bb_upper"][i]), 2) if not pd.isna(_v(features["bb_upper"][i])) else None,
        "bb_lower": round(_v(features["bb_lower"][i]), 2) if not pd.isna(_v(features["bb_lower"][i])) else None,
        "swing_high": round(_v(features["swing_high"][i]), 2) if not pd.isna(_v(features["swing_high"][i])) else None,
        "swing_low": round(_v(features["swing_low"][i]), 2) if not pd.isna(_v(features["swing_low"][i])) else None,
        "ob_high": round(_v(features["ob_high"][i]), 2) if not pd.isna(_v(features["ob_high"][i])) else None,
        "ob_low": round(_v(features["ob_low"][i]), 2) if not pd.isna(_v(features["ob_low"][i])) else None,
        "fvg_high": round(_v(features["fvg_high"][i]), 2) if not pd.isna(_v(features["fvg_high"][i])) else None,
        "fvg_low": round(_v(features["fvg_low"][i]), 2) if not pd.isna(_v(features["fvg_low"][i])) else None,
        "liq_above": round(_v(features["liq_above"][i]), 2) if not pd.isna(_v(features["liq_above"][i])) else None,
        "liq_below": round(_v(features["liq_below"][i]), 2) if not pd.isna(_v(features["liq_below"][i])) else None,
        "rsi_divergence": features.get("rsi_divergence", [None])[i] if i < len(features.get("rsi_divergence", [])) else None,
        "structure_break": features.get("structure_break", [None])[i] if i < len(features.get("structure_break", [])) else None,
    }

def compute_multi_timeframe(df):
    """Compute features for weekly and monthly periods from daily OHLCV.
    Returns dict with 'weekly' and 'monthly' feature dicts."""
    result = {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for period, freq in [("weekly", "W"), ("monthly", "ME")]:
        resampled = df.resample(freq, on="date").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        if len(resampled) < 20:
            result[period] = None
            continue
        resampled["symbol"] = df["symbol"].iloc[0] if "symbol" in df else ""
        features = compute_features(resampled)
        result[period] = get_feature_json(features)

    return result


# ── Internal computation functions ──

def _rsi(series, period, n):
    deltas = np.diff(series)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.full(n, np.nan, dtype=float)
    avg_loss = np.full(n, np.nan, dtype=float)
    if n <= period:
        return avg_gain
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / np.where(avg_loss == 0, 0.001, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    result = np.full(n, np.nan, dtype=float)
    result[period:] = rsi[period:]
    return result

def _sma(series, period, n):
    return pd.Series(series).rolling(window=period).mean().values

def _ema(series, period, n):
    return pd.Series(series).ewm(span=period, adjust=False).mean().values

def _obv(closes, volumes, n):
    obv = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv

def _atr(highs, lows, closes, period, n):
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    atr = np.full(n, np.nan, dtype=float)
    if n <= period:
        return atr
    atr[period] = tr[:period].mean()
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr

def _vwap(closes, volumes, n):
    cum = (closes * volumes).cumsum() / volumes.cumsum()
    return cum

def _detect_structure(closes, highs, lows, features, n):
    """Detect HH/HL (uptrend), LH/LL (downtrend), or range"""
    features["swing_high"] = np.full(n, np.nan, dtype=float)
    features["swing_low"] = np.full(n, np.nan, dtype=float)
    for i in range(2, n - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and \
           highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            features["swing_high"][i] = highs[i]
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and \
           lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            features["swing_low"][i] = lows[i]

    # Detect trend based on swing points
    for i in range(5, n):
        sh = features["swing_high"]
        sl = features["swing_low"]
        last_sh = [sh[j] for j in range(i - 5, i + 1) if not np.isnan(sh[j])]
        last_sl = [sl[j] for j in range(i - 5, i + 1) if not np.isnan(sl[j])]

        if len(last_sh) >= 2 and len(last_sl) >= 2:
            if last_sh[-1] > max(last_sh[:-1]) and last_sl[-1] > max(last_sl[:-1]):
                features["trend"][i] = "uptrend"
            elif last_sh[-1] < min(last_sh[:-1]) and last_sl[-1] < min(last_sl[:-1]):
                features["trend"][i] = "downtrend"
            else:
                features["trend"][i] = "range"

def _find_order_blocks(closes, highs, lows, opens, volumes, features, n):
    """Order blocks: last candle before a strong impulse move"""
    features["ob_high"] = np.full(n, np.nan, dtype=float)
    features["ob_low"] = np.full(n, np.nan, dtype=float)
    vol_avg_series = _sma(volumes, 20, n)
    for i in range(3, n - 1):
        if vol_avg_series[i] is None or np.isnan(vol_avg_series[i]):
            continue
        if closes[i] > highs[i - 1] and volumes[i] > vol_avg_series[i] * 1.5:
            for j in range(i - 1, max(i - 5, 0) - 1, -1):
                if closes[j] < opens[j]:
                    features["ob_high"][i] = highs[j]
                    features["ob_low"][i] = lows[j]
                    break

def _find_ob_mitigation(lows, highs, features, n):
    """Mark order blocks that have been touched by price since formation."""
    for i in range(n):
        ob_h = features["ob_high"][i]
        ob_l = features["ob_low"][i]
        if np.isnan(ob_h) or np.isnan(ob_l):
            continue
        # Check if any later bar entered the OB zone
        for j in range(i + 1, n):
            if lows[j] <= ob_h and highs[j] >= ob_l:
                features["ob_mitigated"][i] = True
                break

def _find_fvg(highs, lows, features, n):
    """Fair Value Gaps: gap between consecutive candle ranges"""
    features["fvg_high"] = np.full(n, np.nan, dtype=float)
    features["fvg_low"] = np.full(n, np.nan, dtype=float)
    for i in range(1, n - 1):
        if lows[i + 1] > highs[i - 1]:
            features["fvg_high"][i] = lows[i + 1]
            features["fvg_low"][i] = highs[i - 1]
        elif highs[i + 1] < lows[i - 1]:
            features["fvg_high"][i] = highs[i - 1]
            features["fvg_low"][i] = lows[i + 1]

def _find_liquidity(highs, lows, features, n):
    """Liquidity zones: above recent swing highs, below recent swing lows"""
    features["liq_above"] = np.full(n, np.nan, dtype=float)
    features["liq_below"] = np.full(n, np.nan, dtype=float)
    for i in range(5, n):
        window = highs[max(0, i - 10):i + 1]
        features["liq_above"][i] = max(window) * 1.001
        window_l = lows[max(0, i - 10):i + 1]
        features["liq_below"][i] = min(window_l) * 0.999

def _find_rsi_divergence(closes, rsi, features, n):
    """Detect regular and hidden RSI divergence."""
    for i in range(10, n):
        # Find last two swing lows and their RSI values
        low_indices = []
        low_rsi = []
        for j in range(i - 10, i + 1):
            if not np.isnan(features["swing_low"][j]):
                low_indices.append(j)
                low_rsi.append(rsi[j])
        # Find last two swing highs and their RSI values
        high_indices = []
        high_rsi = []
        for j in range(i - 10, i + 1):
            if not np.isnan(features["swing_high"][j]):
                high_indices.append(j)
                high_rsi.append(rsi[j])

        # Bullish regular divergence: price lower low, RSI higher low
        if len(low_indices) >= 2 and len(low_rsi) >= 2:
            if closes[low_indices[-1]] < closes[low_indices[-2]] and low_rsi[-1] > low_rsi[-2]:
                current = features["rsi_divergence"][i]
                features["rsi_divergence"][i] = (current + ",bullish_regular") if current else "bullish_regular"

        # Bullish hidden divergence: price higher low, RSI lower low
        if len(low_indices) >= 2 and len(low_rsi) >= 2:
            if closes[low_indices[-1]] > closes[low_indices[-2]] and low_rsi[-1] < low_rsi[-2]:
                current = features["rsi_divergence"][i]
                features["rsi_divergence"][i] = (current + ",bullish_hidden") if current else "bullish_hidden"

        # Bearish regular divergence: price higher high, RSI lower high
        if len(high_indices) >= 2 and len(high_rsi) >= 2:
            if closes[high_indices[-1]] > closes[high_indices[-2]] and high_rsi[-1] < high_rsi[-2]:
                current = features["rsi_divergence"][i]
                features["rsi_divergence"][i] = (current + ",bearish_regular") if current else "bearish_regular"

        # Bearish hidden divergence: price lower high, RSI higher high
        if len(high_indices) >= 2 and len(high_rsi) >= 2:
            if closes[high_indices[-1]] < closes[high_indices[-2]] and high_rsi[-1] > high_rsi[-2]:
                current = features["rsi_divergence"][i]
                features["rsi_divergence"][i] = (current + ",bearish_hidden") if current else "bearish_hidden"

def _find_structure_breaks(highs, lows, features, n):
    """Market Structure Breaks — price breaks through a key swing point."""
    features["structure_break"] = [False] * n
    for i in range(5, n):
        # Uptrend → look for break below last swing low (bearish CHoCH)
        if features["trend"][i] == "uptrend":
            last_swing_low = None
            for j in range(i - 1, max(i - 10, 0) - 1, -1):
                if not np.isnan(features["swing_low"][j]):
                    last_swing_low = features["swing_low"][j]
                    break
            if last_swing_low is not None and lows[i] < last_swing_low:
                features["structure_break"][i] = True
        # Downtrend → look for break above last swing high (bullish CHoCH)
        elif features["trend"][i] == "downtrend":
            last_swing_high = None
            for j in range(i - 1, max(i - 10, 0) - 1, -1):
                if not np.isnan(features["swing_high"][j]):
                    last_swing_high = features["swing_high"][j]
                    break
            if last_swing_high is not None and highs[i] > last_swing_high:
                features["structure_break"][i] = True

def _v(val):
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
    except:
        return None
    return val

def _fmt(val):
    if val is None:
        return "N/A"
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return "N/A"
    except:
        return "N/A"
    if abs(val) > 100:
        return f"{val:.1f}"
    if abs(val) > 1:
        return f"{val:.2f}"
    return f"{val:.3f}"
