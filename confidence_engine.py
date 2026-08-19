"""Confidence Engine — scores every prediction/signal with evidence.
AI never gives raw opinion. Always: signal + confidence + reasoning."""

import logging
from config_manager import get_config

logger = logging.getLogger("confidence")

class ConfidenceEngine:
    def __init__(self):
        cfg = get_config()
        self.ob = cfg.get("confidence.rsi_overbought", 70)
        self.os = cfg.get("confidence.rsi_oversold", 30)
        self.min_conf = cfg.get("confidence.min_confirmation_signals", 3)
        self.max_conflict = cfg.get("confidence.max_conflicting_signals", 2)

    def score_buy_signal(self, features, price=None):
        """Score how strong a BUY signal is. Returns (direction, confidence, reasons, conflicts)"""
        confirmations = []
        conflicts = []

        rsi14 = features.get("rsi_14")
        rsi28 = features.get("rsi_28")
        trend = features.get("trend")
        rvol = features.get("rvol")
        sma_20 = features.get("sma_20")
        sma_50 = features.get("sma_50")
        ob_high = features.get("ob_high")
        ob_low = features.get("ob_low")
        bb_lower = features.get("bb_lower")
        fvg_high = features.get("fvg_high")
        fvg_low = features.get("fvg_low")
        liq_above = features.get("liq_above")
        liq_below = features.get("liq_below")
        swing_high = features.get("swing_high")
        swing_low = features.get("swing_low")
        obv = features.get("obv")
        structure_break = features.get("structure_break")
        confluence = features.get("confluence_score")

        # RSI oversold
        if rsi14 is not None:
            if rsi14 < self.os:
                confirmations.append(f"RSI(14)={rsi14} oversold")
            elif rsi14 > self.ob:
                conflicts.append(f"RSI(14)={rsi14} overbought")

        # Trend
        if trend == "uptrend":
            confirmations.append("Uptrend structure")
        elif trend == "downtrend":
            conflicts.append("Downtrend structure")

        # RSI divergence
        rsi_div = features.get("rsi_divergence") or ""
        if "bullish_regular" in rsi_div or "bullish_hidden" in rsi_div:
            confirmations.append(f"RSI divergence: {rsi_div}")

        # Structure break (CHoCH) — bullish break = buy confirmation
        if structure_break is True:
            confirmations.append("Bullish structure break (CHoCH)")

        # Volume confirmation
        if rvol is not None and rvol > 1.5:
            confirmations.append(f"High volume (RVOL={rvol}x)")
        elif rvol is not None and rvol < 0.5:
            conflicts.append(f"Low volume (RVOL={rvol}x)")

        # Confluence score
        if confluence is not None and abs(confluence) >= 3:
            confirmations.append(f"Strong trend alignment (confluence={confluence})")
        elif confluence is not None and abs(confluence) >= 1:
            confirmations.append(f"Moderate trend alignment (confluence={confluence})")

        # SMA crossover
        if sma_20 is not None and sma_50 is not None:
            if sma_20 > sma_50:
                confirmations.append("SMA20 above SMA50 (bullish)")
            else:
                conflicts.append("SMA20 below SMA50 (bearish)")

        # Price near order block (skip if already mitigated)
        ob_mitigated = features.get("ob_mitigated", False)
        if price is not None and ob_low is not None and ob_high is not None and not ob_mitigated:
            if ob_low <= price <= ob_high:
                confirmations.append("Price at order block zone")
            elif price < ob_low:
                confirmations.append("Below order block — potential bounce")
            elif price > ob_high:
                conflicts.append("Above order block — no support")

        # Price in Fair Value Gap (buy-side)
        if price is not None and fvg_low is not None and fvg_high is not None:
            if fvg_low <= price <= fvg_high:
                confirmations.append("Price in fair value gap")

        # Price at swing low support
        if price is not None and swing_low is not None:
            if abs(price - swing_low) / price < 0.005:
                confirmations.append("Price at swing low support")

        # Price near liquidity below (stop-hunt before reversal)
        if price is not None and liq_below is not None:
            if abs(price - liq_below) / price < 0.005:
                confirmations.append("Near liquidity below — potential stop hunt")

        # Bollinger band
        if price is not None and bb_lower is not None:
            if price <= bb_lower * 1.02:
                confirmations.append("Near lower Bollinger band")

        # OBV divergence check (price making lower low but OBV making higher low)
        if price is not None and obv is not None and rsi14 is not None:
            if rsi14 < 40:
                obv_trend = features.get("obv_trend")
                if obv_trend == "rising":
                    confirmations.append("Bullish OBV divergence")

        return self._compute("BUY", confirmations, conflicts)

    def score_sell_signal(self, features, price=None):
        confirmations = []
        conflicts = []

        rsi14 = features.get("rsi_14")
        rsi28 = features.get("rsi_28")
        trend = features.get("trend")
        rvol = features.get("rvol")
        sma_20 = features.get("sma_20")
        sma_50 = features.get("sma_50")
        ob_high = features.get("ob_high")
        ob_low = features.get("ob_low")
        bb_upper = features.get("bb_upper")
        fvg_high = features.get("fvg_high")
        fvg_low = features.get("fvg_low")
        liq_above = features.get("liq_above")
        liq_below = features.get("liq_below")
        swing_high = features.get("swing_high")
        swing_low = features.get("swing_low")
        obv = features.get("obv")
        structure_break = features.get("structure_break")
        ob_mitigated = features.get("ob_mitigated")
        confluence = features.get("confluence_score")

        if rsi14 is not None:
            if rsi14 > self.ob:
                confirmations.append(f"RSI(14)={rsi14} overbought")
            elif rsi14 < self.os:
                conflicts.append(f"RSI(14)={rsi14} oversold")

        if trend == "downtrend":
            confirmations.append("Downtrend structure")
        elif trend == "uptrend":
            conflicts.append("Uptrend structure")

        # RSI divergence
        rsi_div = features.get("rsi_divergence") or ""
        if "bearish_regular" in rsi_div or "bearish_hidden" in rsi_div:
            confirmations.append(f"RSI divergence: {rsi_div}")

        # Structure break in downtrend = bearish continuation
        if structure_break is True:
            confirmations.append("Bearish structure break")

        # Confluence for sell
        if confluence is not None and abs(confluence) >= 3:
            confirmations.append(f"Strong trend alignment (confluence={confluence})")
        elif confluence is not None and abs(confluence) >= 1:
            if confluence < 0:
                confirmations.append(f"Moderate bearish alignment (confluence={confluence})")

        if rvol is not None and rvol > 1.5:
            confirmations.append(f"High volume selling (RVOL={rvol}x)")

        if sma_20 is not None and sma_50 is not None:
            if sma_20 < sma_50:
                confirmations.append("SMA20 below SMA50 (bearish)")
            else:
                conflicts.append("SMA20 above SMA50 (bullish)")

        ob_mitigated = features.get("ob_mitigated", False)
        if price is not None and ob_high is not None and not ob_mitigated:
            if price >= ob_high:
                confirmations.append("At resistance / order block")

        # Price in Fair Value Gap (sell-side — bearish FVG)
        if price is not None and fvg_low is not None and fvg_high is not None:
            if fvg_low <= price <= fvg_high:
                confirmations.append("Price in bearish fair value gap")

        # Price at swing high resistance
        if price is not None and swing_high is not None:
            if abs(price - swing_high) / price < 0.005:
                confirmations.append("Price at swing high resistance")

        # Price near liquidity above (stop-hunt before reversal down)
        if price is not None and liq_above is not None:
            if abs(price - liq_above) / price < 0.005:
                confirmations.append("Near liquidity above — potential rejection")

        if price is not None and bb_upper is not None:
            if price >= bb_upper * 0.98:
                confirmations.append("Near upper Bollinger band")

        # OBV divergence (price making higher high, OBV making lower high)
        if price is not None and obv is not None and rsi14 is not None:
            if rsi14 > 60:
                obv_trend = features.get("obv_trend")
                if obv_trend == "falling":
                    confirmations.append("Bearish OBV divergence")

        return self._compute("SELL", confirmations, conflicts)

    def analyze_trend_strength(self, features):
        """Pure trend strength analysis, no direction bias"""
        confirmations = []
        conflicts = []

        trend = features.get("trend")
        rsi14 = features.get("rsi_14")
        rvol = features.get("rvol")

        if trend == "uptrend":
            if rsi14 is not None and 40 < rsi14 < 70:
                confirmations.append("Trend up with RSI in healthy range")
        elif trend == "downtrend":
            if rsi14 is not None and 30 < rsi14 < 60:
                confirmations.append("Trend down with RSI in bearish range")

        if rvol is not None and rvol > 1.2:
            confirmations.append(f"Strong volume (RVOL={rvol}x)")

        result = self._compute(trend.upper() if trend else "RANGE", confirmations, conflicts)
        result["trend"] = trend
        return result

    def _compute(self, direction, confirmations, conflicts):
        n_conf = len(confirmations)
        n_conflict = len(conflicts)

        if n_conf == 0 and n_conflict == 0:
            return {
                "direction": direction,
                "confidence": 0,
                "signal": "neutral",
                "confirmations": [],
                "conflicts": [],
                "summary": "No clear signals",
            }

        # Base confidence from confirmations
        base = min(n_conf / (n_conf + n_conflict + 1) * 100, 95)
        # Reduce for conflicts
        penalty = min(n_conflict * 10, 40)
        confidence = max(base - penalty, 5)

        # Determine signal strength
        if confidence >= 75:
            signal = "strong"
        elif confidence >= 55:
            signal = "moderate"
        elif confidence >= 35:
            signal = "weak"
        else:
            signal = "neutral"

        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "signal": signal,
            "confirmations": confirmations,
            "conflicts": conflicts,
            "summary": f"{direction} ({signal}, {round(confidence, 1)}% confidence). "
                       f"{len(confirmations)} confirmations, {len(conflicts)} conflicting signals.",
        }

# Singleton
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = ConfidenceEngine()
    return _engine
