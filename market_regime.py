"""Market Regime Detection — classifies overall market state using ADX, Bollinger width, ATR, RSI.

Regimes:
- Strong Bull: ADX > 25, RSI > 60, price above all SMAs
- Weak Bull: ADX < 25, RSI 50-60, price above SMA50
- Ranging: ADX < 20, Bollinger width < 15%, RSI 40-60
- Weak Bear: ADX < 25, RSI 40-50, price below SMA50
- Strong Bear: ADX > 25, RSI < 40, price below all SMAs
- High Volatility: Bollinger width > 30% or ATR ratio > 1.5
- Low Volatility: Bollinger width < 10% or ATR ratio < 0.5

Stored in market DB: market_regime table (daily regime snapshots).
Wired into auto-trader for dynamic position sizing.
"""

import logging, math
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config

logger = logging.getLogger("market_regime")

REGIME_LABELS = {
    "strong_bull": "Strong Bull",
    "weak_bull": "Weak Bull",
    "ranging": "Ranging",
    "weak_bear": "Weak Bear",
    "strong_bear": "Strong Bear",
    "high_volatility": "High Volatility",
    "low_volatility": "Low Volatility",
}

REGIME_COLORS = {
    "strong_bull": "#238636",
    "weak_bull": "#3fb950",
    "ranging": "#d29922",
    "weak_bear": "#da3633",
    "strong_bear": "#a40e26",
    "high_volatility": "#f0883e",
    "low_volatility": "#58a6ff",
}


class MarketRegime:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.con = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS market_regime (
                date DATE PRIMARY KEY,
                regime VARCHAR,
                adx FLOAT,
                bb_width_pct FLOAT,
                atr_ratio FLOAT,
                rsi_14 FLOAT,
                vwap_position VARCHAR,
                sma_position VARCHAR,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    async def analyze(self):
        """Fetch NIFTY data, compute regime, store, return regime dict."""
        try:
            # Fetch NIFTY daily data
            df = self.con.execute("""
                SELECT date, open, high, low, close, volume
                FROM daily WHERE security_id = '13'
                ORDER BY date
            """).fetchdf()

            if len(df) < 30:
                logger.warning("Not enough NIFTY data for regime detection")
                return {"error": "Not enough NIFTY data (need 30+ days)"}

            closes = df["close"].values.astype(float)
            highs = df["high"].values.astype(float)
            lows = df["low"].values.astype(float)
            volumes = df["volume"].values.astype(float)
            n = len(closes)

            # Compute ADX
            adx = self._compute_adx(highs, lows, closes, 14, n)
            adx_val = float(adx[-1]) if not np.isnan(adx[-1]) else 0

            # Bollinger Band width
            bb_width = self._compute_bb_width(closes, 20, n)
            bb_width_pct = float(bb_width[-1]) if not np.isnan(bb_width[-1]) else 0

            # ATR ratio (current ATR / 20-day average ATR)
            atr = self._compute_atr(highs, lows, closes, 14, n)
            atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else 0
            atr_20d_avg = np.nanmean(atr[-40:-20]) if n >= 40 else atr_val
            atr_ratio = atr_val / max(atr_20d_avg, 0.0001)

            # RSI
            rsi = self._compute_rsi(closes, 14, n)
            rsi_val = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50

            # SMAs
            sma20 = self._sma(closes, 20, n)[-1] if n >= 20 else closes[-1]
            sma50 = self._sma(closes, 50, n)[-1] if n >= 50 else closes[-1]
            sma200 = self._sma(closes, 200, n)[-1] if n >= 200 else closes[-1]
            cp = closes[-1]

            # VWAP position
            vwap = self._compute_vwap(closes, volumes, n)
            vwap_val = float(vwap[-1]) if not np.isnan(vwap[-1]) else cp
            vwap_position = "above" if cp > vwap_val else "below"

            # SMA position
            above_sma20 = cp > sma20 if sma20 else False
            above_sma50 = cp > sma50 if sma50 else False
            above_sma200 = cp > sma200 if sma200 else False
            sma_count = sum([above_sma20, above_sma50, above_sma200])

            if sma_count >= 2:
                sma_position = "bullish"
            elif sma_count <= 0:
                sma_position = "bearish"
            else:
                sma_position = "mixed"

            # Classify regime
            regime = self._classify(adx_val, bb_width_pct, atr_ratio, rsi_val, vwap_position, sma_count)

            # Build details string
            details = (
                f"ADX={adx_val:.1f}, BBWidth={bb_width_pct:.1f}%, ATRRatio={atr_ratio:.2f}, "
                f"RSI={rsi_val:.1f}, VWAP={vwap_position}, SMA={sma_count}/3 bullish"
            )

            # Store in DB
            self.con.execute("""
                INSERT INTO market_regime (date, regime, adx, bb_width_pct, atr_ratio, rsi_14, vwap_position, sma_position, details)
                VALUES (CURRENT_DATE, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (date) DO UPDATE SET
                    regime = EXCLUDED.regime, adx = EXCLUDED.adx,
                    bb_width_pct = EXCLUDED.bb_width_pct, atr_ratio = EXCLUDED.atr_ratio,
                    rsi_14 = EXCLUDED.rsi_14, vwap_position = EXCLUDED.vwap_position,
                    sma_position = EXCLUDED.sma_position, details = EXCLUDED.details
            """, [regime, adx_val, bb_width_pct, atr_ratio, rsi_val, vwap_position, sma_position, details])
            self.con.commit()

            result = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "regime": regime,
                "regime_label": REGIME_LABELS.get(regime, regime),
                "regime_color": REGIME_COLORS.get(regime, "#8b949e"),
                "adx": round(adx_val, 1),
                "bb_width_pct": round(bb_width_pct, 1),
                "atr_ratio": round(atr_ratio, 2),
                "rsi_14": round(rsi_val, 1),
                "vwap_position": vwap_position,
                "sma_position": sma_position,
                "close": round(float(cp), 2),
                "details": details,
            }

            logger.info(f"Market regime: {result['regime_label']} (ADX={adx_val:.1f}, BB={bb_width_pct:.1f}%, RSI={rsi_val:.1f})")
            return result

        except Exception as e:
            logger.warning(f"Regime analysis failed: {e}")
            return {"error": str(e)}

    def _classify(self, adx, bb_width_pct, atr_ratio, rsi, vwap_position, sma_count):
        """Classify market regime based on computed metrics."""
        # Check volatility regimes first (they override trend)
        if bb_width_pct > 30 or atr_ratio > 1.8:
            return "high_volatility"
        if bb_width_pct < 8 or atr_ratio < 0.4:
            return "low_volatility"

        # Strong trend detection
        if adx > 25 and rsi > 60 and sma_count >= 2:
            return "strong_bull"
        if adx > 25 and rsi < 40 and sma_count <= 1:
            return "strong_bear"

        # Weak trend / ranging
        if adx < 20 and 40 <= rsi <= 60:
            return "ranging"

        # Weak directional bias
        if sma_count >= 2 and rsi >= 50:
            return "weak_bull"
        if sma_count <= 1 and rsi < 50:
            return "weak_bear"

        # Fallback based on VWAP
        if vwap_position == "above" and rsi > 50:
            return "weak_bull"
        if vwap_position == "below" and rsi < 50:
            return "weak_bear"

        return "ranging"

    def get_current(self):
        """Get the latest regime from DB (fast, no recomputation)."""
        row = self.con.execute("""
            SELECT date, regime, adx, bb_width_pct, atr_ratio, rsi_14, vwap_position, sma_position, details
            FROM market_regime ORDER BY date DESC LIMIT 1
        """).fetchone()

        if not row:
            return None

        regime = row[1]
        return {
            "date": str(row[0]),
            "regime": regime,
            "regime_label": REGIME_LABELS.get(regime, regime),
            "regime_color": REGIME_COLORS.get(regime, "#8b949e"),
            "adx": row[2],
            "bb_width_pct": row[3],
            "atr_ratio": row[4],
            "rsi_14": row[5],
            "vwap_position": row[6],
            "sma_position": row[7],
            "details": row[8],
        }

    def get_history(self, days=30):
        """Get regime history for display."""
        rows = self.con.execute("""
            SELECT date, regime, adx, bb_width_pct, atr_ratio, rsi_14
            FROM market_regime ORDER BY date DESC LIMIT ?
        """, [days]).fetchdf()
        if len(rows) == 0:
            return []

        result = []
        for _, r in rows.iterrows():
            result.append({
                "date": str(r["date"]),
                "regime": r["regime"],
                "regime_label": REGIME_LABELS.get(r["regime"], r["regime"]),
                "regime_color": REGIME_COLORS.get(r["regime"], "#8b949e"),
                "adx": r["adx"],
                "bb_width_pct": r["bb_width_pct"],
                "rsi_14": r["rsi_14"],
            })
        return result

    def get_regime_df(self, ticker="NIFTY", days=60):
        """Return DataFrame with [date, regime] for regime-conditioned backtesting."""
        ticker = ticker.upper()
        sid = self.con.execute("SELECT security_id FROM securities WHERE symbol = ?", [ticker]).fetchone()
        if not sid:
            if ticker == "NIFTY":
                sid = ("13",)
            elif ticker == "BANKNIFTY":
                sid = ("25",)
            else:
                return None
        df = self.con.execute("""
            SELECT d.date, mr.regime
            FROM daily d
            LEFT JOIN market_regime mr ON d.date = mr.date
            WHERE d.security_id = ? AND mr.regime IS NOT NULL
            ORDER BY d.date DESC LIMIT ?
        """, [sid[0], days]).fetchdf()
        if len(df) == 0:
            return None
        return df.sort_values("date")

    def get_position_sizing_adjustment(self):
        """Get position sizing multiplier based on regime. Returns 0.0-1.5."""
        current = self.get_current()
        if not current:
            return 1.0

        regime = current.get("regime", "ranging")
        adjustments = {
            "strong_bull": 1.5,
            "weak_bull": 1.0,
            "ranging": 0.6,
            "weak_bear": 0.8,
            "strong_bear": 0.4,
            "high_volatility": 0.3,
            "low_volatility": 0.8,
        }
        return adjustments.get(regime, 1.0)

    # ── Technical indicators ──

    @staticmethod
    def _compute_adx(highs, lows, closes, period, n):
        """Compute Average Directional Index."""
        adx = np.full(n, np.nan, dtype=float)
        if n < period * 2:
            return adx

        tr = np.full(n, np.nan, dtype=float)
        up_move = np.full(n, np.nan, dtype=float)
        down_move = np.full(n, np.nan, dtype=float)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1]))
            up_move[i] = highs[i] - highs[i - 1]
            down_move[i] = lows[i - 1] - lows[i]

        # Smoothed ATR and directional movement (Wilder's smoothing)
        atr = np.full(n, np.nan, dtype=float)
        smooth_plus_dm = np.full(n, np.nan, dtype=float)
        smooth_minus_dm = np.full(n, np.nan, dtype=float)
        plus_di = np.full(n, np.nan, dtype=float)
        minus_di = np.full(n, np.nan, dtype=float)

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        atr[period] = np.nanmean(tr[1:period + 1])
        smooth_plus_dm[period] = np.nanmean(plus_dm[1:period + 1])
        smooth_minus_dm[period] = np.nanmean(minus_dm[1:period + 1])

        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            smooth_plus_dm[i] = (smooth_plus_dm[i - 1] * (period - 1) + plus_dm[i]) / period
            smooth_minus_dm[i] = (smooth_minus_dm[i - 1] * (period - 1) + minus_dm[i]) / period

        # Compute +DI, -DI from smoothed values
        for i in range(period, n):
            plus_di[i] = smooth_plus_dm[i] / max(atr[i], 0.001) * 100
            minus_di[i] = smooth_minus_dm[i] / max(atr[i], 0.001) * 100

        # DX = |+DI - -DI| / (+DI + -DI) * 100
        dx = np.full(n, np.nan, dtype=float)
        for i in range(period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100

        # ADX = SMA of DX
        for i in range(period * 2 - 1, n):
            adx[i] = np.nanmean(dx[i - period + 1:i + 1])

        return adx

    @staticmethod
    def _compute_bb_width(closes, period, n):
        """Bollinger Band width as % of middle band."""
        width = np.full(n, np.nan, dtype=float)
        for i in range(period - 1, n):
            mid = np.mean(closes[i - period + 1:i + 1])
            std = np.std(closes[i - period + 1:i + 1])
            if mid > 0:
                width[i] = (std * 4) / mid * 100  # (upper-lower)/mid
        return width

    @staticmethod
    def _compute_atr(highs, lows, closes, period, n):
        tr = np.full(n, np.nan, dtype=float)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1]))
        atr = np.full(n, np.nan, dtype=float)
        if n <= period:
            return atr
        atr[period] = np.nanmean(tr[1:period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def _compute_rsi(closes, period, n):
        rsi = np.full(n, np.nan, dtype=float)
        if n < period + 1:
            return rsi
        gains, losses = 0, 0
        for i in range(1, period + 1):
            diff = closes[i] - closes[i - 1]
            gains += max(diff, 0)
            losses += max(-diff, 0)
        avg_gain = gains / period
        avg_loss = losses / period
        for i in range(period + 1, n):
            diff = closes[i] - closes[i - 1]
            avg_gain = (avg_gain * (period - 1) + max(diff, 0)) / period
            avg_loss = (avg_loss * (period - 1) + max(-diff, 0)) / period
            rs = avg_gain / max(avg_loss, 0.001)
            rsi[i] = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_vwap(closes, volumes, n):
        closes = np.asarray(closes, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        cum_pv = np.cumsum(closes * volumes)
        cum_v = np.cumsum(volumes)
        return cum_pv / cum_v

    @staticmethod
    def _sma(values, period, n):
        sma = np.full(n, np.nan, dtype=float)
        for i in range(period - 1, n):
            sma[i] = np.mean(values[i - period + 1:i + 1])
        return sma

    def close(self):
        try:
            self.con.close()
        except:
            pass


# Singleton
_instance = None

def get_market_regime():
    global _instance
    if _instance is None:
        _instance = MarketRegime()
    return _instance
