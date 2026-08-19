"""Market Overview — single-ping market snapshot for morning review.

Returns:
- Market breadth (% stocks above key SMAs, advancing/declining)
- Top/bottom stocks by change, RSI, volume
- Count of buy/sell/neutral signals across all stocks
- Per-stock quick snapshot (RSI, trend, signal, change%)
"""

import logging
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config

logger = logging.getLogger("market_overview")

class MarketOverview:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.con = duckdb.connect(self.db_path)

    def get_breadth(self):
        """Market breadth: stocks above key SMAs, advancing/declining."""
        latest = self.con.execute("SELECT MAX(date) FROM daily").fetchone()[0]
        if not latest:
            return {}

        # Advancing/declining on latest day (exclude indices)
        adv_dec = self.con.execute("""
            WITH changes AS (
                SELECT d.security_id, s.symbol,
                       (d.close - d_prev.close) / NULLIF(d_prev.close, 0) * 100 AS change_pct
                FROM daily d
                JOIN securities s ON d.security_id = s.security_id
                JOIN daily d_prev ON d.security_id = d_prev.security_id
                    AND d_prev.date = (SELECT MAX(date) FROM daily WHERE security_id = d.security_id AND date < d.date)
                WHERE d.date = ? AND s.is_index = FALSE
            )
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN change_pct > 0 THEN 1 ELSE 0 END) as advancing,
                SUM(CASE WHEN change_pct < 0 THEN 1 ELSE 0 END) as declining,
                SUM(CASE WHEN change_pct > 2 THEN 1 ELSE 0 END) as strong_up,
                SUM(CASE WHEN change_pct < -2 THEN 1 ELSE 0 END) as strong_down,
                AVG(change_pct) as avg_change
            FROM changes
        """, [latest]).fetchone()

        # Stocks above SMA50, SMA200 from indicators table
        above_50 = self.con.execute("""
            SELECT COUNT(*) FROM indicators i
            JOIN daily d ON i.security_id = d.security_id AND i.date = d.date
            WHERE d.date = ? AND d.close > i.sma_50
        """, [latest]).fetchone()[0] or 0

        total_stocks = self.con.execute("SELECT COUNT(*) FROM securities WHERE is_index = FALSE").fetchone()[0] or 1

        return {
            "date": str(latest),
            "total_stocks": int(adv_dec[0]) if adv_dec else 0,
            "advancing": int(adv_dec[1]) if adv_dec else 0,
            "declining": int(adv_dec[2]) if adv_dec else 0,
            "strong_up": int(adv_dec[3]) if adv_dec else 0,
            "strong_down": int(adv_dec[4]) if adv_dec else 0,
            "avg_change_pct": round(float(adv_dec[5]), 2) if adv_dec and adv_dec[5] else 0,
            "above_sma50_pct": round(above_50 / max(total_stocks, 1) * 100, 1),
        }

    def get_top_movers(self, limit=10):
        """Top gainers and losers by change percentage."""
        latest = self.con.execute("SELECT MAX(date) FROM daily").fetchone()[0]
        if not latest:
            return {"gainers": [], "losers": []}

        rows = self.con.execute("""
            WITH changes AS (
                SELECT d.security_id, s.symbol,
                       d.close, d.volume,
                       (d.close - d_prev.close) / NULLIF(d_prev.close, 0) * 100 AS change_pct
                FROM daily d
                JOIN securities s ON d.security_id = s.security_id
                JOIN daily d_prev ON d.security_id = d_prev.security_id
                    AND d_prev.date = (SELECT MAX(date) FROM daily WHERE security_id = d.security_id AND date < d.date)
                WHERE d.date = ? AND s.is_index = FALSE
            )
            SELECT * FROM changes WHERE change_pct IS NOT NULL ORDER BY change_pct DESC
        """, [latest]).fetchdf()

        if len(rows) == 0:
            return {"gainers": [], "losers": []}

        gainers = rows.head(limit).to_dict("records")
        losers = rows.tail(limit).sort_values("change_pct").to_dict("records")
        return {"gainers": gainers, "losers": losers}

    def get_signal_screener(self):
        """Screen all stocks for strongest buy/sell signals.
        Uses confidence engine on latest available indicators."""
        latest = self.con.execute("SELECT MAX(date) FROM daily").fetchone()[0]
        if not latest:
            return []

        try:
            stocks = self.con.execute("""
                SELECT DISTINCT s.symbol, s.security_id
                FROM securities s
                WHERE s.is_index = FALSE AND s.is_active = TRUE
                ORDER BY s.symbol
            """).fetchall()
        except:
            return []

        from confidence_engine import ConfidenceEngine
        engine = ConfidenceEngine()

        results = []
        for symbol, sec_id in stocks:
            try:
                ind = self.con.execute("""
                    SELECT i.*, d.close,
                           d_prev.close AS prev_close,
                           (d.close - d_prev.close) / NULLIF(d_prev.close, 0) * 100 AS change_pct
                    FROM indicators i
                    JOIN daily d ON i.security_id = d.security_id AND i.date = d.date
                    JOIN daily d_prev ON d_prev.security_id = d.security_id
                        AND d_prev.date = (SELECT MAX(date) FROM daily WHERE security_id = d.security_id AND date < d.date)
                    WHERE i.security_id = ? ORDER BY i.date DESC LIMIT 1
                """, [sec_id]).fetchdf()
                if len(ind) == 0:
                    continue
                row = ind.iloc[0]
                price = float(row.get("close", 0)) if row.get("close") and not pd.isna(row.get("close")) else None
                change_pct = float(row.get("change_pct", 0)) if row.get("change_pct") and not pd.isna(row.get("change_pct")) else None

                features = {
                    "rsi_14": float(row["rsi_14"]) if row.get("rsi_14") and not pd.isna(row["rsi_14"]) else None,
                    "rsi_28": float(row["rsi_28"]) if row.get("rsi_28") and not pd.isna(row["rsi_28"]) else None,
                    "trend": None,
                    "rvol": float(row["rvol"]) if row.get("rvol") and not pd.isna(row["rvol"]) else None,
                    "sma_20": float(row["sma_20"]) if row.get("sma_20") and not pd.isna(row["sma_20"]) else None,
                    "sma_50": float(row["sma_50"]) if row.get("sma_50") and not pd.isna(row["sma_50"]) else None,
                    "bb_upper": float(row["bollinger_upper"]) if row.get("bollinger_upper") and not pd.isna(row["bollinger_upper"]) else None,
                    "bb_lower": float(row["bollinger_lower"]) if row.get("bollinger_lower") and not pd.isna(row["bollinger_lower"]) else None,
                    "atr_14": float(row["atr_14"]) if row.get("atr_14") and not pd.isna(row["atr_14"]) else None,
                }

                # Fetch trend from market_structure
                try:
                    ms = self.con.execute("""
                        SELECT trend FROM market_structure
                        WHERE security_id = ? ORDER BY date DESC LIMIT 1
                    """, [sec_id]).fetchone()
                    if ms and ms[0]:
                        features["trend"] = ms[0]
                except:
                    pass

                buy = engine.score_buy_signal(features, price)
                sell = engine.score_sell_signal(features, price)

                if buy["signal"] != "neutral" or sell["signal"] != "neutral":
                    results.append({
                        "symbol": symbol,
                        "price": price,
                        "rsi_14": features["rsi_14"],
                        "rvol": features["rvol"],
                        "change_pct": round(change_pct, 2) if change_pct is not None else None,
                        "buy_signal": buy["signal"],
                        "buy_confidence": buy["confidence"],
                        "sell_signal": sell["signal"],
                        "sell_confidence": sell["confidence"],
                        "top_signal": "BUY" if buy["confidence"] >= sell["confidence"] else "SELL",
                        "top_confidence": max(buy["confidence"], sell["confidence"]),
                    })
            except:
                continue

        # Sort by signal strength
        results.sort(key=lambda r: r["top_confidence"], reverse=True)
        return results[:20]

    def get_all_stocks_snapshot(self):
        """Single table: every Nifty 50 stock with key metrics."""
        latest = self.con.execute("SELECT MAX(date) FROM daily").fetchone()[0]
        if not latest:
            return []

        try:
            rows = self.con.execute("""
                WITH changes AS (
                    SELECT d.security_id,
                           d.close,
                           (d.close - d_prev.close) / NULLIF(d_prev.close, 0) * 100 AS change_pct
                    FROM daily d
                    JOIN daily d_prev ON d.security_id = d_prev.security_id
                        AND d_prev.date = (SELECT MAX(date) FROM daily WHERE security_id = d.security_id AND date < d.date)
                    WHERE d.date = ?
                )
                SELECT s.symbol, c.close, c.change_pct,
                       i.rsi_14, i.rvol, i.sma_20, i.sma_50,
                       i.bollinger_upper, i.bollinger_lower
                FROM securities s
                JOIN changes c ON s.security_id = c.security_id
                LEFT JOIN indicators i ON s.security_id = i.security_id AND i.date = ?
                WHERE s.is_active = TRUE AND s.is_index = FALSE
                ORDER BY s.symbol
            """, [latest, latest]).fetchdf()
            return self._clean_records(rows.to_dict("records"))
        except:
            return []

    def _clean_records(self, records):
        """Replace NaN/Inf with None for JSON safety."""
        import math
        cleaned = []
        for r in records:
            clean = {}
            for k, v in r.items():
                try:
                    f = float(v)
                    if math.isnan(f) or math.isinf(f):
                        clean[k] = None
                    else:
                        clean[k] = v
                except:
                    clean[k] = v
            cleaned.append(clean)
        return cleaned

    def get_full_summary(self):
        """Everything in one call — breadth + movers + signals + all stocks."""
        return {
            "breadth": self.get_breadth(),
            "movers": self.get_top_movers(),
            "signals": self.get_signal_screener(),
            "stocks": self.get_all_stocks_snapshot(),
        }

    def close(self):
        try:
            self.con.close()
        except:
            pass
