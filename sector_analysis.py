"""Sector Analysis — maps Nifty 50 stocks to sectors, detects rotation.

Provides:
- per-sector aggregated performance (daily, weekly, monthly)
- sector rotation detection (which sectors gaining/losing momentum)
- sector-level features for AI context
"""

import logging
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config

logger = logging.getLogger("sector_analysis")

# Nifty 50 sector mapping (cleaned — HDFC removed, no duplicates)
SECTOR_MAP = {
    "RELIANCE":     "Energy",
    "ONGC":         "Energy",
    "NTPC":         "Energy",
    "POWERGRID":    "Energy",
    "BPCL":         "Energy",
    "COALINDIA":    "Energy",
    "TCS":          "IT",
    "INFY":         "IT",
    "WIPRO":        "IT",
    "HCLTECH":      "IT",
    "TECHM":        "IT",
    "HDFCBANK":     "Financial Services",
    "ICICIBANK":    "Financial Services",
    "AXISBANK":     "Financial Services",
    "KOTAKBANK":    "Financial Services",
    "SBIN":         "Financial Services",
    "INDUSINDBK":   "Financial Services",
    "BAJFINANCE":   "Financial Services",
    "BAJAJFINSV":   "Financial Services",
    "MARUTI":       "Automobile",
    "M&M":          "Automobile",
    "BAJAJ-AUTO":   "Automobile",
    "EICHERMOT":    "Automobile",
    "TATAMOTORS":   "Automobile",
    "HINDUNILVR":   "FMCG",
    "ITC":          "FMCG",
    "BRITANNIA":    "FMCG",
    "NESTLEIND":    "FMCG",
    "TATACONSUM":   "FMCG",
    "DABUR":        "FMCG",
    "TATASTEEL":    "Metals & Mining",
    "JSWSTEEL":     "Metals & Mining",
    "HINDALCO":     "Metals & Mining",
    "SUNPHARMA":    "Pharma",
    "DRREDDY":      "Pharma",
    "CIPLA":        "Pharma",
    "DIVISLAB":     "Pharma",
    "APOLLOHOSP":   "Healthcare",
    "BHARTIARTL":   "Telecom",
    "LT":           "Infrastructure",
    "ULTRACEMCO":   "Infrastructure",
    "ADANIPORTS":   "Infrastructure",
    "GRASIM":       "Infrastructure",
    "ADANIENT":     "Infrastructure",
    "ASIANPAINT":   "Consumer Durables",
    "TITAN":        "Consumer Durables",
    "SBILIFE":      "Insurance",
    "HDFCLIFE":     "Insurance",
    "TRENT":        "Consumer Durables",
    "SHRIRAMFIN":   "Financial Services",
    "HEROMOTOCO":   "Automobile",
    "BEL":          "Industrials",
}

SECTOR_GROUPS = {}
for sym, sec in SECTOR_MAP.items():
    SECTOR_GROUPS.setdefault(sec, []).append(sym)


class SectorAnalyzer:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.con = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS sector_performance (
                sector VARCHAR,
                date DATE,
                avg_change_pct FLOAT,
                median_change_pct FLOAT,
                max_change_pct FLOAT,
                min_change_pct FLOAT,
                stocks_up INTEGER,
                stocks_down INTEGER,
                total_stocks INTEGER,
                volume_ratio FLOAT,
                PRIMARY KEY (sector, date)
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS sector_trends (
                sector VARCHAR,
                period VARCHAR,
                period_start DATE,
                period_end DATE,
                avg_change_pct FLOAT,
                total_return_pct FLOAT,
                stocks_up INTEGER,
                stocks_down INTEGER,
                total_stocks INTEGER,
                PRIMARY KEY (sector, period, period_end)
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS sector_rotation (
                id VARCHAR PRIMARY KEY,
                date DATE,
                leading_sector VARCHAR,
                lagging_sector VARCHAR,
                rotation_type VARCHAR,
                strength FLOAT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.con.commit()

    def get_sector(self, ticker):
        return SECTOR_MAP.get(ticker.upper(), "Other")

    def get_stocks_in_sector(self, sector):
        return SECTOR_GROUPS.get(sector, [])

    def compute_sector_performance(self, date=None, include_volume=True):
        """Compute aggregated performance for each sector on a given date."""
        if date is None:
            date = self.con.execute("SELECT MAX(date) FROM daily").fetchone()[0]
        if date is None:
            return pd.DataFrame()

        results = []
        for sector, stocks in SECTOR_GROUPS.items():
            placeholders = ",".join(["?" for _ in stocks])
            query = f"""
                SELECT d.symbol, d.change_pct, d.volume_ratio
                FROM (
                    SELECT s.symbol,
                           (d.close - d_prev.close) / NULLIF(d_prev.close, 0) * 100 AS change_pct,
                           d.volume / NULLIF(vol_avg.avg_vol, 0) AS volume_ratio
                    FROM daily d
                    JOIN (
                        SELECT security_id, MAX(date) as max_date
                        FROM daily WHERE date <= ?
                        GROUP BY security_id
                    ) latest ON d.security_id = latest.security_id AND d.date = latest.max_date
                    LEFT JOIN (
                        SELECT d2.security_id, MAX(d2.date) as prev_date
                        FROM daily d2 WHERE d2.date < ?
                        GROUP BY d2.security_id
                    ) prev ON d.security_id = prev.security_id
                    LEFT JOIN daily d_prev ON d.security_id = d_prev.security_id AND d_prev.date = prev.prev_date
                    JOIN securities s ON d.security_id = s.security_id
                    LEFT JOIN (
                        SELECT security_id, AVG(volume) AS avg_vol
                        FROM daily
                        WHERE date < (SELECT MAX(date) FROM daily)
                          AND date >= (SELECT MAX(date) FROM daily) - INTERVAL 20 DAY
                        GROUP BY security_id
                    ) vol_avg ON d.security_id = vol_avg.security_id
                    WHERE s.symbol IN ({placeholders})
                ) d
            """
            try:
                df = self.con.execute(query, [date, date] + stocks).fetchdf()
                if len(df) == 0:
                    continue
                changes = df["change_pct"].dropna()
                vol_ratio = df["volume_ratio"].dropna().mean() if include_volume and "volume_ratio" in df.columns else None
                stocks_up = int((changes > 0).sum())
                stocks_down = int((changes < 0).sum())
                results.append({
                    "sector": sector,
                    "date": date,
                    "avg_change_pct": round(float(changes.mean()), 2),
                    "median_change_pct": round(float(changes.median()), 2),
                    "max_change_pct": round(float(changes.max()), 2),
                    "min_change_pct": round(float(changes.min()), 2),
                    "stocks_up": stocks_up,
                    "stocks_down": stocks_down,
                    "total_stocks": len(changes),
                    "volume_ratio": round(float(vol_ratio), 2) if vol_ratio is not None and not np.isnan(vol_ratio) else None,
                    "stocks": df.to_dict("records"),
                })
                self.con.execute("""
                    INSERT INTO sector_performance (sector, date, avg_change_pct, median_change_pct,
                        max_change_pct, min_change_pct, stocks_up, stocks_down, total_stocks, volume_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (sector, date) DO UPDATE SET
                        avg_change_pct = EXCLUDED.avg_change_pct,
                        median_change_pct = EXCLUDED.median_change_pct,
                        max_change_pct = EXCLUDED.max_change_pct, min_change_pct = EXCLUDED.min_change_pct,
                        stocks_up = EXCLUDED.stocks_up, stocks_down = EXCLUDED.stocks_down,
                        volume_ratio = EXCLUDED.volume_ratio
                """, [sector, date, results[-1]["avg_change_pct"], results[-1]["median_change_pct"],
                      results[-1]["max_change_pct"], results[-1]["min_change_pct"],
                      stocks_up, stocks_down, len(changes), results[-1]["volume_ratio"]])
            except Exception as e:
                logger.warning(f"Sector {sector} query failed: {e}")
        self.con.commit()
        return pd.DataFrame(results)

    def compute_sector_trends(self, lookback_days=365):
        """Compute weekly and monthly aggregated performance per sector."""
        latest = self.con.execute("SELECT MAX(date) FROM daily").fetchone()[0]
        if latest is None:
            return

        start = latest - timedelta(days=lookback_days)
        for sector, stocks in SECTOR_GROUPS.items():
            placeholders = ",".join(["?" for _ in stocks])
            try:
                df = self.con.execute(f"""
                    SELECT s.symbol, d.date, d.close
                    FROM daily d
                    JOIN securities s ON d.security_id = s.security_id
                    WHERE s.symbol IN ({placeholders}) AND d.date >= ?
                    ORDER BY d.date
                """, stocks + [start]).fetchdf()
                if len(df) == 0:
                    continue
                df["date"] = pd.to_datetime(df["date"])

                for period, freq in [("weekly", "W"), ("monthly", "ME")]:
                    df_pivot = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
                    df_resampled = df_pivot.resample(freq).last()
                    for i in range(1, len(df_resampled)):
                        prev = df_resampled.iloc[i - 1]
                        curr = df_resampled.iloc[i]
                        period_start = df_resampled.index[i - 1].date()
                        period_end = df_resampled.index[i].date()
                        if prev.isna().all() or curr.isna().all():
                            continue
                        returns = (curr - prev) / prev * 100
                        valid = returns.dropna()
                        if len(valid) == 0:
                            continue
                        avg_ret = round(float(valid.mean()), 2)
                        total_ret = avg_ret
                        stocks_up = int((valid > 0).sum())
                        stocks_down = int((valid < 0).sum())
                        self.con.execute("""
                            INSERT INTO sector_trends (sector, period, period_start, period_end,
                                avg_change_pct, total_return_pct, stocks_up, stocks_down, total_stocks)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT (sector, period, period_end) DO UPDATE SET
                                avg_change_pct = EXCLUDED.avg_change_pct,
                                total_return_pct = EXCLUDED.total_return_pct,
                                stocks_up = EXCLUDED.stocks_up,
                                stocks_down = EXCLUDED.stocks_down
                        """, [sector, period, period_start, period_end, avg_ret, total_ret,
                              stocks_up, stocks_down, len(valid)])
            except Exception as e:
                logger.warning(f"Sector {sector} trend query failed: {e}")
        self.con.commit()
        logger.info("Sector trends computed (weekly + monthly)")

    def detect_rotation(self, lookback_days=20):
        """Detect sector rotation by comparing recent vs older momentum (by week)."""
        latest = self.con.execute("SELECT MAX(period_end) FROM sector_trends WHERE period = 'weekly'").fetchone()
        if not latest or not latest[0]:
            return []
        latest_date = latest[0]

        sectors = list(SECTOR_GROUPS.keys())
        results = []
        for sector in sectors:
            data = self.con.execute("""
                SELECT period_end, total_return_pct FROM sector_trends
                WHERE sector = ? AND period = 'weekly' AND period_end <= ?
                ORDER BY period_end DESC LIMIT 8
            """, [sector, latest_date]).fetchdf()
            if len(data) < 3:
                continue
            recent = float(data["total_return_pct"].head(min(3, len(data))).mean())
            older = float(data["total_return_pct"].tail(min(3, len(data))).mean())
            momentum = round(recent - older, 2)
            results.append({"sector": sector, "momentum": momentum, "recent_avg": round(recent, 2)})

        results.sort(key=lambda x: x["momentum"], reverse=True)
        if len(results) >= 2:
            rotation = {
                "date": latest_date,
                "leading_sector": results[0]["sector"],
                "lagging_sector": results[-1]["sector"],
                "rotation_type": "risk_on" if results[0]["momentum"] > 1 else "risk_off",
                "strength": round(float(results[0]["momentum"]) - float(results[-1]["momentum"]), 2),
                "details": f"Top: {results[0]['sector']}(+{results[0]['momentum']:.2f}) → Bottom: {results[-1]['sector']}({results[-1]['momentum']:.2f})",
            }
            rotation["rankings"] = [{"sector": r["sector"], "momentum": round(r["momentum"], 2)} for r in results]
            try:
                from memory_manager import MemoryManager
                mm = MemoryManager(
                    self.db_path.replace("market", "memory").replace(".duckdb", "memory.duckdb")
                    if "market" in self.db_path else None
                )
                mm.store_fact("ALL", "sector_rotation",
                    f"Sector rotation: {results[0]['sector']} leading, {results[-1]['sector']} lagging. "
                    f"Top 3: {', '.join(r['sector'] for r in results[:3])}. "
                    f"Bottom 3: {', '.join(r['sector'] for r in results[-3:])}.",
                    confidence=0.8, source="sector_analyzer")
                mm.close()
            except:
                pass
            return rotation
        return []

    def get_sector_trends(self, period="weekly", top_n=5):
        """Get best/worst performing sectors for a given period (weekly/monthly)."""
        latest = self.con.execute("""
            SELECT period_end FROM sector_trends
            WHERE period = ? ORDER BY period_end DESC LIMIT 1
        """, [period]).fetchone()
        if not latest:
            return {"period": period, "error": "No data"}
        date = latest[0]
        data = self.con.execute("""
            SELECT sector, avg_change_pct, total_return_pct, stocks_up, stocks_down, total_stocks
            FROM sector_trends WHERE period = ? AND period_end = ?
            ORDER BY total_return_pct DESC
        """, [period, date]).fetchdf()
        return {
            "period": period,
            "date": str(date),
            "sectors": [
                {"name": r["sector"], "avg_change_pct": float(r["avg_change_pct"]),
                 "total_return_pct": float(r["total_return_pct"]),
                 "stocks_up": int(r["stocks_up"]), "stocks_down": int(r["stocks_down"]),
                 "total": int(r["total_stocks"])}
                for _, r in data.iterrows()
            ],
        }

    def get_sector_summary(self, as_dict=True):
        """Get current sector performance summary for AI context."""
        latest = self.con.execute("SELECT MAX(date) FROM sector_performance").fetchone()[0]
        if latest is None:
            sectors = self.compute_sector_performance()
            if sectors is None or len(sectors) == 0:
                return {} if as_dict else "No sector data available yet."
            latest = sectors["date"].iloc[0]

        data = self.con.execute("""
            SELECT sector, avg_change_pct, stocks_up, stocks_down, total_stocks, volume_ratio
            FROM sector_performance WHERE date = ?
            ORDER BY avg_change_pct DESC
        """, [latest]).fetchdf()

        if len(data) == 0:
            return {} if as_dict else "No sector data available yet."

        if as_dict:
            # Get latest weekly and monthly performance for each sector
            week_data = {}
            month_data = {}
            try:
                wrows = self.con.execute("""
                    SELECT sector, total_return_pct FROM sector_trends
                    WHERE period = 'weekly' AND period_end = (
                        SELECT MAX(period_end) FROM sector_trends WHERE period = 'weekly'
                    )
                """).fetchall()
                for s, v in wrows:
                    week_data[s] = round(float(v), 2)
            except:
                pass
            try:
                mrows = self.con.execute("""
                    SELECT sector, total_return_pct FROM sector_trends
                    WHERE period = 'monthly' AND period_end = (
                        SELECT MAX(period_end) FROM sector_trends WHERE period = 'monthly'
                    )
                """).fetchall()
                for s, v in mrows:
                    month_data[s] = round(float(v), 2)
            except:
                pass

            return {
                "date": str(latest),
                "sectors": [
                    {
                        "name": r["sector"],
                        "change_pct": round(float(r["avg_change_pct"]), 2),
                        "week_change_pct": week_data.get(r["sector"]),
                        "month_change_pct": month_data.get(r["sector"]),
                        "stocks_up": int(r["stocks_up"]),
                        "stocks_down": int(r["stocks_down"]),
                        "total": int(r["total_stocks"]),
                        "volume_ratio": round(float(r["volume_ratio"]), 2) if r["volume_ratio"] is not None else None,
                    }
                    for _, r in data.iterrows()
                ]
            }
        lines = [f"Sector Performance ({latest}):"]
        for _, r in data.iterrows():
            arrow = "+" if r["avg_change_pct"] > 0 else "-"
            lines.append(f"  {arrow} {r['sector']}: {r['avg_change_pct']:+.2f}% ({r['stocks_up']}/{r['stocks_down']} up/down)")
        return "\n".join(lines)

    def close(self):
        try:
            self.con.close()
        except:
            pass


def get_sector_for_ticker(ticker):
    return SECTOR_MAP.get(ticker.upper(), "Other")

def list_sectors():
    return sorted(SECTOR_GROUPS.keys())

def list_tickers_in_sector(sector):
    return SECTOR_GROUPS.get(sector, [])
