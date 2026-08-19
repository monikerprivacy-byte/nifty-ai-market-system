"""Data Quality Checker — validates downloaded market data integrity.
Checks: missing candles, duplicate rows, holiday gaps, volume anomalies."""

import logging
from datetime import datetime, timedelta, date
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config

logger = logging.getLogger("data_quality")

class DataQualityChecker:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")

    def check_security(self, ticker):
        """Full quality check for one security. Returns dict of issues."""
        results = {
            "ticker": ticker,
            "passed": True,
            "total_rows": 0,
            "date_range": {"first": None, "last": None},
            "issues": [],
            "metrics": {},
        }
        try:
            con = duckdb.connect(self.db_path)
            sid = con.execute("SELECT security_id FROM securities WHERE symbol = ?", [ticker]).fetchone()
            if not sid:
                results["passed"] = False
                results["issues"].append({"type": "not_found", "detail": f"Ticker {ticker} not found"})
                con.close()
                return results
            df = con.execute("""
                SELECT date as timestamp, open, high, low, close, volume
                FROM daily WHERE security_id = ?
                ORDER BY date
            """, [sid[0]]).fetchdf()
            con.close()
        except Exception as e:
            results["passed"] = False
            results["issues"].append({"type": "db_error", "detail": str(e)})
            return results

        if len(df) == 0:
            results["passed"] = False
            results["issues"].append({"type": "no_data", "detail": "No data rows found"})
            return results

        results["total_rows"] = len(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        results["date_range"]["first"] = str(df["timestamp"].min().date())
        results["date_range"]["last"] = str(df["timestamp"].max().date())

        # 1. Duplicate timestamps
        dups = df[df.duplicated(subset=["timestamp"], keep=False)]
        if len(dups) > 0:
            dup_dates = dups["timestamp"].dt.date.unique().tolist()
            results["issues"].append({
                "type": "duplicates",
                "count": len(dups),
                "detail": f"{len(dups)} duplicate rows on {len(dup_dates)} dates: {[str(d) for d in dup_dates[:5]]}",
            })
            results["passed"] = False

        # 2. Missing weekdays (gaps)
        dates_set = set(df["timestamp"].dt.date)
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        all_weekdays = []
        d = min_date
        while d <= max_date:
            if d.weekday() < 5:
                all_weekdays.append(d)
            d += timedelta(days=1)

        missing = [d for d in all_weekdays if d not in dates_set]
        # Indian market holidays (major NSE holidays per year)
        HOLIDAYS = {
            "2020-02-21", "2020-03-10", "2020-04-02", "2020-04-06", "2020-04-10",
            "2020-04-14", "2020-05-01", "2020-05-25", "2020-10-02", "2020-10-26",
            "2020-11-16", "2020-11-30", "2020-12-25",
            "2021-01-26", "2021-03-11", "2021-03-29", "2021-04-02", "2021-04-14",
            "2021-04-21", "2021-05-26", "2021-07-21", "2021-08-19", "2021-10-15",
            "2021-11-05", "2021-11-19", "2021-12-25",
            "2022-01-26", "2022-03-01", "2022-03-18", "2022-04-14", "2022-08-15",
            "2022-08-31", "2022-10-05", "2022-10-26", "2022-11-08",
            "2023-01-26", "2023-03-07", "2023-03-30", "2023-04-04", "2023-04-14",
            "2023-06-28", "2023-08-15", "2023-09-19", "2023-10-24", "2023-11-14",
            "2023-12-25",
            "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29", "2024-04-11",
            "2024-04-17", "2024-05-01", "2024-06-17", "2024-07-17", "2024-08-15",
            "2024-10-01", "2024-10-02", "2024-11-01", "2024-11-15", "2024-12-25",
            "2025-01-29", "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10",
            "2025-04-14", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
            "2025-10-22", "2025-11-05",
            "2026-01-26", "2026-02-17", "2026-03-06", "2026-03-27", "2026-03-30",
            "2026-03-31", "2026-04-02", "2026-04-03", "2026-04-06", "2026-05-01",
            "2026-05-28", "2026-06-26",
        }
        missing = [d for d in missing if d.isoformat() not in HOLIDAYS]
        if len(missing) > 0:
            # Group consecutive missing days into gaps
            gaps = []
            gap_start = missing[0]
            gap_end = missing[0]
            for d in missing[1:]:
                if d == gap_end + timedelta(days=1):
                    gap_end = d
                elif d == gap_end + timedelta(days=3) and gap_end.weekday() == 4 and d.weekday() == 0:
                    gap_end = d
                else:
                    gaps.append((gap_start, gap_end))
                    gap_start = d
                    gap_end = d
            gaps.append((gap_start, gap_end))

            gap_strs = []
            for gs, ge in gaps:
                if gs == ge:
                    gap_strs.append(str(gs))
                else:
                    gap_strs.append(f"{gs} to {ge}")
            results["issues"].append({
                "type": "missing_dates",
                "count": len(missing),
                "detail": f"{len(missing)} missing weekdays in {len(gaps)} gaps: {', '.join(gap_strs[:10])}",
            })
            if len(missing) > 20:
                results["passed"] = False

        # 3. Zero volume (suspicious)
        zero_vol = df[df["volume"] == 0]
        if len(zero_vol) > 0:
            dates = zero_vol["timestamp"].dt.date.unique().tolist()
            results["issues"].append({
                "type": "zero_volume",
                "count": len(zero_vol),
                "detail": f"{len(zero_vol)} rows with zero volume on {[str(d) for d in dates[:5]]}",
            })

        # 4. OHLC consistency: high >= low, close within range
        bad_hl = df[df["high"] < df["low"]]
        if len(bad_hl) > 0:
            results["issues"].append({
                "type": "ohlc_inconsistent",
                "count": len(bad_hl),
                "detail": f"{len(bad_hl)} rows where high < low",
            })
            results["passed"] = False

        bad_close = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
        if len(bad_close) > 0:
            results["issues"].append({
                "type": "close_out_of_range",
                "count": len(bad_close),
                "detail": f"{len(bad_close)} rows where close is outside high-low range",
            })

        # 5. Stale price (no change for 5+ consecutive days)
        df["close_pct"] = df["close"].pct_change().abs()
        stale_runs = (df["close_pct"] < 0.001).astype(int)
        max_stale = 0
        current_stale = 0
        for v in stale_runs:
            if v:
                current_stale += 1
                max_stale = max(max_stale, current_stale)
            else:
                current_stale = 0
        if max_stale >= 5:
            results["issues"].append({
                "type": "stale_data",
                "count": max_stale,
                "detail": f"Price unchanged for {max_stale} consecutive days",
            })
            results["passed"] = False

        # Metrics
        results["metrics"] = {
            "avg_volume": int(df["volume"].mean()),
            "avg_daily_range_pct": float(((df["high"] - df["low"]) / df["close"] * 100).mean()),
            "volatility_pct": float(df["close"].pct_change().std() * 100),
            "missing_pct": round(len(missing) / len(all_weekdays) * 100, 1) if all_weekdays else 0,
        }

        return results

    def check_all(self, tickers=None):
        """Run quality check on all (or specified) securities."""
        from data_manager import DataManager
        db = DataManager(self.db_path)
        if tickers is None:
            secs = db.get_all_securities()
            tickers = secs["symbol"].tolist() if len(secs) > 0 else []

        results = {}
        for t in tickers:
            try:
                results[t] = self.check_security(t)
            except Exception as e:
                results[t] = {"ticker": t, "passed": False, "issues": [{"type": "error", "detail": str(e)}]}
        return results

    def summary_report(self, tickers=None):
        """Generate a human-readable quality report."""
        checks = self.check_all(tickers)
        total = len(checks)
        passed = sum(1 for v in checks.values() if v.get("passed"))
        failed = total - passed

        lines = [f"Data Quality Report ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                 f"Securities checked: {total} | Passed: {passed} | Issues: {failed}"]
        for ticker, result in checks.items():
            if not result.get("passed"):
                issue_types = [i["type"] for i in result.get("issues", [])]
                lines.append(f"  ✗ {ticker}: {', '.join(issue_types)}")
        return "\n".join(lines)


def check_and_report():
    """Convenience: run full check and store results in memory."""
    checker = DataQualityChecker()
    report = checker.summary_report()
    logger.info(f"\n{report}")
    return report
