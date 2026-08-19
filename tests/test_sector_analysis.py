"""Tests for SectorAnalyzer — performance, trends, rotation, mapping."""
import pytest
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from sector_analysis import SectorAnalyzer, SECTOR_MAP, SECTOR_GROUPS, get_sector_for_ticker, list_sectors


class TestMapping:
    def test_all_tickers_mapped(self):
        assert len(SECTOR_MAP) == 52, f"Expected 52, got {len(SECTOR_MAP)}"

    def test_no_duplicates(self):
        assert len(SECTOR_MAP) == len(set(SECTOR_MAP.keys()))

    def test_hdfc_removed(self):
        assert "HDFC" not in SECTOR_MAP, "HDFC should be removed (merged with HDFCBANK)"

    def test_sector_groups_built(self):
        assert len(SECTOR_GROUPS) == 13

    def test_get_sector_for_ticker(self):
        assert get_sector_for_ticker("RELIANCE") == "Energy"
        assert get_sector_for_ticker("UNKNOWN") == "Other"

    def test_list_sectors(self):
        sectors = list_sectors()
        assert "IT" in sectors
        assert "Financial Services" in sectors


@pytest.fixture
def sa():
    path = os.path.join(tempfile.mkdtemp(), "test_sector.duckdb")
    db = SectorAnalyzer.__new__(SectorAnalyzer)
    import duckdb
    db.__dict__["db_path"] = path
    db.__dict__["con"] = duckdb.connect(path)
    db.__dict__["con"].execute("""
        CREATE TABLE IF NOT EXISTS securities (
            security_id VARCHAR PRIMARY KEY, symbol VARCHAR, name VARCHAR,
            segment VARCHAR, instrument_type VARCHAR,
            lot_size INTEGER DEFAULT 1, is_index BOOLEAN DEFAULT FALSE
        )
    """)
    db.__dict__["con"].execute("""
        CREATE TABLE IF NOT EXISTS daily (
            security_id VARCHAR, date DATE, open DOUBLE, high DOUBLE,
            low DOUBLE, close DOUBLE, volume BIGINT,
            PRIMARY KEY (security_id, date)
        )
    """)
    db.__dict__["con"].execute("""
        CREATE TABLE IF NOT EXISTS sector_performance (
            sector VARCHAR, date DATE, avg_change_pct FLOAT,
            median_change_pct FLOAT, max_change_pct FLOAT, min_change_pct FLOAT,
            stocks_up INTEGER, stocks_down INTEGER, total_stocks INTEGER,
            volume_ratio FLOAT, PRIMARY KEY (sector, date)
        )
    """)
    db.__dict__["con"].execute("""
        CREATE TABLE IF NOT EXISTS sector_trends (
            sector VARCHAR, period VARCHAR, period_start DATE, period_end DATE,
            avg_change_pct FLOAT, total_return_pct FLOAT,
            stocks_up INTEGER, stocks_down INTEGER, total_stocks INTEGER,
            PRIMARY KEY (sector, period, period_end)
        )
    """)
    yield db
    db.con.close()
    import shutil; shutil.rmtree(os.path.dirname(path), ignore_errors=True)


class TestPerformance:
    def test_insert_performance(self, sa):
        sa.con.execute("""
            INSERT INTO sector_performance (sector, date, avg_change_pct, median_change_pct,
                max_change_pct, min_change_pct, stocks_up, stocks_down, total_stocks, volume_ratio)
            VALUES ('IT', CURRENT_DATE, 1.5, 1.2, 3.0, -0.5, 3, 2, 5, 1.2)
        """)
        sa.con.commit()
        summary = sa.get_sector_summary(as_dict=True)
        assert "sectors" in summary
        assert len(summary["sectors"]) >= 1

    def test_get_sector_summary_empty(self, sa):
        summary = sa.get_sector_summary(as_dict=True)
        assert summary == {}  # No data, nothing computed

    def test_get_sector_summary_string(self, sa):
        sa.con.execute("""
            INSERT INTO sector_performance (sector, date, avg_change_pct, median_change_pct,
                max_change_pct, min_change_pct, stocks_up, stocks_down, total_stocks)
            VALUES ('IT', CURRENT_DATE, 1.5, 1.2, 3.0, -0.5, 3, 2, 5)
        """)
        sa.con.commit()
        text = sa.get_sector_summary(as_dict=False)
        assert isinstance(text, str)
        assert "IT" in text


class TestTrends:
    def test_insert_trend(self, sa):
        import datetime
        sa.con.execute("""
            INSERT INTO sector_trends (sector, period, period_start, period_end,
                avg_change_pct, total_return_pct, stocks_up, stocks_down, total_stocks)
            VALUES ('IT', 'weekly', '2026-06-28', '2026-07-04', 2.5, 3.0, 4, 1, 5)
        """)
        sa.con.commit()
        trends = sa.get_sector_trends("weekly")
        assert trends["period"] == "weekly"
        assert len(trends["sectors"]) >= 1

    def test_get_trends_missing(self, sa):
        trends = sa.get_sector_trends("monthly")
        assert "error" in trends

    def test_get_trends_invalid_period(self, sa):
        trends = sa.get_sector_trends("invalid")
        assert "error" in trends


class TestRotation:
    def test_rotation_empty_data(self, sa):
        rot = sa.detect_rotation()
        assert rot == []  # No trend data

    def test_rotation_with_data(self, sa):
        import datetime
        sectors = ["IT", "Energy", "Financial Services", "FMCG"]
        for sector in sectors:
            for week in range(4):
                d = datetime.date(2026, 7, 4) - datetime.timedelta(weeks=week)
                start = d - datetime.timedelta(days=6)
                sa.con.execute("""
                    INSERT INTO sector_trends (sector, period, period_start, period_end,
                        avg_change_pct, total_return_pct, stocks_up, stocks_down, total_stocks)
                    VALUES (?, 'weekly', ?, ?, ?, ?, ?, ?, ?)
                """, [sector, start, d, 1.0, 1.5, 3, 2, 5])
        sa.con.commit()
        rot = sa.detect_rotation()
        assert isinstance(rot, dict) or rot == []


class TestGetSector:
    def test_get_sector(self, sa):
        assert sa.get_sector("RELIANCE") == "Energy"

    def test_get_sector_unknown(self, sa):
        assert sa.get_sector("UNKNOWN") == "Other"

    def test_get_sector_lowercase(self, sa):
        assert sa.get_sector("reliance") == "Energy"

    def test_get_stocks_in_sector(self, sa):
        stocks = sa.get_stocks_in_sector("IT")
        assert "TCS" in stocks
        assert "INFY" in stocks
        assert "WIPRO" in stocks

    def test_get_stocks_unknown_sector(self, sa):
        assert sa.get_stocks_in_sector("NONEXISTENT") == []
