"""Tests for DataManager using a temporary DuckDB."""
import pytest
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_manager import DataManager

@pytest.fixture
def dm():
    path = os.path.join(tempfile.mkdtemp(), "test.duckdb")
    db = DataManager(path)
    yield db
    db.close()
    try:
        import shutil; shutil.rmtree(os.path.dirname(path))
    except: pass

class TestDataManager:
    def test_schema_init(self, dm):
        cols = dm.con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='securities'").fetchall()
        names = [c[0] for c in cols]
        assert "security_id" in names
        assert "symbol" in names
        assert "is_index" in names

    def test_store_and_get_securities(self, dm):
        dm.store_securities({"TEST": {"security_id": "99", "name": "Test Corp",
                             "segment": "NSE", "instrument_type": "EQUITY", "lot_size": 1,
                             "is_index": False, "is_active": True}})
        sid = dm.get_security_id("TEST")
        assert sid == "99"

    def test_get_security_id_nonexistent(self, dm):
        sid = dm.get_security_id("NONEXISTENT")
        assert sid is None

    def test_get_all_securities(self, dm):
        dm.store_securities({"TEST": {"security_id": "99", "name": "Test Corp",
                             "segment": "NSE", "instrument_type": "EQUITY", "lot_size": 1,
                             "is_index": False, "is_active": True}})
        df = dm.get_all_securities()
        assert len(df) >= 1
        assert "symbol" in df.columns

    def test_store_daily_data(self, dm):
        dm.store_securities({"TEST": {"security_id": "99", "name": "Test Corp",
                             "segment": "NSE", "instrument_type": "EQUITY", "lot_size": 1,
                             "is_index": False, "is_active": True}})
        import pandas as pd
        df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-01", "2026-07-02"]), "close": [100.0, 102.0],
                           "high": [101, 103], "low": [99, 101], "open": [100, 102],
                           "volume": [1000, 1500]})
        dm.store_daily("99", df)
        rows = dm.get_daily("99", limit=10)
        assert len(rows) == 2

    def test_get_download_progress(self, dm):
        p = dm.get_download_progress()
        assert isinstance(p, dict)
        assert "securities_in_db" in p or "total_securities" in p

    def test_latest_date_nodata(self, dm):
        d = dm.con.execute("SELECT MAX(date) FROM daily").fetchone()[0]
        assert d is None

    def test_get_significant_movers_empty(self, dm):
        df = dm.get_significant_movers(threshold_pct=1.0)
        assert len(df) == 0
