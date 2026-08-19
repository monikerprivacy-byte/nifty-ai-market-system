import asyncio, logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf
from dhan_client import DhanClient, AuthExpiredError
from data_manager import DataManager

logger = logging.getLogger("downloader")

FIVE_YEARS_AGO = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
TODAY = datetime.now().strftime("%Y-%m-%d")
START_DATE = "2020-01-01"

YF_INDEX_MAP = {
    "13": "^NSEI",     # NIFTY 50
    "25": "^NSEBANK",  # BANK NIFTY
}

class DataDownloader:
    def __init__(self, dhan_client: DhanClient, data_mgr: DataManager):
        self.dhan = dhan_client
        self.db = data_mgr
        self.running = False

    async def download_all(self):
        """Download all data: indices, stocks, F&O"""
        self.running = True
        try:
            ok, msg = await self.dhan.verify_connection()
            if not ok:
                logger.error(f"Dhan API connection failed: {msg}")
                return {"status": "error", "message": msg}

            master = await self.dhan.fetch_security_master()
            constituents = self.dhan.get_nifty50_constituents(master)
            # Add indices
            constituents["NIFTY"] = {
                "security_id": "13", "segment": "IDX_I", "instrument_type": "INDEX",
                "lot_size": 1, "is_index": True, "name": "NIFTY 50",
            }
            constituents["BANKNIFTY"] = {
                "security_id": "25", "segment": "IDX_I", "instrument_type": "INDEX",
                "lot_size": 1, "is_index": True, "name": "BANK NIFTY",
            }
            self.db.store_securities(constituents)

            total = 0
            total += await self._download_index("13", "IDX_I", "INDEX", "NIFTY")
            total += await self._download_index("25", "IDX_I", "INDEX", "BANKNIFTY")
            total += await self._download_stocks(constituents)

            logger.info(f"=== Download complete: {total} rows ===")
            self.running = False
            return {"status": "success", "rows_downloaded": total}
        except AuthExpiredError:
            self.running = False
            return {"status": "error", "message": "Dhan API token expired. Regenerate token in Dhan web console."}
        except Exception as e:
            self.running = False
            logger.exception("Download failed")
            return {"status": "error", "message": str(e)}

    async def download_incremental(self):
        """Download only new data since last download"""
        self.running = True
        try:
            total = 0
            stocks = self.db.get_all_securities()
            for _, row in stocks.iterrows():
                sid = row["security_id"]
                latest = self.db.get_latest_date(sid)
                if latest is None:
                    from_date = START_DATE
                else:
                    from_date = (latest - timedelta(days=5)).strftime("%Y-%m-%d")

                if from_date >= TODAY:
                    continue

                seg = row["segment"]
                itype = row["instrument_type"]
                symbol = row.get("symbol")
                rows = await self._download_one(sid, seg, itype, from_date, TODAY, symbol)
                total += rows
                await asyncio.sleep(0.3)

            logger.info(f"Incremental download: {total} new rows")
            self.running = False
            return {"status": "success", "rows_downloaded": total}
        except AuthExpiredError:
            self.running = False
            return {"status": "error", "message": "Token expired"}
        except Exception as e:
            self.running = False
            logger.exception("Incremental download failed")
            return {"status": "error", "message": str(e)}

    async def _yf_to_df(self, ticker, name):
        """Download from Yahoo Finance and convert to standard DataFrame"""
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, lambda: yf.download(
                ticker, start=START_DATE, end=TODAY, progress=False, auto_adjust=True
            ))
            if data is None or len(data) == 0:
                logger.warning(f"  {name}: yfinance returned 0 rows")
                return None
            # Flatten MultiIndex columns: ('Close', '^NSEI') -> 'close'
            flat = data.copy()
            if isinstance(flat.columns, pd.MultiIndex):
                flat.columns = [c[0].lower() for c in flat.columns]
            else:
                flat.columns = [c.lower() for c in flat.columns]
            if "adj close" in flat.columns:
                flat = flat.drop(columns=["adj close"])
            flat.index.name = "timestamp"
            flat = flat.reset_index()
            flat.insert(0, "timestamp", flat.pop("timestamp"))
            return flat
        except Exception as e:
            logger.warning(f"  {name}: yfinance error - {e}")
            return None

    async def _download_index(self, security_id, segment, instrument, name):
        logger.info(f"Downloading {name} (ID={security_id})...")
        # Try yfinance first
        yf_ticker = YF_INDEX_MAP.get(str(security_id))
        if yf_ticker:
            df = await self._yf_to_df(yf_ticker, name)
            if df is not None and len(df) > 0:
                rows = self.db.store_daily(str(security_id), df)
                if rows:
                    self.db.log_download(security_id, instrument, START_DATE, TODAY, rows, "success")
                    logger.info(f"  {name}: {rows} rows (yfinance)")
                    return rows
        # Fall back to Dhan API
        try:
            resp = await self.dhan.historical_daily(security_id, segment, instrument, START_DATE, TODAY)
            df = self.dhan.parse_historical_response(resp)
            if df is not None and len(df) > 0:
                rows = self.db.store_daily(str(security_id), df)
                self.db.log_download(security_id, instrument, START_DATE, TODAY, rows, "success")
                logger.info(f"  {name}: {rows} rows (Dhan)")
                return rows
            self.db.log_download(security_id, instrument, START_DATE, TODAY, 0, "empty", str(resp))
        except Exception as e:
            self.db.log_download(security_id, instrument, START_DATE, TODAY, 0, "error", str(e))
            logger.warning(f"  {name} failed (Dhan): {e}")
        return 0

    async def _download_stocks(self, constituents):
        total = 0
        for symbol, info in constituents.items():
            if info.get("is_index"):
                continue
            sid = info["security_id"]
            logger.info(f"Downloading {symbol} (ID={sid})...")
            rows = await self._download_one(sid, "NSE_EQ", "EQUITY", START_DATE, TODAY, symbol)
            total += rows
            await asyncio.sleep(0.3)
        return total

    async def _download_one(self, security_id, segment, instrument, from_date, to_date, symbol=None):
        # Try yfinance first (stocks use .NS suffix)
        if symbol:
            yf_ticker = f"{symbol}.NS"
            df = await self._yf_to_df(yf_ticker, symbol)
            if df is not None and len(df) > 0:
                rows = self.db.store_daily(str(security_id), df)
                if rows:
                    self.db.log_download(security_id, instrument, from_date, to_date, rows, "success")
                    return rows
        # Fall back to Dhan API
        try:
            resp = await self.dhan.historical_daily(security_id, segment, instrument, from_date, to_date)
            df = self.dhan.parse_historical_response(resp)
            if df is not None and len(df) > 0:
                rows = self.db.store_daily(str(security_id), df)
                self.db.log_download(security_id, instrument, from_date, to_date, rows, "success")
                return rows
            self.db.log_download(security_id, instrument, from_date, to_date, 0, "empty")
        except Exception as e:
            self.db.log_download(security_id, instrument, from_date, to_date, 0, "error", str(e))
            logger.warning(f"  {security_id}: {e}")
        return 0

    async def download_all_indicators(self):
        """Compute indicators for all securities in database"""
        stocks = self.db.get_all_securities()
        total = 0
        for _, row in stocks.iterrows():
            try:
                count = self.db.compute_all_indicators(row["security_id"])
                if count > 0:
                    logger.info(f"  Indicators for {row['symbol']}: {count} rows")
                    total += 1
            except Exception as e:
                logger.warning(f"  Indicators failed for {row['symbol']}: {e}")
        logger.info(f"Indicators computed for {total} securities")
        return total

    async def download_all_intraday(self, intervals=(15, 60)):
        """Download intraday minute data for all securities."""
        from datetime import datetime as _dt, timedelta as _td
        self.running = True
        try:
            ok, _ = await self.dhan.verify_connection()
            if not ok:
                logger.warning("Intraday: Dhan API unavailable, skipping")
                return {"status": "skipped", "reason": "Dhan unavailable"}

            stocks = self.db.get_all_securities()
            total = 0

            for interval in intervals:
                for _, row in stocks.iterrows():
                    sid = row["security_id"]
                    seg = row["segment"]
                    itype = row["instrument_type"]
                    symbol = row.get("symbol", sid)
                    try:
                        latest_ts = self.db.get_latest_intraday_ts(sid)
                        if latest_ts:
                            from_dt = latest_ts - _td(hours=4)
                        else:
                            from_dt = _dt.now() - _td(days=5)
                        to_dt = _dt.now()
                        from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
                        to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")
                        resp = await self.dhan.intraday_minute(sid, seg, itype, from_str, to_str, interval)
                        df = self.dhan.parse_intraday_response(resp)
                        rows = self.db.store_intraday(str(sid), df)
                        if rows:
                            total += rows
                            logger.info(f"  Intraday {symbol} ({interval}m): {rows} rows")
                        await asyncio.sleep(0.3)
                    except Exception as e:
                        logger.debug(f"  Intraday {symbol} ({interval}m): {e}")
            logger.info(f"Intraday download: {total} rows across {len(intervals)} intervals")
            self.running = False
            return {"status": "success", "rows_downloaded": total}
        except Exception as e:
            self.running = False
            logger.exception("Intraday download failed")
            return {"status": "error", "message": str(e)}

    async def download_expired_options(self, underlying, security_id, segment, instrument,
                                        expiry_flag="WEEK", expiry_code=1,
                                        strike="ATM", interval=5,
                                        from_date=None, to_date=None):
        """Download historical expired options data for a specific strike.

        Args:
            strike: Relative strike position - "ATM", "ATM+1", "ATM-1", "ATM+2", etc.
            expiry_code: 0=nearest expired, 1=previous, etc.
            interval: 1, 5, 15, 25, or 60 minutes.
        """
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        total = 0
        for opt_type in ["CALL", "PUT"]:
            try:
                resp = await self.dhan.expired_options(
                    security_id, segment, instrument, interval,
                    expiry_flag, expiry_code, strike, opt_type,
                    from_date, to_date,
                )
                df = self.dhan.parse_expired_options(resp)
                if df is not None and len(df) > 0:
                    rows = self.db.store_options_history(df, underlying, expiry_flag, strike)
                    total += rows
                    logger.info(f"  ExpOpt {underlying} {opt_type} x{strike}: {rows} rows")
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning(f"  ExpOpt {underlying} {opt_type} x{strike} failed: {e}")
        return total
