import asyncio, json, os, io, logging, time
from pathlib import Path
import httpx
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger("dhan_client")
DATA_DIR = Path("/Volumes/Untitled/market_data")
DOWNLOAD_DIR = DATA_DIR / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

class AuthExpiredError(Exception):
    pass

class MaxRetryError(Exception):
    pass

class RateLimitError(Exception):
    pass

class DhanClient:
    BASE = "https://api.dhan.co/v2"
    COMPACT_CSV = "https://images.dhan.co/api-data/api-scrip-master.csv"

    SEGMENT_MAP = {
        "NSE_EQ": "NSE",
        "IDX_I": "NSE",
        "NSE_FNO": "NSE",
    }

    def __init__(self, client_id=None, access_token=None):
        if client_id and access_token:
            self._client_id = client_id
            self._token = access_token
        else:
            from config_manager import get_config
            cfg = get_config()
            self._client_id = client_id or cfg.get("dhan_api.client_id")
            self._token = access_token or cfg.get("dhan_api.access_token")
        if not self._client_id or not self._token:
            raise RuntimeError(
                "Dhan credentials are not configured. Set DHAN_CLIENT_ID and "
                "DHAN_ACCESS_TOKEN in the environment, or provide client_id and "
                "access_token explicitly. Credentials are never hardcoded."
            )
        self.headers = {
            "access-token": self._token,
            "client-id": self._client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._semaphore = asyncio.Semaphore(3)
        self._last_request = 0

    @property
    def CLIENT_ID(self):
        return self._client_id

    @property
    def ACCESS_TOKEN(self):
        return self._token

    async def _throttle(self):
        now = time.monotonic()
        gap = 0.25 - (now - self._last_request)
        if gap > 0:
            await asyncio.sleep(gap)
        self._last_request = time.monotonic()

    async def _request(self, method, path, payload=None, retries=3):
        url = f"{self.BASE}{path}"
        async with self._semaphore:
            for attempt in range(retries):
                await self._throttle()
                try:
                    async with httpx.AsyncClient(timeout=30) as client:
                        if method == "GET":
                            resp = await client.get(url, headers=self.headers, params=payload)
                        else:
                            if payload is not None:
                                payload["dhanClientId"] = self._client_id
                            resp = await client.post(url, headers=self.headers, json=payload)

                    if resp.status_code == 429:
                        wait = min(5 * (2 ** attempt), 60)
                        logger.warning(f"Rate limited (429), waiting {wait}s (attempt {attempt+1})")
                        await asyncio.sleep(wait)
                        continue

                    if resp.status_code >= 500:
                        if attempt < retries - 1:
                            wait = min(2 ** attempt, 30)
                            logger.warning(f"Server error {resp.status_code}, retrying in {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        raise MaxRetryError(f"Server error after {retries} attempts: {resp.status_code}")

                    if resp.status_code == 401:
                        raise AuthExpiredError("Dhan API token expired or invalid")

                    data = resp.json()
                    error_code = data.get("errorCode", "")
                    if error_code in ("807", "808", "809", "810"):
                        raise AuthExpiredError(f"Auth error: {error_code}")
                    if error_code in ("DH-904", "805"):
                        if attempt < retries - 1:
                            await asyncio.sleep(5 * (2 ** attempt))
                            continue
                    if error_code == "813":
                        raise ValueError(f"Invalid security ID in request")
                    if data.get("status") == "error":
                        logger.error(f"Dhan API error: {data}")
                        if attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise MaxRetryError(f"API error: {data}")

                    return data

                except httpx.TimeoutException:
                    if attempt == retries - 1:
                        raise MaxRetryError(f"Timeout after {retries} attempts")
                    await asyncio.sleep(2 ** attempt)
                except httpx.ConnectError:
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)

    async def fetch_security_master(self, force=False):
        """Download and cache security master CSV"""
        cache_path = DOWNLOAD_DIR / "security_master.csv"
        if cache_path.exists() and not force:
            age = time.time() - cache_path.stat().st_mtime
            if age < 86400:
                return pd.read_csv(cache_path, low_memory=False)

        logger.info("Downloading security master CSV...")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(self.COMPACT_CSV)
        df = pd.read_csv(io.BytesIO(resp.content), low_memory=False)
        df.to_csv(cache_path, index=False)
        logger.info(f"Security master: {len(df)} rows")
        return df

    async def historical_daily(self, security_id, segment, instrument, from_date, to_date):
        """Get daily OHLCV data"""
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": segment,
            "instrument": instrument,
            "expiryCode": 0,
            "oi": True if instrument in ("FUTIDX", "OPTIDX", "FUTSTK", "OPTSTK") else False,
            "fromDate": from_date,
            "toDate": to_date,
        }
        return await self._request("POST", "/charts/historical", payload)

    async def intraday_minute(self, security_id, segment, instrument, from_date, to_date, interval=5):
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": segment,
            "instrument": instrument,
            "expiryCode": 0,
            "fromDate": from_date,
            "toDate": to_date,
            "interval": str(interval),
        }
        return await self._request("POST", "/charts/intraday", payload)

    async def expired_options(self, security_id, exchange_segment, instrument, interval,
                               expiry_flag, expiry_code, strike, opt_type, from_date, to_date,
                               required_data=None):
        """Use dhanhq SDK's expired_options_data which validates params correctly."""
        if required_data is None:
            required_data = ["open", "high", "low", "close", "oi", "iv", "volume", "strike", "spot"]
        # dhanhq is synchronous; run in thread to avoid blocking
        import functools
        from dhanhq import DhanContext, dhanhq
        ctx = DhanContext(self._client_id, self._token)
        dh = dhanhq(ctx)
        fn = functools.partial(
            dh.expired_options_data,
            str(security_id), exchange_segment, instrument,
            expiry_flag, expiry_code, strike, opt_type,
            required_data, from_date, to_date, interval,
        )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn)

    async def get_fund_limits(self):
        return await self._request("GET", "/fundlimit")

    async def get_ltp(self, instruments):
        """Get last traded price for up to 1000 instruments.
        instruments: list of dicts [{"securityId": str, "exchangeSegment": str, "instrument": str}]"""
        return await self._request("POST", "/marketfeed/ltp", {"instruments": instruments})

    async def get_quote(self, instruments):
        """Get full quote with market depth for up to 1000 instruments."""
        return await self._request("POST", "/marketfeed/quote", {"instruments": instruments})

    async def get_ohlc(self, instruments):
        """Get OHLC data for up to 1000 instruments."""
        return await self._request("POST", "/marketfeed/ohlc", {"instruments": instruments})

    async def get_holdings(self):
        """Get demat holdings."""
        return await self._request("GET", "/holdings")

    async def get_positions_book(self):
        """Get server-side position book."""
        return await self._request("GET", "/positions")

    async def get_order_book(self):
        """Get all orders for today."""
        return await self._request("GET", "/orders")

    async def get_trade_book(self):
        """Get executed trade history."""
        return await self._request("GET", "/trades")

    async def cancel_order(self, order_id):
        """Cancel a pending order."""
        return await self._request("DELETE", f"/orders/{order_id}")

    async def modify_order(self, order_id, quantity, price, trigger_price=None):
        """Modify a pending order."""
        payload = {"orderId": order_id, "quantity": quantity, "price": price}
        if trigger_price is not None:
            payload["triggerPrice"] = trigger_price
        return await self._request("PUT", f"/orders/{order_id}", payload)

    async def get_live_prices(self, security_ids, segment="NSE_EQ", instrument="EQUITY"):
        """Get LTP for a list of security IDs. Returns dict of {security_id: ltp}."""
        instruments = [{"securityId": str(sid), "exchangeSegment": segment, "instrument": instrument}
                       for sid in security_ids]
        resp = await self.get_ltp(instruments)
        result = {}
        if isinstance(resp, dict):
            data = resp.get("data") or resp
            for item in data if isinstance(data, list) else []:
                if isinstance(item, dict):
                    result[item.get("securityId")] = item.get("ltp") or item.get("close")
        return result

    async def place_super_order(self, symbol, exchange_segment, product_type, order_type,
                                 quantity, price, trigger_price, sl_price, tp_price,
                                 validity="DAY", amo="NO"):
        """Place a bracket order (entry + SL + TP in one request)."""
        payload = {
            "tradingSymbol": symbol,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "validity": validity,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price or "",
            "amo": amo,
            "afterMarketOrder": amo,
            "boProfitValue": tp_price,
            "boLossValue": sl_price,
        }
        return await self._request("POST", "/super/orders", payload)

    async def modify_super_order(self, order_id, quantity, price, sl_price, tp_price):
        """Modify an existing super order."""
        payload = {
            "orderId": order_id,
            "quantity": quantity,
            "price": price,
            "boProfitValue": tp_price,
            "boLossValue": sl_price,
        }
        return await self._request("PUT", f"/super/orders/{order_id}", payload)

    async def cancel_super_order(self, order_id, leg=None):
        """Cancel a super order (or one leg)."""
        path = f"/super/orders/{order_id}"
        if leg:
            path += f"/{leg}"
        return await self._request("DELETE", path)

    async def get_super_orders(self):
        """List all super orders."""
        return await self._request("GET", "/super/orders")

    async def get_security_master_df(self, force=False):
        df = await self.fetch_security_master(force)
        return df

    def get_nifty50_constituents(self, master_df):
        """Extract Nifty 50 constituent stocks from security master"""
        nse_eq = master_df[
            (master_df["SEM_EXM_EXCH_ID"] == "NSE") &
            (master_df["SEM_SEGMENT"] == "E") &
            (master_df["SEM_INSTRUMENT_NAME"] == "EQUITY") &
            (master_df["SEM_SERIES"] == "EQ")
        ]
        symbols = [
            "RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "INFY", "SBIN",
            "BHARTIARTL", "BAJFINANCE", "HINDUNILVR", "ITC", "KOTAKBANK",
            "LT", "WIPRO", "AXISBANK", "TATASTEEL", "ONGC", "NTPC", "M&M",
            "MARUTI", "ADANIENT", "TITAN", "SUNPHARMA", "ULTRACEMCO",
            "ASIANPAINT", "BAJAJFINSV", "HCLTECH", "POWERGRID", "NESTLEIND",
            "JSWSTEEL", "GRASIM", "DRREDDY", "EICHERMOT", "COALINDIA",
            "BRITANNIA", "BPCL", "HINDALCO", "ADANIPORTS", "CIPLA",
            "SBILIFE", "BAJAJ-AUTO", "INDUSINDBK", "DIVISLAB", "APOLLOHOSP",
            "TECHM", "TRENT", "SHRIRAMFIN", "HEROMOTOCO", "BEL",
        ]
        constituents = {}
        for sym in symbols:
            row = nse_eq[nse_eq["SEM_TRADING_SYMBOL"] == sym]
            if len(row) > 0:
                constituents[sym] = {
                    "security_id": str(row.iloc[0]["SEM_SMST_SECURITY_ID"]),
                    "lot_size": int(row.iloc[0]["SEM_LOT_UNITS"]),
                    "name": row.iloc[0].get("SEM_CUSTOM_SYMBOL", sym),
                }
        return constituents

    async def verify_connection(self):
        """Test if API credentials work"""
        try:
            r = await self.get_fund_limits()
            if "dhanClientId" in r:
                balance = r.get("availabelBalance", 0)
                logger.info(f"Dhan API OK. Balance: ₹{balance}")
                return True, balance
            if r.get("status") == "success":
                balance = r.get("data", {}).get("availabelBalance", 0)
                logger.info(f"Dhan API OK. Balance: ₹{balance}")
                return True, balance
            return False, str(r)
        except AuthExpiredError:
            return False, "Token expired"
        except Exception as e:
            return False, str(e)

    async def download_historical_data(self, security_id, segment, instrument, from_date, to_date):
        """Download with caching to avoid repeat downloads"""
        cache_key = f"{security_id}_{segment}_{instrument}_{from_date}_{to_date}"
        cache_file = DOWNLOAD_DIR / f"hist_{cache_key.replace('/', '_')}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        data = await self.historical_daily(security_id, segment, instrument, from_date, to_date)
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data

    def parse_historical_response(self, response):
        """Convert Dhan API response to pandas DataFrame"""
        if response.get("status") != "success":
            return None
        data = response.get("data", {})
        if not data or "timestamp" not in data:
            return None
        df = pd.DataFrame({
            "timestamp": [self._dhan_ts_to_dt(ts) for ts in data["timestamp"]],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data.get("volume", [0] * len(data["timestamp"])),
        })
        if "open_interest" in data:
            df["open_interest"] = data["open_interest"]
        return df

    @staticmethod
    def _dhan_ts_to_dt(ts):
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000).date()
        return ts

    @staticmethod
    def _dhan_intra_ts(ts):
        if isinstance(ts, (int, float)):
            # Expired options returns seconds; intraday returns milliseconds
            if ts > 1e12:
                ts = ts / 1000
            return datetime.fromtimestamp(ts)
        return ts

    def parse_intraday_response(self, response):
        if not isinstance(response, dict) or response.get("status") != "success":
            return None
        data = response.get("data", {})
        if not data or "timestamp" not in data:
            return None
        df = pd.DataFrame({
            "timestamp": [self._dhan_intra_ts(ts) for ts in data["timestamp"]],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data.get("volume", [0] * len(data["timestamp"])),
        })
        return df

    def parse_expired_options(self, response):
        if not isinstance(response, dict):
            return None
        if response.get("status") != "success":
            return None
        inner = response.get("data", {})
        if isinstance(inner, dict) and "data" in inner:
            inner = inner["data"]
        if not isinstance(inner, dict):
            return None
        ce = inner.get("ce") or {}
        pe = inner.get("pe") or {}
        result = []
        for opt_type_raw, opt_data in [("CE", ce), ("PE", pe)]:
            opt_type = "CALL" if opt_type_raw == "CE" else "PUT"
            if not opt_data or not opt_data.get("timestamp"):
                continue
            ts_list = opt_data["timestamp"]
            for i in range(len(ts_list)):
                result.append({
                    "timestamp": self._dhan_intra_ts(ts_list[i]),
                    "option_type": opt_type,
                    "open": opt_data["open"][i] if opt_data.get("open") and i < len(opt_data["open"]) else 0,
                    "high": opt_data["high"][i] if opt_data.get("high") and i < len(opt_data["high"]) else 0,
                    "low": opt_data["low"][i] if opt_data.get("low") and i < len(opt_data["low"]) else 0,
                    "close": opt_data["close"][i] if opt_data.get("close") and i < len(opt_data["close"]) else 0,
                    "volume": opt_data["volume"][i] if opt_data.get("volume") and i < len(opt_data["volume"]) else 0,
                    "oi": opt_data["oi"][i] if opt_data.get("oi") and i < len(opt_data["oi"]) else 0,
                    "iv": opt_data["iv"][i] if opt_data.get("iv") and i < len(opt_data["iv"]) else 0,
                    "strike": opt_data["strike"][i] if opt_data.get("strike") and i < len(opt_data["strike"]) else 0,
                    "spot": opt_data["spot"][i] if opt_data.get("spot") and i < len(opt_data["spot"]) else 0,
                })
        return pd.DataFrame(result) if result else None
