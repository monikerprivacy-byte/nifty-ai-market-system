import json, logging, threading, time
from collections import deque
from datetime import datetime
from typing import Optional

logger = logging.getLogger("dhan_feed")

NIFTY50 = {
    "RELIANCE": "2885", "HDFCBANK": "1333", "ICICIBANK": "4963", "TCS": "11536",
    "INFY": "1594", "ITC": "1660", "HCLTECH": "7229", "BHARTIARTL": "10604",
    "LT": "11483", "SBIN": "3045", "BAJFINANCE": "317", "KOTAKBANK": "1922",
    "AXISBANK": "5900", "TITAN": "3506", "M&M": "2031", "MARUTI": "10999",
    "SUNPHARMA": "3351", "NTPC": "11630", "WIPRO": "3787", "ULTRACEMCO": "11532",
    "ASIANPAINT": "236", "ONGC": "2475", "POWERGRID": "14977", "ADANIPORTS": "15083",
    "TATASTEEL": "3499", "BAJAJFINSV": "16675", "JSWSTEEL": "11723", "HINDUNILVR": "1394",
    "DRREDDY": "881", "NESTLEIND": "17963", "DIVISLAB": "10940", "APOLLOHOSP": "157",
    "HEROMOTOCO": "1348", "CIPLA": "694", "TECHM": "13538", "COALINDIA": "20374",
    "EICHERMOT": "910", "BAJAJ_AUTO": "16669", "BEL": "383",
    "TRENT": "1964", "HINDALCO": "1363", "BRITANNIA": "547", "SBILIFE": "21808",
    "BPCL": "526", "GRASIM": "1232", "INDUSINDBK": "5258", "SHRIRAMFIN": "4306",
}

INDICES = {"NIFTY": "13", "BANKNIFTY": "25"}


class LiveFeed:
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._feed = None
        self.latest = {}
        self.tick_history = deque(maxlen=10000)
        self._lock = threading.Lock()

    def _build_instruments(self):
        instruments = []
        for sym, sid in NIFTY50.items():
            instruments.append((1, sid, 15))
        for sym, sid in INDICES.items():
            instruments.append((0, sid, 17))
        return instruments

    def _on_connect(self, *args):
        logger.info("LiveFeed connected")

    def _on_message(self, *args):
        raw = args[-1] if args else {}
        try:
            sec_id = str(raw.get("security_id", ""))
            ts = int(time.time())
            tick = {
                "security_id": sec_id,
                "ltp": raw.get("ltp") or raw.get("last_price") or raw.get("close", 0),
                "volume": raw.get("volume", 0),
                "high": raw.get("high", 0),
                "low": raw.get("low", 0),
                "open": raw.get("open", 0),
                "close": raw.get("close", 0),
                "oi": raw.get("oi", 0),
                "timestamp": ts,
            }
            if tick["ltp"] and float(tick["ltp"]) > 0:
                with self._lock:
                    self.latest[sec_id] = tick
                    self.tick_history.append(tick)
        except Exception as e:
            logger.debug(f"Feed parse: {e}")

    def _on_error(self, *args):
        logger.error(f"LiveFeed error: {args}")

    def _on_close(self, *args):
        logger.warning("LiveFeed disconnected")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("LiveFeed started")

    def stop(self):
        self._running = False
        if self._feed:
            try:
                self._feed.close_connection()
            except:
                pass

    def _run(self):
        try:
            from dhanhq import DhanContext, MarketFeed
            ctx = DhanContext(self.client_id, self.access_token)
            instruments = self._build_instruments()
            self._feed = MarketFeed(
                ctx, instruments, version="v2",
                on_connect=self._on_connect,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._feed.run_forever()
        except Exception as e:
            logger.error(f"LiveFeed thread: {e}")
            self._running = False

    def get_ltp(self, symbol: str) -> Optional[float]:
        sid = NIFTY50.get(symbol) or INDICES.get(symbol)
        with self._lock:
            t = self.latest.get(sid) if sid else None
        return float(t["ltp"]) if t and t.get("ltp") else None

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        sid = NIFTY50.get(symbol) or INDICES.get(symbol)
        with self._lock:
            t = self.latest.get(sid) if sid else None
        return dict(t) if t else None

    def get_all_prices(self) -> dict:
        result = {}
        with self._lock:
            for sym, sid in {**NIFTY50, **INDICES}.items():
                t = self.latest.get(sid)
                if t and t.get("ltp"):
                    result[sym] = float(t["ltp"])
        return result

    def get_ltp(self, symbol: str) -> Optional[float]:
        sid = NIFTY50.get(symbol) or INDICES.get(symbol)
        with self._lock:
            t = self.latest.get(sid) if sid else None
        return float(t["ltp"]) if t and t.get("ltp") else None

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        sid = NIFTY50.get(symbol) or INDICES.get(symbol)
        with self._lock:
            t = self.latest.get(sid) if sid else None
        return dict(t) if t else None

    @property
    def is_connected(self) -> bool:
        return self._running and self._feed is not None


_live_feed_instance: Optional[LiveFeed] = None


def get_live_feed(client_id: str = None, access_token: str = None) -> LiveFeed:
    global _live_feed_instance
    if _live_feed_instance is None:
        from config_manager import get_config
        cfg = get_config()
        cid = client_id or cfg.get("dhan_api", {}).get("client_id")
        tok = access_token or cfg.get("dhan_api", {}).get("access_token")
        if not cid or not tok:
            raise RuntimeError(
                "Dhan credentials are not configured. Set DHAN_CLIENT_ID and "
                "DHAN_ACCESS_TOKEN in the environment."
            )
        _live_feed_instance = LiveFeed(cid, tok)
    return _live_feed_instance
