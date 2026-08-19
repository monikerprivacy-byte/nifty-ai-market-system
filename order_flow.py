"""Order Flow Analysis — bid/ask depth, cumulative delta, order book imbalance, large trades.

Data sources:
- Dhan API `get_quote()` for bid/ask depth snapshots
- LiveFeed tick history for large trade detection
- Daily OHLC for volume context

Provides:
- market_depth table (bid/ask snapshots per security)
- Cumulative delta trends (buying vs selling pressure)
- Order book imbalance indicator
- Large trade detection (>2x avg trade size)
"""

import logging, json, time
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config

logger = logging.getLogger("order_flow")

# Securities to track depth for (Nifty 50 + indices)
TRACKED_SECURITIES = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "INFY", "ITC",
    "HCLTECH", "BHARTIARTL", "LT", "SBIN", "BAJFINANCE", "KOTAKBANK",
    "AXISBANK", "TITAN", "M&M", "MARUTI", "SUNPHARMA", "NTPC",
    "NIFTY", "BANKNIFTY",
]


class OrderFlowAnalyzer:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.con = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS market_depth (
                id INTEGER PRIMARY KEY,
                security_id VARCHAR,
                symbol VARCHAR,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ltp FLOAT,
                bid_price1 FLOAT, bid_qty1 INTEGER,
                ask_price1 FLOAT, ask_qty1 INTEGER,
                bid_price2 FLOAT, bid_qty2 INTEGER,
                ask_price2 FLOAT, ask_qty2 INTEGER,
                bid_price3 FLOAT, bid_qty3 INTEGER,
                ask_price3 FLOAT, ask_qty3 INTEGER,
                bid_price4 FLOAT, bid_qty4 INTEGER,
                ask_price4 FLOAT, ask_qty4 INTEGER,
                bid_price5 FLOAT, bid_qty5 INTEGER,
                ask_price5 FLOAT, ask_qty5 INTEGER,
                total_bid_qty INTEGER,
                total_ask_qty INTEGER,
                imbalance FLOAT,
                spread FLOAT,
                spread_pct FLOAT
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS order_flow_signals (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cumulative_delta FLOAT,
                delta_5bar FLOAT,
                avg_imbalance FLOAT,
                large_trade_detected BOOLEAN,
                large_trade_size FLOAT,
                signal VARCHAR
            )
        """)
        self.con.commit()

    def _lookup_id(self, ticker):
        ticker = ticker.upper().strip()
        r = self.con.execute(
            "SELECT security_id FROM securities WHERE symbol = ?", [ticker]
        ).fetchone()
        if r:
            return r[0]
        if ticker == "NIFTY":
            return "13"
        if ticker == "BANKNIFTY":
            return "25"
        return None

    def snapshot_depth(self, dhan_client, symbols=None):
        """Fetch bid/ask depth from Dhan API and store snapshot."""
        if symbols is None:
            symbols = TRACKED_SECURITIES
        instruments = []
        symbol_map = {}
        for sym in symbols:
            sid = self._lookup_id(sym)
            if not sid:
                continue
            exchange = 0 if sym in ("NIFTY", "BANKNIFTY") else 1
            instrument_type = 17 if sym in ("NIFTY", "BANKNIFTY") else 15
            instruments.append({"securityId": sid, "exchangeSegment": exchange, "instrument": instrument_type})
            symbol_map[sid] = sym

        try:
            import asyncio
            resp = asyncio.get_event_loop().run_until_complete(dhan_client.get_quote(instruments))
        except RuntimeError:
            resp = None
            logger.warning("No event loop for depth snapshot")
        if not resp or not isinstance(resp, dict):
            return
        data = resp.get("data") or resp.get("results") or resp
        if isinstance(data, dict):
            data = [data]
        for entry in data:
            try:
                sid = str(entry.get("securityId", ""))
                sym = symbol_map.get(sid, sid)
                ltp = float(entry.get("ltp", entry.get("lastPrice", 0)))
                bid_data = entry.get("bid", [])
                ask_data = entry.get("ask", [])
                total_bid = sum(b.get("quantity", 0) for b in bid_data[:5])
                total_ask = sum(a.get("quantity", 0) for a in ask_data[:5])
                imbalance = (total_bid - total_ask) / max(total_bid + total_ask, 1)
                best_bid = float(bid_data[0]["price"]) if bid_data and bid_data[0].get("price") else None
                best_ask = float(ask_data[0]["price"]) if ask_data and ask_data[0].get("price") else None
                spread = (best_ask - best_bid) if best_bid and best_ask else None
                spread_pct = (spread / ltp * 100) if spread and ltp else None

                cols = ["security_id", "symbol", "ltp",
                        "bid_price1", "bid_qty1", "ask_price1", "ask_qty1",
                        "bid_price2", "bid_qty2", "ask_price2", "ask_qty2",
                        "bid_price3", "bid_qty3", "ask_price3", "ask_qty3",
                        "bid_price4", "bid_qty4", "ask_price4", "ask_qty4",
                        "bid_price5", "bid_qty5", "ask_price5", "ask_qty5",
                        "total_bid_qty", "total_ask_qty", "imbalance", "spread", "spread_pct"]
                vals = [sid, sym, ltp]
                for i in range(5):
                    b = bid_data[i] if i < len(bid_data) else {}
                    a = ask_data[i] if i < len(ask_data) else {}
                    vals += [b.get("price"), b.get("quantity"), a.get("price"), a.get("quantity")]
                vals += [total_bid, total_ask, round(imbalance, 4),
                         round(spread, 2) if spread else None,
                         round(spread_pct, 4) if spread_pct else None]
                self.con.execute(f"""
                    INSERT INTO market_depth ({", ".join(cols)})
                    VALUES ({", ".join(["?" for _ in cols])})
                """, vals)
            except Exception as e:
                logger.debug(f"Depth parse failed for {entry.get('securityId')}: {e}")
        self.con.commit()
        logger.info(f"Depth snapshot for {len(data)} securities")

    def compute_delta_trend(self, symbol, lookback_minutes=60):
        """Compute cumulative delta from recent depth snapshots."""
        sid = self._lookup_id(symbol)
        if not sid:
            return None
        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        rows = self.con.execute("""
            SELECT timestamp, imbalance, total_bid_qty, total_ask_qty
            FROM market_depth
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp
        """, [symbol, cutoff]).fetchdf()
        if len(rows) < 2:
            return None
        rows["delta"] = rows["total_bid_qty"].astype(float) - rows["total_ask_qty"].astype(float)
        cum_delta = rows["delta"].sum()
        delta_5bar = rows["delta"].tail(5).sum() if len(rows) >= 5 else rows["delta"].sum()
        avg_imbalance = rows["imbalance"].mean()
        signal = "bullish" if delta_5bar > 0 and avg_imbalance > 0.1 else \
                 "bearish" if delta_5bar < 0 and avg_imbalance < -0.1 else "neutral"
        result = {
            "symbol": symbol,
            "snapshots": len(rows),
            "cumulative_delta": round(float(cum_delta), 2),
            "delta_5bar": round(float(delta_5bar), 2),
            "avg_imbalance": round(float(avg_imbalance), 4),
            "signal": signal,
        }
        self.con.execute("""
            INSERT INTO order_flow_signals
                (symbol, cumulative_delta, delta_5bar, avg_imbalance, signal)
            VALUES (?, ?, ?, ?, ?)
        """, [symbol, result["cumulative_delta"], result["delta_5bar"],
              result["avg_imbalance"], signal])
        self.con.commit()
        return result

    def detect_large_trades(self, symbol, threshold_mult=2.0, lookback_minutes=30):
        """Detect large trades from LiveFeed tick history.
        Compares each tick's volume increment against the rolling average.
        """
        from dhan_feed import get_live_feed
        feed = get_live_feed()
        sid = self._lookup_id(symbol)
        if not sid or not feed or not feed.tick_history:
            return []
        ticks = list(feed.tick_history)
        if len(ticks) < 10:
            return []
        sec_ticks = [t for t in ticks if str(t.get("security_id")) == str(sid)]
        if len(sec_ticks) < 3:
            return []
        volumes = []
        last_vol = 0
        for t in sec_ticks:
            vol = float(t.get("volume", 0))
            delta = vol - last_vol
            if delta > 0:
                volumes.append(delta)
            last_vol = vol
        if len(volumes) < 3:
            return []
        avg_vol = np.mean(volumes)
        threshold = avg_vol * threshold_mult
        large = [v for v in volumes if v > threshold]
        if large:
            count = len(large)
            max_size = max(large)
            result = {
                "symbol": symbol,
                "large_trades": count,
                "largest_size": int(max_size),
                "threshold": int(threshold),
                "signal": "large_trade_detected" if count >= 2 else "normal",
            }
            self.con.execute("""
                INSERT INTO order_flow_signals
                    (symbol, large_trade_detected, large_trade_size, signal)
                VALUES (?, TRUE, ?, ?)
            """, [symbol, int(max_size), result["signal"]])
            self.con.commit()
            return result
        return {"symbol": symbol, "large_trades": 0, "signal": "normal"}

    def get_recent_signals(self, symbol=None, limit=20):
        """Get recent order flow signals."""
        if symbol:
            rows = self.con.execute("""
                SELECT * FROM order_flow_signals
                WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?
            """, [symbol, limit]).fetchdf()
        else:
            rows = self.con.execute("""
                SELECT * FROM order_flow_signals
                ORDER BY timestamp DESC LIMIT ?
            """, [limit]).fetchdf()
        if len(rows) == 0:
            return []
        return rows.to_dict("records")

    def get_depth_summary(self, symbol=None):
        """Get latest depth snapshot summary for a symbol or all tracked."""
        cutoff = datetime.now() - timedelta(hours=2)
        if symbol:
            rows = self.con.execute("""
                SELECT symbol, ltp, imbalance, spread_pct, total_bid_qty, total_ask_qty, timestamp
                FROM market_depth WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT 1
            """, [symbol, cutoff]).fetchdf()
        else:
            rows = self.con.execute("""
                SELECT symbol, ltp, imbalance, spread_pct, total_bid_qty, total_ask_qty, timestamp
                FROM market_depth WHERE timestamp >= ?
                ORDER BY symbol, timestamp DESC
            """, [cutoff]).fetchdf()
            rows = rows.drop_duplicates("symbol") if len(rows) > 0 else rows
        if len(rows) == 0:
            return []
        result = []
        for _, r in rows.iterrows():
            imb = float(r["imbalance"]) if r["imbalance"] is not None else 0
            result.append({
                "symbol": r["symbol"],
                "ltp": float(r["ltp"]),
                "imbalance": round(imb, 4),
                "spread_pct": round(float(r["spread_pct"]), 4) if r["spread_pct"] is not None else None,
                "total_bid_qty": int(r["total_bid_qty"]),
                "total_ask_qty": int(r["total_ask_qty"]),
                "bid_pressure": "buying" if imb > 0.1 else "selling" if imb < -0.1 else "neutral",
                "timestamp": str(r["timestamp"]),
            })
        return result

    def close(self):
        try:
            self.con.close()
        except:
            pass
