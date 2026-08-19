import json, logging
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger("agent_tools")

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": "OHLCV data for any stock or index. Returns recent price history with dates. Use for technical analysis, trend identification, and price pattern recognition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Symbol like NIFTY, RELIANCE, HDFCBANK, ICICIBANK, TCS, SBIN etc."},
                    "days": {"type": "integer", "description": "Number of days of data (default 30, max 500)", "default": 30},
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_indicators",
            "description": "Pre-computed technical indicators for any stock or index: RSI(14,28), OBV, VWAP, Bollinger Bands, SMA(20,50,200), ATR, Market Structure (uptrend/downtrend/range), RVOL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Symbol like NIFTY, RELIANCE, HDFCBANK etc."},
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "option_chain",
            "description": "Get option chain for NIFTY or BANKNIFTY from Dhan API. Returns OI, IV, greeks, bid/ask across strikes. Use for option trading analysis, PCR, max pain, OI buildup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "string", "enum": ["NIFTY", "BANKNIFTY"]},
                    "expiry": {"type": "string", "description": "Expiry type: 'weekly' for nearest, or date like '2026-07-09'"},
                },
                "required": ["index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Semantic search across all past analysis, patterns detected, market observations, and expert insights stored in AI's persistent memory. Use to find relevant historical context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for - describe what you want to find"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_insight",
            "description": "Store an important market insight, pattern, or analysis finding into persistent memory. The AI will remember this for future conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title for the insight"},
                    "content": {"type": "string", "description": "Detailed insight or pattern found"},
                    "ticker": {"type": "string", "description": "Related stock/index symbol"},
                },
                "required": ["title", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for latest news, data, or information. Use for current events, news about specific stocks, market updates, and company announcements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for on the web"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_movers",
            "description": "Get stocks with biggest price movement (>1.5% change) — top gainers and losers. Use for 'sabse zyada move karne wale stocks' or 'biggest movers' queries. NOT for buy/sell signals — use screen_stocks for that.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Lookback days", "default": 1},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_regime",
            "description": "Get the current overall market regime (Strong Bull, Weak Bull, Ranging, Weak Bear, Strong Bear, High/Low Volatility) with ADX, Bollinger width, ATR ratio, RSI. Use for context before any stock analysis.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_signal",
            "description": "Score a buy/sell signal for any stock using confidence engine. Returns direction, confidence (0-95%), signal strength (strong/moderate/weak/neutral), and list of confirmations and conflicts. Use for trade decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Symbol like RELIANCE, HDFCBANK, TCS etc."},
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screen_stocks",
            "description": "Screen Nifty 50 stocks for strongest buy/sell signals using confidence engine. Returns top 20 stocks ranked by score with RSI, change%, signal direction, and confidence level. NOT for biggest movers — use get_top_movers for that.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_performance",
            "description": "Get current sector performance summary — which sectors are leading/lagging today, with change %, stocks up/down counts. Use for sector rotation analysis and identifying where to focus.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_orderflow_delta",
            "description": "Get order flow delta trend for a stock — cumulative delta, delta 5-bar, average imbalance, and bullish/bearish/neutral signal. Shows buying vs selling pressure. Use to confirm trade direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Symbol like RELIANCE, HDFCBANK, TCS etc."},
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_breadth",
            "description": "Get market breadth — advancing/declining count, strong up/down moves, average change %, and % of stocks above SMA50. Use to gauge overall market health and participation.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
]

class ToolRouter:
    def __init__(self, dhan, data_mgr, memory_mgr, browser=None):
        self.dhan = dhan
        self.db = data_mgr
        self.memory = memory_mgr
        self.browser = browser

    async def execute(self, tool_name, args):
        handler = getattr(self, f"_exec_{tool_name}", None)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = await handler(**args)
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception as e:
            logger.exception(f"Tool {tool_name} failed")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _lookup_security_id(self, ticker):
        ticker = ticker.upper().strip()
        try:
            r = self.db.con.execute(
                "SELECT security_id FROM securities WHERE symbol = ?", [ticker]
            ).fetchone()
            if r:
                return r[0]
        except:
            pass
        # Case-insensitive fallback
        try:
            r = self.db.con.execute(
                "SELECT security_id FROM securities WHERE UPPER(symbol) = ?", [ticker]
            ).fetchone()
            if r:
                return r[0]
        except:
            pass
        if ticker == "NIFTY":
            return "13"
        if ticker == "BANKNIFTY":
            return "25"
        # Last resort: search by name
        try:
            r = self.db.con.execute(
                "SELECT security_id FROM securities WHERE UPPER(name) LIKE ? LIMIT 1",
                [f"%{ticker}%"]
            ).fetchone()
            if r:
                return r[0]
        except:
            pass
        raise ValueError(f"Security not found: {ticker}")

    async def _exec_get_market_data(self, ticker, days=30):
        sid = self._lookup_security_id(ticker)
        try:
            df = self.db.get_daily(sid, limit=int(days) + 10)
            if len(df) == 0:
                return {"ticker": ticker, "data": [], "message": "No data available. Download first."}
            records = df.head(min(int(days), 500)).to_dict("records")
            return {
                "ticker": ticker,
                "data_points": len(records),
                "from": str(records[-1]["date"]) if records else None,
                "to": str(records[0]["date"]) if records else None,
                "latest_close": float(records[0]["close"]) if records else None,
                "data": [{k: str(v) if hasattr(v, 'strftime') else v for k, v in r.items()} for r in records[:30]],
            }
        except Exception as e:
            return {"error": str(e)}

    async def _exec_get_indicators(self, ticker):
        sid = self._lookup_security_id(ticker)
        try:
            df = self.db.get_indicators(sid)
            if len(df) == 0:
                return {"ticker": ticker, "message": "Indicators not computed yet. Run download_and_analyze first."}
            latest = df.iloc[0]
            ms = self.db.get_market_structure(sid, limit=1)
            trend = None
            if len(ms) > 0:
                trend = ms.iloc[0].get("trend")

            def _g(row, col):
                v = row.get(col)
                if v is None:
                    return None
                try:
                    fv = float(v)
                    import math
                    return None if (math.isnan(fv) or math.isinf(fv)) else fv
                except:
                    return None

            result = {
                "ticker": ticker,
                "date": str(latest.get("date", "")),
                "close": _g(latest, "close"),  # Will be None since indicators table has no close — keep for confidence_engine
                "rsi_14": _g(latest, "rsi_14"),
                "rsi_28": _g(latest, "rsi_28"),
                "obv": _g(latest, "obv"),
                "vwap": _g(latest, "vwap"),
                "sma_20": _g(latest, "sma_20"),
                "sma_50": _g(latest, "sma_50"),
                "sma_200": _g(latest, "sma_200"),
                "bollinger_upper": _g(latest, "bollinger_upper"),
                "bollinger_lower": _g(latest, "bollinger_lower"),
                "atr_14": _g(latest, "atr_14"),
                "rvol": _g(latest, "rvol"),
                "volume_avg_20": _g(latest, "volume_avg_20"),
                "trend": trend,
            }
            return result
        except Exception as e:
            return {"error": str(e)}

    async def _exec_search_memory(self, query):
        semantic = self.memory.search_knowledge(query, top_k=5)
        return {"results": semantic[:8], "total_found": len(semantic)}

    async def _exec_store_insight(self, title, content, ticker=""):
        ok = self.memory.store_knowledge(title, content, category="insight", ticker=ticker)
        return {"stored": ok, "title": title}

    async def _exec_option_chain(self, index, expiry="weekly"):
        try:
            from dhanhq import DhanContext, dhanhq
            ctx = DhanContext(self.dhan.CLIENT_ID, self.dhan.ACCESS_TOKEN)
            dhan = dhanhq(ctx)
            sec_id = 13 if index.upper() == "NIFTY" else 25
            if expiry == "weekly":
                expiries = dhan.expiry_list(sec_id, "IDX_I")
                raw = expiries.get("data", {})
                if expiries.get("status") == "success" and raw and raw.get("data"):
                    expiry = raw["data"][0]
            chain = dhan.option_chain(sec_id, "IDX_I", expiry)
            raw = chain.get("data", {})
            if chain.get("status") == "success" and raw and raw.get("data"):
                data = raw["data"]
                spot = float(data.get("last_price", 0))
                strikes_raw = data.get("oc", {})
                result = {
                    "underlying": index,
                    "expiry": expiry,
                    "spot_price": spot,
                    "strikes": [],
                }
                for strike_str, sd in list(strikes_raw.items())[:20]:
                    ce = sd.get("CE", {}) or {}
                    pe = sd.get("PE", {}) or {}
                    result["strikes"].append({
                        "strike": float(strike_str),
                        "ce_ltp": float(ce.get("ltp", 0)),
                        "ce_oi": int(ce.get("oi", 0)),
                        "ce_iv": float(ce.get("iv", 0)),
                        "pe_ltp": float(pe.get("ltp", 0)),
                        "pe_oi": int(pe.get("oi", 0)),
                        "pe_iv": float(pe.get("iv", 0)),
                    })
                return result
            return {"error": "Failed to fetch option chain", "response": str(chain)}
        except Exception as e:
            return {"error": str(e)}

    async def _exec_web_search(self, query):
        try:
            from browser_research import BrowserResearch
            br = BrowserResearch()
            result = await br.search(query)
            return {"results": result, "source": "web_search"}
        except Exception as e:
            return {"error": f"Search failed: {e}"}

    async def _exec_get_top_movers(self, days=1, limit=10):
        try:
            if int(days) <= 1:
                df = self.db.get_significant_movers(threshold_pct=1.5)
                if len(df) > 0:
                    return {
                        "movers": [
                            {"symbol": r["symbol"], "change_pct": float(r["change_pct"])}
                            for _, r in df.head(int(limit)).iterrows()
                        ]
                    }
                return {"movers": [], "message": "No significant movers found"}
            # Multi-day: compute avg change over N days
            daily = self.db.con.execute(f"""
                WITH date_range AS (
                    SELECT DISTINCT date FROM daily WHERE date >= (SELECT MAX(date) FROM daily) - INTERVAL '{int(days)} days'
                ),
                first_last AS (
                    SELECT d.security_id, s.symbol,
                           FIRST(d.close ORDER BY d.date) AS first_close,
                           LAST(d.close ORDER BY d.date) AS last_close
                    FROM daily d
                    JOIN securities s ON d.security_id = s.security_id
                    WHERE d.date IN (SELECT date FROM date_range) AND s.is_index = FALSE
                    GROUP BY d.security_id, s.symbol
                    HAVING COUNT(*) >= 2
                )
                SELECT symbol, first_close, last_close,
                       ROUND((last_close - first_close) / NULLIF(first_close, 0) * 100, 2) AS change_pct
                FROM first_last
                ORDER BY ABS(change_pct) DESC LIMIT {int(limit)}
            """).fetchdf()
            return {
                "movers": [
                    {"symbol": r["symbol"], "change_pct": float(r["change_pct"])}
                    for _, r in daily.iterrows()
                ],
                "period_days": int(days),
            }
        except Exception as e:
            return {"error": str(e)}

    async def _exec_get_market_regime(self):
        try:
            from market_regime import MarketRegime
            mr = MarketRegime()
            result = mr.get_current()
            mr.close()
            if result is None:
                return {"message": "Market regime not yet computed. Run auto-learner first."}
            return result
        except Exception as e:
            return {"error": str(e)}

    async def _exec_analyze_signal(self, ticker):
        try:
            sid = self._lookup_security_id(ticker)
            ind = self.db.get_indicators(sid)
            if len(ind) == 0:
                return {"ticker": ticker, "message": "Indicators not available. Download first."}
            row = ind.iloc[0]
            # Get latest close from daily (indicators table has no close column)
            daily = self.db.con.execute("""
                SELECT close, date FROM daily WHERE security_id = ? ORDER BY date DESC LIMIT 1
            """, [str(sid)]).fetchdf()
            price = float(daily.iloc[0]["close"]) if len(daily) > 0 and not pd.isna(daily.iloc[0]["close"]) else None
            features = {
                "rsi_14": float(row["rsi_14"]) if row.get("rsi_14") and not pd.isna(row["rsi_14"]) else None,
                "rsi_28": float(row["rsi_28"]) if row.get("rsi_28") and not pd.isna(row["rsi_28"]) else None,
                "rvol": float(row["rvol"]) if row.get("rvol") and not pd.isna(row["rvol"]) else None,
                "sma_20": float(row["sma_20"]) if row.get("sma_20") and not pd.isna(row["sma_20"]) else None,
                "sma_50": float(row["sma_50"]) if row.get("sma_50") and not pd.isna(row["sma_50"]) else None,
                "bb_upper": float(row["bollinger_upper"]) if row.get("bollinger_upper") and not pd.isna(row["bollinger_upper"]) else None,
                "bb_lower": float(row["bollinger_lower"]) if row.get("bollinger_lower") and not pd.isna(row["bollinger_lower"]) else None,
                "obv": float(row["obv"]) if row.get("obv") and not pd.isna(row["obv"]) else None,
                "atr_14": float(row["atr_14"]) if row.get("atr_14") and not pd.isna(row["atr_14"]) else None,
            }
            ms = self.db.get_market_structure(sid, limit=1)
            if len(ms) > 0:
                features["trend"] = ms.iloc[0].get("trend")
            from confidence_engine import ConfidenceEngine
            ce = ConfidenceEngine()
            buy = ce.score_buy_signal(features, price)
            sell = ce.score_sell_signal(features, price)
            return {
                "ticker": ticker,
                "price": price,
                "buy": buy,
                "sell": sell,
                "recommendation": "BUY" if buy["confidence"] >= sell["confidence"] else "SELL",
                "top_confidence": max(buy["confidence"], sell["confidence"]),
            }
        except Exception as e:
            return {"error": str(e)}

    async def _exec_screen_stocks(self):
        try:
            from market_overview import MarketOverview
            mo = MarketOverview()
            results = mo.get_signal_screener()
            mo.close()
            if not results:
                return {"stocks": [], "message": "No signals found. Run indicators first."}
            return {
                "stocks": results,
                "total_signals": len(results),
                "buy_signals": sum(1 for r in results if r["top_signal"] == "BUY"),
                "sell_signals": sum(1 for r in results if r["top_signal"] == "SELL"),
            }
        except Exception as e:
            return {"error": str(e)}

    async def _exec_get_sector_performance(self):
        try:
            from sector_analysis import SectorAnalyzer
            sa = SectorAnalyzer()
            summary = sa.get_sector_summary(as_dict=True)
            sa.close()
            if not summary or not summary.get("sectors"):
                return {"message": "No sector data available. Run auto-learner first."}
            return summary
        except Exception as e:
            return {"error": str(e)}

    async def _exec_get_orderflow_delta(self, ticker):
        try:
            from order_flow import OrderFlowAnalyzer
            of = OrderFlowAnalyzer()
            delta = of.compute_delta_trend(ticker)
            of.close()
            if delta is None:
                return {"ticker": ticker, "message": "No order flow data available. Depth snapshots not yet collected."}
            return delta
        except Exception as e:
            return {"error": str(e)}

    async def _exec_get_market_breadth(self):
        try:
            from market_overview import MarketOverview
            mo = MarketOverview()
            breadth = mo.get_breadth()
            mo.close()
            if not breadth:
                return {"message": "No market data for breadth calculation."}
            return breadth
        except Exception as e:
            return {"error": str(e)}
