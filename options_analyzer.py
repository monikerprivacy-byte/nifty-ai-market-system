"""Options Analysis Pipeline — fetches option chain daily, computes key metrics.

Computes:
- Put-Call Ratio (PCR) by OI and premium
- Top OI buildup strikes (support/resistance levels)
- IV skew / smile
- Max Pain estimate (buyer's pain: price where total option buyer payout is minimized)
- Stores signals in memory as market facts
"""

import logging, json, uuid
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config

logger = logging.getLogger("options_analyzer")

class OptionsAnalyzer:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.con = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS options_snapshots (
                id VARCHAR PRIMARY KEY,
                underlying VARCHAR,
                expiry DATE,
                spot_price FLOAT,
                pcr_oi FLOAT,
                pcr_premium FLOAT,
                max_pain_strike FLOAT,
                total_ce_oi BIGINT,
                total_pe_oi BIGINT,
                iv_skew FLOAT,
                snapshot_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            self.con.execute("ALTER TABLE options_snapshots RENAME COLUMN pcr_volume TO pcr_premium")
        except:
            pass
        try:
            self.con.execute("ALTER TABLE options_snapshots ADD COLUMN IF NOT EXISTS pcr_premium FLOAT")
        except:
            pass
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS options_strikes (
                id VARCHAR PRIMARY KEY,
                snapshot_id VARCHAR,
                underlying VARCHAR,
                expiry DATE,
                strike FLOAT,
                ce_oi BIGINT, ce_ltp FLOAT, ce_iv FLOAT, ce_delta FLOAT,
                ce_gamma FLOAT, ce_theta FLOAT, ce_vega FLOAT,
                pe_oi BIGINT, pe_ltp FLOAT, pe_iv FLOAT, pe_delta FLOAT,
                pe_gamma FLOAT, pe_theta FLOAT, pe_vega FLOAT,
                snapshot_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for col in ["ce_gamma", "ce_theta", "ce_vega", "pe_gamma", "pe_theta", "pe_vega"]:
            try:
                self.con.execute(f"ALTER TABLE options_strikes ADD COLUMN IF NOT EXISTS {col} FLOAT")
            except:
                pass
        self.con.commit()

    async def fetch_and_analyze(self, underlying="NIFTY", expiry="weekly"):
        """Fetch option chain from Dhan API and compute metrics."""
        from dhan_client import DhanClient
        dhan = DhanClient()
        sec_id = 13 if underlying.upper() == "NIFTY" else 25

        try:
            # Get expiry list
            from dhanhq import DhanContext, dhanhq
            ctx = DhanContext(dhan.CLIENT_ID, dhan.ACCESS_TOKEN)
            dh = dhanhq(ctx)

            if expiry == "weekly":
                expiries = dh.expiry_list(sec_id, "IDX_I")
                raw = expiries.get("data", {})
                if expiries.get("status") != "success" or not raw or not raw.get("data"):
                    return {"error": "No expiries found"}
                expiry_date = raw["data"][0]
            else:
                expiry_date = expiry

            # Fetch chain
            chain = dh.option_chain(sec_id, "IDX_I", expiry_date)
            raw = chain.get("data", {})
            if chain.get("status") != "success" or not raw or not raw.get("data"):
                return {"error": f"Option chain fetch failed: {chain}"}

            data = raw.get("data", {})
            spot = float(data.get("last_price", 0))
            strikes_raw = data.get("oc", {})

            if not strikes_raw:
                return {"error": "No strike data"}

            # Parse strikes (Dhan API: key=strike_price, value=expiry_data dict)
            strikes = []
            for strike_str, sd in strikes_raw.items():
                strike = float(strike_str)
                ce = sd.get("CE", {}) or sd.get("ce", {}) or {}
                pe = sd.get("PE", {}) or sd.get("pe", {}) or {}
                cg = ce.get("greeks", {}) or {}
                pg = pe.get("greeks", {}) or {}
                strikes.append({
                    "strike": strike,
                    "ce_oi": int(ce.get("oi", 0)),
                    "ce_ltp": float(ce.get("last_price", 0)),
                    "ce_iv": float(ce.get("implied_volatility", 0)),
                    "ce_delta": float(cg.get("delta", 0)),
                    "ce_gamma": float(cg.get("gamma", 0)),
                    "ce_theta": float(cg.get("theta", 0)),
                    "ce_vega": float(cg.get("vega", 0)),
                    "pe_oi": int(pe.get("oi", 0)),
                    "pe_ltp": float(pe.get("last_price", 0)),
                    "pe_iv": float(pe.get("implied_volatility", 0)),
                    "pe_delta": float(pg.get("delta", 0)),
                    "pe_gamma": float(pg.get("gamma", 0)),
                    "pe_theta": float(pg.get("theta", 0)),
                    "pe_vega": float(pg.get("vega", 0)),
                })

            df = pd.DataFrame(strikes)
            snapshot_id = uuid.uuid4().hex

            # Compute metrics
            total_ce_oi = int(df["ce_oi"].sum())
            total_pe_oi = int(df["pe_oi"].sum())
            total_ce_premium = int(df["ce_ltp"].sum())
            total_pe_premium = int(df["pe_ltp"].sum())
            pcr_oi = round(total_pe_oi / max(total_ce_oi, 1), 4)
            pcr_premium = round(total_pe_premium / max(total_ce_premium, 1), 4)

            # Max pain: expiry price P where total buyer payout across ALL strikes is minimized.
            # total_payout(P) = Σ[max(0, P−Ki) × call_OIi + max(0, Ki−P) × put_OIi]
            total_oi = int(df["ce_oi"].sum()) + int(df["pe_oi"].sum())
            if total_oi == 0:
                max_pain = spot
            else:
                strikes_arr = df["strike"].values
                ce_oi_arr = df["ce_oi"].values
                pe_oi_arr = df["pe_oi"].values
                best_pain, best_val = spot, float("inf")
                for P in strikes_arr:
                    payout = (
                        np.sum(np.maximum(P - strikes_arr, 0) * ce_oi_arr) +
                        np.sum(np.maximum(strikes_arr - P, 0) * pe_oi_arr)
                    )
                    if 0 < payout < best_val:
                        best_val = payout
                        best_pain = P
                if best_val == float("inf"):
                    max_pain = spot
                else:
                    max_pain = float(best_pain)

            # IV skew: difference between OTMs (only strikes with non-zero IV)
            atm_strike = round(spot / 50) * 50
            otm_puts = df[(df["strike"] < atm_strike) & (df["pe_iv"] > 0)]
            otm_calls = df[(df["strike"] > atm_strike) & (df["ce_iv"] > 0)]
            otm_put_iv = float(otm_puts["pe_iv"].mean()) if len(otm_puts) > 0 else 0
            otm_call_iv = float(otm_calls["ce_iv"].mean()) if len(otm_calls) > 0 else 0
            iv_skew = round(otm_put_iv - otm_call_iv, 2) if otm_call_iv else 0

            # Top OI strikes relative to spot (resistance above / support below)
            above_spot = df[df["strike"] > spot].sort_values("ce_oi", ascending=False)
            below_spot = df[df["strike"] < spot].sort_values("pe_oi", ascending=False)
            top_ce_resistance = above_spot.head(3)[["strike", "ce_oi"]].to_dict("records") if len(above_spot) > 0 else []
            top_pe_support = below_spot.head(3)[["strike", "pe_oi"]].to_dict("records") if len(below_spot) > 0 else []

            # Store snapshot
            self.con.execute("""
                INSERT INTO options_snapshots (id, underlying, expiry, spot_price,
                    pcr_oi, pcr_premium, max_pain_strike, total_ce_oi, total_pe_oi, iv_skew)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [snapshot_id, underlying.upper(), expiry_date, spot,
                  pcr_oi, pcr_premium, max_pain, total_ce_oi, total_pe_oi, iv_skew])

            # Store strikes
            from data_manager import DataManager
            dm = DataManager()
            for s in strikes:
                self.con.execute("""
                    INSERT INTO options_strikes (id, snapshot_id, underlying, expiry, strike,
                        ce_oi, ce_ltp, ce_iv, ce_delta, ce_gamma, ce_theta, ce_vega,
                        pe_oi, pe_ltp, pe_iv, pe_delta, pe_gamma, pe_theta, pe_vega)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [uuid.uuid4().hex, snapshot_id, underlying.upper(), expiry_date,
                      s["strike"], s["ce_oi"], s["ce_ltp"], s["ce_iv"], s["ce_delta"],
                      s["ce_gamma"], s["ce_theta"], s["ce_vega"],
                      s["pe_oi"], s["pe_ltp"], s["pe_iv"], s["pe_delta"],
                      s["pe_gamma"], s["pe_theta"], s["pe_vega"]])
                dm.store_options(underlying.upper(), expiry_date, s["strike"], "CE",
                    s["ce_ltp"], s["ce_oi"], s["ce_iv"], s["ce_delta"],
                    s["ce_gamma"], s["ce_theta"], s["ce_vega"])
                dm.store_options(underlying.upper(), expiry_date, s["strike"], "PE",
                    s["pe_ltp"], s["pe_oi"], s["pe_iv"], s["pe_delta"],
                    s["pe_gamma"], s["pe_theta"], s["pe_vega"])
            dm.close()
            self.con.commit()

            # Store signals in memory
            await self._store_signals(underlying, expiry_date, spot, pcr_oi, pcr_premium, max_pain, iv_skew, top_ce_resistance, top_pe_support)

            result = {
                "underlying": underlying.upper(),
                "expiry": expiry_date,
                "spot": spot,
                "pcr_oi": pcr_oi,
                "pcr_premium": pcr_premium,
                "max_pain": max_pain,
                "iv_skew": iv_skew,
                "total_ce_oi": total_ce_oi,
                "total_pe_oi": total_pe_oi,
                "top_ce_resistance": [s["strike"] for s in top_ce_resistance[:2]],
                "top_pe_support": [s["strike"] for s in top_pe_support[:2]],
                "interpretation": self._interpret(pcr_oi, pcr_premium, iv_skew, spot, max_pain),
            }

            logger.info(f"Options analysis: {underlying} PCR={pcr_oi} Spot={spot} MP={max_pain}")
            return result

        except Exception as e:
            logger.warning(f"Options analysis failed for {underlying}: {e}")
            return {"error": str(e)}

    async def _store_signals(self, underlying, expiry, spot, pcr_oi, pcr_premium, max_pain, iv_skew, top_ce_resistance, top_pe_support):
        try:
            from memory_manager import MemoryManager
            cfg = get_config()
            mem_path = cfg.get("databases.memory", "/Volumes/Untitled/market_data/memory.duckdb")
            mm = MemoryManager(mem_path)

            # Resistance (highest CE OI above spot)
            if top_ce_resistance:
                res_strikes = [s["strike"] for s in top_ce_resistance[:2]]
                mm.store_fact(underlying, "options_resistance",
                    f"Max CE OI above spot at {res_strikes} — resistance zone",
                    confidence=0.7, source="options_analyzer")

            # Support (highest PE OI below spot)
            if top_pe_support:
                sup_strikes = [s["strike"] for s in top_pe_support[:2]]
                mm.store_fact(underlying, "options_support",
                    f"Max PE OI below spot at {sup_strikes} — support zone",
                    confidence=0.7, source="options_analyzer")

            # PCR interpretation
            if pcr_oi < 0.5:
                mm.store_fact(underlying, "options_pcr",
                    f"PCR-OI={pcr_oi} — extremely bearish (too many calls)",
                    confidence=0.8, source="options_analyzer")
            elif pcr_oi > 1.5:
                mm.store_fact(underlying, "options_pcr",
                    f"PCR-OI={pcr_oi} — extremely bullish (too many puts)",
                    confidence=0.8, source="options_analyzer")

            # Max pain vs spot — use absolute price distance
            diff = spot - max_pain
            if abs(diff) > max_pain * 0.02:  # >2% away
                direction = "above" if diff > 0 else "below"
                mm.store_fact(underlying, "options_max_pain",
                    f"Spot (₹{spot}) is {abs(diff):.0f} pts {direction} max pain (₹{max_pain}) — market may pull toward max pain",
                    confidence=0.6, source="options_analyzer")

            # IV skew
            if abs(iv_skew) > 5:
                skew_dir = "puts expensive (fear)" if iv_skew > 0 else "calls expensive (greed)"
                mm.store_fact(underlying, "options_iv_skew",
                    f"IV skew={iv_skew:+.1f} — {skew_dir}",
                    confidence=0.6, source="options_analyzer")

            mm.close()
        except Exception as e:
            logger.debug(f"Options signal store failed: {e}")

    def get_latest_snapshot(self, underlying="NIFTY"):
        """Get the latest stored snapshot."""
        return self.con.execute("""
            SELECT * FROM options_snapshots
            WHERE underlying = ?
            ORDER BY snapshot_ts DESC LIMIT 1
        """, [underlying.upper()]).fetchdf()

    def get_strikes(self, snapshot_id):
        """Get strikes for a given snapshot."""
        return self.con.execute("""
            SELECT * FROM options_strikes
            WHERE snapshot_id = ?
            ORDER BY strike
        """, [snapshot_id]).fetchdf()

    def get_weekly_trend(self, underlying="NIFTY", weeks=4):
        """Get PCR trend over last N weeks."""
        return self.con.execute("""
            SELECT snapshot_ts, spot_price, pcr_oi, pcr_premium, max_pain_strike, iv_skew
            FROM options_snapshots
            WHERE underlying = ?
            ORDER BY snapshot_ts DESC LIMIT ?
        """, [underlying.upper(), weeks]).fetchdf()

    @staticmethod
    def _interpret(pcr_oi, pcr_premium, iv_skew, spot, max_pain):
        parts = []
        # PCR
        if pcr_oi < 0.5:
            parts.append("Extremely bearish (too many calls, PCR<0.5)")
        elif pcr_oi < 0.8:
            parts.append("Slightly bearish")
        elif pcr_oi < 1.2:
            parts.append("Neutral range")
        elif pcr_oi < 1.5:
            parts.append("Slightly bullish")
        else:
            parts.append("Extremely bullish (too many puts, PCR>1.5)")
        # Max pain
        diff = spot - max_pain
        if abs(diff) > max_pain * 0.03:  # >3% away
            parts.append(f"Spot (₹{spot:.0f}) is {abs(diff):.0f}pts {'above' if diff>0 else 'below'} max pain (₹{max_pain:.0f}) — expected pull toward max pain")
        # IV skew
        if abs(iv_skew) > 5:
            skew_desc = "puts priced higher (fear)" if iv_skew > 0 else "calls priced higher (greed)"
            parts.append(f"IV skew {iv_skew:+.1f} - {skew_desc}")
        return ". ".join(parts)

    def analyze_expired_history(self, underlying="NIFTY", days=30):
        """Analyze expired options history for sentiment signals.

        Reads from options_history table and computes:
        - Daily IV trend (put IV - call IV spread over time)
        - Daily OI change rate
        - Put/Call premium ratio trend
        - Volume-weighted sentiment score
        """
        df = self.con.execute("""
            SELECT date(ts) as day, option_type,
                   avg(iv) as avg_iv, avg(oi) as avg_oi, avg(close) as avg_premium,
                   sum(volume) as total_vol
            FROM options_history
            WHERE underlying = ?
            GROUP BY day, option_type
            ORDER BY day
        """, [underlying.upper()]).fetchdf()
        if len(df) == 0:
            return {}
        calls = df[df["option_type"] == "CALL"].set_index("day")
        puts = df[df["option_type"] == "PUT"].set_index("day")
        common_days = sorted(set(calls.index) & set(puts.index))
        if not common_days:
            return {}
        signals = []
        for d in common_days:
            c = calls.loc[d]
            p = puts.loc[d]
            iv_spread = float(p["avg_iv"] - c["avg_iv"])
            oi_ratio = float(p["avg_oi"] / max(c["avg_oi"], 1))
            premium_ratio = float(p["avg_premium"] / max(c["avg_premium"], 1))
            vol_ratio = float(p["total_vol"] / max(c["total_vol"], 1))
            signals.append({
                "date": str(d),
                "iv_spread": round(iv_spread, 2),
                "oi_ratio": round(oi_ratio, 4),
                "premium_ratio": round(premium_ratio, 4),
                "vol_ratio": round(vol_ratio, 4),
                "call_iv": round(float(c["avg_iv"]), 2),
                "put_iv": round(float(p["avg_iv"]), 2),
            })
        latest = signals[-1] if signals else {}
        trend_iv = signals[-3:] if len(signals) >= 3 else signals
        iv_trend = [s["iv_spread"] for s in trend_iv]
        oi_trend = [s["oi_ratio"] for s in trend_iv]
        result = {
            "underlying": underlying.upper(),
            "days_analyzed": len(signals),
            "latest": latest,
            "iv_spread_trend": iv_trend,
            "oi_ratio_trend": oi_trend,
            "signals": signals[-10:],  # last 10 days
        }
        # Sentiment interpretation
        latest_iv = latest.get("iv_spread", 0)
        if latest_iv > 5:
            result["sentiment"] = "bearish (puts expensive, fear)"
        elif latest_iv < -3:
            result["sentiment"] = "bullish (calls expensive, greed)"
        else:
            result["sentiment"] = "neutral"
        if abs(latest_iv) > 5 and len(iv_trend) >= 3:
            widening = all(abs(iv_trend[i]) <= abs(iv_trend[i+1]) for i in range(len(iv_trend)-1))
            result["iv_spread_trend_dir"] = "widening" if widening else "narrowing"
        return result

    def close(self):
        try:
            self.con.close()
        except:
            pass
