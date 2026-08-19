"""Unified Data Manager — single market.duckdb for all market data."""

import os, uuid, json, logging
from pathlib import Path
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config

logger = logging.getLogger("data_manager")

class DataManager:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.con = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        # ── Securities master ──
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS securities (
                security_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                name VARCHAR,
                segment VARCHAR,
                instrument_type VARCHAR,
                lot_size INTEGER DEFAULT 1,
                tick_size FLOAT DEFAULT 0.05,
                is_index BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # ── Daily OHLCV — single table for all instruments ──
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS daily (
                security_id VARCHAR,
                date DATE,
                open FLOAT, high FLOAT, low FLOAT, close FLOAT,
                volume BIGINT DEFAULT 0,
                open_interest BIGINT DEFAULT 0,
                PRIMARY KEY (security_id, date)
            )
        """)
        # ── Intraday minute data ──
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS intraday (
                security_id VARCHAR,
                ts TIMESTAMP,
                open FLOAT, high FLOAT, low FLOAT, close FLOAT,
                volume BIGINT DEFAULT 0,
                PRIMARY KEY (security_id, ts)
            )
        """)
        # ── Options data snapshot ──
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS options (
                id VARCHAR PRIMARY KEY,
                underlying VARCHAR,
                expiry DATE,
                strike FLOAT,
                option_type VARCHAR,
                ltp FLOAT,
                oi BIGINT,
                iv FLOAT,
                delta FLOAT,
                gamma FLOAT,
                theta FLOAT,
                vega FLOAT,
                snapshot_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for col in ["gamma", "theta", "vega"]:
            try:
                self.con.execute(f"ALTER TABLE options ADD COLUMN IF NOT EXISTS {col} FLOAT")
            except:
                pass
        self.con.execute("DROP TABLE IF EXISTS options_history")
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS options_history (
                underlying VARCHAR,
                expiry_flag VARCHAR,
                strike_label VARCHAR,
                option_type VARCHAR,
                ts TIMESTAMP,
                open FLOAT, high FLOAT, low FLOAT, close FLOAT,
                volume BIGINT, oi BIGINT, iv FLOAT, strike FLOAT, spot FLOAT
            )
        """)
        # ── Technical indicators (pre-computed) ──
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                security_id VARCHAR,
                date DATE,
                rsi_14 FLOAT, rsi_28 FLOAT,
                obv BIGINT,
                vwap FLOAT,
                sma_20 FLOAT, sma_50 FLOAT, sma_200 FLOAT,
                ema_20 FLOAT, ema_50 FLOAT,
                bollinger_upper FLOAT, bollinger_mid FLOAT, bollinger_lower FLOAT,
                volume_avg_20 FLOAT, rvol FLOAT,
                atr_14 FLOAT,
                PRIMARY KEY (security_id, date)
            )
        """)
        # ── Market structure (SMC) ──
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS market_structure (
                security_id VARCHAR,
                date DATE,
                trend VARCHAR,
                swing_high FLOAT, swing_low FLOAT,
                order_block_high FLOAT, order_block_low FLOAT,
                fvg_high FLOAT, fvg_low FLOAT,
                liquidity_above FLOAT, liquidity_below FLOAT,
                structure_break BOOLEAN,
                PRIMARY KEY (security_id, date)
            )
        """)
        # ── AI predictions (versioned, never overwritten) ──
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ai_predictions (
                id VARCHAR PRIMARY KEY,
                version INTEGER DEFAULT 1,
                ticker VARCHAR,
                prediction_type VARCHAR,
                direction VARCHAR,
                entry FLOAT, target FLOAT, stop FLOAT,
                confidence FLOAT,
                reasoning TEXT,
                indicators_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                resolved_outcome VARCHAR,
                accuracy FLOAT
            )
        """)
        # ── Data quality log ──
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                id VARCHAR PRIMARY KEY,
                check_type VARCHAR,
                security_id VARCHAR,
                date DATE,
                status VARCHAR,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # ── Download log ──
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS download_log (
                id VARCHAR PRIMARY KEY,
                security_id VARCHAR,
                instrument_type VARCHAR,
                from_date DATE, to_date DATE,
                rows_count INTEGER,
                status VARCHAR,
                error TEXT,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.con.commit()
        logger.info("Schema initialized")

    # ── CRUD Operations ──

    def store_securities(self, securities_dict):
        for symbol, info in securities_dict.items():
            self.con.execute("""
                INSERT INTO securities (security_id, symbol, name, segment, instrument_type, lot_size, is_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (security_id) DO UPDATE SET
                    symbol = EXCLUDED.symbol, name = EXCLUDED.name,
                    segment = EXCLUDED.segment, instrument_type = EXCLUDED.instrument_type,
                    lot_size = EXCLUDED.lot_size, is_index = EXCLUDED.is_index, is_active = TRUE
            """, [info["security_id"], symbol, info.get("name", symbol),
                  info.get("segment", "NSE_EQ"), info.get("instrument_type", "EQUITY"),
                  info.get("lot_size", 1), info.get("is_index", False)])
        self.con.commit()

    def store_daily(self, security_id, df):
        if df is None or len(df) == 0:
            return 0
        count = 0
        for _, row in df.iterrows():
            try:
                date_val = row.get("timestamp")
                if hasattr(date_val, "strftime"):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_val)[:10]
                self.con.execute("""
                    INSERT INTO daily (security_id, date, open, high, low, close, volume, open_interest)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (security_id, date) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest
                """, [str(security_id), date_str,
                      float(row["open"]), float(row["high"]), float(row["low"]),
                      float(row["close"]), int(row.get("volume", 0)), int(row.get("open_interest", 0))])
                count += 1
            except Exception as e:
                logger.warning(f"store_daily error: {e}")
        self.con.commit()
        return count

    def store_intraday(self, security_id, df):
        if df is None or len(df) == 0:
            return 0
        count = 0
        for _, row in df.iterrows():
            try:
                ts_val = row.get("timestamp")
                if hasattr(ts_val, "strftime"):
                    ts_str = ts_val.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = str(ts_val)[:19]
                self.con.execute("""
                    INSERT INTO intraday (security_id, ts, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (security_id, ts) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, volume = EXCLUDED.volume
                """, [str(security_id), ts_str,
                      float(row["open"]), float(row["high"]), float(row["low"]),
                      float(row["close"]), int(row.get("volume", 0))])
                count += 1
            except Exception as e:
                logger.warning(f"store_intraday error: {e}")
        self.con.commit()
        return count

    def store_options(self, underlying, expiry, strike, opt_type, ltp, oi, iv, delta, gamma, theta, vega):
        import uuid
        oid = uuid.uuid4().hex
        self.con.execute("""
            INSERT INTO options (id, underlying, expiry, strike, option_type, ltp, oi, iv, delta, gamma, theta, vega)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
        """, [oid, underlying, expiry, strike, opt_type, ltp, oi, iv, delta, gamma, theta, vega])

    def get_intraday(self, security_id, limit=500):
        return self.con.execute("""
            SELECT * FROM intraday WHERE security_id = ? ORDER BY ts DESC LIMIT ?
        """, [str(security_id), limit]).fetchdf()

    def store_options_history(self, df, underlying, expiry_flag="WEEK", strike_label="ATM"):
        if df is None or len(df) == 0:
            return 0
        count = 0
        for _, row in df.iterrows():
            try:
                self.con.execute("""
                    INSERT INTO options_history
                        (underlying, expiry_flag, strike_label, option_type, ts,
                         open, high, low, close, volume, oi, iv, strike, spot)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [underlying, expiry_flag, strike_label, row.get("option_type", ""),
                      row["timestamp"], float(row["open"]), float(row["high"]),
                      float(row["low"]), float(row["close"]), int(row.get("volume", 0)),
                      int(row.get("oi", 0)), float(row.get("iv", 0)),
                      float(row.get("strike", 0)), float(row.get("spot", 0))])
                count += 1
            except Exception as e:
                pass
        self.con.commit()
        return count

    def get_options_history(self, underlying, option_type=None, limit=500):
        query = "SELECT * FROM options_history WHERE underlying = ?"
        params = [underlying]
        if option_type:
            query += " AND option_type = ?"
            params.append(option_type)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return self.con.execute(query, params).fetchdf()

    # ── Query Methods ──

    def get_latest_date(self, security_id):
        r = self.con.execute("SELECT MAX(date) FROM daily WHERE security_id = ?", [str(security_id)]).fetchone()
        return r[0] if r and r[0] else None

    def get_latest_intraday_ts(self, security_id):
        r = self.con.execute("SELECT MAX(ts) FROM intraday WHERE security_id = ?", [str(security_id)]).fetchone()
        return r[0] if r and r[0] else None

    def get_daily(self, security_id, limit=500):
        return self.con.execute("""
            SELECT * FROM daily WHERE security_id = ? ORDER BY date DESC LIMIT ?
        """, [str(security_id), limit]).fetchdf()

    def get_daily_asc(self, security_id, limit=500):
        return self.con.execute("""
            SELECT * FROM daily WHERE security_id = ? ORDER BY date ASC LIMIT ?
        """, [str(security_id), limit]).fetchdf()

    def get_all_securities(self):
        return self.con.execute("SELECT * FROM securities ORDER BY symbol").fetchdf()

    def get_security_by_symbol(self, symbol):
        return self.con.execute("SELECT * FROM securities WHERE symbol = ?", [symbol]).fetchone()

    def get_security_id(self, symbol):
        r = self.con.execute("SELECT security_id FROM securities WHERE symbol = ?", [symbol]).fetchone()
        return r[0] if r else None

    def get_indicators(self, security_id, limit=10):
        return self.con.execute("""
            SELECT * FROM indicators WHERE security_id = ? ORDER BY date DESC LIMIT ?
        """, [str(security_id), limit]).fetchdf()

    def get_market_structure(self, security_id, limit=10):
        return self.con.execute("""
            SELECT * FROM market_structure WHERE security_id = ? ORDER BY date DESC LIMIT ?
        """, [str(security_id), limit]).fetchdf()

    def get_significant_movers(self, threshold_pct=1.5, limit=10):
        return self.con.execute("""
            WITH changes AS (
                SELECT d.security_id, d.date, d.close,
                       d_prev.close as prev_close,
                       (d.close - d_prev.close) / NULLIF(d_prev.close, 0) * 100 as change_pct
                FROM daily d
                JOIN daily d_prev ON d.security_id = d_prev.security_id
                    AND d_prev.date = (
                        SELECT MAX(date) FROM daily d2
                        WHERE d2.security_id = d.security_id AND d2.date < d.date
                    )
                WHERE d.date = (SELECT MAX(date) FROM daily)
            )
            SELECT c.security_id, s.symbol, c.change_pct
            FROM changes c
            JOIN securities s ON c.security_id = s.security_id
            WHERE s.is_index = FALSE AND ABS(c.change_pct) >= ?
            ORDER BY ABS(c.change_pct) DESC LIMIT ?
        """, [threshold_pct, limit]).fetchdf()

    def get_download_progress(self):
        total_securities = self.con.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
        securities_with_data = self.con.execute(
            "SELECT COUNT(DISTINCT security_id) FROM daily"
        ).fetchone()[0]
        total_rows = self.con.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
        latest = self.con.execute("SELECT MAX(date) FROM daily").fetchone()[0]
        idx_rows = self.con.execute(
            "SELECT SUM(rows_count) FROM download_log WHERE status='success'"
        ).fetchone()[0] or 0
        return {
            "securities_in_db": securities_with_data,
            "expected_stocks": total_securities,
            "total_rows": total_rows,
            "latest_date": str(latest) if latest else None,
            "total_downloaded_rows": int(idx_rows),
            "completion_pct": round(securities_with_data / max(total_securities, 1) * 100, 1),
        }

    # ── Feature Computation (Stored in indicators + market_structure tables) ──

    def compute_all_indicators(self, security_id):
        df = self.get_daily_asc(security_id, limit=1000)
        if len(df) < 30:
            return 0

        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        from feature_engine import compute_features
        features = compute_features(df)

        count = 0
        for i in range(len(df)):
            if features["rsi_14"][i] is None or pd.isna(features["rsi_14"][i]):
                continue
            try:
                date_str = str(df.iloc[i]["date"])
                # Store indicators
                self.con.execute("""
                    INSERT INTO indicators VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (security_id, date) DO UPDATE SET
                        rsi_14 = EXCLUDED.rsi_14, rsi_28 = EXCLUDED.rsi_28,
                        obv = EXCLUDED.obv, vwap = EXCLUDED.vwap,
                        sma_20 = EXCLUDED.sma_20, sma_50 = EXCLUDED.sma_50,
                        sma_200 = EXCLUDED.sma_200, ema_20 = EXCLUDED.ema_20,
                        ema_50 = EXCLUDED.ema_50,
                        bollinger_upper = EXCLUDED.bollinger_upper,
                        bollinger_mid = EXCLUDED.bollinger_mid,
                        bollinger_lower = EXCLUDED.bollinger_lower,
                        volume_avg_20 = EXCLUDED.volume_avg_20,
                        rvol = EXCLUDED.rvol, atr_14 = EXCLUDED.atr_14
                """, [str(security_id), date_str,
                      _v(features["rsi_14"][i]), _v(features["rsi_28"][i]),
                      _v(features["obv"][i], int), _v(features["vwap"][i]),
                      _v(features["sma_20"][i]), _v(features["sma_50"][i]), _v(features["sma_200"][i]),
                      _v(features["ema_20"][i]), _v(features["ema_50"][i]),
                      _v(features["bb_upper"][i]), _v(features["bb_mid"][i]), _v(features["bb_lower"][i]),
                      _v(features["volume_avg"][i]), _v(features["rvol"][i]), _v(features["atr"][i])])

                # Store market structure
                self.con.execute("""
                    INSERT INTO market_structure (security_id, date, trend, swing_high, swing_low,
                        order_block_high, order_block_low, fvg_high, fvg_low,
                        liquidity_above, liquidity_below, structure_break)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (security_id, date) DO UPDATE SET
                        trend = EXCLUDED.trend, swing_high = EXCLUDED.swing_high,
                        swing_low = EXCLUDED.swing_low,
                        order_block_high = EXCLUDED.order_block_high,
                        order_block_low = EXCLUDED.order_block_low,
                        fvg_high = EXCLUDED.fvg_high, fvg_low = EXCLUDED.fvg_low,
                        liquidity_above = EXCLUDED.liquidity_above,
                        liquidity_below = EXCLUDED.liquidity_below,
                        structure_break = EXCLUDED.structure_break
                """, [str(security_id), date_str,
                      features.get("trend", [None])[i] if isinstance(features.get("trend"), list) else None,
                      _v(features.get("swing_high", [None])[i]) if isinstance(features.get("swing_high"), list) else None,
                      _v(features.get("swing_low", [None])[i]) if isinstance(features.get("swing_low"), list) else None,
                      _v(features.get("ob_high", [None])[i]) if isinstance(features.get("ob_high"), list) else None,
                      _v(features.get("ob_low", [None])[i]) if isinstance(features.get("ob_low"), list) else None,
                      _v(features.get("fvg_high", [None])[i]) if isinstance(features.get("fvg_high"), list) else None,
                      _v(features.get("fvg_low", [None])[i]) if isinstance(features.get("fvg_low"), list) else None,
                      _v(features.get("liq_above", [None])[i]) if isinstance(features.get("liq_above"), list) else None,
                      _v(features.get("liq_below", [None])[i]) if isinstance(features.get("liq_below"), list) else None,
                      features.get("structure_break", [False])[i] if isinstance(features.get("structure_break"), list) else False])
                count += 1
            except Exception as e:
                logger.warning(f"Indicator store error: {e}")
        self.con.commit()
        return count

    def store_prediction(self, ticker, pred_type, direction, entry, target, stop,
                         confidence, reasoning, indicators_used="", version=1):
        self.con.execute("""
            INSERT INTO ai_predictions (id, version, ticker, prediction_type, direction,
                entry, target, stop, confidence, reasoning, indicators_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [uuid.uuid4().hex, version, ticker, pred_type, direction,
              _v(entry), _v(target), _v(stop), confidence, reasoning[:5000],
              json.dumps(indicators_used)[:1000]])
        self.con.commit()

    def log_download(self, security_id, instrument, from_date, to_date, rows, status, error=""):
        self.con.execute("""
            INSERT INTO download_log (id, security_id, instrument_type, from_date, to_date, rows_count, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [uuid.uuid4().hex, str(security_id), instrument, str(from_date)[:10], str(to_date)[:10],
              rows, status, str(error)[:500]])
        self.con.commit()

    def log_quality(self, check_type, security_id, date_val, status, details=""):
        self.con.execute("""
            INSERT INTO data_quality (id, check_type, security_id, date, status, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [uuid.uuid4().hex, check_type, str(security_id), str(date_val)[:10], status, str(details)[:1000]])
        self.con.commit()

    def close(self):
        try:
            self.con.close()
        except:
            pass

def _v(val, cast=float):
    if val is None:
        return None
    try:
        f = float(val)
        if pd.isna(f) or np.isnan(f) or np.isinf(f):
            return None
        return cast(val)
    except:
        return None
