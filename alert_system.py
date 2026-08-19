"""Alert System — desktop notifications for strong market signals + alert history.

When confidence engine finds a strong (≥75%) buy/sell signal, this module:
1. Fires a macOS desktop notification via `osascript`
2. Stores in alert history (DuckDB)
3. Deduplicates to avoid spamming the same signal
"""

import os, json, logging, uuid, subprocess
from datetime import datetime, timedelta
import duckdb
from config_manager import get_config

logger = logging.getLogger("alert_system")

class AlertSystem:
    def __init__(self, db_path=None):
        cfg = get_config()
        if db_path is None:
            data_dir = cfg.get("app.data_dir", "/Volumes/Untitled/market_data")
            db_path = os.path.join(data_dir, "alerts.duckdb")
        self.db_path = db_path
        self.con = duckdb.connect(self.db_path)
        self._init_schema()
        self._min_interval = cfg.get("alert.min_interval_minutes", 60)
        self._enabled = cfg.get("alert.enabled", True)
        self._sound = cfg.get("alert.sound", "default")

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id VARCHAR PRIMARY KEY,
                ticker VARCHAR,
                alert_type VARCHAR,
                direction VARCHAR,
                confidence FLOAT,
                signal_strength VARCHAR,
                price FLOAT,
                message TEXT,
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS alert_suppression (
                signal_key VARCHAR PRIMARY KEY,
                last_fired TIMESTAMP
            )
        """)
        self.con.commit()

    def _can_fire(self, signal_key):
        """Check if enough time has passed since last identical signal."""
        r = self.con.execute(
            "SELECT last_fired FROM alert_suppression WHERE signal_key = ?",
            [signal_key]
        ).fetchone()
        if not r:
            return True
        elapsed = datetime.now() - r[0]
        return elapsed.total_seconds() >= self._min_interval * 60

    def _mark_fired(self, signal_key):
        self.con.execute("""
            INSERT INTO alert_suppression (signal_key, last_fired)
            VALUES (?, CURRENT_TIMESTAMP)
            ON CONFLICT (signal_key) DO UPDATE SET last_fired = CURRENT_TIMESTAMP
        """, [signal_key])
        self.con.commit()

    def fire(self, ticker, alert_type, direction, confidence, signal_strength,
             price=None, message="", reasoning=""):
        """Fire an alert. Returns True if alert was sent."""
        if not self._enabled:
            return False

        signal_key = f"{ticker}:{direction}:{alert_type}"
        if not self._can_fire(signal_key):
            logger.debug(f"Alert suppressed (cooldown): {signal_key}")
            return False

        alert_id = uuid.uuid4().hex
        msg = message or f"{direction} signal for {ticker} ({signal_strength}, {confidence:.0f}% confidence)"
        if price:
            msg += f" @ ₹{price}"

        # Store
        self.con.execute("""
            INSERT INTO alerts (id, ticker, alert_type, direction, confidence, signal_strength,
                price, message, reasoning, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [alert_id, ticker, alert_type, direction, confidence, signal_strength,
              float(price) if price else None, msg[:500], reasoning[:2000]])
        self.con.commit()

        # macOS notification
        self._notify_macos(ticker, direction, signal_strength, confidence, msg)

        self._mark_fired(signal_key)
        logger.info(f"Alert fired: {signal_key} ({signal_strength})")
        # Publish to event bus for WebSocket
        try:
            from event_bus import get_event_bus
            get_event_bus().emit("alert", {
                "ticker": ticker, "direction": direction,
                "confidence": confidence, "signal": signal_strength,
                "message": msg[:200],
            })
        except:
            pass
        return True

    def _notify_macos(self, ticker, direction, signal_strength, confidence, message):
        """Fire a macOS desktop notification."""
        try:
            title = f"{'🟢' if direction == 'BUY' else '🔴'} {direction} {ticker}"
            subtitle = f"{signal_strength.upper()} ({confidence:.0f}% confidence)"
            script = f'display notification "{message}" with title "{title}" subtitle "{subtitle}"'
            if self._sound != "none":
                script += f' sound name "{self._sound}"'
            subprocess.run(["osascript", "-e", script],
                         capture_output=True, timeout=5)
        except Exception as e:
            logger.warning(f"macOS notification failed: {e}")

    def check_and_alert(self, ticker, indicators, price=None):
        """Convenience: run confidence engine and fire alert if strong signal."""
        from confidence_engine import get_engine
        engine = get_engine()

        buy = engine.score_buy_signal(indicators, price)
        sell = engine.score_sell_signal(indicators, price)

        fired = False
        if buy["signal"] == "strong":
            fired |= self.fire(ticker, "technical", "BUY", buy["confidence"],
                               buy["signal"], price, reasoning=buy["summary"])
        if sell["signal"] == "strong":
            fired |= self.fire(ticker, "technical", "SELL", sell["confidence"],
                               sell["signal"], price, reasoning=sell["summary"])
        return fired

    def get_alerts(self, limit=50, unread_only=False):
        """Get recent alerts."""
        query = "SELECT * FROM alerts"
        if unread_only:
            query += " WHERE acknowledged = FALSE"
        query += " ORDER BY created_at DESC LIMIT ?"
        rows = self.con.execute(query, [limit]).fetchall()
        return [
            {
                "id": r[0], "ticker": r[1], "type": r[2], "direction": r[3],
                "confidence": r[4], "signal": r[5], "price": r[6],
                "message": r[7], "reasoning": r[8][:200] if r[8] else "",
                "time": str(r[9]), "acknowledged": r[10],
            }
            for r in rows
        ]

    def acknowledge(self, alert_id):
        self.con.execute("UPDATE alerts SET acknowledged = TRUE WHERE id = ?", [alert_id])
        self.con.commit()

    def acknowledge_all(self, ticker=""):
        if ticker:
            self.con.execute("UPDATE alerts SET acknowledged = TRUE WHERE ticker = ?", [ticker])
        else:
            self.con.execute("UPDATE alerts SET acknowledged = TRUE")
        self.con.commit()

    def get_stats(self):
        total = self.con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        unread = self.con.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = FALSE").fetchone()[0]
        by_type = self.con.execute("""
            SELECT direction, signal_strength, COUNT(*)
            FROM alerts GROUP BY direction, signal_strength ORDER BY COUNT(*) DESC
        """).fetchall()
        return {
            "total": total,
            "unread": unread,
            "by_signal": [{"direction": r[0], "strength": r[1], "count": r[2]} for r in by_type],
        }

    def close(self):
        try:
            self.con.close()
        except:
            pass


# Singleton
_instance = None

def get_alert_system():
    global _instance
    if _instance is None:
        _instance = AlertSystem()
    return _instance
