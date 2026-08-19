"""Trade Executor — paper/live order placement, position tracking, P&L monitoring.

Architecture:
- Paper mode by default (config: trading.mode = "paper" | "live")
- All orders recorded in market.duckdb (orders + positions tables)
- Event Bus integration for WebSocket push
- Risk checks: max position size, max daily loss, max concentration

Dhan API endpoints used:
- POST /orders          → place order
- GET /orders           → list orders
- GET /orders/{id}      → order detail
- DELETE /orders/{id}   → cancel order
- GET /positions        → position book
"""

import asyncio, json, logging, time, math
from datetime import datetime
from enum import Enum
import duckdb
import pandas as pd
from config_manager import get_config
from dhan_client import DhanClient, AuthExpiredError

logger = logging.getLogger("trade_executor")

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "STOP_LOSS"
    SL_MARKET = "STOP_LOSS_MARKET"

class OrderDuration(Enum):
    DAY = "DAY"
    IOC = "IOC"

class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"

POSITION_TYPE_LONG = "LONG"
POSITION_TYPE_SHORT = "SHORT"


class TradeExecutor:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.mode = cfg.get("trading.mode", "paper")
        self.max_position_pct = float(cfg.get("trading.max_position_pct", 10))
        self.max_daily_loss = float(cfg.get("trading.max_daily_loss_pct", 2))
        self.max_concentration = float(cfg.get("trading.max_concentration_pct", 30))
        self.con = duckdb.connect(self.db_path)
        self._dhan = DhanClient()
        self._init_db()
        logger.info(f"TradeExecutor initialized (mode={self.mode})")

    def _init_db(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                dhan_order_id TEXT,
                security_id TEXT,
                symbol TEXT,
                direction TEXT,
                order_type TEXT,
                quantity INTEGER,
                price REAL,
                trigger_price REAL,
                status TEXT DEFAULT 'PENDING',
                filled_qty INTEGER DEFAULT 0,
                avg_price REAL,
                mode TEXT DEFAULT 'paper',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                executed_at TIMESTAMP,
                cancelled_at TIMESTAMP,
                reason TEXT,
                sl_price REAL,
                tp_price REAL
            )
        """)
        # Add columns if they don't exist (for existing DBs)
        for col in ["sl_price REAL", "tp_price REAL"]:
            try:
                self.con.execute(f"ALTER TABLE orders ADD COLUMN {col}")
            except:
                pass
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                security_id TEXT,
                symbol TEXT,
                direction TEXT,
                quantity INTEGER,
                entry_price REAL,
                current_price REAL,
                unrealized_pnl REAL,
                realized_pnl REAL DEFAULT 0,
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                closed_at TIMESTAMP,
                status TEXT DEFAULT 'OPEN'
            )
        """)
        # Track daily P&L for max loss check
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date DATE DEFAULT CURRENT_DATE,
                total_realized REAL DEFAULT 0,
                total_unrealized REAL DEFAULT 0,
                peak_value REAL DEFAULT 0,
                PRIMARY KEY (date)
            )
        """)

    def _next_id(self):
        ts = int(time.time() * 1000)
        return f"ORD_{ts}_{hash(self) % 10000:04d}"

    # ── Order Placement ──

    async def place_order(self, symbol, direction, quantity, order_type=OrderType.MARKET,
                          price=None, trigger_price=None, duration=OrderDuration.DAY):
        """Place an order (paper or live based on mode)."""
        # Resolve security
        sid = self._resolve_security_id(symbol)
        if not sid:
            return {"error": f"Unknown symbol: {symbol}"}

        # Risk checks
        risk_ok = await self._check_risk(symbol, direction, quantity, price)
        if risk_ok and risk_ok.get("blocked"):
            logger.warning(f"Order blocked by risk: {risk_ok.get('reason')}")
            return {"error": risk_ok.get("reason")}

        order_id = self._next_id()

        if self.mode == "live":
            if os.environ.get("ALLOW_LIVE_TRADING") != "YES_I_UNDERSTAND":
                logger.error("Live trading blocked: ALLOW_LIVE_TRADING=YES_I_UNDERSTAND not set")
                self._record_order(order_id, sid, symbol, direction, order_type, quantity, price, trigger_price,
                                   status=OrderStatus.REJECTED, reason="ALLOW_LIVE_TRADING not set")
                return {"error": "Live trading disabled. Set ALLOW_LIVE_TRADING=YES_I_UNDERSTAND to enable."}
            dhan_resp = await self._place_dhan_order(sid, symbol, direction, quantity, order_type, price, trigger_price, duration)
            if dhan_resp.get("error"):
                self._record_order(order_id, sid, symbol, direction, order_type, quantity, price, trigger_price,
                                   status=OrderStatus.REJECTED, reason=dhan_resp["error"])
                return dhan_resp
            dhan_order_id = dhan_resp.get("orderId", "")
            filled_qty = dhan_resp.get("filledQty", quantity)
            avg_price = dhan_resp.get("avgPrice", price or 0)
            self._record_order(order_id, sid, symbol, direction, order_type, quantity, price, trigger_price,
                               status=OrderStatus.EXECUTED, dhan_order_id=dhan_order_id,
                               filled_qty=filled_qty, avg_price=avg_price, executed_at=datetime.now())
            self._update_position(symbol, direction, filled_qty, avg_price)
            return {"order_id": order_id, "dhan_order_id": dhan_order_id, "status": "EXECUTED",
                    "filled_qty": filled_qty, "avg_price": avg_price}
        else:
            # Paper: simulate market order execution at current price
            exec_price = price or await self._get_last_price(sid)
            if not exec_price:
                return {"error": "Could not determine execution price"}
            self._record_order(order_id, sid, symbol, direction, order_type, quantity, exec_price, trigger_price,
                               status=OrderStatus.EXECUTED, filled_qty=quantity, avg_price=exec_price,
                               executed_at=datetime.now())
            self._update_position(symbol, direction, quantity, exec_price)
            logger.info(f"[PAPER] {direction} {quantity} {symbol} @ ₹{exec_price}")
            return {"order_id": order_id, "dhan_order_id": "", "status": "EXECUTED",
                    "filled_qty": quantity, "avg_price": exec_price, "mode": "paper"}

    async def place_bracket_order(self, symbol, direction, quantity, sl_price, tp_price,
                                   order_type=OrderType.MARKET, price=None):
        """Place a bracket order with SL/TP (live uses Dhan Super Orders, paper simulates)."""
        sid = self._resolve_security_id(symbol)
        if not sid:
            return {"error": f"Unknown symbol: {symbol}"}

        risk_ok = await self._check_risk(symbol, direction, quantity, price)
        if risk_ok and risk_ok.get("blocked"):
            return {"error": risk_ok.get("reason")}

        order_id = self._next_id()

        if self.mode == "live":
            dhan_resp = await self._place_dhan_super_order(sid, symbol, direction, quantity,
                                                           order_type, price, sl_price, tp_price)
            if dhan_resp.get("error"):
                self._record_order(order_id, sid, symbol, direction, order_type, quantity, price, None,
                                   status=OrderStatus.REJECTED, reason=dhan_resp["error"])
                return dhan_resp
            dhan_order_id = dhan_resp.get("orderId", "")
            filled_qty = dhan_resp.get("filledQty", quantity)
            avg_price = dhan_resp.get("avgPrice", price or 0)
            self._record_order(order_id, sid, symbol, direction, order_type, quantity, price, None,
                               status=OrderStatus.EXECUTED, dhan_order_id=dhan_order_id,
                               filled_qty=filled_qty, avg_price=avg_price, executed_at=datetime.now(),
                               sl_price=sl_price, tp_price=tp_price)
            self._update_position(symbol, direction, filled_qty, avg_price)
            return {"order_id": order_id, "dhan_order_id": dhan_order_id, "status": "EXECUTED",
                    "filled_qty": filled_qty, "avg_price": avg_price, "sl_price": sl_price, "tp_price": tp_price}
        else:
            exec_price = price or await self._get_last_price(sid)
            if not exec_price:
                return {"error": "Could not determine execution price"}
            self._record_order(order_id, sid, symbol, direction, order_type, quantity, exec_price, None,
                               status=OrderStatus.EXECUTED, filled_qty=quantity, avg_price=exec_price,
                               executed_at=datetime.now(), sl_price=sl_price, tp_price=tp_price)
            self._update_position(symbol, direction, quantity, exec_price)
            logger.info(f"[PAPER BRACKET] {direction} {quantity} {symbol} @ ₹{exec_price} SL={sl_price} TP={tp_price}")
            return {"order_id": order_id, "dhan_order_id": "", "status": "EXECUTED",
                    "filled_qty": quantity, "avg_price": exec_price, "mode": "paper",
                    "sl_price": sl_price, "tp_price": tp_price}

    async def _place_dhan_super_order(self, security_id, symbol, direction, quantity, order_type, price, sl_price, tp_price):
        """Place a live super order via Dhan API."""
        try:
            from dhan_client import DhanClient
            dhan = DhanClient()
            dir_val = direction.value if hasattr(direction, 'value') else direction
            seg = "NSE_EQ"
            prod = "INTRADAY"
            ot = "MARKET" if order_type == OrderType.MARKET else "LIMIT"
            if dir_val in ("BUY", "SELL"):
                symbol_info = self.con.execute(
                    "SELECT symbol FROM securities WHERE security_id = ?", [security_id]
                ).fetchone()
                if symbol_info:
                    symbol = symbol_info[0]
            resp = await dhan.place_super_order(
                symbol=symbol,
                exchange_segment=seg,
                product_type=prod,
                order_type=ot,
                quantity=quantity,
                price=price or 0,
                trigger_price=None,
                sl_price=sl_price,
                tp_price=tp_price,
            )
            if isinstance(resp, dict) and "orderId" in resp:
                return {"orderId": resp["orderId"], "filledQty": quantity, "avgPrice": price or 0}
            error = resp.get("remark") or resp.get("errorMessage") or str(resp)
            return {"error": error}
        except Exception as e:
            logger.warning(f"Super order failed: {e}")
            return {"error": str(e)}

    async def _place_dhan_order(self, security_id, symbol, direction, quantity, order_type, price, trigger_price, duration):
        """Place real order via Dhan API."""
        try:
            payload = {
                "dhanClientId": self._dhan.CLIENT_ID,
                "transactionType": direction.value if isinstance(direction, TradeDirection) else direction,
                "exchangeSegment": "NSE_EQ",
                "productType": "CNC",
                "orderType": order_type.value if isinstance(order_type, OrderType) else order_type,
                "validity": duration.value if isinstance(duration, OrderDuration) else duration,
                "securityId": security_id,
                "quantity": quantity,
            }
            if price and order_type in (OrderType.LIMIT, OrderType.SL, OrderType.SL_MARKET):
                payload["price"] = price
            if trigger_price:
                payload["triggerPrice"] = trigger_price
            # Map to Dhan API format
            result = await self._dhan._request("POST", "/orders", payload)
            return result
        except AuthExpiredError:
            return {"error": "Dhan API token expired"}
        except Exception as e:
            return {"error": str(e)}

    # ── Position Management ──

    def _update_position(self, symbol, direction, quantity, price):
        """Update or create position. BUY increases long / reduces short, SELL reduces long / increases short."""
        direction = direction.value if isinstance(direction, TradeDirection) else direction
        pos = self.con.execute(
            "SELECT position_id, direction, quantity, entry_price FROM positions WHERE symbol = ? AND status = 'OPEN'",
            [symbol]
        ).fetchone()

        if direction == TradeDirection.BUY.value:
            if pos:
                pos_id, pos_dir, pos_qty, pos_entry = pos
                if pos_dir == POSITION_TYPE_LONG:
                    # Add to position (average)
                    new_qty = pos_qty + quantity
                    avg_price = ((pos_entry * pos_qty) + (price * quantity)) / new_qty
                    self.con.execute("UPDATE positions SET quantity=?, entry_price=?, updated_at=CURRENT_TIMESTAMP WHERE position_id=?",
                                     [new_qty, avg_price, pos_id])
                else:
                    # Reduce or close short
                    remaining = pos_qty - quantity
                    if remaining > 0:
                        realized = (pos_entry - price) * quantity  # Short: sell high, buy back low
                        self.con.execute("UPDATE positions SET quantity=?, realized_pnl=realized_pnl+?, updated_at=CURRENT_TIMESTAMP WHERE position_id=?",
                                         [remaining, realized, pos_id])
                        self._update_daily_pnl(realized)
                    elif remaining == 0:
                        realized = (pos_entry - price) * quantity
                        self.con.execute("UPDATE positions SET status='CLOSED', realized_pnl=realized_pnl+?, current_price=?, closed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE position_id=?",
                                         [realized, price, pos_id])
                        self._update_daily_pnl(realized)
                    else:
                        # Close short + open long
                        realized = (pos_entry - price) * pos_qty
                        self.con.execute("UPDATE positions SET status='CLOSED', realized_pnl=realized_pnl+?, current_price=?, closed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE position_id=?",
                                         [realized, price, pos_id])
                        self._update_daily_pnl(realized)
                        new_qty = abs(remaining)
                        new_id = f"POS_{symbol}_{int(time.time() * 1000)}"
                        self.con.execute("INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,NULL,NULL,'OPEN')",
                                         [new_id, self._resolve_security_id(symbol), symbol, POSITION_TYPE_LONG, new_qty, price, price, 0, 0])
        else:  # SELL
            if pos:
                pos_id, pos_dir, pos_qty, pos_entry = pos
                if pos_dir == POSITION_TYPE_SHORT:
                    new_qty = pos_qty + quantity
                    avg_price = ((pos_entry * pos_qty) + (price * quantity)) / new_qty
                    self.con.execute("UPDATE positions SET quantity=?, entry_price=?, updated_at=CURRENT_TIMESTAMP WHERE position_id=?",
                                     [new_qty, avg_price, pos_id])
                else:
                    remaining = pos_qty - quantity
                    if remaining > 0:
                        realized = (price - pos_entry) * quantity
                        self.con.execute("UPDATE positions SET quantity=?, realized_pnl=realized_pnl+?, updated_at=CURRENT_TIMESTAMP WHERE position_id=?",
                                         [remaining, realized, pos_id])
                        self._update_daily_pnl(realized)
                    elif remaining == 0:
                        realized = (price - pos_entry) * quantity
                        self.con.execute("UPDATE positions SET status='CLOSED', realized_pnl=realized_pnl+?, current_price=?, closed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE position_id=?",
                                         [realized, price, pos_id])
                        self._update_daily_pnl(realized)
                    else:
                        realized = (price - pos_entry) * pos_qty
                        self.con.execute("UPDATE positions SET status='CLOSED', realized_pnl=realized_pnl+?, current_price=?, closed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE position_id=?",
                                         [realized, price, pos_id])
                        self._update_daily_pnl(realized)
                        new_qty = abs(remaining)
                        new_id = f"POS_{symbol}_{int(time.time() * 1000)}"
                        self.con.execute("INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,NULL,NULL,'OPEN')",
                                         [new_id, self._resolve_security_id(symbol), symbol, POSITION_TYPE_SHORT, new_qty, price, price, 0, 0])
            else:
                # No existing position, sell short
                new_id = f"POS_{symbol}_{int(time.time() * 1000)}"
                self.con.execute("INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,NULL,NULL,'OPEN')",
                                 [new_id, self._resolve_security_id(symbol), symbol, POSITION_TYPE_SHORT, quantity, price, price, 0, 0])

    # ── Risk Checks ──

    async def _check_risk(self, symbol, direction, quantity, price=None):
        """Check trading limits. Returns None if OK, dict with blocked=True if rejected."""
        portfolio_value = await self._get_portfolio_value()
        position_value = (price or 0) * quantity

        # Max position size per security
        if portfolio_value > 0 and position_value > portfolio_value * (self.max_position_pct / 100):
            return {"blocked": True, "reason": f"Position size ₹{position_value:.0f} exceeds {self.max_position_pct}% of portfolio (₹{portfolio_value:.0f})"}

        # Max daily loss
        dpnl = self.con.execute("SELECT total_realized FROM daily_pnl WHERE date = CURRENT_DATE").fetchone()
        if dpnl and dpnl[0] and abs(dpnl[0]) > portfolio_value * (self.max_daily_loss / 100):
            return {"blocked": True, "reason": f"Daily loss limit ({self.max_daily_loss}%) breached"}

        # Concentration: single stock max % of portfolio
        open_pos = self.con.execute("SELECT symbol, quantity, entry_price FROM positions WHERE status = 'OPEN'").fetchall()
        if open_pos:
            total_exposure = sum(abs(r[1] * r[2]) for r in open_pos if r[2])
            new_exposure = position_value
            if total_exposure + new_exposure > 0 and (total_exposure + new_exposure) > portfolio_value * (self.max_concentration / 100):
                return {"blocked": True, "reason": f"Total exposure would breach {self.max_concentration}% concentration limit"}

        # Advanced risk checks via RiskManager
        try:
            from risk_manager import RiskManager
            rm = RiskManager(self.db_path)
            current_positions = self.con.execute(
                "SELECT security_id, symbol, direction, quantity, entry_price FROM positions WHERE status = 'OPEN'"
            ).fetchdf()
            if len(current_positions) > 0:
                # Sector exposure check for the proposed trade
                try:
                    from sector_analysis import get_sector_for_ticker
                    new_sector = get_sector_for_ticker(symbol)
                    sector_positions = current_positions[current_positions["symbol"] != symbol]
                    if len(sector_positions) > 0:
                        sector_values = {}
                        for _, r in sector_positions.iterrows():
                            sec = get_sector_for_ticker(r["symbol"])
                            sector_values[sec] = sector_values.get(sec, 0) + abs(r["quantity"] * r["entry_price"])
                        new_sector_value = sector_values.get(new_sector, 0) + position_value
                        if portfolio_value > 0 and new_sector_value / portfolio_value > 0.4:
                            rm.con.close()
                            return {"blocked": True, "reason": f"Sector {new_sector} exposure would exceed 40%"}
                except:
                    pass

                # Correlation check: if highly correlated position exists, block
                try:
                    if len(current_positions) >= 2:
                        corr = rm.position_correlation(current_positions.to_dict("records"))
                        if corr.get("avg_correlation", 0) > 0.85:
                            rm.con.close()
                            return {"blocked": True, "reason": f"Portfolio correlation too high ({corr['avg_correlation']:.2f})"}
                except:
                    pass

                # Stress test: block if -5% drawdown would exceed max loss
                try:
                    stress = rm.stress_test(current_positions.to_dict("records"))
                    for s in stress:
                        if s["scenario"] == "-5%" and abs(s["total_pnl"]) > portfolio_value * 0.08:
                            rm.con.close()
                            return {"blocked": True, "reason": f"5% market drop would exceed 8% portfolio loss"}
                except:
                    pass
            rm.con.close()
        except:
            pass

        return None

    async def _get_portfolio_value(self):
        """Estimate portfolio value from positions + cash."""
        # For paper mode, use a fixed starting capital
        if self.mode == "paper":
            return float(get_config().get("trading.paper_capital", 100000))

        # For live, try Dhan API
        try:
            limits = await self._dhan.get_fund_limits()
            return float(limits.get("availabelBalance", 0))
        except Exception as e:
            logger.debug(f"Live portfolio value failed: {e}")
            return 0

    # ── Helpers ──

    def _resolve_security_id(self, symbol):
        r = self.con.execute("SELECT security_id FROM securities WHERE symbol = ?", [symbol.upper()]).fetchone()
        if r:
            return str(r[0])
        return None

    async def _get_last_price(self, security_id):
        r = self.con.execute(
            "SELECT close FROM daily WHERE security_id = ? ORDER BY date DESC LIMIT 1", [security_id]
        ).fetchone()
        return float(r[0]) if r else None

    def _record_order(self, order_id, security_id, symbol, direction, order_type, quantity, price, trigger_price,
                      status=OrderStatus.PENDING, dhan_order_id="", filled_qty=0, avg_price=0,
                      executed_at=None, reason="", sl_price=None, tp_price=None):
        self.con.execute("""
            INSERT OR REPLACE INTO orders
            (order_id, dhan_order_id, security_id, symbol, direction, order_type, quantity, price, trigger_price,
             status, filled_qty, avg_price, mode, created_at, executed_at, reason, sl_price, tp_price)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,?,?,?,?)
        """, [order_id, dhan_order_id, security_id, symbol, direction,
              order_type.value if isinstance(order_type, OrderType) else order_type,
              quantity, price or 0, trigger_price or 0,
              status.value if isinstance(status, OrderStatus) else status,
              filled_qty, avg_price, self.mode,
              executed_at or None, reason, sl_price, tp_price])

    def _update_daily_pnl(self, realized):
        self.con.execute("""
            INSERT INTO daily_pnl (date, total_realized) VALUES (CURRENT_DATE, ?)
            ON CONFLICT (date) DO UPDATE SET total_realized = total_realized + ?
        """, [realized, realized])

    # ── Query Methods ──

    def get_open_positions(self):
        rows = self.con.execute("""
            SELECT p.*, d.close as current_price FROM positions p
            LEFT JOIN (SELECT security_id, close, date FROM daily WHERE date = (SELECT MAX(date) FROM daily)) d
            ON p.security_id = d.security_id
            WHERE p.status = 'OPEN'
            ORDER BY p.opened_at DESC
        """).fetchdf()
        if len(rows) > 0:
            rows["unrealized_pnl"] = rows.apply(
                lambda r: (r["current_price"] - r["entry_price"]) * r["quantity"] if r["direction"] == "LONG"
                else (r["entry_price"] - r["current_price"]) * r["quantity"], axis=1
            )
        return rows.to_dict("records") if len(rows) > 0 else []

    def get_order_history(self, limit=50):
        rows = self.con.execute("""
            SELECT * FROM orders ORDER BY created_at DESC LIMIT ?
        """, [limit]).fetchdf()
        return rows.to_dict("records") if len(rows) > 0 else []

    def get_daily_pnl(self, days=7):
        rows = self.con.execute("""
            SELECT * FROM daily_pnl ORDER BY date DESC LIMIT ?
        """, [days]).fetchdf()
        return rows.to_dict("records") if len(rows) > 0 else []

    async def get_portfolio_summary(self):
        positions = self.con.execute("SELECT * FROM positions WHERE status = 'OPEN'").fetchdf()
        orders_today = self.con.execute(
            "SELECT COUNT(*) as cnt, SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed FROM orders WHERE date(created_at) = CURRENT_DATE"
        ).fetchone()

        total_invested = 0
        total_pnl = 0
        if len(positions) > 0:
            for _, r in positions.iterrows():
                val = r["quantity"] * r["entry_price"]
                total_invested += val
                if r["direction"] == POSITION_TYPE_LONG:
                    total_pnl += (r.get("current_price", r["entry_price"]) - r["entry_price"]) * r["quantity"]
                else:
                    total_pnl += (r["entry_price"] - r.get("current_price", r["entry_price"])) * r["quantity"]

        return {
            "mode": self.mode,
            "capital": await self._get_portfolio_value(),
            "invested": total_invested,
            "unrealized_pnl": round(total_pnl, 2),
            "unrealized_pnl_pct": round((total_pnl / total_invested * 100) if total_invested > 0 else 0, 2),
            "open_positions": len(positions),
            "orders_today": orders_today[0] if orders_today else 0,
            "executed_today": orders_today[1] if orders_today else 0,
        }

    def close(self):
        try:
            self.con.close()
        except:
            pass
