"""Auto-Trader — watches confidence engine signals, places paper/live trades, monitors positions.

Architecture:
- After each analysis cycle, AutoTrader checks for high-confidence (≥75%) trade signals
- Places order via TradeExecutor (paper by default)
- Monitors open positions on a separate loop (SL/TP check every 60s)
- Stores each trade decision in memory as a "trade_decision" fact
- Prevents overtrading: max 1 trade per symbol per day, max 3 concurrent positions

Config:
  trading.auto_trade: true|false (default: false until proven)
  trading.min_confidence: 75 (minimum confidence % to auto-trade)
  trading.max_concurrent: 3 (max open positions)
  trading.max_daily_trades: 5 (max trades per day)
  trading.sl_atr_multiplier: 1.5 (stop-loss = ATR × this multiplier)
  trading.tp_atr_multiplier: 3.0 (take-profit = ATR × this multiplier)
"""

import asyncio, logging, math
from datetime import datetime, date
from config_manager import get_config
from trade_executor import TradeExecutor, TradeDirection, OrderType, OrderDuration
from confidence_engine import get_engine

logger = logging.getLogger("auto_trader")


class AutoTrader:
    def __init__(self):
        cfg = get_config()
        self.enabled = cfg.get("trading.auto_trade", False)
        self.min_confidence = float(cfg.get("trading.min_confidence", 75))
        self.max_concurrent = int(cfg.get("trading.max_concurrent", 3))
        self.max_daily_trades = int(cfg.get("trading.max_daily_trades", 5))
        self.sl_atr = float(cfg.get("trading.sl_atr_multiplier", 1.5))
        self.tp_atr = float(cfg.get("trading.tp_atr_multiplier", 3.0))
        self._monitor_task = None
        self.running = False

    def start(self):
        if not self._monitor_task or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info(f"Auto-trader started (enabled={self.enabled}, min_conf={self.min_confidence}%)")

    def stop(self):
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()

    async def evaluate_and_trade(self, ticker, features, price, signal_result):
        """Called by auto-learner after analysis. Checks if we should place a trade."""
        if not self.enabled:
            return None

        if not signal_result or signal_result.get("signal") != "strong":
            return None

        confidence = signal_result.get("confidence", 0)
        if confidence < self.min_confidence:
            return None

        direction = signal_result.get("direction", "")
        if direction not in ("BUY", "SELL"):
            return None

        # Check limits before trading
        limit_check = await self._check_trade_limits(ticker)
        if limit_check.get("blocked"):
            logger.info(f"  Auto-trade blocked for {ticker}: {limit_check['reason']}")
            return limit_check

        # Get ATR for position sizing
        atr = features.get("atr_14")
        if not atr or math.isnan(atr) or atr <= 0:
            atr = price * 0.015  # default 1.5% of price

        # Position sizing: risk 1% of capital per trade, adjusted by regime
        cfg = get_config()
        capital = float(cfg.get("trading.paper_capital", 100000))
        risk_per_trade = capital * 0.01
        sl_distance = atr * self.sl_atr

        # Regime-based sizing adjustment
        try:
            from market_regime import MarketRegime
            mr = MarketRegime()
            sizing_adj = mr.get_position_sizing_adjustment()
            risk_per_trade *= sizing_adj
        except Exception as e:
            logger.debug(f"Regime sizing failed: {e}")

        quantity = max(1, int(risk_per_trade / sl_distance))

        # Calculate SL and TP prices
        if direction == "BUY":
            sl_price = price - sl_distance
            tp_price = price + (atr * self.tp_atr)
        else:
            sl_price = price + sl_distance
            tp_price = price - (atr * self.tp_atr)

        # Use bracket order (with SL/TP) if available
        te = TradeExecutor()
        try:
            if te.mode == "live":
                result = await te.place_bracket_order(
                    ticker,
                    TradeDirection.BUY if direction == "BUY" else TradeDirection.SELL,
                    quantity,
                    round(sl_price, 2),
                    round(tp_price, 2),
                    OrderType.MARKET,
                )
            else:
                result = await te.place_order(
                    ticker,
                    TradeDirection.BUY if direction == "BUY" else TradeDirection.SELL,
                    quantity,
                    OrderType.MARKET,
                    duration=OrderDuration.DAY,
                )
            te.close()

            if result.get("error"):
                logger.warning(f"  Auto-trade failed for {ticker}: {result['error']}")
                return result

            # Enrich result with SL/TP
            result["sl_price"] = round(sl_price, 2)
            result["tp_price"] = round(tp_price, 2)
            result["direction"] = direction
            result["reasoning"] = signal_result.get("summary", "")
            result["confirmations"] = signal_result.get("confirmations", [])
            result["confidence"] = confidence

            # Store trade decision in memory
            self._store_trade_decision(ticker, result)

            logger.info(
                f"  [AUTO-TRADE] {direction} {quantity} {ticker} @ ₹{price:.2f} "
                f"| SL ₹{sl_price:.2f} | TP ₹{tp_price:.2f} | "
                f"Confidence: {confidence}%"
            )

            # Broadcast via event bus
            try:
                from event_bus import get_event_bus
                get_event_bus().emit("auto_trade", result)
            except Exception as e:
                logger.debug(f"Event bus emit failed: {e}")

            # WhatsApp notification
            try:
                from whatsapp_notifier import send_trade_execution
                await send_trade_execution(ticker, direction, quantity, price, result.get("order_id", ""))
            except Exception as e:
                logger.debug(f"WhatsApp notify failed: {e}")

            return result

        except Exception as e:
            logger.warning(f"  Auto-trade error for {ticker}: {e}")
            return {"error": str(e)}

    async def _check_trade_limits(self, ticker):
        """Check all trading limits before placing a trade."""
        te = TradeExecutor()
        try:
            # Max concurrent positions
            positions = te.get_open_positions()
            if len(positions) >= self.max_concurrent:
                return {"blocked": True, "reason": f"Max concurrent positions ({self.max_concurrent}) reached"}

            # Already have a position in this ticker?
            for p in positions:
                if p.get("symbol") == ticker:
                    return {"blocked": True, "reason": f"Already have an open position in {ticker}"}

            # Max daily trades
            today = date.today().isoformat()
            orders = te.get_order_history(limit=100)
            today_trades = sum(1 for o in orders if o.get("created_at", "").startswith(today) and o.get("status") == "EXECUTED")
            if today_trades >= self.max_daily_trades:
                return {"blocked": True, "reason": f"Daily trade limit ({self.max_daily_trades}) reached"}

            return {}
        finally:
            te.close()

    async def check_positions(self):
        """Periodic check: update prices, check SL/TP, close if needed."""
        te = TradeExecutor()
        try:
            positions = te.get_open_positions()
            if not positions:
                return

            db = te.con
            for pos in positions:
                sid = pos.get("security_id")
                if not sid:
                    continue
                current = db.execute(
                    "SELECT close FROM daily WHERE security_id = ? ORDER BY date DESC LIMIT 1",
                    [sid]
                ).fetchone()
                if not current:
                    continue
                current_price = float(current[0])

                direction = pos.get("direction")
                entry = pos.get("entry_price", 0)
                qty = pos.get("quantity", 0)
                symbol = pos.get("symbol", "?")

                # Calculate SL and TP from stored metadata
                # We store them in orders table for the position
                order = db.execute(
                    "SELECT price, reason FROM orders WHERE symbol = ? AND status = 'EXECUTED' ORDER BY created_at DESC LIMIT 1",
                    [symbol]
                ).fetchone()

                if not order:
                    continue

                # Simple SL: 1.5x ATR from entry
                atr_row = db.execute(
                    "SELECT high, low, close FROM daily WHERE security_id = ? ORDER BY date DESC LIMIT 20",
                    [sid]
                ).fetchall()
                if atr_row and len(atr_row) >= 14:
                    highs = [float(r[0]) for r in atr_row]
                    lows = [float(r[1]) for r in atr_row]
                    closes = [float(r[2]) for r in atr_row]
                    atr_val = self._compute_atr(highs, lows, closes)
                    sl_dist = atr_val * self.sl_atr
                    tp_dist = atr_val * self.tp_atr
                    # Apply regime-based adjustment
                    try:
                        from market_regime import MarketRegime
                        mr = MarketRegime()
                        sizing_adj = mr.get_position_sizing_adjustment()
                        sl_dist *= (1 / max(sizing_adj, 0.3))
                        tp_dist *= sizing_adj
                    except:
                        pass
                else:
                    sl_dist = entry * 0.02
                    tp_dist = entry * 0.04
                    atr_val = entry * 0.01

                # Check for persisted SL/TP from bracket order first
                try:
                    persisted = db.execute(
                        "SELECT sl_price, tp_price FROM orders WHERE symbol = ? AND status = 'EXECUTED' ORDER BY created_at DESC LIMIT 1",
                        [symbol]
                    ).fetchone()
                    if persisted and persisted[0]:
                        sl_price = float(persisted[0])
                        tp_price = float(persisted[1]) if persisted[1] else tp_price
                except:
                    pass

                if direction == "LONG":
                    sl_price = sl_price or (entry - sl_dist)
                    tp_price = tp_price or (entry + tp_dist)
                    hit_sl = current_price <= sl_price
                    hit_tp = current_price >= tp_price
                    # Trailing stop: if price moved up > 1x ATR, trail SL to entry + 0.5x ATR
                    if current_price > entry + atr_val:
                        trail_sl = entry + atr_val * 0.5
                        if trail_sl > sl_price:
                            sl_price = trail_sl
                            logger.debug(f"  Trail SL for {symbol}: {sl_price:.2f}")
                else:
                    sl_price = sl_price or (entry + sl_dist)
                    tp_price = tp_price or (entry - tp_dist)
                    hit_sl = current_price >= sl_price
                    hit_tp = current_price <= tp_price
                    # Trailing stop: if price moved down > 1x ATR, trail SL to entry - 0.5x ATR
                    if current_price < entry - atr_val:
                        trail_sl = entry - atr_val * 0.5
                        if trail_sl < sl_price:
                            sl_price = trail_sl
                            logger.debug(f"  Trail SL for {symbol}: {sl_price:.2f}")

                if hit_sl:
                    logger.info(f"  [AUTO-TRADE] SL hit for {symbol} ({direction}) @ ₹{current_price:.2f}")
                    await te.place_order(
                        symbol,
                        TradeDirection.SELL if direction == "LONG" else TradeDirection.BUY,
                        qty,
                        OrderType.MARKET,
                        duration=OrderDuration.IOC,
                    )
                    self._store_trade_outcome(symbol, "stop_loss", entry, current_price, qty, direction)
                    try:
                        from event_bus import get_event_bus
                        get_event_bus().emit("auto_trade_close", {"symbol": symbol, "reason": "stop_loss", "price": current_price})
                    except:
                        pass
                    try:
                        from whatsapp_notifier import send_trade_close
                        pnl = (current_price - entry) * qty * (-1 if direction == "SELL" else 1)
                        await send_trade_close(symbol, "stop_loss", entry, current_price, pnl)
                    except:
                        pass

                elif hit_tp:
                    logger.info(f"  [AUTO-TRADE] TP hit for {symbol} ({direction}) @ ₹{current_price:.2f}")
                    await te.place_order(
                        symbol,
                        TradeDirection.SELL if direction == "LONG" else TradeDirection.BUY,
                        qty,
                        OrderType.MARKET,
                        duration=OrderDuration.IOC,
                    )
                    self._store_trade_outcome(symbol, "take_profit", entry, current_price, qty, direction)
                    try:
                        from event_bus import get_event_bus
                        get_event_bus().emit("auto_trade_close", {"symbol": symbol, "reason": "take_profit", "price": current_price})
                    except:
                        pass
                    try:
                        from whatsapp_notifier import send_trade_close
                        pnl = (current_price - entry) * qty * (-1 if direction == "SELL" else 1)
                        await send_trade_close(symbol, "take_profit", entry, current_price, pnl)
                    except:
                        pass
        finally:
            te.close()

    async def _monitor_loop(self):
        """Background loop monitoring open positions for SL/TP."""
        self.running = True
        while self.running:
            try:
                if self.enabled:
                    await self.check_positions()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Auto-trader monitor error: {e}")
                await asyncio.sleep(120)

    def _store_trade_decision(self, ticker, result):
        """Store trade rationale in memory for later review."""
        try:
            from memory_manager import MemoryManager
            mm = MemoryManager()
            reasoning = (
                f"Auto-trade: {result.get('direction')} {ticker} @ ₹{result.get('avg_price',0):.2f} "
                f"x {result.get('filled_qty',0)} shares. "
                f"Confidence: {result.get('confidence',0)}%. "
                f"SL: ₹{result.get('sl_price',0):.2f}, TP: ₹{result.get('tp_price',0):.2f}. "
                f"Reasons: {'; '.join(result.get('confirmations', []))}. "
                f"Order ID: {result.get('order_id', '')}"
            )
            mm.store_fact(ticker, "trade_decision", reasoning,
                          confidence=result.get("confidence", 0) / 100,
                          source="auto_trader")
            mm.store_knowledge(
                f"Trade Decision: {ticker} ({result.get('direction')})",
                reasoning,
                category="trade_decision", ticker=ticker,
                source="auto_trader",
                tags=f"{result.get('direction','')},{ticker}"
            )
        except Exception as e:
            logger.debug(f"Store trade decision: {e}")

    def _store_trade_outcome(self, ticker, reason, entry, exit_price, qty, direction):
        """Store trade outcome in memory for self-review."""
        try:
            pnl = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
            from memory_manager import MemoryManager
            mm = MemoryManager()
            outcome = (
                f"Trade closed: {ticker} {direction} x{qty}. "
                f"Entry: ₹{entry:.2f}, Exit: ₹{exit_price:.2f}. "
                f"Reason: {reason}. P&L: ₹{pnl:.2f}"
            )
            mm.store_fact(ticker, "trade_outcome", outcome,
                          confidence=0.9, source="auto_trader")
            mm.store_knowledge(
                f"Trade Outcome: {ticker} ({reason})",
                outcome,
                category="trade_outcome", ticker=ticker,
                source="auto_trader",
                tags=f"{reason},{ticker}"
            )
        except Exception as e:
            logger.debug(f"Store trade outcome: {e}")

    @staticmethod
    def _compute_atr(highs, lows, closes):
        if len(closes) < 2:
            return 0
        tr_sum = 0
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
            tr_sum += tr
        return tr_sum / (len(closes) - 1)


# Singleton
_instance = None

def get_auto_trader():
    global _instance
    if _instance is None:
        _instance = AutoTrader()
    return _instance
