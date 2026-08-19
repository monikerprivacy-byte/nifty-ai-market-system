"""Strategy Backtester — test trading strategies on historical data.

Built-in strategies:
- RSI Mean Reversion (RSI < oversold → buy, RSI > overbought → sell)
- SMA Crossover (SMA20 crosses above SMA50 → buy, reverse → sell)
- Bollinger Bounce (price touches lower band → buy, upper band → sell)
- Trend Following (uptrend → buy, downtrend → sell)

Returns: trade log, win rate, profit factor, max drawdown, Sharpe ratio.
"""

import logging, json, math
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config

logger = logging.getLogger("backtester")

STRATEGIES = {
    "rsi_mean_reversion": {
        "name": "RSI Mean Reversion",
        "description": "Buy when RSI(14) < 30 (oversold), sell when RSI(14) > 70 (overbought)",
        "params": {"rsi_period": 14, "oversold": 30, "overbought": 70, "sl_pct": None, "tp_pct": None},
    },
    "sma_crossover": {
        "name": "SMA Crossover",
        "description": "Buy when SMA20 crosses above SMA50, sell when SMA20 crosses below SMA50",
        "params": {"fast_period": 20, "slow_period": 50, "sl_pct": None, "tp_pct": None},
    },
    "bollinger_bounce": {
        "name": "Bollinger Bounce",
        "description": "Buy when price touches lower band, sell when price touches upper band",
        "params": {"bollinger_period": 20, "bollinger_std": 2, "sl_pct": None, "tp_pct": None},
    },
    "trend_follow": {
        "name": "Trend Following",
        "description": "Buy when structure is uptrend and RSI > 50, sell when structure is downtrend and RSI < 50",
        "params": {"sl_pct": None, "tp_pct": None},
    },
    "vwap_reversion": {
        "name": "VWAP Reversion",
        "description": "Buy when price < VWAP by 2x ATR, sell when price > VWAP by 2x ATR",
        "params": {"atr_multiplier": 2, "sl_pct": None, "tp_pct": None},
    },
}


class Backtester:
    def __init__(self, db_path=None, cost_bps=20):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.con = duckdb.connect(self.db_path)
        self.cost_bps = cost_bps  # brokerage + STT + slippage in basis points (20 = 0.2%)

    def list_strategies(self):
        return [{"id": k, **v} for k, v in STRATEGIES.items()]

    def get_strategy(self, strategy_id):
        return STRATEGIES.get(strategy_id)

    async def run(self, ticker, strategy_id, start_date=None, end_date=None, params=None):
        """Run backtest for a strategy on a ticker. Returns detailed results."""
        strategy = STRATEGIES.get(strategy_id)
        if not strategy:
            return {"error": f"Unknown strategy: {strategy_id}"}

        sid = self._lookup_id(ticker)
        if not sid:
            return {"error": f"Ticker not found: {ticker}"}

        # Merge user params with defaults
        s_params = dict(strategy["params"])
        if params:
            s_params.update(params)

        # Fetch daily data
        df = self.con.execute("""
            SELECT date, open, high, low, close, volume
            FROM daily WHERE security_id = ?
            ORDER BY date
        """, [sid]).fetchdf()

        if len(df) < 50:
            return {"error": f"Not enough data for {ticker} ({len(df)} rows, need 50+)"}

        df["date"] = pd.to_datetime(df["date"])
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        # Compute strategy signals
        signals = self._compute_signals(df, strategy_id, s_params)
        df["signal"] = signals

        # Degenerate parameter validation
        validation = self._validate_params(strategy_id, s_params)
        if validation.get("warnings"):
            for w in validation["warnings"]:
                logger.warning(f"Param warning for {strategy_id}/{ticker}: {w}")

        # Generate trades with SL/TP
        sl_pct = s_params.get("sl_pct")
        tp_pct = s_params.get("tp_pct")
        trades = self._generate_trades(df, sl_pct=sl_pct, tp_pct=tp_pct)
        metrics = self._compute_metrics(trades, df)

        result = {
            "ticker": ticker,
            "strategy": strategy["name"],
            "strategy_id": strategy_id,
            "period": {
                "from": str(df["date"].min().date()),
                "to": str(df["date"].max().date()),
                "trading_days": len(df),
            },
            "trades": trades,
            "metrics": metrics,
            "parameters": s_params,
        }
        return Backtester._clean_numpy(result)

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

    def _compute_signals(self, df, strategy_id, params):
        """Return series of signals: 1=buy, -1=sell, 0=hold."""
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        n = len(df)
        signals = np.zeros(n)

        if strategy_id == "rsi_mean_reversion":
            period = int(params.get("rsi_period", 14))
            overbought = float(params.get("overbought", 70))
            oversold = float(params.get("oversold", 30))
            rsi = self._rsi(closes, period, n)
            for i in range(period + 1, n):
                if rsi[i] is not None and not np.isnan(rsi[i]):
                    if rsi[i] < oversold:
                        signals[i] = 1
                    elif rsi[i] > overbought:
                        signals[i] = -1

        elif strategy_id == "sma_crossover":
            fast = int(params.get("fast_period", 20))
            slow = int(params.get("slow_period", 50))
            sma_fast = self._sma(closes, fast, n)
            sma_slow = self._sma(closes, slow, n)
            for i in range(slow + 1, n):
                if sma_fast[i] and sma_slow[i] and sma_fast[i - 1] and sma_slow[i - 1]:
                    if sma_fast[i - 1] <= sma_slow[i - 1] and sma_fast[i] > sma_slow[i]:
                        signals[i] = 1  # Golden cross
                    elif sma_fast[i - 1] >= sma_slow[i - 1] and sma_fast[i] < sma_slow[i]:
                        signals[i] = -1  # Death cross

        elif strategy_id == "bollinger_bounce":
            period = int(params.get("bollinger_period", 20))
            std_mult = float(params.get("bollinger_std", 2))
            mid = self._sma(closes, period, n)
            rolling_std = np.full(n, np.nan, dtype=float)
            for i in range(period - 1, n):
                rolling_std[i] = np.std(closes[i - period + 1:i + 1])
            for i in range(period, n):
                if mid[i] and not np.isnan(mid[i]) and rolling_std[i] and not np.isnan(rolling_std[i]):
                    upper = mid[i] + rolling_std[i] * std_mult
                    lower = mid[i] - rolling_std[i] * std_mult
                    if closes[i] <= lower:
                        signals[i] = 1
                    elif closes[i] >= upper:
                        signals[i] = -1

        elif strategy_id == "trend_follow":
            rsi = self._rsi(closes, 14, n)
            trend = self._detect_trend(closes, highs, lows, n)
            for i in range(30, n):
                if trend[i] == "uptrend" and rsi[i] is not None and rsi[i] > 50:
                    signals[i] = 1
                elif trend[i] == "downtrend" and rsi[i] is not None and rsi[i] < 50:
                    signals[i] = -1

        elif strategy_id == "vwap_reversion":
            atr_mult = float(params.get("atr_multiplier", 2))
            vwap = self._vwap(closes, volumes, n)
            atr = self._atr(highs, lows, closes, 14, n)
            for i in range(20, n):
                if vwap[i] and atr[i] and not np.isnan(vwap[i]) and not np.isnan(atr[i]):
                    if closes[i] < vwap[i] - atr[i] * atr_mult:
                        signals[i] = 1
                    elif closes[i] > vwap[i] + atr[i] * atr_mult:
                        signals[i] = -1

        return signals

    def _generate_trades(self, df, sl_pct=None, tp_pct=None):
        """Convert signal series to trade list with optional SL/TP and transaction costs."""
        cost_factor = self.cost_bps / 10000  # Convert bps to decimal (20 bps = 0.002)
        trades = []
        in_position = False
        entry_price = 0
        entry_date = None
        entry_signal = 0
        intra_trade_dd = 0  # Track max intra-trade drawdown

        for i in range(len(df)):
            sig = df["signal"].iloc[i]

            if not in_position:
                if sig == 1 or sig == -1:
                    in_position = True
                    entry_price = df["close"].iloc[i]
                    entry_date = df["date"].iloc[i]
                    entry_signal = 1 if sig == 1 else -1
                    intra_trade_dd = 0
                continue

            current_price = df["close"].iloc[i]

            # Check SL/TP (use high/low to simulate intraday hits)
            if sl_pct is not None and entry_signal == 1:
                sl_price = entry_price * (1 - sl_pct)
                if df["low"].iloc[i] <= sl_price:
                    exit_price = sl_price
                    pnl_pct = (exit_price - entry_price) / entry_price * 100 - cost_factor * 200
                    trades.append(self._make_trade(entry_date, df["date"].iloc[i], entry_signal, entry_price, exit_price, pnl_pct, "SL"))
                    in_position = False
                    continue
            if sl_pct is not None and entry_signal == -1:
                sl_price = entry_price * (1 + sl_pct)
                if df["high"].iloc[i] >= sl_price:
                    exit_price = sl_price
                    pnl_pct = (exit_price - entry_price) / entry_price * 100 * (-1) - cost_factor * 200
                    trades.append(self._make_trade(entry_date, df["date"].iloc[i], entry_signal, entry_price, exit_price, pnl_pct, "SL"))
                    in_position = False
                    continue

            if tp_pct is not None and entry_signal == 1:
                tp_price = entry_price * (1 + tp_pct)
                if df["high"].iloc[i] >= tp_price:
                    exit_price = tp_price
                    pnl_pct = (exit_price - entry_price) / entry_price * 100 - cost_factor * 200
                    trades.append(self._make_trade(entry_date, df["date"].iloc[i], entry_signal, entry_price, exit_price, pnl_pct, "TP"))
                    in_position = False
                    continue
            if tp_pct is not None and entry_signal == -1:
                tp_price = entry_price * (1 - tp_pct)
                if df["low"].iloc[i] <= tp_price:
                    exit_price = tp_price
                    pnl_pct = (exit_price - entry_price) / entry_price * 100 * (-1) - cost_factor * 200
                    trades.append(self._make_trade(entry_date, df["date"].iloc[i], entry_signal, entry_price, exit_price, pnl_pct, "TP"))
                    in_position = False
                    continue

            # Track intra-trade drawdown
            if entry_signal == 1:
                dd = (entry_price - current_price) / entry_price * 100
                intra_trade_dd = min(intra_trade_dd, dd)
            else:
                dd = (current_price - entry_price) / entry_price * 100
                intra_trade_dd = min(intra_trade_dd, dd)

            # Exit on opposite signal or neutral signal (0)
            if sig != 0 and sig != entry_signal:
                exit_price = current_price
                pnl_pct = (exit_price - entry_price) / entry_price * 100 * entry_signal - cost_factor * 200
                trades.append(self._make_trade(entry_date, df["date"].iloc[i], entry_signal, entry_price, exit_price, pnl_pct, "signal",
                                               intra_trade_dd if intra_trade_dd < 0 else None))
                in_position = False

        # Close open position at end
        if in_position:
            exit_price = df["close"].iloc[-1]
            pnl_pct = (exit_price - entry_price) / entry_price * 100 * entry_signal - cost_factor * 100
            trades.append({
                "entry_date": str(entry_date.date()),
                "exit_date": str(df["date"].iloc[-1].date()),
                "direction": "LONG" if entry_signal == 1 else "SHORT",
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "holding_days": (df["date"].iloc[-1] - entry_date).days,
                "reason": "end_of_data",
                "open": True,
            })

        return trades

    @staticmethod
    def _make_trade(entry_date, exit_date, entry_signal, entry_price, exit_price, pnl_pct, reason, max_dd=None):
        trade = {
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "direction": "LONG" if entry_signal == 1 else "SHORT",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pct": round(pnl_pct, 2),
            "holding_days": (exit_date - entry_date).days,
            "reason": reason,
        }
        if max_dd is not None:
            trade["max_intra_dd_pct"] = round(max_dd, 2)
        return trade

    @staticmethod
    def _validate_params(strategy_id, params):
        """Check for degenerate/contradictory parameter combinations."""
        warnings = []
        if strategy_id == "rsi_mean_reversion":
            oversold = params.get("oversold", 30)
            overbought = params.get("overbought", 70)
            if oversold >= overbought:
                warnings.append(f"oversold ({oversold}) >= overbought ({overbought})")
            if oversold < 1 or overbought > 99:
                warnings.append(f"extremes out of [1,99]: oversold={oversold}, overbought={overbought}")
        elif strategy_id == "sma_crossover":
            fast = params.get("fast_period", 20)
            slow = params.get("slow_period", 50)
            if fast >= slow:
                warnings.append(f"fast_period ({fast}) >= slow_period ({slow})")
            if fast < 2 or slow < 5:
                warnings.append(f"periods too short: fast={fast}, slow={slow}")
        elif strategy_id == "bollinger_bounce":
            period = params.get("bollinger_period", 20)
            std = params.get("bollinger_std", 2)
            if period < 5:
                warnings.append(f"bollinger_period too short: {period}")
            if std <= 0:
                warnings.append(f"bollinger_std must be positive: {std}")
        elif strategy_id == "vwap_reversion":
            mult = params.get("atr_multiplier", 2)
            if mult <= 0:
                warnings.append(f"atr_multiplier must be positive: {mult}")
        sl = params.get("sl_pct")
        tp = params.get("tp_pct")
        if sl is not None and tp is not None and sl >= tp:
            warnings.append(f"sl_pct ({sl}) >= tp_pct ({tp})")
        if sl is not None and (sl <= 0 or sl >= 50):
            warnings.append(f"sl_pct out of realistic range (0,50): {sl}")
        if tp is not None and (tp <= 0 or tp >= 200):
            warnings.append(f"tp_pct out of realistic range (0,200): {tp}")
        if sl is not None and params.get("atr_multiplier") is not None:
            warnings.append("ATR-based and fixed SL both set — will use fixed SL")
        return {"warnings": warnings, "valid": len(warnings) == 0}

    def _compute_metrics(self, trades, df):
        """Compute performance metrics from trade list."""
        if not trades:
            return {
                "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "avg_win": 0, "avg_loss": 0, "max_drawdown": 0,
                "sharpe_ratio": 0, "total_return_pct": 0,
                "avg_intra_dd": 0, "max_intra_dd_pct": 0,
            }

        closed = [t for t in trades if not t.get("open")]
        if not closed:
            return {"total_trades": len(trades), "note": "All positions still open"}

        wins = [t for t in closed if t["pnl_pct"] > 0]
        losses = [t for t in closed if t["pnl_pct"] <= 0]
        total_return = (np.prod([1 + t["pnl_pct"] / 100 for t in closed]) - 1) * 100 if closed else 0

        # Win rate
        win_rate = len(wins) / len(closed) * 100 if closed else 0

        # Profit factor
        gross_profit = sum(t["pnl_pct"] for t in wins)
        gross_loss = abs(sum(t["pnl_pct"] for t in losses))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        # Avg win/loss
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0

        # Max drawdown (equity curve)
        equity = 100
        peak = 100
        max_dd = 0
        for t in closed:
            equity *= (1 + t["pnl_pct"] / 100)
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)

        # Sharpe ratio (from strategy equity curve, not raw asset returns)
        sharpe = 0
        eq_curve = [100]
        for t in closed:
            eq_curve.append(eq_curve[-1] * (1 + t["pnl_pct"] / 100))
        if len(eq_curve) > 2:
            eq_returns = np.diff(eq_curve) / eq_curve[:-1]
            if np.std(eq_returns) > 0:
                sharpe = np.mean(eq_returns) / np.std(eq_returns) * math.sqrt(252)

        # Intra-trade drawdown stats
        intra_dds = [t.get("max_intra_dd_pct", 0) for t in closed if "max_intra_dd_pct" in t]
        max_intra_dd = min(intra_dds) if intra_dds else 0
        avg_intra_dd = np.mean(intra_dds) if intra_dds else 0

        # Payoff ratio
        avg_win_pct = avg_win if wins else 0
        avg_loss_pct = avg_loss if losses else 0
        payoff_ratio = avg_win_pct / abs(avg_loss_pct) if avg_loss_pct != 0 else 0

        # Expectancy per trade
        expectancy = (win_rate / 100 * avg_win_pct) - ((1 - win_rate / 100) * abs(avg_loss_pct)) if closed else 0

        # Cost as % of gross PnL
        gross_pnl = sum(t["pnl_pct"] for t in closed)
        cost_pct = 0
        if gross_pnl != 0:
            cost_estimate = self.cost_bps / 10000 * 100 * 2 * len(closed)  # ~cost bps * 2 (entry+exit) per trade
            cost_pct = round(abs(cost_estimate / gross_pnl) * 100, 1) if gross_pnl != 0 else 0

        return {
            "total_trades": len(closed),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "expectancy_pct": round(expectancy, 2),
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "avg_intra_dd_pct": round(avg_intra_dd, 2),
            "max_intra_dd_pct": round(max_intra_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_return_pct": round(total_return, 2),
            "best_trade_pct": round(max(t["pnl_pct"] for t in closed), 2) if closed else 0,
            "worst_trade_pct": round(min(t["pnl_pct"] for t in closed), 2) if closed else 0,
            "cost_bps": self.cost_bps,
            "cost_pct_of_gross": cost_pct,
        }

    @staticmethod
    def _clean_numpy(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: Backtester._clean_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Backtester._clean_numpy(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 6)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.generic,)):
            return obj.item()
        return obj

# ── Technical helpers ──

    @staticmethod
    def _rsi(closes, period, n):
        rsi = np.full(n, np.nan, dtype=float)
        if n < period + 1:
            return rsi
        gains, losses = 0, 0
        for i in range(1, period + 1):
            diff = closes[i] - closes[i - 1]
            gains += max(diff, 0)
            losses += max(-diff, 0)
        avg_gain = gains / period
        avg_loss = losses / period
        for i in range(period + 1, n):
            diff = closes[i] - closes[i - 1]
            avg_gain = (avg_gain * (period - 1) + max(diff, 0)) / period
            avg_loss = (avg_loss * (period - 1) + max(-diff, 0)) / period
            rs = avg_gain / max(avg_loss, 0.001)
            rsi[i] = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _sma(values, period, n):
        sma = [None] * n
        for i in range(period - 1, n):
            sma[i] = np.mean(values[i - period + 1:i + 1])
        return sma

    @staticmethod
    def _vwap(closes, volumes, n):
        vwap = [None] * n
        cum_pv = 0
        cum_v = 0
        for i in range(n):
            cum_pv += closes[i] * volumes[i]
            cum_v += volumes[i]
            if cum_v > 0:
                vwap[i] = cum_pv / cum_v
        return vwap

    @staticmethod
    def _atr(highs, lows, closes, period, n):
        atr = np.full(n, np.nan, dtype=float)
        tr = np.full(n, np.nan, dtype=float)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1]))
        for i in range(period, n):
            atr[i] = np.mean(tr[i - period + 1:i + 1])
        return atr

    @staticmethod
    def _detect_trend(closes, highs, lows, n):
        trend = [None] * n
        for i in range(10, n):
            recent_highs = highs[max(0, i - 10):i + 1]
            recent_lows = lows[max(0, i - 10):i + 1]
            if closes[i] > np.mean(recent_highs):
                trend[i] = "uptrend"
            elif closes[i] < np.mean(recent_lows):
                trend[i] = "downtrend"
            else:
                trend[i] = "range"
        return trend

    async def run_regime_conditioned(self, ticker, strategy_id, regime_df=None, start_date=None, end_date=None, params=None):
        """Run backtest but only take signals when regime matches strategy.

        regime_df: DataFrame with columns [date, regime]
          where regime is 'uptrend', 'downtrend', or 'range'
        """
        result = await self.run(ticker, strategy_id, start_date, end_date, params)
        if "error" in result:
            return result

        if regime_df is not None and not result["trades"]:
            return result

        if regime_df is not None:
            # Filter trades to only those where regime was favorable
            regime_map = dict(zip(pd.to_datetime(regime_df["date"]), regime_df["regime"]))
            filtered = []
            for t in result["trades"]:
                t_date = pd.to_datetime(t["entry_date"])
                regime = regime_map.get(t_date)
                if regime is None:
                    continue
                if strategy_id in ("trend_follow", "sma_crossover"):
                    if regime in ("uptrend", "downtrend"):
                        filtered.append(t)
                elif strategy_id in ("rsi_mean_reversion", "bollinger_bounce", "vwap_reversion"):
                    if regime == "range":
                        filtered.append(t)
                else:
                    filtered.append(t)
            result["trades"] = filtered
            result["regime_filter"] = f"Kept {len(filtered)}/{len(result['trades'])} trades"
            result["metrics"] = self._compute_metrics(filtered, None)

        result["conditioned"] = regime_df is not None
        return result

    async def walk_forward(self, ticker, strategy_id, param_grid, n_splits=4,
                           start_date=None, end_date=None, metric="sharpe_ratio"):
        """Walk-forward optimization: train on rolling windows, test on next fold.

        param_grid: dict of param_name -> list of values to search
        n_splits: number of folds (train = first k-1 folds, test = kth)
        Returns: best params (from training avg), fold results, out-of-sample metrics
        """
        strategy = STRATEGIES.get(strategy_id)
        if not strategy:
            return {"error": f"Unknown strategy: {strategy_id}"}

        sid = self._lookup_id(ticker)
        if not sid:
            return {"error": f"Ticker not found: {ticker}"}

        df = self.con.execute("""
            SELECT date, open, high, low, close, volume
            FROM daily WHERE security_id = ?
            ORDER BY date
        """, [sid]).fetchdf()
        df["date"] = pd.to_datetime(df["date"])

        if len(df) < 100:
            return {"error": f"Not enough data ({len(df)} rows)"}

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        # Generate folds
        dates = df["date"].values
        total = len(dates)
        fold_size = total // n_splits
        folds = []
        for k in range(n_splits):
            test_end = min((k + 1) * fold_size, total)
            test_start = k * fold_size
            train_end = test_start
            folds.append({
                "train": (dates[0], dates[train_end - 1]) if train_end > 0 else None,
                "test": (dates[test_start], dates[test_end - 1]),
            })

        # Generate all param combinations (limit to 200 max)
        param_names = list(param_grid.keys())
        if not param_names:
            return {"error": "Empty param_grid"}

        import itertools
        combos = list(itertools.product(*[param_grid[p] for p in param_names]))
        if len(combos) > 200:
            # Sample evenly
            indices = np.linspace(0, len(combos) - 1, 200, dtype=int)
            combos = [combos[i] for i in indices]

        best_combo = None
        best_avg_score = -float("inf")
        fold_results = []

        for combo in combos:
            params = dict(zip(param_names, combo))
            fold_scores = {"train": [], "test": []}
            for fold in folds:
                for phase, period in [("train", fold["train"]), ("test", fold["test"])]:
                    if period is None:
                        continue
                    start_d, end_d = period
                    fold_df = df[(df["date"] >= pd.to_datetime(start_d)) &
                                 (df["date"] <= pd.to_datetime(end_d))]
                    if len(fold_df) < 50:
                        continue
                    signals = self._compute_signals(fold_df, strategy_id, params)
                    fold_df = fold_df.copy()
                    fold_df["signal"] = signals
                    trades = self._generate_trades(fold_df, sl_pct=params.get("sl_pct"), tp_pct=params.get("tp_pct"))
                    metrics = self._compute_metrics(trades, fold_df)
                    fold_scores[phase].append(metrics.get(metric, 0))

            avg_train = np.mean(fold_scores["train"]) if fold_scores["train"] else -999
            avg_test = np.mean(fold_scores["test"]) if fold_scores["test"] else -999

            if avg_train > best_avg_score:
                best_avg_score = avg_train
                best_combo = combo

            fold_results.append({
                "params": params,
                "avg_train_score": round(avg_train, 4),
                "avg_test_score": round(avg_test, 4),
            })

        # Run best params on full OOS (last fold test set)
        best_params = dict(zip(param_names, best_combo))
        last_fold = folds[-1]
        test_df = df[(df["date"] >= last_fold["test"][0]) &
                     (df["date"] <= last_fold["test"][1])]
        signals = self._compute_signals(test_df, strategy_id, best_params)
        test_df = test_df.copy()
        test_df["signal"] = signals
        trades = self._generate_trades(test_df, sl_pct=best_params.get("sl_pct"), tp_pct=best_params.get("tp_pct"))
        oos_metrics = self._compute_metrics(trades, test_df)

        # Sort fold results by avg_test_score descending for analysis
        fold_results.sort(key=lambda x: x["avg_test_score"], reverse=True)
        top_n = [r for r in fold_results[:10]]

        return {
            "ticker": ticker,
            "strategy": strategy["name"],
            "strategy_id": strategy_id,
            "n_splits": n_splits,
            "param_grid": param_grid,
            "combos_evaluated": len(combos),
            "best_params": best_params,
            "best_avg_train_score": round(best_avg_score, 4),
            "oos_period": {
                "from": str(pd.Timestamp(last_fold["test"][0]).date()),
                "to": str(pd.Timestamp(last_fold["test"][1]).date()),
            },
            "oos_metrics": oos_metrics,
            "top_combos": top_n,
        }

    def close(self):
        try:
            self.con.close()
        except:
            pass
