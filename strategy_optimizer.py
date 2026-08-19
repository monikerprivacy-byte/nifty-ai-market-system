"""Strategy Optimizer — brute-force parameter optimization for trading strategies.

For each strategy, tries all parameter combinations and returns top results
ranked by Sharpe ratio, profit factor, and win rate.

Strategies and their parameter grids:
- rsi_mean_reversion: rsi_period [10,14,21], oversold [25,30,35], overbought [65,70,75]
- sma_crossover: fast_period [10,20,30,50], slow_period [30,50,100,200]
- bollinger_bounce: bollinger_period [10,20,30], bollinger_std [1.5,2.0,2.5]
- trend_follow: (no params to optimize) — uses existing SMC structure
- vwap_reversion: atr_multiplier [1,1.5,2,3]
"""

import asyncio, json, logging, itertools, math
from datetime import datetime
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config

logger = logging.getLogger("strategy_optimizer")

PARAM_GRIDS = {
    "rsi_mean_reversion": {
        "rsi_period": [10, 14, 21],
        "oversold": [25, 30, 35],
        "overbought": [65, 70, 75],
        "sl_pct": [None, 0.02, 0.03],
        "tp_pct": [None, 0.04, 0.06],
    },
    "sma_crossover": {
        "fast_period": [10, 20, 30, 50],
        "slow_period": [30, 50, 100, 200],
        "sl_pct": [None, 0.02, 0.03],
        "tp_pct": [None, 0.04, 0.06],
    },
    "bollinger_bounce": {
        "bollinger_period": [10, 20, 30],
        "bollinger_std": [1.5, 2.0, 2.5],
        "sl_pct": [None, 0.02, 0.03],
        "tp_pct": [None, 0.04, 0.06],
    },
    "trend_follow": {
        "sl_pct": [None, 0.02, 0.03],
        "tp_pct": [None, 0.04, 0.06],
    },
    "vwap_reversion": {
        "atr_multiplier": [1, 1.5, 2, 3],
        "sl_pct": [None, 0.02, 0.03],
        "tp_pct": [None, 0.04, 0.06],
    },
}

# Cost in bps (brokerage + STT + slippage)
DEFAULT_COST_BPS = 20


class StrategyOptimizer:
    def __init__(self, db_path=None, cost_bps=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.cost_bps = cost_bps or DEFAULT_COST_BPS
        self.con = duckdb.connect(self.db_path)

    @staticmethod
    def _filter_degenerate(strategy_id, params, keys, values, combos):
        """Remove degenerate parameter combinations."""
        from backtester import Backtester
        valid = []
        for combo in combos:
            p = dict(zip(keys, combo))
            validation = Backtester._validate_params(strategy_id, p)
            if validation["valid"]:
                valid.append(combo)
        return valid

    async def optimize(self, ticker, strategy_id, metric="sharpe_ratio", top_n=5,
                       walk_forward=False, n_splits=4):
        """Run full optimization for a strategy on a ticker.

        walk_forward: use walk-forward optimization instead of in-sample only
        n_splits: number of folds for walk-forward
        """
        from backtester import Backtester

        params_grid = PARAM_GRIDS.get(strategy_id)
        if not params_grid:
            return {"error": f"Unknown strategy: {strategy_id}"}

        ticker = ticker.upper()
        sid = self._lookup_id(ticker)
        if not sid:
            return {"error": f"Unknown ticker: {ticker}"}

        # Check data availability
        count = self.con.execute(
            "SELECT COUNT(*) FROM daily WHERE security_id = ?", [sid]
        ).fetchone()[0]
        if count < 100:
            return {"error": f"Not enough data for {ticker} ({count} rows, need 100+)"}

        keys = list(params_grid.keys())
        values = list(params_grid.values())
        raw_combos = list(itertools.product(*values))
        total_raw = len(raw_combos)

        # Filter degenerate combos
        combinations = self._filter_degenerate(strategy_id, params_grid, keys, values, raw_combos)
        total = len(combinations)
        filtered_count = total_raw - total

        if total == 1:
            bt = Backtester(self.db_path, cost_bps=self.cost_bps)
            result = await bt.run(ticker, strategy_id)
            bt.close()
            if "metrics" in result:
                m = result["metrics"]
                return {
                    "ticker": ticker,
                    "strategy": strategy_id,
                    "total_combinations": 1,
                    "degenerate_filtered": filtered_count,
                    "best": {
                        "params": {},
                        metric: m.get(metric, 0),
                        "win_rate": m.get("win_rate", 0),
                        "profit_factor": m.get("profit_factor", 0),
                        "total_return_pct": m.get("total_return_pct", 0),
                        "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                        "total_trades": m.get("total_trades", 0),
                    },
                    "results": [],
                }
            return {"error": "Backtest failed"}

        if total == 0:
            return {"error": "All parameter combinations degenerate — check param grid"}

        logger.info(f"Optimizing {strategy_id} on {ticker}: "
                    f"{total} valid combos (filtered {filtered_count} degenerate)")

        # If walk-forward, delegate to Backtester.walk_forward for each combo group
        if walk_forward:
            bt = Backtester(self.db_path, cost_bps=self.cost_bps)
            try:
                wf_result = await bt.walk_forward(
                    ticker, strategy_id, params_grid,
                    n_splits=n_splits, metric=metric
                )
                bt.close()
                return wf_result
            except Exception as e:
                bt.close()
                logger.error(f"Walk-forward failed: {e}")
                # Fall through to regular optimization

        results = []
        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            params = {k: int(v) if isinstance(v, (np.integer,)) else v
                     for k, v in params.items()}
            # Keep None values for sl_pct/tp_pct
            params = {k: (None if v == "None" or v is None else v) for k, v in params.items()}

            bt = Backtester(self.db_path, cost_bps=self.cost_bps)
            try:
                result = await bt.run(ticker, strategy_id, params=params)
                bt.close()
                if "metrics" in result:
                    m = result["metrics"]
                    total_trades = m.get("total_trades", 0)
                    if total_trades >= 3:
                        results.append({
                            "params": {k: ("None" if v is None else str(v)) for k, v in params.items()},
                            "sharpe_ratio": m.get("sharpe_ratio", 0),
                            "win_rate": m.get("win_rate", 0),
                            "profit_factor": m.get("profit_factor", 0),
                            "total_return_pct": m.get("total_return_pct", 0),
                            "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                            "total_trades": total_trades,
                            "avg_win_pct": m.get("avg_win_pct", 0),
                            "avg_loss_pct": m.get("avg_loss_pct", 0),
                            "expectancy_pct": m.get("expectancy_pct", 0),
                        })
            except Exception as e:
                bt.close()
                logger.debug(f"  Combo {idx+1}/{total} failed: {e}")

            if (idx + 1) % 20 == 0:
                logger.info(f"  Progress: {idx+1}/{total}")

        if not results:
            return {"error": "No valid parameter combinations found (need 3+ trades)"}

        reverse = metric not in ("max_drawdown_pct", "avg_loss_pct")
        results.sort(key=lambda r: r.get(metric, 0), reverse=reverse)

        best = results[0]
        logger.info(f"Optimization complete. Best: {best['params']} → {metric}={best.get(metric, '?')}")

        try:
            from memory_manager import MemoryManager
            mm = MemoryManager()
            mm.store_knowledge(
                f"Optimized: {ticker} {strategy_id}",
                json.dumps(best, indent=2),
                category="strategy_optimization", ticker=ticker,
                source="strategy_optimizer",
                tags=f"{strategy_id},{metric},{'wfo' if walk_forward else 'full'}"
            )
        except:
            pass

        return {
            "ticker": ticker,
            "strategy": strategy_id,
            "metric": metric,
            "total_combinations": total_raw,
            "degenerate_filtered": filtered_count,
            "valid_combinations": len(results),
            "walk_forward": walk_forward,
            "cost_bps": self.cost_bps,
            "best": best,
            "results": results[:top_n],
        }

    def get_top_strategies(self, ticker, top_n=3):
        """Quick comparison: run each strategy with default params, return ranking."""
        import asyncio

        async def _run_all():
            from backtester import Backtester
            bt = Backtester(self.db_path, cost_bps=self.cost_bps)
            results = []
            for sid in PARAM_GRIDS.keys():
                try:
                    result = await bt.run(ticker, sid)
                    if "metrics" in result:
                        m = result["metrics"]
                        results.append({
                            "strategy": sid,
                            "sharpe_ratio": m.get("sharpe_ratio", 0),
                            "win_rate": m.get("win_rate", 0),
                            "profit_factor": m.get("profit_factor", 0),
                            "total_return_pct": m.get("total_return_pct", 0),
                            "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                            "total_trades": m.get("total_trades", 0),
                            "expectancy_pct": m.get("expectancy_pct", 0),
                        })
                except:
                    pass
            bt.close()
            results.sort(key=lambda r: r.get("sharpe_ratio", 0), reverse=True)
            return results[:top_n]

        return asyncio.run(_run_all())

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

    def close(self):
        try:
            self.con.close()
        except:
            pass
