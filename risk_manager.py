"""Risk Manager — portfolio risk analytics: VAR, sector exposure, correlation, concentration.

Metrics:
- 1-day 95% parametric VAR (per position + portfolio)
- Sector exposure breakdown (₹ and %)
- Position correlation matrix (daily returns, last 60 days)
- Concentration: top 1/3/5 positions as % of portfolio
- Position sizing check vs max_position_pct
- Maximum sector exposure alert
- Sharpe/Sortino ratios
- Stress test: -5%, -10%, -20% market drop scenarios
"""

import logging, math
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np
from config_manager import get_config
from sector_analysis import get_sector_for_ticker

logger = logging.getLogger("risk_manager")

RISK_FREE_RATE = 0.065  # ~6.5% Indian risk-free rate


class RiskManager:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.market", "/Volumes/Untitled/market_data/market.duckdb")
        self.con = duckdb.connect(self.db_path)

    # ── Portfolio-Level Risk ──

    def portfolio_var(self, positions, confidence=0.95, days=1):
        """Compute parametric VAR for portfolio."""
        if not positions:
            return {"var_95_1d": 0, "var_95_1d_pct": 0, "method": "parametric"}

        returns = []
        weights = []
        total_value = sum(abs(p.get("quantity", 0) * p.get("entry_price", 0)) for p in positions)
        if total_value == 0:
            return {"var_95_1d": 0, "var_95_1d_pct": 0}

        for p in positions:
            sid = p.get("security_id")
            if not sid:
                continue
            daily_rets = self.con.execute("""
                SELECT close FROM daily
                WHERE security_id = ? ORDER BY date DESC LIMIT 60
            """, [sid]).fetchdf()
            if len(daily_rets) < 10:
                continue
            rets = daily_rets["close"].pct_change().dropna().values
            if len(rets) < 5:
                continue
            returns.append(rets)
            weights.append(abs(p.get("quantity", 0) * p.get("entry_price", 0)) / total_value)

        if not returns:
            return {"var_95_1d": 0, "var_95_1d_pct": 0}

        # Portfolio variance = w^T * Cov * w
        min_len = min(len(r) for r in returns)
        aligned = np.array([r[-min_len:] for r in returns])
        cov_matrix = np.cov(aligned)
        weights_arr = np.array(weights)
        portfolio_std = math.sqrt(weights_arr.T @ cov_matrix @ weights_arr)

        z = {0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
        var_pct = z * portfolio_std * math.sqrt(days)
        var_amount = var_pct * total_value

        return {
            "var_95_1d": round(var_amount, 2),
            "var_95_1d_pct": round(var_pct * 100, 2),
            "total_exposure": round(total_value, 2),
            "portfolio_std": round(portfolio_std * 100, 2),
            "confidence": confidence,
            "days": days,
            "method": "parametric",
        }

    def per_position_var(self, positions):
        """VAR contribution per position."""
        results = []
        for p in positions:
            sid = p.get("security_id")
            if not sid:
                continue
            daily_rets = self.con.execute("""
                SELECT close FROM daily
                WHERE security_id = ? ORDER BY date DESC LIMIT 60
            """, [sid]).fetchdf()
            if len(daily_rets) < 10:
                continue
            rets = daily_rets["close"].pct_change().dropna().values
            if len(rets) < 5:
                continue
            std = np.std(rets)
            var_pct = 1.645 * std  # 95% VAR, 1 day
            pos_value = abs(p.get("quantity", 0) * p.get("entry_price", 0))
            var_amount = var_pct * pos_value
            results.append({
                "symbol": p.get("symbol"),
                "direction": p.get("direction"),
                "value": round(pos_value, 2),
                "volatility_pct": round(std * 100, 2),
                "var_95_pct": round(var_pct * 100, 2),
                "var_95_amount": round(var_amount, 2),
                "quantity": p.get("quantity"),
                "entry_price": p.get("entry_price"),
                "current_price": p.get("current_price"),
            })
        return results

    # ── Sector Exposure ──

    def sector_exposure(self, positions):
        """Exposure breakdown by sector."""
        sectors = {}
        total = 0
        for p in positions:
            symbol = p.get("symbol", "?")
            value = abs(p.get("quantity", 0) * p.get("entry_price", 0))
            sector = get_sector_for_ticker(symbol)
            sectors.setdefault(sector, {"exposure": 0, "positions": []})
            sectors[sector]["exposure"] += value
            sectors[sector]["positions"].append({
                "symbol": symbol,
                "direction": p.get("direction"),
                "value": round(value, 2),
            })
            total += value

        if total == 0:
            return {"sectors": [], "total": 0}

        result = []
        for sector, data in sorted(sectors.items(), key=lambda x: x[1]["exposure"], reverse=True):
            result.append({
                "sector": sector,
                "exposure": round(data["exposure"], 2),
                "pct": round(data["exposure"] / total * 100, 1),
                "positions": data["positions"],
            })

        return {"sectors": result, "total": round(total, 2)}

    def max_sector_exposure(self, positions):
        """Check if any single sector exceeds concentration limit."""
        cfg = get_config()
        max_pct = float(cfg.get("trading.max_concentration_pct", 30))
        sectors = self.sector_exposure(positions)
        warnings = []
        for s in sectors.get("sectors", []):
            if s["pct"] > max_pct:
                warnings.append({
                    "sector": s["sector"],
                    "exposure_pct": s["pct"],
                    "limit_pct": max_pct,
                    "excess_pct": round(s["pct"] - max_pct, 1),
                })
        return warnings

    # ── Correlation ──

    def position_correlation(self, positions):
        """Correlation matrix between position returns (last 60 days)."""
        if len(positions) < 2:
            return {"matrix": [], "note": "Need ≥2 positions for correlation"}

        symbols = []
        returns_list = []
        for p in positions:
            sid = p.get("security_id")
            if not sid:
                continue
            df = self.con.execute("""
                SELECT close FROM daily
                WHERE security_id = ? ORDER BY date DESC LIMIT 60
            """, [sid]).fetchdf()
            if len(df) < 10:
                continue
            rets = df["close"].pct_change().dropna().values
            if len(rets) < 5:
                continue
            symbols.append(p.get("symbol", "?"))
            returns_list.append(rets)

        if len(symbols) < 2:
            return {"matrix": [], "note": "Insufficient return data for correlation"}

        min_len = min(len(r) for r in returns_list)
        aligned = np.array([r[-min_len:] for r in returns_list])
        corr = np.corrcoef(aligned)

        matrix = []
        for i, sym_i in enumerate(symbols):
            row = {"symbol": sym_i}
            for j, sym_j in enumerate(symbols):
                row[sym_j] = round(float(corr[i][j]), 3)
            matrix.append(row)

        avg_corr = (np.sum(corr) - len(symbols)) / (len(symbols) * (len(symbols) - 1)) if len(symbols) > 1 else 0

        return {
            "matrix": matrix,
            "symbols": symbols,
            "avg_correlation": round(float(avg_corr), 3),
            "observation_days": min_len,
        }

    # ── Concentration ──

    def concentration(self, positions):
        """Top 1/3/5 position concentration."""
        if not positions:
            return {"top1_pct": 0, "top3_pct": 0, "top5_pct": 0}

        sorted_pos = sorted(
            positions,
            key=lambda p: abs(p.get("quantity", 0) * p.get("entry_price", 0)),
            reverse=True,
        )
        total = sum(abs(p.get("quantity", 0) * p.get("entry_price", 0)) for p in sorted_pos)
        if total == 0:
            return {"top1_pct": 0, "top3_pct": 0, "top5_pct": 0}

        def pct_of_top(n):
            val = sum(abs(p.get("quantity", 0) * p.get("entry_price", 0)) for p in sorted_pos[:n])
            return round(val / total * 100, 1)

        return {
            "top1_pct": pct_of_top(1),
            "top3_pct": pct_of_top(3),
            "top5_pct": pct_of_top(5),
            "total_positions": len(positions),
        }

    # ── Position Sizing ──

    def position_sizing_check(self, positions):
        """Check each position against max_position_pct."""
        cfg = get_config()
        max_pct = float(cfg.get("trading.max_position_pct", 10))
        capital = float(cfg.get("trading.paper_capital", 100000))
        warnings = []

        for p in positions:
            value = abs(p.get("quantity", 0) * p.get("entry_price", 0))
            pct = value / capital * 100 if capital > 0 else 0
            if pct > max_pct:
                warnings.append({
                    "symbol": p.get("symbol"),
                    "value": round(value, 2),
                    "pct": round(pct, 1),
                    "limit_pct": max_pct,
                    "excess_pct": round(pct - max_pct, 1),
                    "suggested_qty": max(1, int((max_pct / 100 * capital) / p.get("entry_price", 1))),
                })
        return warnings

    # ── Scenario / Stress Test ──

    def stress_test(self, positions, scenarios=None):
        """Simulate P&L impact under different market scenarios."""
        if scenarios is None:
            scenarios = [
                {"name": "Market Crash -5%", "shock_pct": -5},
                {"name": "Correction -10%", "shock_pct": -10},
                {"name": "Bear Market -20%", "shock_pct": -20},
                {"name": "Rally +3%", "shock_pct": 3},
                {"name": "Rally +5%", "shock_pct": 5},
            ]

        results = []
        for scenario in scenarios:
            total_pnl = 0
            details = []
            for p in positions:
                value = abs(p.get("quantity", 0) * p.get("entry_price", 0))
                beta = self._estimate_beta(p.get("security_id"))
                shock = scenario["shock_pct"] * beta
                pnl = value * shock / 100
                direction_mult = 1 if p.get("direction") == "LONG" else -1
                pnl *= direction_mult
                total_pnl += pnl
                details.append({
                    "symbol": p.get("symbol"),
                    "pnl": round(pnl, 2),
                })

            results.append({
                "scenario": scenario["name"],
                "shock_pct": scenario["shock_pct"],
                "total_pnl": round(total_pnl, 2),
                "details": details,
            })

        return results

    def _estimate_beta(self, security_id):
        """Estimate beta vs NIFTY (simplified: covariance/variance of last 60 days)."""
        nifty_id = "13"
        stock_data = self.con.execute("""
            SELECT close FROM daily WHERE security_id = ? ORDER BY date ASC LIMIT 60
        """, [security_id]).fetchdf()
        market_data = self.con.execute("""
            SELECT close FROM daily WHERE security_id = ? ORDER BY date ASC LIMIT 60
        """, [nifty_id]).fetchdf()
        if len(stock_data) < 10 or len(market_data) < 10:
            return 1.0
        stock_rets = stock_data["close"].pct_change().dropna().values
        market_rets = market_data["close"].pct_change().dropna().values
        min_len = min(len(stock_rets), len(market_rets))
        if min_len < 5:
            return 1.0
        stock_rets = stock_rets[-min_len:]
        market_rets = market_rets[-min_len:]
        cov = np.cov(stock_rets, market_rets)[0][1]
        var = np.var(market_rets)
        if var == 0:
            return 1.0
        beta = round(float(cov / var), 2)
        return max(-5, min(5, beta))

    # ── Sharpe / Sortino ──

    def portfolio_ratios(self, positions):
        """Compute Sharpe and Sortino ratios for the portfolio."""
        if not positions:
            return {"sharpe_ratio": 0, "sortino_ratio": 0}

        # Get daily returns for each position, compute weighted average
        total_value = sum(abs(p.get("quantity", 0) * p.get("entry_price", 0)) for p in positions)
        if total_value == 0:
            return {"sharpe_ratio": 0, "sortino_ratio": 0}

        portfolio_returns = None
        for p in positions:
            sid = p.get("security_id")
            if not sid:
                continue
            df = self.con.execute("""
                SELECT close FROM daily WHERE security_id = ? ORDER BY date ASC
            """, [sid]).fetchdf()
            if len(df) < 10:
                continue
            rets = df["close"].pct_change().dropna().values
            weight = abs(p.get("quantity", 0) * p.get("entry_price", 0)) / total_value
            if portfolio_returns is None:
                portfolio_returns = rets * weight
            else:
                min_len = min(len(portfolio_returns), len(rets))
                portfolio_returns = portfolio_returns[-min_len:] + (rets[-min_len:] * weight)

        if portfolio_returns is None or len(portfolio_returns) < 5:
            return {"sharpe_ratio": 0, "sortino_ratio": 0}

        excess_returns = portfolio_returns - (RISK_FREE_RATE / 252)
        sharpe = np.mean(excess_returns) / max(np.std(portfolio_returns), 0.0001) * math.sqrt(252)

        # Sortino: only downside deviation
        downside = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(downside) if len(downside) > 1 else np.std(portfolio_returns)
        sortino = np.mean(excess_returns) / max(downside_std, 0.0001) * math.sqrt(252)

        return {
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "daily_volatility_pct": round(float(np.std(portfolio_returns) * 100), 2),
            "annualized_volatility_pct": round(float(np.std(portfolio_returns) * math.sqrt(252) * 100), 2),
            "observation_days": len(portfolio_returns),
        }

    # ── Full Report ──

    def full_report(self, positions):
        """Generate complete risk report."""
        return {
            "generated_at": datetime.now().isoformat(),
            "var": self.portfolio_var(positions),
            "per_position_var": self.per_position_var(positions),
            "sector_exposure": self.sector_exposure(positions),
            "max_sector_warnings": self.max_sector_exposure(positions),
            "correlation": self.position_correlation(positions),
            "concentration": self.concentration(positions),
            "position_sizing_warnings": self.position_sizing_check(positions),
            "stress_test": self.stress_test(positions),
            "ratios": self.portfolio_ratios(positions),
        }

    def close(self):
        try:
            self.con.close()
        except:
            pass
