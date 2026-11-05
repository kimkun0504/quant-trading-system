"""Daily-bar backtest engine and performance metrics."""

from qts.backtest.metrics import (
    annualized_return,
    annualized_sharpe,
    annualized_vol,
    breakeven_cost_bps,
    calmar,
    cost_sensitivity,
    drawdown_series,
    hit_rate,
    max_drawdown,
    newey_west_tstat,
    sharpe_t_stat,
    sortino,
    turnover_stats,
)
from qts.backtest.result import BacktestResult, BacktestSummary
from qts.backtest.simulator import DailyBarSimulator
from qts.backtest.spec import BacktestSpec

__all__ = [
    "BacktestResult",
    "BacktestSpec",
    "BacktestSummary",
    "DailyBarSimulator",
    "annualized_return",
    "annualized_sharpe",
    "annualized_vol",
    "breakeven_cost_bps",
    "calmar",
    "cost_sensitivity",
    "drawdown_series",
    "hit_rate",
    "max_drawdown",
    "newey_west_tstat",
    "sharpe_t_stat",
    "sortino",
    "turnover_stats",
]
