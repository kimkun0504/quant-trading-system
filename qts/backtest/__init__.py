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
from qts.backtest.simulator import BacktestResult, DailyBarSimulator

__all__ = [
    "BacktestResult",
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
