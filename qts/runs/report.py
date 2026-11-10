"""Plain-text report card of a pipeline run (:func:`format_result`)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from qts.validation import HurdleReport

if TYPE_CHECKING:
    from qts.pipeline import PipelineResult


def _pct(v: object) -> str:
    x = float(v) if isinstance(v, (int, float)) else float("nan")
    return f"{100 * x:.0f}%" if math.isfinite(x) else "n/a"


def _vol_ratio_text(s: Mapping[str, object], rv: Mapping[str, float] | None = None) -> str:
    """``realized_vs_exante_vol_ratio`` over invested days (with the two annual
    vols when ``rv`` is given), or why it is undefined."""
    v = s.get("realized_vs_exante_vol_ratio", float("nan"))
    x = float(v) if isinstance(v, (int, float)) else float("nan")
    if math.isfinite(x):
        if rv and math.isfinite(rv.get("realized_vol_annual", float("nan"))):
            return f"{x:.2f} ({100 * rv['realized_vol_annual']:.1f}% / {100 * rv['exante_vol_annual']:.1f}%)"
        return f"{x:.2f}"
    f = s.get("invested_day_frac", float("nan"))
    fx = float(f) if isinstance(f, (int, float)) else float("nan")
    return "n/a (too few invested days)" if math.isfinite(fx) else "n/a"


def format_result(res: PipelineResult) -> str:
    s = res.summary
    icr = res.ic_report
    bt = getattr(res, "backtest", None)
    rv = bt.realized_vs_exante() if bt is not None and hasattr(bt, "realized_vs_exante") else None
    lines = [
        "=== Backtest summary ===",
        f"period            {s['start']} -> {s['end']}  ({s['n_days']} days)"
        + (f"   signal lag {res.signal_lag_days}d" if res.signal_lag_days else ""),
        f"Sharpe gross/net  {s['sharpe_gross']:.2f} / {s['sharpe_net']:.2f}   (NW t = {s['newey_west_t_net']:.2f})",
        f"CAGR gross/net    {100 * s['cagr_gross']:.2f}% / {100 * s['cagr_net']:.2f}%   vol {100 * s['ann_vol_net']:.1f}%",
        f"max drawdown      {100 * s['max_drawdown_net']:.1f}%   calmar {s['calmar_net']:.2f}",
        f"turnover/day      {100 * s['avg_daily_turnover']:.2f}%   effective holding {s['effective_holding_days']:.0f} d",
        f"cost per side     {s['avg_cost_bps_per_side']:.1f} bp   gross exp {s['avg_gross_exposure']:.2f}  net exp {s['avg_net_exposure']:+.2f}",
        f"transfer coeff    {s.get('transfer_coefficient_mean', float('nan')):.2f}   "
        f"realized/ex-ante vol {_vol_ratio_text(s, rv)}   invested {_pct(s.get('invested_day_frac'))} of days",
        f"sizing            mean gamma {s.get('mean_risk_aversion', float('nan')):.2f}   "
        f"vol target capped on {100 * s.get('vol_target_caps_share', float('nan')):.0f}% of solves",
        "",
        "=== Alpha IC (score vs residualised target, sampled every horizon) ===",
        f"mean rank IC {icr.mean_ic:.4f}  ICIR {icr.icir:.2f} "
        f"(annualised at {icr.periods_per_year:.1f}/yr)  t {icr.t_stat:.2f}  NW t {icr.t_stat_nw:.2f} "
        f"(L={icr.nw_lags})  n {icr.n}",
    ]
    if len(icr.decay):
        lines.append(
            "IC decay (horizon d: mean IC)  " + "  ".join(f"{int(h)}d: {v:.4f}" for h, v in icr.decay.items())
        )
    if res.decile is not None:
        lines.append(
            f"decile L/S (VW)  mean {100 * res.decile.mean:.2f}%/period  Sharpe {res.decile.sharpe:.2f}  "
            f"NW t {res.decile.t_stat:.2f} (L={res.decile.nw_lags})"
        )
    lines += [
        f"DSR {res.dsr:.3f}  (n_trials {res.dsr_inputs.get('n_trials', 1):.1f}, var SR {res.dsr_inputs.get('var_trials_sr_annual', 0):.3f})"
        + (f"  PBO {res.pbo:.3f}" if res.pbo is not None else "  PBO skipped"),
        f"config {res.config_hash}  data {res.data_hash}  git {res.git_sha[:12]}"
        + (f"  run {res.manifest.run_id}" if res.manifest is not None else ""),
        "",
        "=== Fold (purged walk-forward test block) net Sharpe ===",
        (
            res.folds.assign(
                start=res.folds["start"].astype(str).str[:10], end=res.folds["end"].astype(str).str[:10]
            )
            .round(2)
            .to_string(index=False)
            if len(res.folds)
            else "(no folds)"
        ),
        "",
        "=== Cost sensitivity (net Sharpe at cost multiples) ===",
        res.cost_sensitivity.round(3).to_string(),
        "",
        "=== Plan hurdles ===",
        HurdleReport.format(res.hurdles),
    ]
    if res.notes:
        lines += ["", "NOTES: " + " | ".join(res.notes)]
    if res.timings:
        lines += ["", "timings (s): " + ", ".join(f"{k}={v:.1f}" for k, v in res.timings.items())]
    if res.run_dir:
        lines += [f"run directory: {res.run_dir}"]
    return "\n".join(lines)
