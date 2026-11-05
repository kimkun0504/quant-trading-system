"""Golden-file regression of the research pipeline's headline numbers.

Why this exists
---------------
The package is being refactored (one cost contract, de-duplicated helpers,
typed results, a DPP-parametrised optimizer, vectorised loops...).  Each of
those changes is meant to be *behaviour-preserving*, and unit tests alone do
not prove that: a refactor can shift a Sharpe ratio in the 4th decimal while
every invariant test stays green.  This test runs
``ResearchPipeline(use_ml=False).run(md, max_days=60)`` on a fixed synthetic
market (``conftest.GOLDEN_SPEC``: 100 names x 700 days, seed 11, top-80
universe) and compares a curated set of backtest summary metrics, IC
statistics, the decile spread, the DSR and the hurdle statuses to
``tests/golden/pipeline_summary.json``.  Any unintended numerical drift fails
loudly with the list of fields that moved.

Tolerances: every float compares with ``rtol=1e-6`` (plus ``atol=1e-9`` so
quantities that are ~0 do not demand relative precision); counts, strings,
the data hash and hurdle statuses compare exactly.  NaN/inf are stored as the
strings ``"nan"`` / ``"inf"`` / ``"-inf"`` and must match exactly.  Floats are
deterministic on one machine (see ``tests/test_determinism.py``); a different
BLAS / cvxpy / solver version may move the optimizer's solution in the 7th+
digit, which ``rtol=1e-6`` absorbs.

Updating the golden file (only for an *intended* behaviour change)
------------------------------------------------------------------
    python -m pytest tests/test_golden.py --update-golden
    git diff tests/golden/pipeline_summary.json   # review every moved number

then commit the JSON together with the change that explains it (say why in
the commit message).  The data hash is checked first: if the synthetic
generator itself changed, every other number moves too and the message says so.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import GOLDEN_MAX_DAYS, GOLDEN_SPEC, GOLDEN_TOP_N

GOLDEN = Path(__file__).parent / "golden" / "pipeline_summary.json"
RTOL, ATOL = 1e-6, 1e-9

SUMMARY_KEYS = (
    "start",
    "end",
    "n_days",
    "final_nav",
    "cagr_gross",
    "cagr_net",
    "ann_vol_gross",
    "ann_vol_net",
    "sharpe_gross",
    "sharpe_net",
    "sharpe_net_lo",
    "sharpe_t_stat_net",
    "newey_west_t_net",
    "max_drawdown_net",
    "calmar_net",
    "sortino_net",
    "hit_rate_net",
    "avg_gross_exposure",
    "avg_net_exposure",
    "avg_daily_turnover",
    "effective_holding_days",
    "avg_cost_bps_per_side",
    "avg_daily_cost_bps_of_nav",
    "total_costs_usd",
    "transfer_coefficient_mean",
    "realized_vs_exante_vol_ratio",
    "breakeven_cost_bps",
    "n_delistings",
    "n_failed_solves",
    "vol_target_caps_share",
    "mean_risk_aversion",
)
IC_KEYS = ("n", "mean_ic", "ic_sd", "icir", "t_stat", "hit_rate", "periods_per_year")
# Gates added after the golden file was written.  They are *not* numerical drift, so
# they are left out of the comparison and pinned by test_golden_run_is_not_degenerate
# instead (regenerating the golden file for a new gate would re-bless every number).
GATES_NOT_IN_GOLDEN = ("failed_solves",)


def _enc(v):
    """JSON-safe scalar: numpy -> python, non-finite floats -> strings."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else str(v)
    return v


def curated(pipe, res) -> dict:
    s, icr = res.summary, res.ic_report
    bt = res.backtest
    nav = bt.nav.to_numpy()
    return {
        "data_hash": res.data_hash,
        "summary": {k: _enc(s[k]) for k in SUMMARY_KEYS},
        "ic_report": {k: _enc(icr[k]) for k in IC_KEYS},
        "decile": {
            "mean": _enc(res.decile.mean),
            "sharpe": _enc(res.decile.sharpe),
            "t_stat": _enc(res.decile.t_stat),
        },
        "dsr": _enc(res.dsr),
        "hurdles": {
            "overall": _enc(res.hurdles["overall"]),
            "status": {
                k: res.hurdles["status"][k]
                for k in sorted(res.hurdles["status"])
                if k not in GATES_NOT_IN_GOLDEN
            },
        },
        "backtest": {
            "n_nav": len(nav),
            "nav_quartiles": [_enc(nav[int(q * (len(nav) - 1))]) for q in (0.25, 0.5, 0.75)],
            "sum_abs_weights": _enc(np.abs(bt.weights.to_numpy()).sum()),
            "sum_costs_by_kind": {c: _enc(bt.costs[c].sum()) for c in ("spread", "impact", "fees", "borrow")},
        },
    }


def _compare(path: str, got, exp, errors: list) -> None:
    if isinstance(exp, dict):
        if set(got) != set(exp):
            errors.append(f"{path}: keys {sorted(set(got) ^ set(exp))} added/removed")
        for k in exp:
            if k in got:
                _compare(f"{path}.{k}", got[k], exp[k], errors)
    elif isinstance(exp, list):
        if len(got) != len(exp):
            errors.append(f"{path}: length {len(got)} != {len(exp)}")
        for i, (g, e) in enumerate(zip(got, exp, strict=False)):  # length mismatch reported above
            _compare(f"{path}[{i}]", g, e, errors)
    elif isinstance(exp, float) and not isinstance(exp, bool) and isinstance(got, (int, float)):
        if not math.isclose(got, exp, rel_tol=RTOL, abs_tol=ATOL):
            errors.append(
                f"{path}: {got!r} != golden {exp!r} (rel diff {abs(got - exp) / max(abs(exp), 1e-300):.2e})"
            )
    elif got != exp:
        errors.append(f"{path}: {got!r} != golden {exp!r}")


def test_pipeline_matches_golden(golden_pipeline_run, update_golden):
    pipe, res = golden_pipeline_run
    got = curated(pipe, res)
    if update_golden or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_meta": {
                "spec": GOLDEN_SPEC,
                "top_n_by_mcap": GOLDEN_TOP_N,
                "max_days": GOLDEN_MAX_DAYS,
                "use_ml": False,
                "rtol": RTOL,
                "atol": ATOL,
                "regenerate": "python -m pytest tests/test_golden.py --update-golden",
            },
            **got,
        }
        GOLDEN.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        if not update_golden:
            pytest.fail(f"golden file was missing and has been written to {GOLDEN}; review and commit it")
        pytest.skip(f"golden file regenerated at {GOLDEN}")
    exp = json.loads(GOLDEN.read_text())
    exp.pop("_meta", None)
    if got["data_hash"] != exp["data_hash"]:
        pytest.fail(
            "synthetic market changed (data_hash "
            f"{got['data_hash']} != golden {exp['data_hash']}): qts.data.synthetic.generate output moved; "
            "every golden number will differ. If intended, regenerate with --update-golden."
        )
    errors: list = []
    _compare("golden", got, exp, errors)
    assert not errors, (
        "pipeline output drifted from tests/golden/pipeline_summary.json "
        "(regenerate with --update-golden only if the change is intended):\n  " + "\n  ".join(errors)
    )


def test_golden_run_is_not_degenerate(golden_pipeline_run):
    """Guard against a golden file that pins a do-nothing run (e.g. a cost
    or universe change that leaves the book flat would still 'match')."""
    _, res = golden_pipeline_run
    s = res.summary
    assert s["n_days"] == GOLDEN_MAX_DAYS and s["n_failed_solves"] == 0
    assert s["avg_gross_exposure"] > 0.5 and s["avg_daily_turnover"] > 0.0
    assert s["total_costs_usd"] > 0.0
    assert res.ic_report["n"] >= 3
    # optimizer health gate (not in the golden file, see GATES_NOT_IN_GOLDEN)
    assert res.hurdles["status"]["failed_solves"] == "PASS"
    assert res.backtest.failed_solve_fraction() == 0.0
