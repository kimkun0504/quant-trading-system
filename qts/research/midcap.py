"""The pre-registered mid-cap confirmation of the frozen baseline (``qts check midcap``).

Pre-registration (criteria, thresholds, what each outcome means):
``docs/notes/midcap_check_2026-11-13.md``.  In short: ``configs/baseline.yaml`` with
``universe.rank_offset = 1000`` (stocks ranked 1001-2000 by market cap, never traded
by any earlier run), run **once** over the full history (exempt from the holdout seal
by design), logged as a one-shot confirmation in its own ledger family
``confirm:midcap:<market_hash[:12]>``.

* PASS = C1 net Sharpe > 0 and Newey-West t of the daily net return >= 2.0,
  C2 traded-score IC > 0 in >= 4 of 6 purged folds, C3 net Sharpe > 0 in >= 4 of 6
  folds;
* INCONCLUSIVE when a validity condition fails (V1 median universe >= 900 names,
  V2 load coverage checked before the run, V3 failed solves <= 2 %, risk-model
  sanity gate not FAIL, invested >= 90 % of days);
* reported: S1 net Sharpe at 2x the modelled cost per side ("PASS, cost-fragile" when
  <= 0), breakeven cost, the PSR, and a side-by-side with the large-cap reference run.

Library entry points (JSON-serialisable via ``to_dict()``): :func:`prefilter_coverage`
-> :class:`CoverageReport`; :func:`evaluate_midcap` -> :class:`MidcapVerdict`;
:func:`run_midcap_check` -> :class:`MidcapCheckResult`; :func:`previous_midcap_records`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from qts.research.common import (
    BASELINE_CONFIG,
    adv_cap_share,
    fmt_float,
    json_safe,
    pipeline_config,
    resolve_repo_path,
    write_json,
)
from qts.research.data import Printer, StoreSpec, _quiet

if TYPE_CHECKING:
    from qts.core.config import SystemConfig
    from qts.pipeline import PipelineResult, ResearchPipeline

#: The check's configuration change: skip the 1,000 largest names.
MIDCAP_RANK_OFFSET = 1000
#: Default prefilter of the check: ``--max-rank`` (screened market-cap rank) 2,750 -- every
#: name that ranks <= 2,750 among the names passing the price / ADV screens on some date:
#: rank 2,000 is the last the universe takes, 2,500 = 2,000 / (1 - 0.2) keep the bottom-size
#: cut below it, +10 % margin for the screens' approximation in SQL
#: (docs/notes/research_tools_2026-11-13.md "Mid-cap load").
MIDCAP_DEFAULT_MAX_RANK = 2750
#: Pre-registration note (its text above the "## Result" heading is hashed onto the record).
PREREGISTRATION = "docs/notes/midcap_check_2026-11-13.md"
#: The large-cap reference run the check is compared with (``summary.json``).
REFERENCE_RUN_ID = "20261113T023313333450Z_e4667d90"
#: Ledger family prefix of the confirmation.
FAMILY_PREFIX = "confirm:midcap:"
#: V2: the store's screened top ``COVERAGE_RANK`` names (the universe's last rank) must be
#: loaded on at least ``COVERAGE_MIN_SHARE`` of dates.
COVERAGE_RANK = 2000
COVERAGE_MIN_SHARE = 0.99
#: Thresholds of the pre-registration.
C1_MIN_NW_T = 2.0
FOLD_PASS_FRACTION = 2.0 / 3.0
#: V1: median eligible names >= this fraction of ``universe.top_n_by_mcap`` (900 of 1,000)
V1_MIN_UNIVERSE_FRAC = 0.9
V3_MAX_FAILED_SOLVES = 0.02
V3_MIN_INVESTED = 0.90
S1_COST_MULTIPLE = 2.0


class MidcapCheckError(RuntimeError):
    """The check refuses to run (already run, coverage too low, wrong config)."""


# ----------------------------------------------------------------------------- coverage
@dataclass(frozen=True)
class CoverageReport:
    """V2: does the prefiltered load contain every name the universe can pick?

    Ranks are *screened* ranks (``ParquetStore.screened_rank_sql``): by market cap among
    the names that pass the universe's price and ADV screens on the date, i.e. the
    ranks the universe rule uses (before its bottom-size cut, which only removes
    names from the bottom).  ``share_dates_full``: share of dates on which every one of
    the store's screened top-``rank`` names (``rank`` = the universe's last rank, 2,000)
    is loaded; ``min_coverage`` the worst date's share; ``min_loaded`` /
    ``median_loaded``: loaded names passing the screens per date -- the bottom-size cut
    is a quantile of these, so on every date they must reach ``needed_loaded`` = last
    rank / (1 - quantile) = 2,500, or every passing name of the store when it has fewer
    (``min_store``: the store's own minimum; the load then cannot do better, and a thin
    universe shows up in V1).  ``n_short_dates``: dates failing that.
    ``est_load_gb``: rough peak memory of the load
    (:func:`qts.research.data.estimated_load_gb`)."""

    rank: int
    prefilter: str
    n_loaded_ids: int
    n_dates: int
    share_dates_full: float
    min_coverage: float
    worst_dates: list[dict[str, Any]]
    min_loaded: int
    median_loaded: float
    needed_loaded: int
    est_load_gb: float = float("nan")
    min_store: int = 0
    n_short_dates: int = 0
    min_share: float = COVERAGE_MIN_SHARE

    @property
    def ok(self) -> bool:
        return bool(self.n_dates > 0 and self.share_dates_full >= self.min_share and self.n_short_dates == 0)

    def to_dict(self) -> dict[str, Any]:
        return json_safe({**asdict(self), "ok": self.ok})

    def format(self) -> str:
        lines = [
            "=== Load coverage (V2, checked before the run) ===",
            f"prefilter {self.prefilter}: {self.n_loaded_ids} ids x {self.n_dates} dates "
            f"(estimated peak memory of the load ~{self.est_load_gb:.1f} GB, before the fits)",
            f"store screened top-{self.rank} (price / ADV screens, by market cap) fully loaded on "
            f"{100 * self.share_dates_full:.2f}% of dates (need >= {100 * self.min_share:.0f}%)  "
            f"worst date {100 * self.min_coverage:.1f}%",
            f"loaded names passing the screens per date: min {self.min_loaded}  median {self.median_loaded:.0f}  "
            f"(need >= {self.needed_loaded}, or all of the store's when fewer, so the bottom-size cut stays "
            f"below rank {self.rank}; short on {self.n_short_dates} dates)",
            f"store names passing the screens per date: min {self.min_store}"
            + (
                f"  (< {self.needed_loaded}: on those dates the universe can be short of names whatever "
                "is loaded; V1 judges it)"
                if self.min_store < self.needed_loaded
                else ""
            ),
        ]
        if self.worst_dates:
            lines.append(
                "worst dates: "
                + ", ".join(
                    f"{w['date']} {100 * w['coverage']:.1f}% ({w['loaded']} passing loaded)"
                    for w in self.worst_dates
                )
            )
        lines.append("coverage OK" if self.ok else "coverage NOT OK: raise --max-rank (or --max-names)")
        return "\n".join(lines)


def prefilter_coverage(
    spec: StoreSpec,
    rank: int = COVERAGE_RANK,
    needed_loaded: int = 2500,
    n_worst: int = 5,
    end: str | None = None,
) -> CoverageReport:
    """Measure V2 on the store with DuckDB (no panel is pivoted): the prefilter's ids
    (:func:`qts.research.data.prefilter_ids`), then per date the store-wide screened
    top-``rank`` names and how many of them are loaded, and the number of loaded names
    passing the screens.  The screens are ``spec``'s (``StoreSpec.with_screens``)."""
    from qts.data.store import ParquetStore
    from qts.research.data import estimated_load_gb, prefilter_ids

    store = ParquetStore(spec.store)
    e = end if end is not None else spec.end
    ids = prefilter_ids(spec, end=e)
    ranked = store.screened_rank_sql(spec.start, e, spec.min_price, spec.min_adv_usd, spec.adv_window)
    if spec.max_rank is not None:
        how = f"--max-rank {spec.max_rank}"
    elif ids is None:
        how = "none (every id)"
    else:
        how = f"--max-names {spec.max_names} --min-mcap-ever {spec.min_mcap_ever}"
    con = store.duckdb_connection()
    try:
        if ids is None:
            con.execute("CREATE TEMP TABLE loaded AS SELECT DISTINCT ticker FROM prices_close")
        else:
            con.execute("CREATE TEMP TABLE loaded (ticker VARCHAR)")
            con.executemany("INSERT INTO loaded VALUES (?)", [(t,) for t in ids])
        n_ids = int((con.execute("SELECT count(*) FROM loaded").fetchone() or (0,))[0])
        con.execute(f"CREATE TEMP TABLE ranked AS {ranked}")
        df = con.execute(
            f"""
            SELECT r.date AS date,
                   count(*) FILTER (WHERE r.rk <= {int(rank)}) AS n_top,
                   count(l.ticker) FILTER (WHERE r.rk <= {int(rank)}) AS n_top_loaded,
                   count(l.ticker) AS n_loaded,
                   count(*) AS n_store
            FROM ranked r LEFT JOIN loaded l ON r.ticker = l.ticker
            GROUP BY r.date ORDER BY r.date
            """
        ).df()
        n_days = int((con.execute("SELECT count(DISTINCT date) FROM prices_close").fetchone() or (0,))[0])
    finally:
        con.close()
    est = estimated_load_gb(n_ids, n_days)
    if df.empty:
        return CoverageReport(rank, how, n_ids, 0, 0.0, 0.0, [], 0, 0.0, needed_loaded, est)
    df["date"] = pd.to_datetime(df["date"])
    cov = df["n_top_loaded"] / df["n_top"].clip(lower=1)
    short = df["n_loaded"] < np.minimum(int(needed_loaded), df["n_store"])
    worst = df.assign(cov=cov).nsmallest(n_worst, "cov")
    return CoverageReport(
        rank=int(rank),
        prefilter=how,
        n_loaded_ids=n_ids,
        n_dates=len(df),
        share_dates_full=float((cov >= 1.0 - 1e-12).mean()),
        min_coverage=float(cov.min()),
        worst_dates=[
            {"date": str(r.date.date()), "coverage": float(r.cov), "loaded": int(r.n_loaded)}
            for r in worst.itertuples()
            if float(r.cov) < 1.0
        ],
        min_loaded=int(df["n_loaded"].min()),
        median_loaded=float(df["n_loaded"].median()),
        needed_loaded=int(needed_loaded),
        est_load_gb=est,
        min_store=int(df["n_store"].min()),
        n_short_dates=int(short.sum()),
    )


def needed_rank(cfg: SystemConfig) -> int:
    """The universe's last screened rank: ``rank_offset + top_n_by_mcap`` (2,000)."""
    return int(cfg.universe.rank_offset) + int(cfg.universe.top_n_by_mcap)


def needed_loaded_names(cfg: SystemConfig) -> int:
    """Loaded names with a market cap needed per date so that the universe's bottom
    size-quantile cut (computed on the loaded cross-section) stays below the last rank
    taken: ``(rank_offset + top_n) / (1 - exclude_bottom_size_quantile)``."""
    u = cfg.universe
    last = int(u.rank_offset) + int(u.top_n_by_mcap)
    q = float(u.exclude_bottom_size_quantile)
    return math.ceil(last / max(1.0 - q, 1e-9))


# ----------------------------------------------------------------------------- verdict
@dataclass(frozen=True)
class Criterion:
    """One pre-registered criterion or validity condition."""

    id: str
    name: str
    value: str
    threshold: str
    passed: bool | None

    @property
    def mark(self) -> str:
        return "n/a " if self.passed is None else ("PASS" if self.passed else "FAIL")


@dataclass(frozen=True)
class MidcapVerdict:
    """Outcome of the pre-registered check (:func:`evaluate_midcap`)."""

    outcome: str  #: "PASS", "PASS, cost-fragile", "FAIL" or "INCONCLUSIVE"
    criteria: list[Criterion]
    validity: list[Criterion]
    secondary: dict[str, Any]
    side_by_side: list[dict[str, Any]]
    reference_run: str | None
    run_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("criteria", "validity"):
            d[k] = [dict(c, mark=Criterion(**c).mark) for c in d[k]]
        return json_safe(d)

    def format(self) -> str:
        lines = [
            "",
            "=== Mid-cap confirmation: pre-registered criteria (docs/notes/midcap_check_2026-11-13.md) ===",
        ]
        for c in [*self.criteria, *self.validity]:
            lines.append(f"{c.id:3s} {c.mark}  {c.name:44s} {c.value:>28s}   need {c.threshold}")
        s = self.secondary
        lines.append(
            f"S1  net Sharpe at {S1_COST_MULTIPLE:g}x cost {fmt_float(s.get('net_sharpe_cost_x2'))}   breakeven cost "
            f"{fmt_float(s.get('breakeven_cost_bps'), '.1f')} bp/side vs modelled "
            f"{fmt_float(s.get('cost_bps_per_side'), '.1f')} bp"
        )
        lines.append(
            f"S2  PSR (DSR at N = 1) {fmt_float(s.get('psr'), '.3f')}   ADV cap: {fmt_float(100 * s.get('traded_at_cap', float('nan')), '.1f')}% "
            f"of traded names at the cap per solve"
        )
        if self.side_by_side:
            lines += ["", "=== Side by side (S3) ===", f"{'':28s}{'mid caps':>12s}{'large caps':>12s}"]
            lines.append(f"{'':28s}{'(this run)':>12s}{'(reference)':>12s}")
            for r in self.side_by_side:
                lines.append(f"{r['metric']:28s}{r['midcap']:>12s}{r['largecap']:>12s}")
            if self.reference_run is None:
                lines.append("(reference run not found: large-cap column empty; pass --reference-run DIR)")
        if self.notes:
            lines += ["", "notes: " + " | ".join(self.notes)]
        lines += ["", f"VERDICT: {self.outcome}"]
        return "\n".join(lines)


def _fold_counts(values: pd.Series) -> tuple[int, int]:
    v = pd.Series(values, dtype=float).dropna()
    return int((v > 0).sum()), len(v)


def _needed(n: int) -> int:
    return math.ceil(FOLD_PASS_FRACTION * n - 1e-12)


SIDE_BY_SIDE: tuple[tuple[str, str, str, str], ...] = (
    # (label, section, key, format)
    ("net Sharpe", "backtest", "sharpe_net", ".2f"),
    ("gross Sharpe", "backtest", "sharpe_gross", ".2f"),
    ("NW t (net)", "backtest", "newey_west_t_net", ".2f"),
    ("net CAGR %", "backtest", "cagr_net", "%"),
    ("ann. vol net %", "backtest", "ann_vol_net", "%"),
    ("max drawdown %", "backtest", "max_drawdown_net", "%"),
    ("mean rank IC", "ic", "mean_ic", ".4f"),
    ("IC NW t", "ic", "t_stat_nw", ".2f"),
    ("turnover / day %", "backtest", "avg_daily_turnover", "%"),
    ("cost per side bp", "backtest", "avg_cost_bps_per_side", ".1f"),
    ("breakeven cost bp", "", "breakeven_cost_bps", ".1f"),
    ("effective names", "backtest", "effective_n_names", ".0f"),
    ("invested days %", "backtest", "invested_day_frac", "%"),
    ("transfer coefficient", "backtest", "transfer_coefficient_mean", ".2f"),
    ("DSR (own family)", "", "dsr", ".3f"),
)


def _get(summary: Mapping[str, Any] | None, section: str, key: str) -> Any:
    if not summary:
        return None
    src = summary.get(section, {}) if section else summary
    return src.get(key) if isinstance(src, Mapping) else None


def _fmt(v: Any, spec: str) -> str:
    if spec == "%":
        try:
            return fmt_float(100 * float(v), ".1f")
        except (TypeError, ValueError):
            return "n/a"
    return fmt_float(v, spec)


def side_by_side(mid: Mapping[str, Any], ref: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Rows ``{"metric", "midcap", "largecap", "midcap_value", "largecap_value"}`` from two
    ``summary.json`` payloads (the reference may be ``None``)."""
    rows = []
    for label, section, key, spec in SIDE_BY_SIDE:
        a, b = _get(mid, section, key), _get(ref, section, key)
        rows.append(
            {
                "metric": label,
                "midcap": _fmt(a, spec),
                "largecap": _fmt(b, spec) if ref else "",
                "midcap_value": a,
                "largecap_value": b,
            }
        )
    return rows


def load_reference_summary(
    path: str | Path | None, runs_root: str | Path = "runs"
) -> tuple[dict | None, str | None]:
    """``(summary.json payload, run dir)`` of the large-cap reference run, or
    ``(None, None)`` when it is not on this machine."""
    cands = (
        [Path(path)]
        if path
        else [Path(runs_root) / REFERENCE_RUN_ID, resolve_repo_path(f"runs/{REFERENCE_RUN_ID}")]
    )
    for p in cands:
        f = p / "summary.json"
        if f.exists():
            from qts.runs.registry import _from_json_scalar

            return _from_json_scalar(json.loads(f.read_text())), str(p)
    return None, None


def evaluate_midcap(
    res: PipelineResult,
    pipe: ResearchPipeline,
    reference_summary: Mapping[str, Any] | None = None,
    reference_run: str | None = None,
    coverage: CoverageReport | None = None,
) -> MidcapVerdict:
    """Apply the pre-registered criteria to a finished mid-cap run."""
    from qts.pipeline import purged_fold_blocks
    from qts.runs.registry import run_summary_dict
    from qts.validation.deflated_sharpe import deflated_sharpe_ratio_annual

    cfg = pipe.cfg
    s = res.summary
    bt = res.backtest
    net = bt.returns["net"].dropna()
    v = cfg.validation
    blocks = purged_fold_blocks(
        pd.DatetimeIndex(net.index), v.n_splits, v.embargo_days, cfg.alpha.horizon_days, v.min_fold_days
    )
    ic_folds = pipe._ic_by_fold(res.ic_report.ic_series, blocks)
    sr_pos, sr_n = _fold_counts(res.fold_sharpes)
    ic_pos, ic_n = _fold_counts(ic_folds)
    sharpe_net = float(s["sharpe_net"])
    nw_t = float(s["newey_west_t_net"])
    c1 = math.isfinite(sharpe_net) and math.isfinite(nw_t) and sharpe_net > 0 and nw_t >= C1_MIN_NW_T
    criteria = [
        Criterion(
            "C1",
            "net Sharpe > 0 and NW t (net) >= 2.0",
            f"SR {fmt_float(sharpe_net)}  t {fmt_float(nw_t)}",
            f">0 and t >= {C1_MIN_NW_T:g}",
            bool(c1),
        ),
        Criterion(
            "C2",
            "traded-score IC > 0 by purged fold",
            f"{ic_pos} of {ic_n} folds",
            f">= {_needed(ic_n)} of {ic_n}" if ic_n else "folds (none measurable)",
            None if ic_n == 0 else bool(ic_pos >= _needed(ic_n)),
        ),
        Criterion(
            "C3",
            "net Sharpe > 0 by purged fold",
            f"{sr_pos} of {sr_n} folds",
            f">= {_needed(sr_n)} of {sr_n}" if sr_n else "folds (none measurable)",
            None if sr_n == 0 else bool(sr_pos >= _needed(sr_n)),
        ),
    ]
    # validity
    assert pipe.md_u is not None
    elig = pipe.md_u.eligibility()
    bt_dates = pd.DatetimeIndex(bt.weights.index).intersection(elig.index)
    median_u = float(elig.loc[bt_dates].sum(axis=1).median()) if len(bt_dates) else float("nan")
    failed = bt.failed_solve_fraction()
    sanity = res.hurdles.passed.get("risk_model_sanity")
    invested = float(s.get("invested_day_frac", float("nan")))
    v3 = (
        (failed is None or failed <= V3_MAX_FAILED_SOLVES)
        and sanity is not False
        and math.isfinite(invested)
        and invested >= V3_MIN_INVESTED
    )
    v1_min = V1_MIN_UNIVERSE_FRAC * float(cfg.universe.top_n_by_mcap)
    validity = [
        Criterion(
            "V1",
            "median eligible names per backtest day",
            fmt_float(median_u, ".0f"),
            f">= {v1_min:.0f}",
            bool(math.isfinite(median_u) and median_u >= v1_min),
        ),
        Criterion(
            "V2",
            f"screened top-{coverage.rank if coverage is not None else COVERAGE_RANK} loaded (share of dates)",
            "not measured" if coverage is None else f"{100 * coverage.share_dates_full:.2f}%",
            f">= {100 * COVERAGE_MIN_SHARE:.0f}%",
            None if coverage is None else bool(coverage.ok),
        ),
        Criterion(
            "V3",
            "failed solves / risk sanity / invested days",
            f"{fmt_float(100 * (failed or 0.0), '.1f')}% / {'n/a' if sanity is None else ('ok' if sanity else 'FAIL')} / "
            f"{fmt_float(100 * invested, '.0f')}%",
            f"<= {100 * V3_MAX_FAILED_SOLVES:.0f}% / not FAIL / >= {100 * V3_MIN_INVESTED:.0f}%",
            bool(v3),
        ),
    ]
    # secondary
    cs = res.cost_sensitivity
    sr2 = float(cs.loc[S1_COST_MULTIPLE, "net_sharpe"]) if S1_COST_MULTIPLE in cs.index else float("nan")
    psr = deflated_sharpe_ratio_annual(
        sharpe_net,
        n_trials=1.0,
        var_trials_sr_annual=0.0,
        n_obs=len(net),
        skew=float(net.skew()),
        kurt=float(net.kurt() + 3),
        periods_per_year=cfg.trading_days_per_year,
    )
    cap = adv_cap_share(bt)
    secondary = {
        "net_sharpe_cost_x2": sr2,
        "breakeven_cost_bps": float(res.breakeven_cost_bps),
        "cost_bps_per_side": float(s.get("avg_cost_bps_per_side", float("nan"))),
        "psr": float(psr),
        "traded_at_cap": cap["traded_at_cap"],
        "solves_with_cap": cap["solves_with_cap"],
        "median_universe": median_u,
        "ic_by_fold": {str(k.date()): float(x) for k, x in ic_folds.items()},
        "sharpe_by_fold": {str(pd.Timestamp(k).date()): float(x) for k, x in res.fold_sharpes.items()},
    }
    if any(c.passed is False for c in validity) or any(c.passed is None for c in criteria):
        # a failed validity condition, or a criterion that could not be measured
        outcome = "INCONCLUSIVE"
    elif all(c.passed for c in criteria):
        outcome = "PASS" if (math.isfinite(sr2) and sr2 > 0) else "PASS, cost-fragile"
    else:
        outcome = "FAIL"
    notes = []
    if coverage is None:
        notes.append("V2 was not measured (library call without a coverage report)")
    return MidcapVerdict(
        outcome=outcome,
        criteria=criteria,
        validity=validity,
        secondary=secondary,
        side_by_side=side_by_side(run_summary_dict(res), reference_summary),
        reference_run=reference_run,
        run_id=None if res.manifest is None else res.manifest.run_id,
        notes=notes,
    )


# ----------------------------------------------------------------------------- once-only
def preregistration_sha256(path: str | Path = PREREGISTRATION) -> str:
    """SHA-256 of the pre-registration text above its ``## Result`` heading (appending
    the result later does not change it); ``""`` when the file is missing."""
    p = resolve_repo_path(path)
    if not p.exists():
        return ""
    text = p.read_text()
    head = text.split("\n## Result", 1)[0]
    return hashlib.sha256(head.encode()).hexdigest()


def previous_midcap_records(ledger_path: str | None, config_hash: str) -> list[dict]:
    """Earlier mid-cap confirmations of the same strategy: records of any
    ``confirm:midcap:*`` family whose configuration, re-hashed with the current
    ``SystemConfig`` (:func:`qts.validation.ledger.normalized_config_hash`), equals
    ``config_hash`` (so a config field added since does not hide an earlier run)."""
    if not ledger_path or not Path(ledger_path).exists():
        return []
    from qts.validation import TrialLedger
    from qts.validation.ledger import normalized_config_hash

    out = []
    for r in TrialLedger(ledger_path).raw_records():
        if not str(r.get("family") or "").startswith(FAMILY_PREFIX):
            continue
        if r.get("config_hash") == config_hash or normalized_config_hash(r) == config_hash:
            out.append(r)
    return out


def midcap_config(
    config_path: str | Path = BASELINE_CONFIG,
    overrides: Mapping[str, Any] | None = None,
    rank_offset: int = MIDCAP_RANK_OFFSET,
) -> SystemConfig:
    """The check's configuration: the baseline with ``universe.rank_offset = 1000``
    (``rank_offset`` / ``overrides`` exist for scaled-down tests; any other value is a
    different check with a different config hash).  A baseline that already sets a
    rank offset is refused."""
    from qts.core.config import SystemConfig

    base = SystemConfig.from_yaml(resolve_repo_path(config_path))
    if int(base.universe.rank_offset) != 0:
        raise MidcapCheckError(f"{config_path} already sets universe.rank_offset={base.universe.rank_offset}")
    return base.with_overrides({"universe.rank_offset": int(rank_offset), **dict(overrides or {})})


# ----------------------------------------------------------------------------- the run
@dataclass
class MidcapCheckResult:
    """What :func:`run_midcap_check` did (``dry_run``: the plan and the coverage only)."""

    config_hash: str
    config_path: str
    store: dict[str, Any]
    coverage: CoverageReport | None
    preregistration_sha256: str
    previous_runs: list[dict[str, Any]]
    dry_run: bool
    forced: bool = False
    family: str | None = None
    run_dir: str | None = None
    verdict: MidcapVerdict | None = None
    report: str = ""
    json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "config_hash": self.config_hash,
                "config_path": self.config_path,
                "store": self.store,
                "coverage": None if self.coverage is None else self.coverage.to_dict(),
                "preregistration": PREREGISTRATION,
                "preregistration_sha256": self.preregistration_sha256,
                "previous_runs": self.previous_runs,
                "dry_run": self.dry_run,
                "forced": self.forced,
                "family": self.family,
                "run_dir": self.run_dir,
                "verdict": None if self.verdict is None else self.verdict.to_dict(),
                "json_path": self.json_path,
            }
        )


def _record_summary(r: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ts": r.get("ts"),
        "family": r.get("family"),
        "config_hash": r.get("config_hash"),
        "sharpe_annual": (r.get("metrics") or {}).get("sharpe_annual"),
        "forced": bool(r.get("midcap_forced", False)),
    }


def run_midcap_check(
    spec: StoreSpec,
    *,
    config_path: str | Path = BASELINE_CONFIG,
    ledger_path: str | None = "runs/trial_ledger.jsonl",
    runs_root: str | None = "runs",
    cache_root: str | None = None,
    nav: float = 5e6,
    use_ml: bool = True,
    force: bool = False,
    dry_run: bool = False,
    reference_run: str | Path | None = None,
    progress: bool = False,
    log: Printer = _quiet,
    md: Any = None,
    coverage_rank: int | None = None,
    rank_offset: int = MIDCAP_RANK_OFFSET,
    max_days: int | None = None,
) -> MidcapCheckResult:
    """Run the check once (see the module docstring).  ``md``: an already loaded
    full-history ``MarketData`` (tests); otherwise the store is loaded with
    :func:`qts.research.data.load_store_market`.  ``rank_offset`` / ``max_days`` /
    ``coverage_rank`` are for scaled-down tests only (the check is 1000 / all days /
    2200).  Raises :class:`MidcapCheckError` when it refuses to run (a previous run of
    the same configuration without ``force``, insufficient load coverage, no
    pre-registration or no ledger)."""
    from qts.core.manifest import hash_market
    from qts.data.universe import survivorship_check
    from qts.pipeline import ResearchPipeline, format_result
    from qts.research.data import load_store_market
    from qts.runs.context import RunContext

    cfg = midcap_config(config_path, rank_offset=rank_offset)
    chash = cfg.hash()
    sha = preregistration_sha256()
    prev = previous_midcap_records(ledger_path, chash)
    log(f"mid-cap check: config {config_path} + universe.rank_offset={rank_offset}  hash {chash}")
    log(f"pre-registration {PREREGISTRATION}  sha256 {sha[:16] or '(missing)'}")
    if not sha:
        raise MidcapCheckError(
            f"pre-registration {PREREGISTRATION} not found: the check must not run without it"
        )
    if not ledger_path:
        raise MidcapCheckError("the mid-cap check must be logged: give a trial ledger (--ledger)")
    if prev and not force:
        lines = "\n".join(f"  {r.get('ts')}  {r.get('family')}" for r in prev)
        raise MidcapCheckError(
            f"the mid-cap check of this configuration already ran:\n{lines}\n"
            "a second run is no longer out-of-sample; --force runs it anyway (flagged on the record)"
        )
    if prev:
        log(
            "WARNING: --force: the mid-cap check of this configuration already ran "
            f"{len(prev)} time(s); this repeat is NOT out-of-sample and is flagged as such"
        )
    coverage = None
    spec = spec.with_screens(cfg.universe)
    if md is None:
        log("measuring load coverage (DuckDB over the store; a minute or two on the full store) ...")
        coverage = prefilter_coverage(
            spec, rank=coverage_rank or needed_rank(cfg), needed_loaded=needed_loaded_names(cfg)
        )
        log(coverage.format())
    result = MidcapCheckResult(
        config_hash=chash,
        config_path=str(config_path),
        store=spec.to_dict(),
        coverage=coverage,
        preregistration_sha256=sha,
        previous_runs=[_record_summary(r) for r in prev],
        dry_run=dry_run,
        forced=bool(prev) and force,
    )
    if coverage is not None and not coverage.ok and not dry_run:
        raise MidcapCheckError("load coverage too low (V2): " + coverage.format().splitlines()[-1])
    if dry_run:
        log("dry run: nothing loaded, nothing logged")
        return result
    if md is None:
        md = load_store_market(spec, log=log)
    survivorship_check(md)
    mh = hash_market(md)
    family = f"{FAMILY_PREFIX}{mh[:12]}"
    seal_fields = _seal_fields()
    ctx = RunContext(
        family=family,
        seal_exempt="midcap",
        holdout_start=seal_fields.get("holdout_start"),
        unsealed=seal_fields.get("holdout_start") is not None,
        record_extra={
            "check": "midcap",
            "preregistration": PREREGISTRATION,
            "preregistration_sha256": sha,
            "midcap_forced": bool(prev),
            "midcap_previous_runs": len(prev),
        },
    )
    cache = cache_root if cache_root is not None else (f"{runs_root}/_cache" if runs_root else None)
    run_cfg = pipeline_config(cfg, md, cache)
    pipe = ResearchPipeline(
        run_cfg, use_ml=use_ml, ledger_path=ledger_path, nav0=nav, runs_root=runs_root, context=ctx
    )
    res = pipe.run(md, progress=progress, max_days=max_days)
    ref, ref_dir = load_reference_summary(reference_run, runs_root or "runs")
    verdict = evaluate_midcap(res, pipe, ref, ref_dir, coverage)
    report = format_result(res) + "\n" + verdict.format()
    result.family = family
    result.run_dir = res.run_dir
    result.verdict = verdict
    result.report = report
    if runs_root:
        out = Path(runs_root) / "_checks" / f"midcap_{verdict.run_id or 'run'}.json"
        result.json_path = str(out)
        write_json(out, result.to_dict())
        if res.run_dir:
            write_json(Path(res.run_dir) / "midcap_check.json", result.to_dict())
    log(report)
    return result


def _seal_fields() -> dict[str, Any]:
    """``{"holdout_start": ISO date}`` of the seal in force, when the seal module and
    file exist (the check is exempt; the record notes the seal it bypassed)."""
    try:
        from qts.research.seal import load_seal  # type: ignore[import-not-found]
    except ImportError:
        return {}
    seal = load_seal()
    return {} if seal is None else {"holdout_start": seal.holdout_start_iso}


def criteria_table() -> list[dict[str, Any]]:
    """The pre-registered thresholds as data (API / UI)."""
    return [
        {"id": "C1", "rule": f"net Sharpe > 0 and Newey-West t of daily net return >= {C1_MIN_NW_T:g}"},
        {"id": "C2", "rule": "traded-score rank IC > 0 in >= 2/3 of purged walk-forward folds"},
        {"id": "C3", "rule": "net Sharpe > 0 in >= 2/3 of purged walk-forward folds"},
        {
            "id": "V1",
            "rule": f"median eligible names per backtest day >= {V1_MIN_UNIVERSE_FRAC:g} x top_n_by_mcap (900)",
        },
        {
            "id": "V2",
            "rule": f"every name ranked <= {COVERAGE_RANK} by market cap among names passing the price / ADV "
            f"screens loaded on >= {100 * COVERAGE_MIN_SHARE:.0f}% of dates, and >= 2,500 such names loaded per date",
        },
        {
            "id": "V3",
            "rule": f"failed solves <= {100 * V3_MAX_FAILED_SOLVES:.0f}%, risk_model_sanity not FAIL, "
            f"invested >= {100 * V3_MIN_INVESTED:.0f}% of days",
        },
        {
            "id": "S1",
            "rule": f"reported: net Sharpe at {S1_COST_MULTIPLE:g}x modelled cost (PASS, cost-fragile if <= 0)",
        },
    ]


__all__ = [
    "COVERAGE_RANK",
    "MIDCAP_DEFAULT_MAX_RANK",
    "MIDCAP_RANK_OFFSET",
    "CoverageReport",
    "Criterion",
    "MidcapCheckError",
    "MidcapCheckResult",
    "MidcapVerdict",
    "criteria_table",
    "evaluate_midcap",
    "midcap_config",
    "needed_loaded_names",
    "prefilter_coverage",
    "preregistration_sha256",
    "previous_midcap_records",
    "run_midcap_check",
    "side_by_side",
]
