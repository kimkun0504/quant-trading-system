"""Load-coverage guard: refuse a store load whose prefilter misses names the universe picks.

Why (docs/notes/prefilter_lookahead_2026-11-14.md): ``--max-names N --min-mcap-ever X``
keeps the N ids with the largest *peak* market cap over the load window.  A name that
was in the point-in-time top 1,000 on some date but never grew enough to rank among the
3,000 largest peaks is not loaded, so on that date the universe is chosen partly by how
big names became *later* (look-ahead), and the bottom-size cut, a quantile of the
loaded cross-section, moves into the top 1,000 when too few names are loaded.  Measured
on the Sharadar store (2026-11-14): ``--max-names 3000`` loaded the whole screened
top 1,000 on only 23.7% of dates (worst date 82%), and was short of names for the
bottom-size cut on 1,080 dates.  It also made the universe depend on the window's end,
so a sealed re-run of the same configuration traded a different universe on the same
dates.

The guard, called by :func:`qts.research.data.load_research_market` (``qts run --store``,
signal tests, sweeps, capacity, risk calibration, the diagnostics loader, the API's
store jobs) before anything is loaded:

* ``--max-rank K`` (every id that ranked <= K by screened market cap on at least one
  date of the window): covers a universe of screened ranks ``offset+1 .. offset+top_n``
  by construction when ``K >= (offset + top_n) / (1 - exclude_bottom_size_quantile)``
  (:func:`qts.research.midcap.needed_loaded_names`; 1,250 for the baseline).  Checked
  without touching the store; below :data:`MARGIN` x that a warning is printed (the
  SQL screens and the panel's differ slightly for names with data gaps).
* the peak-cap prefilter: measured with :func:`qts.research.midcap.prefilter_coverage`
  (DuckDB over the store, a minute or two on the full store) over the window actually
  loaded (the sealed window under the seal); refused unless every criterion holds.
* no prefilter: every id is loaded, covered.

A refused load raises :class:`LoadCoverageError` (a ``ValueError``); ``allow_partial``
(``qts run --allow-partial-coverage``) loads anyway, prints a warning, and the run's
ledger record carries ``load_coverage.ok = false``.  ``$QTS_LOAD_COVERAGE``:
``enforce`` (default), ``warn`` (never refuse) or ``off`` (skip the check; the test
suite's default, see tests/conftest.py)."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

ENV = "QTS_LOAD_COVERAGE"
#: --max-rank headroom above the strict minimum before a warning
MARGIN = 1.2

Printer = Callable[[str], None]


class LoadCoverageError(ValueError):
    """The store prefilter would load an incomplete universe."""


@dataclass(frozen=True)
class LoadCoverage:
    """Result of the guard (``to_dict`` goes on the run's ledger record)."""

    prefilter: str
    method: str  # "by construction" | "measured" | "no prefilter" | "skipped"
    ok: bool
    needed_rank: int
    needed_loaded: int
    recommended_max_rank: int
    share_dates_full: float | None = None
    min_coverage: float | None = None
    n_short_dates: int | None = None
    n_loaded_ids: int | None = None
    window_end: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        if self.method == "skipped":
            return f"load coverage: not checked (${ENV}=off)"
        head = f"load coverage ({self.prefilter}, {self.method}): {'OK' if self.ok else 'NOT OK'}"
        if self.share_dates_full is not None:
            head += (
                f" -- screened top {self.needed_rank} fully loaded on {100 * self.share_dates_full:.2f}% of dates"
                f" (worst {100 * float(self.min_coverage or 0):.1f}%), short of names for the bottom-size cut on"
                f" {self.n_short_dates} dates"
            )
        return head + (f"; {self.detail}" if self.detail else "")


def recommended_max_rank(universe: Any) -> int:
    """``MARGIN`` x the strict minimum, rounded up to a multiple of 250 (1,500 for the baseline)."""
    from qts.research.midcap import needed_loaded_names

    need = needed_loaded_names(_as_system(universe))
    return int(math.ceil(need * MARGIN / 250.0) * 250)


def _as_system(universe: Any) -> Any:
    """An object with a ``.universe`` (what ``needed_rank`` / ``needed_loaded_names`` read):
    a ``SystemConfig`` as is, a ``UniverseConfig`` wrapped, ``None`` = the default universe."""
    from types import SimpleNamespace

    from qts.core.config import UniverseConfig

    if universe is None:
        return SimpleNamespace(universe=UniverseConfig())
    if hasattr(universe, "universe"):
        return universe
    return SimpleNamespace(universe=universe)


def mode() -> str:
    m = os.environ.get(ENV, "enforce").strip().lower() or "enforce"
    if m not in ("enforce", "warn", "off"):
        raise LoadCoverageError(f"${ENV} must be enforce, warn or off (got {m!r})")
    return m


def check_load_coverage(spec: Any, universe: Any = None, end: str | None = None) -> LoadCoverage:
    """Coverage of ``spec``'s prefilter for ``universe`` (a ``UniverseConfig`` or
    ``SystemConfig``; ``None`` = the default universe) over ``[spec.start, end]``."""
    from qts.research.midcap import needed_loaded_names, needed_rank, prefilter_coverage

    cfg = _as_system(universe)
    rank, need = needed_rank(cfg), needed_loaded_names(cfg)
    rec = recommended_max_rank(cfg)
    common = {"needed_rank": rank, "needed_loaded": need, "recommended_max_rank": rec, "window_end": end}
    if spec.max_rank is not None:
        k = int(spec.max_rank)
        ok = k >= need
        detail = (
            f"--max-rank {k} < {need} = universe last rank / (1 - bottom-size quantile)"
            if not ok
            else (
                f"--max-rank {k} is under {MARGIN:g}x that minimum ({rec} recommended)"
                if k < need * MARGIN
                else ""
            )
        )
        return LoadCoverage(
            prefilter=f"--max-rank {k}", method="by construction", ok=ok, detail=detail, **common
        )
    if spec.max_names is None and spec.min_mcap_ever is None:
        return LoadCoverage(prefilter="none (every id)", method="no prefilter", ok=True, **common)
    rep = prefilter_coverage(spec.with_screens(cfg.universe), rank=rank, needed_loaded=need, end=end)
    return LoadCoverage(
        prefilter=rep.prefilter,
        method="measured",
        ok=bool(rep.ok),
        share_dates_full=float(rep.share_dates_full),
        min_coverage=float(rep.min_coverage),
        n_short_dates=int(rep.n_short_dates),
        n_loaded_ids=int(rep.n_loaded_ids),
        **common,
    )


def require_load_coverage(
    spec: Any,
    universe: Any = None,
    *,
    end: str | None = None,
    allow_partial: bool = False,
    log: Printer = lambda _m: None,
) -> LoadCoverage:
    """:func:`check_load_coverage`, refusing an incomplete load (see the module docstring)."""
    m = mode()
    if m == "off":
        cfg = _as_system(universe)
        from qts.research.midcap import needed_loaded_names, needed_rank

        return LoadCoverage(
            prefilter="?",
            method="skipped",
            ok=True,
            needed_rank=needed_rank(cfg),
            needed_loaded=needed_loaded_names(cfg),
            recommended_max_rank=recommended_max_rank(cfg),
            window_end=end,
        )
    if spec.max_rank is None and (spec.max_names is not None or spec.min_mcap_ever is not None):
        log(
            "load coverage: measuring the peak-cap prefilter against the universe (DuckDB, a minute or two) ..."
        )
    cov = check_load_coverage(spec, universe, end=end)
    log(cov.line())
    if cov.ok:
        return cov
    advice = (
        f"the store prefilter would load an incomplete universe ({cov.line()}).  "
        f"Use --max-rank {cov.recommended_max_rank} instead of --max-names / --min-mcap-ever "
        "(docs/notes/prefilter_lookahead_2026-11-14.md)"
    )
    if allow_partial or m == "warn":
        log(f"WARNING: {advice}; loading anyway (flagged on the run's ledger record)")
        return cov
    raise LoadCoverageError(advice + "; --allow-partial-coverage loads anyway (flagged)")


__all__ = [
    "ENV",
    "MARGIN",
    "LoadCoverage",
    "LoadCoverageError",
    "check_load_coverage",
    "recommended_max_rank",
    "require_load_coverage",
]
