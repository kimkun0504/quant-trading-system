"""The one store loader of the research tools: :class:`StoreSpec` (the ``qts run``
store flags as a JSON-serialisable value) and :func:`load_store_market`.

``load_store_market(spec)`` reproduces ``qts run --store`` exactly: the peak-market-cap
prefilter (``ParquetStore.tickers_by_market_cap`` over ``[start, end]``) when
``max_names`` or ``min_mcap_ever`` is set, then ``ParquetStore.as_market_data`` of
those ids, so the market hash (and with it the ledger family) of a research command
equals the one of a ``qts run`` with the same flags.

``max_rank`` (``--max-rank K``; research commands only) is the alternative
*screened-rank* prefilter (``ParquetStore.tickers_by_screened_rank``): the ids that
rank ``<= K`` by market cap among the names passing the universe's price / ADV
screens on at least one date.  It is the right prefilter for a universe that does
not start at rank 1 (``universe.rank_offset``): a peak-cap cut can miss a name that
was a mid cap in 2003 and never grew, which a rank cut cannot.  It cannot be
combined with ``max_names`` / ``min_mcap_ever``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from qts.core.types import MarketData

Printer = Callable[[str], None]


def _quiet(_msg: str) -> None:
    return None


@dataclass(frozen=True)
class StoreSpec:
    """Which ids and dates to load from a ParquetStore (the ``qts run`` store flags)."""

    store: str
    start: str | None = None
    end: str | None = None
    max_names: int | None = None
    min_mcap_ever: float | None = None
    #: screened-rank prefilter (module docstring); the screens are the universe's
    #: ``min_price`` / ``min_adv_usd`` / ``adv_window`` (defaults: ``UniverseConfig()``)
    max_rank: int | None = None
    min_price: float = 5.0
    min_adv_usd: float = 2e6
    adv_window: int = 63

    def __post_init__(self) -> None:
        if self.max_rank is not None and (self.max_names is not None or self.min_mcap_ever is not None):
            raise ValueError("--max-rank cannot be combined with --max-names / --min-mcap-ever")

    def with_screens(self, universe: Any) -> StoreSpec:
        """The spec with the screened-rank prefilter's screens taken from a
        ``UniverseConfig`` (``min_price``, ``min_adv_usd``, ``adv_window``)."""
        from dataclasses import replace

        return replace(
            self,
            min_price=float(universe.min_price),
            min_adv_usd=float(universe.min_adv_usd),
            adv_window=int(universe.adv_window),
        )

    @classmethod
    def from_args(cls, a: Any) -> StoreSpec:
        """From an ``argparse`` namespace with ``store``, ``start``, ``end``,
        ``max_names``, ``min_mcap_ever`` (missing attributes -> ``None``)."""
        return cls(
            store=str(a.store),
            start=getattr(a, "start", None),
            end=getattr(a, "end", None),
            max_names=getattr(a, "max_names", None),
            min_mcap_ever=getattr(a, "min_mcap_ever", None),
            max_rank=getattr(a, "max_rank", None),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StoreSpec:
        keys = (
            "store",
            "start",
            "end",
            "max_names",
            "min_mcap_ever",
            "max_rank",
            "min_price",
            "min_adv_usd",
            "adv_window",
        )
        return cls(**{k: d[k] for k in keys if k in d})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cli_flags(self) -> str:
        """The flags as they are typed on the command line (for printed commands)."""
        parts = [f"--store {self.store}"]
        if self.start:
            parts.append(f"--start {self.start}")
        if self.end:
            parts.append(f"--end {self.end}")
        if self.max_names is not None:
            parts.append(f"--max-names {int(self.max_names)}")
        if self.min_mcap_ever is not None:
            parts.append(f"--min-mcap-ever {self.min_mcap_ever:g}")
        if self.max_rank is not None:
            parts.append(f"--max-rank {int(self.max_rank)}")
        return " ".join(parts)


class StoreLoadError(RuntimeError):
    """The store or its prefilter produced no data."""


def prefilter_ids(spec: StoreSpec, end: str | None = None) -> list[str] | None:
    """Ids the peak-market-cap prefilter keeps over ``[spec.start, end or spec.end]``;
    ``None`` when the spec has no prefilter (every id is loaded)."""
    from qts.data.store import ParquetStore

    store = ParquetStore(spec.store)
    if spec.max_rank is not None:
        return store.tickers_by_screened_rank(
            int(spec.max_rank),
            spec.start,
            end if end is not None else spec.end,
            min_price=spec.min_price,
            min_adv_usd=spec.min_adv_usd,
            adv_window=spec.adv_window,
        )
    if spec.min_mcap_ever is None and spec.max_names is None:
        return None
    return store.tickers_by_market_cap(
        spec.start,
        end if end is not None else spec.end,
        min_mcap_ever=spec.min_mcap_ever,
        max_names=spec.max_names,
    )


def load_store_market(spec: StoreSpec, end: str | None = None, log: Printer = _quiet) -> MarketData:
    """``MarketData`` for ``spec`` (exactly as ``qts run --store`` loads it); ``end``
    overrides ``spec.end`` for both the prefilter and the load.  Raises
    :class:`StoreLoadError` when the prefilter leaves no id."""
    from qts.data.store import ParquetStore

    e = end if end is not None else spec.end
    tickers = prefilter_ids(spec, end=e)
    if tickers is not None:
        how = (
            f"screened market-cap rank <= {spec.max_rank}" if spec.max_rank is not None else "peak market cap"
        )
        log(f"prefilter: {len(tickers)} candidate ids by {how}")
        if not tickers:
            raise StoreLoadError("the prefilter left no ids")
    md = ParquetStore(spec.store).as_market_data(universe_tickers=tickers, start=spec.start, end=e)
    if len(md.dates):
        log(
            f"store: {md.close.shape[1]} ids x {md.close.shape[0]} days "
            f"({md.dates[0].date()}..{md.dates[-1].date()})"
        )
    return md


# ----------------------------------------------------------------------------- the seal
def _store_fingerprint(spec: StoreSpec, datasets: tuple[str, ...]) -> list[list[Any]]:
    """Cheap identity of the stored files a market hash depends on (relative path,
    size, mtime in ns): a rewrite of any of them invalidates a cached hash."""
    import os

    from qts.data.store import ParquetStore

    st = ParquetStore(spec.store)
    out: list[list[Any]] = []
    for name in datasets:
        if not st.exists(name):
            continue
        for f in st._files(name):
            s = os.stat(f)
            out.append([os.path.relpath(f, st.root), s.st_size, s.st_mtime_ns])
    return out


def full_history_market_hash(spec: StoreSpec, cache_dir: str | None = None, log: Printer = _quiet) -> str:
    """:func:`qts.core.manifest.hash_market` of the full-history load with ``spec``'s flags
    (prefilter over ``[start, end]``, close panel ``[start, end]``) -- what ``qts run``
    with the same flags and no seal hashes -- computed from ``prices_close`` alone (no
    other panel is read).  Cached as JSON under ``cache_dir`` keyed on the flags and the
    files' sizes / modification times, so a second sealed run does not read the store
    again."""
    import hashlib
    import json
    from pathlib import Path

    from qts.core.manifest import hash_close
    from qts.data.store import ParquetStore

    deps = ("prices_close", "market_cap", "prices_close_raw", "prices_volume")
    key_src = json.dumps({"spec": spec.to_dict(), "files": _store_fingerprint(spec, deps)}, sort_keys=True)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:24]
    path = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if path is not None and path.exists():
        try:
            return str(json.loads(path.read_text())["market_hash"])
        except (OSError, ValueError, KeyError):
            pass
    log("seal: hashing the full-history close panel of this load (once per store snapshot and flags) ...")
    ids = prefilter_ids(spec)
    close = ParquetStore(spec.store).read_wide("prices_close", spec.start, spec.end, ids)
    mh = hash_close(close)
    if path is not None:
        from qts.research.common import write_json

        write_json(path, {"market_hash": mh, "spec": spec.to_dict(), "n_ids": int(close.shape[1])})
    return mh


@dataclass
class ResearchLoad:
    """A store load under the research seal (:func:`load_research_market`).

    ``sealed``: the cut was applied (``md`` ends before ``seal.holdout_start``);
    ``unsealed``: a seal is in force but the full history was loaded (``--unseal`` or
    an exempt command); ``market_hash_full``: the full-history identity of a sealed
    load (its ledger family; ``""`` otherwise)."""

    md: MarketData
    seal: Any = None
    sealed: bool = False
    unsealed: bool = False
    exempt: str = ""
    market_hash_full: str = ""
    spec: StoreSpec | None = None

    def context(self, phase: int, **kw: Any) -> Any:
        """The :class:`~qts.runs.context.RunContext` of a pipeline run on this load:
        seal fields, and for a sealed load the full-history family
        ``pipeline_phase{phase}:{market_hash_full[:12]}``.  ``kw`` adds / overrides
        context fields (``reference``, ``record_kind``, ``family``, ...)."""
        from qts.runs.context import RunContext
        from qts.validation.ledger import research_family

        d: dict[str, Any] = {}
        if self.seal is not None:
            d["holdout_start"] = self.seal.holdout_start_iso
            d["unsealed"] = bool(self.unsealed)
            d["seal_exempt"] = self.exempt
            if self.sealed:
                d["sealed_end"] = str(self.md.dates[-1].date()) if len(self.md.dates) else self.seal.load_end_iso
                d["market_hash_full"] = self.market_hash_full
                d["family"] = research_family(int(phase), self.market_hash_full)
        d.update(kw)
        return RunContext(**d)

    def family_full(self, phase: int) -> str | None:
        from qts.validation.ledger import research_family

        return research_family(int(phase), self.market_hash_full) if self.sealed else None


def load_research_market(
    spec: StoreSpec,
    *,
    seal: Any = "default",
    unseal: bool = False,
    exempt: str = "",
    hash_cache_dir: str | None = "runs/_cache/market_hash",
    log: Printer = _quiet,
) -> ResearchLoad:
    """Load ``spec`` under the research seal (:mod:`qts.research.seal`).

    With a seal in force (``seal="default"``: :func:`qts.research.seal.load_seal`;
    ``None``: no seal) and neither ``unseal`` nor ``exempt``: the prefilter runs over
    ``[start, load_end]`` and every panel is loaded through ``load_end`` (the day
    before ``holdout_start``; an explicit ``spec.end`` earlier than that is kept), then
    :func:`qts.research.seal.apply_seal` asserts nothing at or after the holdout
    survived; the full-history market hash of the same flags is computed for the
    ledger family.  Otherwise the full history is loaded (``unsealed`` set when a seal
    is in force)."""
    from qts.research.seal import apply_seal, load_seal

    s = load_seal() if seal == "default" else seal
    if s is None:
        return ResearchLoad(md=load_store_market(spec, log=log), spec=spec)
    if unseal or exempt:
        md = load_store_market(spec, log=log)
        return ResearchLoad(md=md, seal=s, unsealed=True, exempt=exempt, spec=spec)
    end = s.load_end_iso
    if spec.end is not None and pd.Timestamp(spec.end) < s.load_end:
        end = str(pd.Timestamp(spec.end).date())
    log(f"seal: holdout from {s.holdout_start_iso} sealed; loading data through {end} ({s.path})")
    md = apply_seal(load_store_market(spec, end=end, log=log), s)
    mh_full = full_history_market_hash(spec, cache_dir=hash_cache_dir, log=log)
    return ResearchLoad(md=md, seal=s, sealed=True, market_hash_full=mh_full, spec=spec)


#: Peak RSS of ``as_market_data`` + the universe mask, measured with
#: ``scripts/bench_store_load.py`` (docs/notes/research_tools_2026-11-13.md "Mid-cap load"):
#: about ``LOAD_FIXED_GB + LOAD_BYTES_PER_CELL x ids x days``.
LOAD_FIXED_GB = 0.3
LOAD_BYTES_PER_CELL = 200.0


def estimated_load_gb(n_ids: int, n_days: int) -> float:
    """Rough peak memory (GB) of loading ``n_ids`` x ``n_days`` (before any fit)."""
    return LOAD_FIXED_GB + LOAD_BYTES_PER_CELL * float(n_ids) * float(n_days) / 2**30


__all__ = [
    "Printer",
    "ResearchLoad",
    "StoreLoadError",
    "StoreSpec",
    "estimated_load_gb",
    "full_history_market_hash",
    "load_research_market",
    "load_store_market",
    "prefilter_ids",
]
