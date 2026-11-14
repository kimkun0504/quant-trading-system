"""Serving the frozen models: walk-forward fold models persisted per refit, and
day-``s`` signals (alpha + risk) built from data through ``s`` only.

Why this is cheap enough to run daily
-------------------------------------
The batch pipeline fits the risk model (~3 min on the real store) and the alpha
model (~21 min) in one go.  Almost all of the alpha time is the walk-forward ML
sleeve: one LightGBM + MLP + ridge ensemble per refit (``alpha.refit_frequency_days``
= 252 trading days from a fixed anchor, the first feature date).  The model of the
refit that predicts from position ``p0`` is trained only on label rows realised
strictly before ``p0`` (:mod:`qts.alpha.trainer`: embargo ``h + L``, rows decided
before ``p0``), so it is the *same model* whether the data end a day or a year after
``p0``.  :class:`FrozenFoldStore` persists every fold model the first time it is
trained (``models/fold_<train_end>.pkl`` + a JSON sidecar) and serves it on every
later fit: :func:`fit_signals` runs the ordinary
:class:`~qts.pipeline.ResearchPipeline` fit on the data through ``s`` with an
``ml_factory`` whose models look themselves up in the store (:class:`FoldModelProxy`;
pure composition through the pipeline's public ``ml_factory`` argument, no change
to the pipeline or the trainer).  Everything else -- features through ``s``, labels,
composite, the traded score's neutralisation and smoothing, the trailing IC, the
half-life, Grinold scaling, the risk model -- is recomputed by the unchanged batch
code on the data through ``s``.

* A **normal day** costs one risk fit + the non-ML part of the alpha fit + ML
  predictions with the stored models.  Hold days (``rebalance_every_days`` = 5) need
  no fit at all (:mod:`qts.paper.engine`).
* A **refit day** (the data end crosses the next scheduled refit position) trains
  exactly one new fold -- the batch's own fold for that refit -- and stores it.
* The **first fit** of a book trains every historical fold once (the batch alpha
  fit's ~21 minutes) and stores them.

Freezing rule.  A stored fold is served as long as the book's frozen config text
and the feature columns match; a fold whose training rows have changed since it was
stored (vendor restatement) is still served -- that is what "frozen" means for a live
book -- and the fit records a warning (``FoldModelProxy`` compares row count and two
moments of the label against the sidecar).

Point in time.  ``fit_signals(md, ...)`` sees only the data it is given; the paper
engine always passes ``md`` cut at the signal date.  ``tests/test_paper.py`` checks
that day-``s`` outputs equal the batch fit's ``as_of(s)`` and do not move when data
after ``s`` is perturbed.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qts.core.config import SystemConfig
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput
from qts.paper.book import PaperError, atomic_write_bytes, atomic_write_json, iso, utc_now

log = logging.getLogger(__name__)

#: relative tolerance of the training-data fingerprint (label moments); a mismatch
#: means the fold's training rows were restated after the fold was frozen
FINGERPRINT_RTOL = 1e-6


# --------------------------------------------------------------------------- fold store
class FrozenFoldStore:
    """Fitted walk-forward fold models, one file per refit (key: the fold's last
    training date).  ``identity`` is what must match for a stored fold to be served
    (the book's frozen config text hash and the inner factory); it is written into
    every sidecar."""

    def __init__(self, root: str | Path, identity: dict[str, Any]):
        self.root = Path(root)
        self.identity = {str(k): v for k, v in identity.items()}
        self.loaded: list[str] = []
        self.trained: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    def reset_stats(self) -> None:
        self.loaded, self.trained, self.warnings = [], [], []

    def _paths(self, train_end: str) -> tuple[Path, Path]:
        return self.root / f"fold_{train_end}.pkl", self.root / f"fold_{train_end}.json"

    def keys(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem.removeprefix("fold_") for p in self.root.glob("fold_*.pkl"))

    def sidecars(self) -> list[dict[str, Any]]:
        out = []
        for k in self.keys():
            _, side = self._paths(k)
            try:
                out.append(json.loads(side.read_text()))
            except (OSError, ValueError):
                continue
        return out

    def load(self, train_end: str, columns: list[str]) -> tuple[Any, dict[str, Any]] | None:
        pkl, side = self._paths(train_end)
        if not (pkl.exists() and side.exists()):
            return None
        meta = json.loads(side.read_text())
        ident = meta.get("identity", {})
        bad = {k: (ident.get(k), v) for k, v in self.identity.items() if ident.get(k) != v}
        if bad:
            raise PaperError(
                f"frozen fold model {pkl.name} was trained under a different book identity {bad}; "
                "the book's models are frozen -- start a new book instead of mixing models"
            )
        if list(meta.get("columns", [])) != list(columns):
            raise PaperError(
                f"frozen fold model {pkl.name} was trained on features {meta.get('columns')} but the "
                f"current feature set is {list(columns)} (the feature code changed since the book was "
                "initialised); the frozen model cannot score these inputs -- start a new book"
            )
        with open(pkl, "rb") as fh:
            model = pickle.load(fh)  # trusted local file written by this class
        return model, meta

    def save(self, train_end: str, model: Any, meta: dict[str, Any]) -> None:
        pkl, side = self._paths(train_end)
        atomic_write_bytes(pkl, pickle.dumps(model, protocol=5))
        atomic_write_json(side, {**meta, "identity": self.identity})

    def factory(self, inner: Callable[[], object]) -> FrozenFactory:
        return FrozenFactory(self, inner)


def _fingerprint(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    v = y.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return {
        "n_rows": len(X),
        "n_labels": len(v),
        "label_sum": float(v.sum()),
        "label_sumsq": float((v * v).sum()),
    }


def _fingerprint_differs(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if int(a.get("n_rows", -1)) != int(b.get("n_rows", -2)):
        return True
    if int(a.get("n_labels", -1)) != int(b.get("n_labels", -2)):
        return True
    for k in ("label_sum", "label_sumsq"):
        x, y = float(a.get(k, np.nan)), float(b.get(k, np.nan))
        if not np.isclose(x, y, rtol=FINGERPRINT_RTOL, atol=1e-9):
            return True
    return False


class FrozenFactory:
    """``ml_factory`` for :class:`~qts.alpha.AlphaModel`: every call returns a
    :class:`FoldModelProxy` bound to the store (the trainer calls it once per fold)."""

    def __init__(self, store: FrozenFoldStore, inner: Callable[[], object]):
        self.store = store
        self.inner = inner

    def __call__(self) -> FoldModelProxy:
        return FoldModelProxy(self.store, self.inner)


class FoldModelProxy:
    """A fold model that is loaded from the store when the fold (identified by its
    last training date) was trained before, and trained + stored otherwise.  Every
    attribute the alpha layer reads after the fit (``models``, ``weights``,
    ``feature_importance``, ``feature_names_``, ...) is delegated to the real model."""

    def __init__(self, store: FrozenFoldStore, inner: Callable[[], object]):
        self._store = store
        self._inner = inner
        self._model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series, valid_mask: Any = None, sample_weight: Any = None):
        dates = pd.DatetimeIndex(X.index.get_level_values("date"))
        train_end = iso(dates.max())
        columns = [str(c) for c in X.columns]
        fp = _fingerprint(X, y)
        hit = self._store.load(train_end, columns)
        if hit is not None:
            self._model, meta = hit
            self._store.loaded.append(train_end)
            if _fingerprint_differs(meta.get("fingerprint", {}), fp):
                self._store.warnings.append(
                    f"fold {train_end}: training rows differ from when the model was frozen "
                    f"({meta.get('fingerprint', {}).get('n_rows')} -> {fp['n_rows']} rows; vendor "
                    "restatement?); the frozen model is served unchanged"
                )
            return self
        t0 = time.perf_counter()
        model = self._inner()
        if sample_weight is None:
            model.fit(X, y, valid_mask)  # type: ignore[attr-defined]
        else:
            model.fit(X, y, valid_mask, sample_weight=sample_weight)  # type: ignore[attr-defined]
        sec = time.perf_counter() - t0
        vm = None if valid_mask is None else pd.Series(valid_mask).reindex(X.index).fillna(False)
        valid_start = None
        if vm is not None and bool(vm.any()):
            valid_start = iso(dates[vm.to_numpy(dtype=bool)].min())
        meta = {
            "train_end": train_end,
            "train_start": iso(dates.min()),
            "valid_start": valid_start,
            "columns": columns,
            "fingerprint": fp,
            "weighted": sample_weight is not None,
            "fit_seconds": sec,
            "trained_at": utc_now(),
            "model_class": type(model).__name__,
        }
        self._store.save(train_end, model, meta)
        self._store.trained.append({"train_end": train_end, "seconds": sec, "n_rows": fp["n_rows"]})
        log.warning(
            "walk-forward refit: trained and froze the fold model with training data through %s "
            "(%d rows, %.0fs)",
            train_end,
            fp["n_rows"],
            sec,
        )
        self._model = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("FoldModelProxy.predict before fit")
        return self._model.predict(X)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        model = self.__dict__.get("_model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    def __getstate__(self) -> dict[str, Any]:  # pickle the real model only
        return {"_store": None, "_inner": None, "_model": self._model}

    def __setstate__(self, st: dict[str, Any]) -> None:
        self.__dict__.update(st)


# --------------------------------------------------------------------------- signals
@dataclass
class SignalFit:
    """The fitted pipeline on data through ``asof`` and the day-``s`` signal
    functions the engine needs (``s <= asof``)."""

    asof: pd.Timestamp
    pipe: Any  #: qts.pipeline.ResearchPipeline (fitted)
    seconds: float
    trained: list[dict[str, Any]] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def md_u(self) -> MarketData:
        return self.pipe.md_u

    @property
    def cfg(self) -> SystemConfig:
        return self.pipe.cfg

    def alpha_as_of(self, d: pd.Timestamp) -> AlphaOutput:
        """``AlphaModel.as_of(d)`` restricted to names eligible at ``d`` (the
        pipeline's ``alpha_fn`` wrapper, :meth:`ResearchPipeline._eligible_alpha_fn`)."""
        out = self.pipe.alpha.as_of(d)
        elig = self.pipe.md_u.eligibility()
        ok = elig.loc[out.date]
        keep = out.score.index[ok.reindex(out.score.index).fillna(False).astype(bool).to_numpy()]
        if len(keep) == len(out.score):
            return out
        return dataclasses.replace(
            out,
            score=out.score.loc[keep],
            expected_return=out.expected_return.loc[keep],
            components={k: v.reindex(keep) for k, v in out.components.items()},
        )

    def risk_as_of(self, d: pd.Timestamp, alpha: pd.Series | None) -> RiskModelOutput:
        """``risk.as_of(d, alpha)`` exactly as the pipeline's ``risk_fn``."""
        return self.pipe.risk.as_of(d, alpha=alpha if self.cfg.risk.alignment else None)

    def folds(self) -> list[dict[str, Any]]:
        alpha = getattr(self.pipe, "alpha", None)
        tr = getattr(alpha, "trainer", None)
        out = []
        for f in getattr(tr, "folds_", []) or []:
            out.append(
                {
                    "refit_date": iso(f.pred_start),
                    "pred_end": iso(f.pred_end),
                    "train_start": iso(f.train_start),
                    "train_end": iso(f.train_end),
                }
            )
        return out

    def fold_for(self, d: pd.Timestamp) -> dict[str, Any] | None:
        best = None
        for f in self.folds():
            if pd.Timestamp(f["refit_date"]) <= d:
                best = f
        return best

    def realised_ic(self, start: pd.Timestamp) -> pd.Series:
        """Daily OOS rank IC of the traded score on dates ``>= start`` whose labels
        are fully realised by ``asof`` (``AlphaModel.oos_ic_`` and ``label_ready_``):
        the live book's realised IC, lagged by the label horizon."""
        a = self.pipe.alpha
        ic = getattr(a, "oos_ic_", None)
        ready = getattr(a, "label_ready_", None)
        if ic is None or ready is None:
            return pd.Series(dtype=float)
        ic = ic[ic.index >= start].dropna()
        rd = pd.to_datetime(ready.reindex(ic.index))
        ok = (rd.notna() & (rd <= self.asof)).to_numpy()
        return ic[ok]


def effective_config(cfg: SystemConfig, md: MarketData) -> SystemConfig:
    """The config the batch command line would run: ``qts run`` caps
    ``universe.top_n_by_mcap`` at the number of loaded names, and the stage cache is
    off (every paper fit is on a new data snapshot)."""
    n = int(md.close.shape[1])
    return cfg.with_overrides(
        {"universe.top_n_by_mcap": min(int(cfg.universe.top_n_by_mcap), n), "cache.enabled": False}
    )


def fit_signals(
    md: MarketData,
    cfg: SystemConfig,
    *,
    use_ml: bool,
    store: FrozenFoldStore | None,
    inner_factory: Callable[[], object] | None,
    nav0: float,
) -> SignalFit:
    """Fit the frozen strategy on ``md`` (data through the signal date) with the
    fold models served from ``store`` (see the module docstring)."""
    from qts.alpha.alpha_model import default_ml_factory
    from qts.pipeline import ResearchPipeline

    t0 = time.perf_counter()
    inner = inner_factory or default_ml_factory(cfg.alpha)
    factory: Callable[[], object] | None = None
    if use_ml:
        factory = store.factory(inner) if store is not None else inner
    if store is not None:
        store.reset_stats()
    pipe = ResearchPipeline(cfg, use_ml=use_ml, ml_factory=factory, nav0=nav0)
    pipe.fit(md)
    sec = time.perf_counter() - t0
    return SignalFit(
        asof=pd.Timestamp(md.dates[-1]),
        pipe=pipe,
        seconds=sec,
        trained=list(store.trained) if store is not None else [],
        loaded=list(store.loaded) if store is not None else [],
        warnings=list(store.warnings) if store is not None else [],
    )


class SignalProvider:
    """Signal fits for the engine.  ``loader(s)`` returns the market as the job of day
    ``s`` sees it (data through ``s``).  ``mode="daily"`` (default) fits on
    ``loader(s)`` for each signal date -- exactly what a job running that evening
    could see.  ``mode="single"`` fits once on ``loader(end)`` and answers every
    earlier signal date from that fit's ``as_of`` (relies on the fits being point in
    time, which ``tests/test_paper.py`` checks; a fast catch-up over many days)."""

    def __init__(
        self,
        loader: Callable[[pd.Timestamp], MarketData],
        cfg: SystemConfig,
        *,
        use_ml: bool,
        store: FrozenFoldStore | None,
        inner_factory: Callable[[], object] | None,
        nav0: float,
        mode: str = "daily",
        end: pd.Timestamp | None = None,
    ):
        if mode not in ("daily", "single"):
            raise PaperError(f"unknown signal mode {mode!r} (daily | single)")
        if mode == "single" and end is None:
            raise PaperError("signal mode 'single' needs the last date to fit on")
        self.loader = loader
        self.cfg = cfg
        self.use_ml = use_ml
        self.store = store
        self.inner_factory = inner_factory
        self.nav0 = nav0
        self.mode = mode
        self.end = None if end is None else pd.Timestamp(end)
        self._cache: SignalFit | None = None
        self.fits: list[dict[str, Any]] = []  # one entry per fit performed (for the journal)

    @property
    def latest(self) -> SignalFit | None:
        return self._cache

    def fit_for(self, s: pd.Timestamp) -> SignalFit:
        s = pd.Timestamp(s)
        if self.mode == "single":
            assert self.end is not None
            if self._cache is None:
                self._cache = self._fit(self.loader(self.end))
            if s > self._cache.asof:
                raise PaperError(f"signal date {iso(s)} is after the loaded data ({iso(self._cache.asof)})")
            return self._cache
        if self._cache is not None and self._cache.asof == s:
            return self._cache
        md = self.loader(s)
        if len(md.dates) == 0 or md.dates[-1] != s:
            raise PaperError(f"no data on the signal date {iso(s)}")
        self._cache = self._fit(md)
        return self._cache

    def _fit(self, md: MarketData) -> SignalFit:
        fit = fit_signals(
            md,
            self.cfg,
            use_ml=self.use_ml,
            store=self.store,
            inner_factory=self.inner_factory,
            nav0=self.nav0,
        )
        self.fits.append(
            {
                "asof": iso(fit.asof),
                "seconds": round(fit.seconds, 2),
                "stages": {
                    k: round(float(v), 2) for k, v in (getattr(fit.pipe, "timings_", {}) or {}).items()
                },
                "trained": fit.trained,
                "n_loaded": len(fit.loaded),
                "warnings": fit.warnings,
            }
        )
        return fit
