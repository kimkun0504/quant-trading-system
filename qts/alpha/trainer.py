"""Walk-forward training (BUILD_PLAN §2.3 "Training schedule", §2.4 Gate 5).

Protocol
--------
* Expanding window, refit every ``cfg.refit_frequency_days`` trading days,
  first fit once ``cfg.train_min_days`` usable training dates exist.
* **Embargo / purge** (exact): a label at date ``s`` is realised over
  ``(s, s+h]`` (``h`` = embargo = ``cfg.embargo_days or cfg.horizon_days``).
  For a prediction block starting at ``t``, only labels with
  ``pos(s) + h < pos(t)`` are allowed, i.e. the label's last return day is
  strictly before the first prediction date.  In index positions this is
  ``pos(s) <= pos(t) - h - 1``.
* Inside the training window the trailing 12 months (252 dates) are the
  validation block used for early stopping / hyper-parameter choice.  The
  same purge is applied between the fit rows and the validation rows.
* Models are re-created from ``model_factory()`` at every refit (no parameter
  carry-over between windows).
* Only out-of-sample predictions are returned; dates before the first fit are
  NaN.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from qts.core.config import AlphaConfig
from qts.core.types import Long, Wide

from .features import feature_signature

log = logging.getLogger(__name__)


@dataclass
class FoldInfo:
    pred_start: pd.Timestamp
    pred_end: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: Optional[pd.Timestamp]
    n_train_rows: int
    n_valid_rows: int
    extra: Dict = field(default_factory=dict)


def embargoed_train_mask(label_pos: np.ndarray, pred_start_pos: int, horizon: int) -> np.ndarray:
    """Boolean mask over label positions that may be used to train a model that
    predicts from ``pred_start_pos`` onward: ``pos(s) + h < pred_start_pos``."""
    return label_pos + horizon < pred_start_pos


class WalkForwardTrainer:
    def __init__(self, cfg: AlphaConfig, valid_days: int = 252, verbose: bool = False):
        self.cfg = cfg
        self.valid_days = valid_days
        self.verbose = verbose
        self.folds_: List[FoldInfo] = []
        self.models_: List = []

    @property
    def embargo(self) -> int:
        return int(self.cfg.embargo_days or self.cfg.horizon_days)

    def schedule(self, dates: pd.DatetimeIndex) -> List[int]:
        """Positions (into ``dates``) at which a refit happens."""
        h = self.embargo
        first = self.cfg.train_min_days + h  # pos(t) - h - 1 >= train_min_days - 1
        if first >= len(dates):
            return []
        return list(range(first, len(dates), self.cfg.refit_frequency_days))

    def run(self, X: Long, y: pd.Series, model_factory: Callable[[], object],
            dates: Optional[pd.DatetimeIndex] = None) -> Wide:
        """Return a wide (date x ticker) panel of strictly out-of-sample predictions.

        Rows with any NaN feature are dropped up front (``FeatureBuilder.process``
        emits eligible rows only, so this is a guard against hand-built frames);
        rows with a NaN label are excluded from training but still predicted.
        """
        y = y.reindex(X.index)
        finite = X.notna().all(axis=1).to_numpy()
        if not finite.all():
            X, y = X.loc[finite], y.loc[finite]
        all_dates = X.index.get_level_values("date")
        uniq = pd.DatetimeIndex(all_dates.unique()).sort_values()
        if dates is not None:
            dates = pd.DatetimeIndex(dates)
        pos_of = pd.Series(np.arange(len(uniq)), index=uniq)
        row_pos = pos_of.reindex(all_dates).to_numpy()
        h = self.embargo
        preds: List[pd.Series] = []
        self.folds_, self.models_ = [], []
        sched = self.schedule(uniq)
        for k, p0 in enumerate(sched):
            p1 = sched[k + 1] if k + 1 < len(sched) else len(uniq)
            train_rows = embargoed_train_mask(row_pos, p0, h) & y.notna().to_numpy()
            if train_rows.sum() == 0:
                continue
            train_pos = row_pos[train_rows]
            last_train_pos = train_pos.max()
            valid_start_pos = last_train_pos - self.valid_days + 1
            # fit rows must also be purged w.r.t. the validation block
            valid_mask_arr = train_rows & (row_pos >= valid_start_pos)
            fit_ok = train_rows & embargoed_train_mask(row_pos, valid_start_pos, h)
            use = valid_mask_arr | fit_ok
            if valid_mask_arr.sum() == 0 or fit_ok.sum() == 0:
                use, valid_mask_arr = train_rows, np.zeros(len(X), bool)
            Xtr, ytr = X[use], y[use]
            vmask = pd.Series(valid_mask_arr[use], index=Xtr.index)
            model = model_factory()
            model.fit(Xtr, ytr, vmask)
            self.models_.append(model)
            pred_rows = (row_pos >= p0) & (row_pos < p1)
            if dates is not None:
                pred_rows &= all_dates.isin(dates)
            if pred_rows.any():
                preds.append(model.predict(X[pred_rows]))
            self.folds_.append(FoldInfo(
                pred_start=uniq[p0], pred_end=uniq[p1 - 1], train_start=uniq[train_pos.min()],
                train_end=uniq[last_train_pos],
                valid_start=uniq[valid_start_pos] if valid_mask_arr.any() else None,
                n_train_rows=int(fit_ok.sum() if valid_mask_arr.any() else train_rows.sum()),
                n_valid_rows=int(valid_mask_arr.sum()),
                extra={"feature_hash": feature_signature(X.columns)}))
            f = self.folds_[-1]
            log.log(logging.INFO if self.verbose else logging.DEBUG,
                    "fold %d: train %s..%s (valid from %s) -> predict %s..%s", k, f.train_start.date(),
                    f.train_end.date(), f.valid_start, f.pred_start.date(), f.pred_end.date())
        if not preds:
            return pd.DataFrame(np.nan, index=uniq, columns=X.index.get_level_values("ticker").unique())
        long = pd.concat(preds).sort_index()
        wide = long.unstack("ticker").reindex(uniq)
        wide.index.name, wide.columns.name = "date", "ticker"
        return wide


def train_live_parity_check(model, X_date: pd.DataFrame) -> str:
    """Assert that the live feature frame has exactly the columns/order the model
    was trained on (process parity, Numerai 2025 lesson).  Returns the hash."""
    live = feature_signature(X_date.columns)
    trained = getattr(model, "feature_hash_", None)
    if trained is None:
        names = getattr(model, "feature_names_", None)
        trained = None if names is None else feature_signature(names)
    if trained is None:
        raise AssertionError("model has no recorded feature signature; was it fit?")
    if live != trained:
        raise AssertionError(f"train/live feature mismatch: live={live} trained={trained}")
    return live
