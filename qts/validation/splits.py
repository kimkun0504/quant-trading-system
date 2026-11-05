"""Purged / embargoed cross-validation splitters for overlapping-label time series.

All splitters operate on a ``pd.DatetimeIndex`` of *label dates* (the date a
prediction is made) and a ``label_horizon`` (how many trading days the label
looks ahead).  A train observation with label date ``s`` "uses" the forward
window ``[s, s + h]`` (in trading-day positions).  It is **purged** if that
window overlaps the test window ``[t0, t1 + h]`` and **embargoed** if it falls
within ``embargo_days`` after the end of the test window's forward reach.

Conventions
-----------
* ``split(dates)`` yields ``(train_idx, test_idx)`` as integer position arrays
  into ``dates``.  ``dates`` may also be any sized object; only ``len`` is used
  for the position arithmetic, so the splitters are scikit-learn compatible
  (``split(X, y=None, groups=None)``).
* Positions are trading-day positions; "days" are therefore trading days.

Classes
-------
``PurgedWalkForward``     expanding train, contiguous forward test blocks.
``PurgedKFold``           contiguous K blocks, train = everything else purged.
``CombinatorialPurgedCV`` all C(n_groups, k_test) combinations of test groups
                          (López de Prado, *Advances in Financial ML*, ch. 12),
                          with ``backtest_paths()`` to assemble the
                          ``n_paths = C(N,k) * k / N`` out-of-sample paths.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

IdxPair = tuple[np.ndarray, np.ndarray]


def _n(dates) -> int:
    return len(dates)


def purge_train(train_pos: np.ndarray, test_blocks: Sequence[tuple[int, int]],
                label_horizon: int, embargo_days: int, n: int) -> np.ndarray:
    """Remove training positions whose label window overlaps any test block.

    ``test_blocks`` is a list of inclusive ``(start, end)`` position ranges.
    A train position ``s`` is dropped if ``[s, s+h]`` intersects
    ``[t0, t1 + h]`` (overlap on either side), or if ``s`` lies in the embargo
    window ``(t1 + h, t1 + h + embargo]``.
    """
    h = int(label_horizon)
    keep = np.ones(len(train_pos), dtype=bool)
    for t0, t1 in test_blocks:
        lo = t0 - h                 # any s with s + h >= t0 overlaps
        hi = t1 + h + embargo_days  # test forward reach + embargo
        keep &= ~((train_pos >= lo) & (train_pos <= hi))
    return train_pos[keep]


def label_windows_overlap(train_pos: np.ndarray, test_pos: np.ndarray, label_horizon: int) -> bool:
    """True if any train label window [s, s+h] intersects any test window [t, t+h]."""
    if len(train_pos) == 0 or len(test_pos) == 0:
        return False
    h = int(label_horizon)
    test_pos = np.sort(np.asarray(test_pos))
    # test sets may be non-contiguous (CPCV): check each contiguous run
    for run in np.split(test_pos, np.where(np.diff(test_pos) > 1)[0] + 1):
        t_lo, t_hi = run.min(), run.max() + h
        # train windows: [s, s+h].  Overlap iff s <= t_hi and s + h >= t_lo
        if np.any((train_pos <= t_hi) & (train_pos + h >= t_lo)):
            return True
    return False


@dataclass
class PurgedWalkForward:
    """Expanding-window walk-forward with purge and embargo.

    The test region (positions after ``min_train_days``) is cut into
    ``n_splits`` contiguous blocks.  For block ``i`` the train set is every
    position *before* the block, purged of label windows that reach into it.
    Nothing after the test block is ever in train (strict forward-only).
    """

    n_splits: int = 6
    embargo_days: int = 20
    label_horizon: int = 20
    min_train_days: int = 252

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def test_blocks(self, dates) -> list[tuple[int, int]]:
        n = _n(dates)
        start = int(self.min_train_days)
        if n - start < self.n_splits:
            raise ValueError("not enough observations for the requested splits")
        edges = np.linspace(start, n, self.n_splits + 1).astype(int)
        return [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(self.n_splits)]

    def split(self, dates, y=None, groups=None) -> Iterator[IdxPair]:
        n = _n(dates)
        for t0, t1 in self.test_blocks(dates):
            test = np.arange(t0, t1 + 1)
            train = np.arange(0, t0)
            train = purge_train(train, [(t0, t1)], self.label_horizon, self.embargo_days, n)
            yield train, test

    @property
    def n_paths(self) -> int:
        return 1


@dataclass
class PurgedKFold:
    """K contiguous folds; train = all other folds, purged and embargoed on both sides."""

    n_splits: int = 5
    embargo_days: int = 20
    label_horizon: int = 20

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def test_blocks(self, dates) -> list[tuple[int, int]]:
        n = _n(dates)
        edges = np.linspace(0, n, self.n_splits + 1).astype(int)
        return [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(self.n_splits)]

    def split(self, dates, y=None, groups=None) -> Iterator[IdxPair]:
        n = _n(dates)
        for t0, t1 in self.test_blocks(dates):
            test = np.arange(t0, t1 + 1)
            train = np.setdiff1d(np.arange(n), test)
            train = purge_train(train, [(t0, t1)], self.label_horizon, self.embargo_days, n)
            yield train, test

    @property
    def n_paths(self) -> int:
        return 1


@dataclass
class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation (AFML ch. 12).

    The sample is cut into ``n_groups`` contiguous groups; every combination of
    ``k_test`` groups is a test set (the remaining groups, purged and
    embargoed around each test group, form train).  There are
    ``C(n_groups, k_test)`` splits and ``n_paths = C(n_groups,k_test) * k_test
    / n_groups`` complete out-of-sample backtest paths, each path containing
    every group exactly once.
    """

    n_groups: int = 8
    k_test: int = 2
    embargo: int = 20
    label_horizon: int = 20

    def __post_init__(self):
        if not 0 < self.k_test < self.n_groups:
            raise ValueError("need 0 < k_test < n_groups")

    # -- sizes ------------------------------------------------------------
    @property
    def n_splits(self) -> int:
        return comb(self.n_groups, self.k_test)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    @property
    def n_paths(self) -> int:
        return self.n_splits * self.k_test // self.n_groups

    # -- structure --------------------------------------------------------
    def group_blocks(self, dates) -> list[tuple[int, int]]:
        n = _n(dates)
        edges = np.linspace(0, n, self.n_groups + 1).astype(int)
        return [(int(edges[g]), int(edges[g + 1]) - 1) for g in range(self.n_groups)]

    def test_combinations(self) -> list[tuple[int, ...]]:
        return list(combinations(range(self.n_groups), self.k_test))

    def split(self, dates, y=None, groups=None) -> Iterator[IdxPair]:
        n = _n(dates)
        blocks = self.group_blocks(dates)
        for combo in self.test_combinations():
            test_blocks = [blocks[g] for g in combo]
            test = np.concatenate([np.arange(t0, t1 + 1) for t0, t1 in test_blocks])
            train = np.setdiff1d(np.arange(n), test)
            train = purge_train(train, test_blocks, self.label_horizon, self.embargo, n)
            yield train, test

    def split_with_groups(self, dates) -> Iterator[tuple[int, tuple[int, ...], IdxPair]]:
        """Like ``split`` but also yields (split_id, test_group_tuple)."""
        for i, (combo, pair) in enumerate(zip(self.test_combinations(), self.split(dates), strict=True)):
            yield i, combo, pair

    def backtest_paths(self) -> np.ndarray:
        """Path assignment matrix of shape ``(n_groups, n_paths)``.

        ``paths[g, p]`` is the split index whose test set supplies group ``g``
        for path ``p``.  Each group appears in exactly ``n_paths`` test sets
        (once per path); assignment is the AFML greedy fill: for each split,
        each of its test groups goes into the lowest-numbered path that does
        not yet have that group.
        """
        G, P = self.n_groups, self.n_paths
        paths = -np.ones((G, P), dtype=int)
        for split_id, combo in enumerate(self.test_combinations()):
            for g in combo:
                p = int(np.argmax(paths[g] < 0))
                if paths[g, p] >= 0:
                    raise RuntimeError("path assembly overflow (should not happen)")
                paths[g, p] = split_id
        if (paths < 0).any():
            raise RuntimeError("path assembly incomplete (should not happen)")
        return paths

    def assemble_paths(self, dates, oos_predictions: Sequence[pd.Series]) -> list[pd.Series]:
        """Stitch per-split OOS prediction series (indexed by position) into
        ``n_paths`` complete series.  ``oos_predictions[i]`` must be indexed by
        the integer test positions of split ``i``."""
        blocks = self.group_blocks(dates)
        paths = self.backtest_paths()
        out = []
        for p in range(self.n_paths):
            parts = []
            for g in range(self.n_groups):
                t0, t1 = blocks[g]
                s = oos_predictions[paths[g, p]]
                parts.append(s.loc[t0:t1])
            out.append(pd.concat(parts).sort_index())
        return out


def check_no_overlap(splitter, dates, label_horizon: int | None = None) -> bool:
    """Exhaustive check: for every fold, no train label window overlaps the test windows."""
    h = label_horizon if label_horizon is not None else splitter.label_horizon
    return all(not label_windows_overlap(train, test, h) for train, test in splitter.split(dates))
