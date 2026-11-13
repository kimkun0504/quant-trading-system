"""Content-addressed cache for the expensive deterministic pipeline stages.

What is cached
--------------
:class:`~qts.pipeline.ResearchPipeline` can store and re-use the result of two
stages of :meth:`~qts.pipeline.ResearchPipeline.fit`:

* ``risk_fit``  -- the fitted risk model (exposure history, factor-return and
  specific-return history, the statistical companion's per-refit state);
* ``alpha_fit`` -- the fitted alpha model (raw and processed feature panels,
  labels, theme scores, composite / ML score panels, OOS IC history).

Both are pure functions of (input data, config sections, code, library
versions), so the key is a SHA-256 over exactly those inputs
(:meth:`StageCache.key`):

* the **full** SHA-256 of the input ``MarketData`` panels (every panel, index and
  column labels, the eligibility panel -- :func:`qts.core.manifest.hash_market_data`)
  plus its first / last date and shape (the *as-of span* of the snapshot).
  Reference tables that only one stage reads are left out of this panel hash
  (:data:`STAGE_EXCLUDED_DATA`: ``delisting_reasons``, ``earnings_dates``,
  ``earnings_fundamentals``) and keyed on the stage that reads them (``alpha_fit``:
  the labels; the earnings tables only under ``alpha.feature_set != "v1"``), so
  adding or re-deriving such a table does not invalidate the risk fit;
* the hash of every config section the stage reads (``universe`` + ``risk``
  for ``risk_fit``; additionally ``alpha``, the delisting returns by reason and
  the delisting reasons, the signal lag and the pipeline's constructor knobs
  for ``alpha_fit``, chained on the ``risk_fit`` key because labels are
  residualised on the risk model);
* the code version **of the packages the stage can import**
  (:data:`STAGE_CODE_PACKAGES`): SHA-256 of their ``*.py`` files (so an
  uncommitted edit invalidates the cache, unlike the git sha) and the versions
  of the numerical libraries (:data:`qts.core.manifest.TRACKED_LIBRARIES`).

Code component per stage
------------------------
Until 2026-11-12 the code component hashed every ``qts/**/*.py`` file, so an
edit to the simulator or the optimizer forced a refit of both stages (25 minutes
on the real universe) although neither can change a fit.  It is now the hash of
the packages in the stage's import closure, checked by
``tests/test_stage_cache.py::test_stage_packages_cover_the_import_graph``
(static scan of every ``import`` in those packages):

* ``risk_fit``  -> ``qts/core``, ``qts/data``, ``qts/risk``
  (``qts.risk`` imports ``qts.core`` and ``qts.data``; ``qts.data`` imports ``qts.core``);
* ``alpha_fit`` -> the same three + ``qts/alpha`` (``qts.alpha`` imports only
  ``qts.core``; the risk packages are included because the labels are
  residualised on the fitted risk model and the universe mask comes from ``qts.data``).

Top-level modules of the package (``qts/__init__.py``) are always included.
What is **not** in the code hash: ``qts/pipeline.py`` as a whole (it also holds
the evaluation and reporting code and changes with every simulator-side
feature).  The part of it that builds the two fits is covered separately: the
pipeline adds the source text of its fit methods to the key parts
(``fit_glue`` in ``ResearchPipeline._cache_parts``).  An ML factory outside the
hashed packages is identified by ``callable_identity`` (module + qualname), as before.
Consequence to remember: a change in ``qts/backtest``, ``qts/portfolio``,
``qts/tcost``, ``qts/execution``, ``qts/validation``, ``qts/runs`` or the CLI
never invalidates a fit; delete ``<runs_root>/_cache`` if you ever make one of
those packages an input of a fit without adding it to :data:`STAGE_CODE_PACKAGES`.

Point-in-time safety
--------------------
A fitted stage only answers ``as_of(t)`` queries from data ``<= t`` (that is
the stage's own PIT contract, unchanged here).  The cache adds one rule: an
entry is only ever served for the *identical* data snapshot.  Any change to the
data -- one more day appended, a revised price, a different universe, a
different start date -- changes the data hash or the span and therefore the
key.  The entry's metadata repeats the data hash and span and is checked again
on load; a mismatch is treated as a miss (and logged).  So an entry built on
data through 2024-06-28 can never serve a run on data through 2024-07-01, even
though the two snapshots agree on every earlier day.

Storage
-------
``<root>/<stage>/<key[:2]>/<key>.pkl`` (pickle protocol 5) with a JSON sidecar
``<key>.json`` (stage, key parts, data span, sizes, creation time).  Writes go
to a temporary file and are moved into place atomically, so concurrent runs
cannot read a half-written entry (two runs that miss at the same time both
compute and the second rename wins; both values are identical).  The default
root is ``<runs_root>/_cache`` (``runs/`` is git-ignored).  Delete the directory
to clear the cache; nothing else refers to it.

Objects the stage shares with the caller (the input ``MarketData``, the ML
factory callable) are *not* pickled: they are written as references
(``persistent_id``) and re-attached from the current run on load, which keeps
entries small and guarantees the loaded model points at this run's objects.

Pickle executes code on load: only point the cache at a directory you trust
(it is a local, per-user artefact store, not an interchange format).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pickle
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger("qts.runs.cache")

#: Bump when the entry layout or the key recipe changes (old entries become misses).
#: 2 (2026-11-12): per-stage code hash, per-stage data hash, fit-glue part.
CACHE_SCHEMA_VERSION = 2
#: Stages ``ResearchPipeline`` knows how to cache.
CACHEABLE_STAGES = ("risk_fit", "alpha_fit")
#: Default cache directory name under the runs root.
DEFAULT_CACHE_DIRNAME = "_cache"


#: Sub-packages of ``qts`` whose source enters each stage's key (the stage's
#: import closure; module docstring "Code component per stage").
STAGE_CODE_PACKAGES: dict[str, tuple[str, ...]] = {
    "risk_fit": ("core", "data", "risk"),
    "alpha_fit": ("core", "data", "risk", "alpha"),
}
#: ``MarketData`` fields left out of the panel hash shared by every stage and
#: keyed on the stage that reads them (``ResearchPipeline._cache_parts``).
STAGE_EXCLUDED_DATA: frozenset[str] = frozenset(
    {"delisting_reasons", "earnings_dates", "earnings_fundamentals"}
)


def package_root() -> Path:
    """Directory of the ``qts`` package (a function so tests can point it elsewhere)."""
    return Path(__file__).resolve().parents[1]


def _hash_files(root: Path, files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _source_hash_of(root: Path, packages: tuple[str, ...] | None) -> str:
    if packages is None:
        files = [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    else:
        files = [p for p in root.glob("__init__.py")]
        for pkg in packages:
            files += [p for p in (root / pkg).rglob("*.py") if "__pycache__" not in p.parts]
    return _hash_files(root, files)


@lru_cache(maxsize=8)
def source_hash(stage: str | None = None) -> str:
    """SHA-256 over the relative path and bytes of the ``*.py`` files a stage can
    depend on: the packages of :data:`STAGE_CODE_PACKAGES` ``[stage]`` plus
    ``qts/__init__.py``.  ``stage=None`` (or a stage without an entry): every
    ``*.py`` file of the ``qts`` package.  Changes with any edit to those
    files, committed or not; cached per process."""
    return _source_hash_of(package_root(), STAGE_CODE_PACKAGES.get(stage) if stage else None)


@lru_cache(maxsize=1)
def _library_versions() -> tuple[tuple[str, str], ...]:
    from qts.core.manifest import library_versions

    return tuple(sorted(library_versions().items()))


def code_identity(stage: str | None = None) -> dict[str, Any]:
    """Code part of a stage's key: source hash of the stage's packages
    (:func:`source_hash`) + numerical library versions."""
    return {"source": source_hash(stage), "libraries": dict(_library_versions())}


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class _RefPickler(pickle.Pickler):
    """Pickler that writes the objects in ``refs`` (matched by identity) as
    named references instead of their contents."""

    def __init__(self, f: io.BufferedIOBase, refs: Mapping[str, Any]):
        super().__init__(f, protocol=5)
        self._by_id = {id(v): k for k, v in refs.items() if v is not None}

    def persistent_id(self, obj: Any) -> str | None:
        return self._by_id.get(id(obj))


class _RefUnpickler(pickle.Unpickler):
    def __init__(self, f: io.BufferedIOBase, refs: Mapping[str, Any]):
        super().__init__(f)
        self._refs = refs

    def persistent_load(self, pid: Any) -> Any:
        if pid not in self._refs:
            raise pickle.UnpicklingError(
                f"stage cache entry references {pid!r}, which the caller did not supply"
            )
        return self._refs[pid]


class StageCache:
    """Content-addressed store of fitted pipeline stages (see module docstring)."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.hits = 0
        self.misses = 0
        self.stores = 0

    # ------------------------------------------------------------------ keys
    @staticmethod
    def key(stage: str, parts: Mapping[str, Any]) -> str:
        """SHA-256 (hex) of the canonical JSON of ``stage``, the schema version,
        the code identity and ``parts`` (data hash, span, config-section hashes...)."""
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "stage": stage,
            "code": code_identity(stage),
            "parts": dict(parts),
        }
        return hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def _paths(self, stage: str, key: str) -> tuple[Path, Path]:
        d = self.root / stage / key[:2]
        return d / f"{key}.pkl", d / f"{key}.json"

    # ------------------------------------------------------------------ load / store
    def load(
        self,
        stage: str,
        key: str,
        expect: Mapping[str, Any],
        refs: Mapping[str, Any] | None = None,
    ) -> Any | None:
        """The cached object for ``(stage, key)``, or ``None`` on a miss.

        ``expect``: metadata values the entry must carry (at least the data hash
        and the data span); any mismatch, a missing sidecar or an unreadable
        pickle is a miss.  ``refs``: the objects to re-attach for references
        written by :meth:`store`."""
        pkl, meta_path = self._paths(stage, key)
        if not pkl.exists() or not meta_path.exists():
            self.misses += 1
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError) as e:
            log.warning("stage cache: unreadable metadata %s (%s); recomputing", meta_path, e)
            self.misses += 1
            return None
        bad = {
            k: (meta.get("expect", {}).get(k), v)
            for k, v in expect.items()
            if meta.get("expect", {}).get(k) != v
        }
        if meta.get("key") != key or meta.get("stage") != stage or bad:
            log.warning(
                "stage cache: entry %s/%s does not match this run %s; recomputing", stage, key[:12], bad
            )
            self.misses += 1
            return None
        t0 = time.perf_counter()
        try:
            with open(pkl, "rb") as f:
                obj = _RefUnpickler(f, refs or {}).load()  # type: ignore[arg-type]
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as e:
            log.warning("stage cache: unreadable entry %s (%s); recomputing", pkl, e)
            self.misses += 1
            return None
        self.hits += 1
        log.info("stage cache hit: %s %s (%.2fs)", stage, key[:12], time.perf_counter() - t0)
        return obj

    def store(
        self,
        stage: str,
        key: str,
        obj: Any,
        expect: Mapping[str, Any],
        refs: Mapping[str, Any] | None = None,
        parts: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Write ``obj`` atomically; ``refs`` objects are written as references.
        Returns the entry path, or ``None`` when writing failed (logged; a
        cache that cannot be written never fails the run)."""
        pkl, meta_path = self._paths(stage, key)
        try:
            pkl.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            _RefPickler(buf, refs or {}).dump(obj)  # type: ignore[arg-type]
            data = buf.getvalue()
            meta = {
                "schema": CACHE_SCHEMA_VERSION,
                "stage": stage,
                "key": key,
                "expect": dict(expect),
                "parts": dict(parts or {}),
                "refs": sorted((refs or {}).keys()),
                "bytes": len(data),
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            for target, payload in ((pkl, data), (meta_path, _canonical(meta).encode())):
                fd, tmp = tempfile.mkstemp(dir=pkl.parent, prefix=".tmp-")
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(payload)
                    os.replace(tmp, target)
                except BaseException:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise
        except (OSError, pickle.PicklingError, TypeError, AttributeError) as e:
            log.warning("stage cache: could not store %s %s (%s)", stage, key[:12], e)
            return None
        self.stores += 1
        log.info("stage cache store: %s %s (%.1f MB)", stage, key[:12], len(data) / 1e6)
        return pkl

    def entries(self, stage: str | None = None) -> list[Path]:
        """Entry files (``*.pkl``) under the root, optionally of one stage."""
        base = self.root / stage if stage else self.root
        return sorted(base.rglob("*.pkl")) if base.exists() else []


__all__ = [
    "CACHEABLE_STAGES",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_DIRNAME",
    "STAGE_CODE_PACKAGES",
    "STAGE_EXCLUDED_DATA",
    "StageCache",
    "code_identity",
    "source_hash",
]
