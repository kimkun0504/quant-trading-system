"""Trial ledger: the append-only registry of every backtest / experiment.

**This file is the input to the deflated Sharpe ratio and must never be
reset.**  DSR deflates the best observed Sharpe by the expected maximum of
``N`` trials; if trials are deleted (or not logged) the deflation is too weak
and the selection bias goes unpunished.  Treat the JSONL file like an
accounting ledger: append-only, version-controlled or backed up, and shared
across every researcher/agent touching the same data.

Each line is one JSON record::

    {"ts": ISO-8601, "name": str, "family": str, "config": {...},
     "metrics": {"sharpe_annual": float, "sharpe_daily": float, "n_obs": int,
                 "skew": float, "kurt": float, ...},
     "data_hash": str, "code_version": str, "notes": str}

``family`` groups trials that belong to the same search (e.g. all parameter
settings of one signal); it defaults to ``name.split(':')[0]`` so that
``"mom:lb=12"`` and ``"mom:lb=6"`` share the family ``"mom"``.

``effective_trials`` implements Nyholt's (2004) effective number of
independent tests from the eigenvalues of the correlation matrix of trial
return streams: ``M_eff = 1 + (M - 1) * (1 - Var(lambda) / M)`` (sample
variance of the eigenvalues).  This is the
right ``n_trials`` for DSR when trials are highly correlated variants.
Records may carry ``returns_monthly`` (month -> return) so that
:meth:`effective_trials_for` can build that correlation matrix from the ledger
itself.

Identity / de-duplication (audit 2 findings 2-3)
------------------------------------------------
The file is **strictly append-only**: every :meth:`TrialLedger.log` appends one
line under an exclusive ``fcntl.flock`` on ``<path>.lock`` (one ``write`` of a
complete line), and nothing ever rewrites it, so concurrent writers (several
researchers / agents / processes) cannot lose each other's trials.

``log(..., replace_identical=True)`` marks the record with a ``dedup_key``:
SHA-256 of ``(name, config_hash, data_hash, code_version, run_spec_hash)``,
where ``run_spec`` holds every setting outside the config that changes the
numbers (the pipeline passes ``ResearchPipeline.run_spec``: ``use_ml``,
``ml_factory`` identity, ``composite_weight``, ``rebalance_every``, ``venue``,
``nav0``, ``warmup_days``, backtest dates, signal lag).  De-duplication happens
at **read** time: records sharing a ``dedup_key`` collapse into the latest one
(at the position of the first), carrying ``n_reruns`` (= occurrences - 1) and
``ts_first``.  A record is never de-duplicated (always its own trial) when
``replace_identical`` is False, ``data_hash`` is empty, or the code version is
empty, ``"unknown"`` or dirty (``...+dirty``): a dirty tree or unknown code may
differ from the last run in ways nobody recorded.  :meth:`raw_records` returns
the undeduplicated lines.

Research-family identity (2026-11-12)
------------------------------------
The DSR's ``N`` must count every strategy configuration tried on the same
price history.  ``ResearchPipeline`` therefore logs under
:func:`research_family` = ``pipeline_phase{p}:{market_hash[:12]}``, where
``market_hash`` (:func:`qts.core.manifest.hash_market`) covers the
adjusted-close panel only: values, dates, ids.  The record still stores the full
``data_hash`` (every panel; reproducibility and the re-run de-duplication key)
next to ``market_hash``.  Until 2026-11-12 the family was keyed on
``data_hash[:12]``, so adding an auxiliary table (``delisting_reasons``) started
a new family and reset ``N`` to 1 on the same market.  Three tools repair and
maintain the families (handbook 10 "Research family"):

* :meth:`TrialLedger.rekey_families` (``qts ledger rekey [--dry-run]``): moves
  records from ``pipeline_phase{p}:{data_hash[:12]}`` to the market-hash family
  wherever the record's market hash is *known*: stored on the record, or found
  in a ``data_hash -> market_hash`` map built from other records, run manifests
  and (``--store``) the hashes recomputed from the data.  A record stores only
  hashes, so a data hash nobody can map stays where it is and is listed;
* :meth:`TrialLedger.merge_families` (``qts ledger merge-families --into F A B``):
  the owner's explicit statement that families are one search (the fallback
  for unmappable records, and the tool after a data refresh, which changes the
  adjusted closes and therefore the market hash);
* :meth:`TrialLedger.families` (``qts ledger list``): every family with its
  record / trial counts and hashes.

Both rewrites keep every record and every other field, note the previous
family on the record (``family_rekeyed_from`` / ``family_merged_from``), write a
backup ``<path>.bak-<UTC timestamp>`` first and are idempotent.  They can only
*pool* trials: nothing here deletes a trial or splits a family.

Record kinds and reference re-runs (2026-11-13)
-----------------------------------------------
Not every line is an alpha trial.  A record may carry ``kind`` (absent =
``"trial"``): ``"unseal"`` (an event: someone loaded the sealed holdout,
``qts.research.seal``), ``"capacity"`` (the same strategy re-run at another NAV),
``"calibration"`` (a risk-forecast diagnostic).  A trial may carry
``reference: true``: a re-run of a strategy that is already counted, on a subset of
the same history (the baseline re-run under the seal).  Every query that feeds the
DSR (:meth:`TrialLedger.n_trials`, :meth:`TrialLedger.records`,
:meth:`TrialLedger.trial_sharpes_daily`, :meth:`TrialLedger.returns_matrix`,
:meth:`TrialLedger.dsr_inputs`, ...) sees trials only (:func:`is_trial_record`);
``records(include_non_trials=True)`` and :meth:`TrialLedger.raw_records` see
everything.  A non-default kind / ``reference`` is part of the de-duplication key,
so such a record never replaces a trial in the de-duplicated view.
:func:`normalized_config_hash` re-hashes a record's stored config with the current
``SystemConfig`` (new fields at their defaults): the strategy identity used by the
once-only rules (unseal, the mid-cap confirmation) and by the ``reference`` check,
stable when a config field is added.

Family names and the one-off migration
--------------------------------------
Between 2026-11-05 and 2026-11-12 ``ResearchPipeline`` logged under
``pipeline_phase{p}:{data_hash[:12]}``.  Records written before 2026-11-05 carry the
bare ``pipeline_phase{p}`` (and the earliest ones an empty ``data_hash``), so
they pool runs on different data into one family.  :meth:`TrialLedger.migrate_family_names`
(CLI: ``qts ledger migrate``) is the single sanctioned rewrite of the file: it
renames a bare pipeline family to the new format when the record has a data
hash and to :data:`LEGACY_FAMILY` otherwise, records the old name in
``family_migrated_from``, keeps every record (no trial is dropped) and writes a
backup first.  It is idempotent.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from qts.validation.deflated_sharpe import deannualize_sharpe, deflated_sharpe_ratio


def callable_identity(fn: Callable[..., Any] | None) -> str | None:
    """Stable identity of a factory callable for the ledger key: module +
    qualified name, plus a hash of the code object (so two different lambdas /
    closures with the same name differ) and, for ``functools.partial``, of its
    bound arguments.  ``None`` -> ``None``."""
    if fn is None:
        return None
    if isinstance(fn, functools.partial):
        inner = callable_identity(fn.func)
        bound = repr((fn.args, sorted((fn.keywords or {}).items())))
        return f"partial({inner};{hashlib.sha256(bound.encode()).hexdigest()[:12]})"
    target = fn if (hasattr(fn, "__code__") or isinstance(fn, type)) else type(fn)
    name = f"{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', repr(target))}"
    code = getattr(fn, "__code__", None)
    if code is not None:
        payload = code.co_code + repr(code.co_consts).encode() + repr(code.co_names).encode()
        cells = getattr(fn, "__closure__", None) or ()
        for c in cells:
            try:
                v = c.cell_contents
            except ValueError:  # empty cell
                continue
            payload += (callable_identity(v) if callable(v) else repr(v)).encode()
        name += "#" + hashlib.sha256(payload).hexdigest()[:12]
    return name


#: Family of pre-November pipeline records that have no data hash (cannot be keyed).
LEGACY_FAMILY = "legacy"
_BARE_PIPELINE_FAMILY = re.compile(r"^pipeline_phase(\d+)$")


def migrated_family(rec: dict) -> str | None:
    """New family of a legacy record (module docstring), or ``None`` if it needs none."""
    fam = str(rec.get("family") or "")
    m = _BARE_PIPELINE_FAMILY.match(fam)
    if not m:
        return None
    dh = str(rec.get("data_hash") or "")
    return f"pipeline_phase{m.group(1)}:{dh[:12]}" if dh else LEGACY_FAMILY


_PIPELINE_FAMILY = re.compile(r"^pipeline_phase(\d+):([0-9a-f]+)$")


def research_family(phase: int, market_hash: str) -> str:
    """Trial family of a pipeline run: ``pipeline_phase{p}:{market_hash[:12]}``
    (module docstring "Research-family identity")."""
    return f"pipeline_phase{int(phase)}:{(market_hash or 'nodata')[:12]}"


def rekeyed_family(rec: dict, market_of: dict[str, str] | None = None) -> tuple[str, str] | None:
    """``(new family, market_hash)`` of a record still keyed on its data hash,
    or ``None`` when it needs no change or its market hash is unknown.

    Only families of the form ``pipeline_phase{p}:{data_hash[:12]}`` are touched
    (the suffix must be the record's own data hash; a hand-made or merged family
    is left alone).  The market hash is the record's ``market_hash`` or
    ``market_of[data_hash]``."""
    m = _PIPELINE_FAMILY.match(str(rec.get("family") or ""))
    dh = str(rec.get("data_hash") or "")
    if not m or not dh or m.group(2) != dh[:12]:
        return None
    mh = str(rec.get("market_hash") or "") or str((market_of or {}).get(dh) or "")
    if not mh:
        return None
    new = research_family(int(m.group(1)), mh)
    return None if new == rec.get("family") else (new, mh)


def _is_clean_code_version(code_version: str) -> bool:
    cv = str(code_version or "")
    return bool(cv) and cv != "unknown" and not cv.endswith("+dirty")


#: Record fields :meth:`TrialLedger.log` sets itself (``extra`` may not use them).
_RESERVED_KEYS = frozenset(
    {
        "ts",
        "name",
        "family",
        "config",
        "config_hash",
        "metrics",
        "data_hash",
        "code_version",
        "notes",
        "market_hash",
        "run_spec",
        "returns_monthly",
        "dedup_key",
        "n_reruns",
        "ts_first",
        "family_rekeyed_from",
        "family_merged_from",
        "family_migrated_from",
    }
)


def record_kind(rec: dict) -> str:
    """``"trial"`` (also for every record written before kinds existed), or the
    record's ``kind`` (``"unseal"``, ``"capacity"``, ``"calibration"``, ...)."""
    return str(rec.get("kind") or "trial")


def is_trial_record(rec: dict) -> bool:
    """True when the record counts towards a family's DSR inputs: an ordinary trial
    that is not a ``reference`` re-run of an already counted strategy."""
    return record_kind(rec) == "trial" and not bool(rec.get("reference"))


def normalized_config_hash(rec: dict) -> str | None:
    """The record's configuration hashed by the *current* ``SystemConfig``: the stored
    ``config`` dict rebuilt with ``SystemConfig.from_dict`` (fields added since the
    record was written take their defaults) and hashed.  Two records of the same
    strategy written before and after a config field was added therefore share it,
    although their stored ``config_hash`` differs.  ``None`` when the stored dict no
    longer parses (a removed field) or there is none."""
    from qts.core.config import SystemConfig

    cfg = rec.get("config")
    if not isinstance(cfg, dict) or not cfg:
        return None
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return SystemConfig.from_dict(cfg).hash()
    except (KeyError, TypeError, ValueError):
        return None


class TrialLedger:
    def __init__(self, path: str):
        self.path = str(path)
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(self.path):
            open(self.path, "a").close()

    # ------------------------------------------------------------------ io
    @contextmanager
    def _locked(self, exclusive: bool) -> Iterator[None]:
        """``fcntl.flock`` on ``<path>.lock`` (exclusive for writers, shared for readers)."""
        fd = os.open(self.path + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def raw_records(self) -> list[dict]:
        """Every line of the file, in order (no de-duplication)."""
        out = []
        with self._locked(exclusive=False), open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _records(self, include_non_trials: bool = False) -> list[dict]:
        """Trials: raw records with identical re-runs (same ``dedup_key``) collapsed
        into the latest record at the first one's position (``n_reruns``, ``ts_first``).

        Records that are not alpha trials are left out unless ``include_non_trials``
        (module docstring "Record kinds"): ``kind`` other than ``"trial"`` (unseal
        events, capacity and calibration runs) and ``reference`` re-runs."""
        out = self._dedup_records()
        if include_non_trials:
            return out
        return [r for r in out if is_trial_record(r)]

    def _dedup_records(self) -> list[dict]:
        out: list[dict] = []
        where: dict[str, int] = {}
        for r in self.raw_records():
            k = r.get("dedup_key")
            if not k:
                out.append(r)
                continue
            if k not in where:
                where[k] = len(out)
                out.append(
                    dict(r, n_reruns=int(r.get("n_reruns", 0)), ts_first=r.get("ts_first", r.get("ts")))
                )
                continue
            first = out[where[k]]
            out[where[k]] = dict(r, n_reruns=int(first["n_reruns"]) + 1, ts_first=first["ts_first"])
        return out

    @staticmethod
    def dedup_key(
        name: str, config_hash: str, data_hash: str, code_version: str, run_spec: dict | None = None
    ) -> str:
        """Identity of an experiment (see the module docstring)."""
        spec = json.dumps(_jsonable_nested(run_spec or {}), sort_keys=True, default=str)
        spec_hash = hashlib.sha256(spec.encode()).hexdigest()[:16]
        ident = json.dumps([str(name), str(config_hash), str(data_hash), str(code_version), spec_hash])
        return hashlib.sha256(ident.encode()).hexdigest()[:24]

    def log(
        self,
        name: str,
        config: dict,
        metrics: dict,
        data_hash: str = "",
        code_version: str = "",
        notes: str = "",
        family: str | None = None,
        config_hash: str | None = None,
        returns_monthly: pd.Series | None = None,
        replace_identical: bool = False,
        run_spec: dict | None = None,
        market_hash: str = "",
        extra: dict | None = None,
    ) -> dict:
        """Append one trial.

        ``market_hash`` (:func:`qts.core.manifest.hash_market`): the price history
        the trial was run on; stored next to ``data_hash`` so that
        :meth:`rekey_families` can map the one to the other.

        ``extra``: further top-level fields (e.g. ``kind``, ``reference``, the seal
        fields of :class:`qts.runs.context.RunContext`); a key that collides with a
        standard field raises ``ValueError``.  A non-default ``kind`` or
        ``reference`` enters the de-duplication key, so such a record can never
        collapse into (and hide) an ordinary trial with the same identity.

        With ``replace_identical=True`` (and a non-empty ``data_hash`` and a clean
        ``code_version``) the record carries a ``dedup_key`` so that an identical
        re-run is counted once at read time (module docstring); otherwise it is
        always a new trial.  ``run_spec``: settings outside ``config`` that change
        the result (hashed into the key, stored on the record).

        ``metrics`` should include ``sharpe_annual`` (or ``sharpe_daily``),
        ``n_obs`` and optionally ``skew``/``kurt``.  ``config`` should be the
        *full* configuration (``SystemConfig.to_dict()``); ``config_hash``
        defaults to a hash of its canonical JSON.  ``returns_monthly`` (a
        month-indexed Series) enables :meth:`effective_trials_for`.
        """
        cfg_json = _jsonable_nested(config)
        if config_hash is None:
            config_hash = hashlib.sha256(
                json.dumps(cfg_json, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
        rec: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "name": str(name),
            "family": family if family is not None else str(name).split(":")[0],
            "config": cfg_json,
            "config_hash": str(config_hash),
            "metrics": _jsonable(metrics),
            "data_hash": str(data_hash),
            "code_version": str(code_version),
            "notes": str(notes),
        }
        if market_hash:
            rec["market_hash"] = str(market_hash)
        if run_spec is not None:
            rec["run_spec"] = _jsonable_nested(run_spec)
        if returns_monthly is not None and len(returns_monthly):
            rm = pd.Series(returns_monthly, dtype=float)
            rec["returns_monthly"] = {str(pd.Timestamp(k).strftime("%Y-%m")): float(v) for k, v in rm.items()}
        ex = _jsonable_nested(extra or {})
        clash = sorted(set(ex) & (set(rec) | _RESERVED_KEYS))
        if clash:
            raise ValueError(f"TrialLedger.log: extra fields {clash} collide with standard record fields")
        rec.update(ex)
        if replace_identical and data_hash and _is_clean_code_version(code_version):
            spec_key = run_spec
            kind = str(ex.get("kind") or "trial")
            if kind != "trial" or ex.get("reference"):
                spec_key = {**(run_spec or {}), "_record_kind": kind, "_reference": bool(ex.get("reference"))}
            rec["dedup_key"] = self.dedup_key(name, str(config_hash), data_hash, code_version, spec_key)
        line = (json.dumps(rec, default=str) + "\n").encode()
        with self._locked(exclusive=True):
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                view = memoryview(line)
                while view:
                    n = os.write(fd, view)
                    view = view[n:]
                os.fsync(fd)
            finally:
                os.close(fd)
        return rec

    def _rewrite_families(
        self, new_family: Callable[[dict], tuple[str, dict[str, Any]] | None], dry_run: bool
    ) -> list[dict]:
        """Shared body of the sanctioned rewrites.  ``new_family(rec)`` returns
        ``(family, extra fields)`` or ``None`` (unchanged).  Under the exclusive
        lock: every line is kept, only ``family`` (and ``name`` when it equals the
        old family), plus the extra fields, change; unless ``dry_run`` the file is
        first copied to ``<path>.bak-<UTC timestamp>`` and then replaced atomically.
        Returns one ``{"line", "ts", "old", "new"}`` per changed record."""
        changes: list[dict] = []
        with self._locked(exclusive=True):
            with open(self.path) as f:
                lines = f.readlines()
            out: list[str] = []
            for i, line in enumerate(lines):
                if not line.strip():
                    out.append(line)
                    continue
                rec = json.loads(line)
                res = new_family(rec)
                if res is None or res[0] == rec.get("family"):
                    out.append(line)
                    continue
                new, extra = res
                old = rec.get("family")
                changes.append({"line": i + 1, "ts": rec.get("ts"), "old": old, "new": new})
                rec["family"] = new
                if rec.get("name") == old:
                    rec["name"] = new
                rec.update(extra)
                out.append(json.dumps(rec, default=str) + "\n")
            if dry_run or not changes:
                return changes
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            shutil.copy2(self.path, f"{self.path}.bak-{stamp}")
            tmp = f"{self.path}.tmp-{os.getpid()}"
            with open(tmp, "w") as f:
                f.writelines(out)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        return changes

    def migrate_family_names(self, dry_run: bool = False) -> list[dict]:
        """One-off rewrite of legacy family names (module docstring).

        Returns one ``{"line", "ts", "old", "new"}`` per changed record.  Unless
        ``dry_run``, copies the file to ``<path>.bak-<UTC timestamp>`` and replaces
        it atomically (temp file + ``os.replace``) under the exclusive lock; the
        record count and every other field are unchanged."""

        def new_family(rec: dict) -> tuple[str, dict[str, Any]] | None:
            new = migrated_family(rec)
            return None if new is None else (new, {"family_migrated_from": rec.get("family")})

        return self._rewrite_families(new_family, dry_run)

    def known_market_hashes(self) -> dict[str, str]:
        """``data_hash -> market_hash`` from the records that carry both."""
        out: dict[str, str] = {}
        for r in self.raw_records():
            dh, mh = str(r.get("data_hash") or ""), str(r.get("market_hash") or "")
            if dh and mh:
                out[dh] = mh
        return out

    def rekey_families(self, market_of: dict[str, str] | None = None, dry_run: bool = False) -> list[dict]:
        """Move records keyed on their data hash to the research family of their
        market hash (module docstring "Research-family identity").

        ``market_of``: extra ``data_hash -> market_hash`` entries (run manifests,
        hashes recomputed from the data); the ledger's own records that carry both
        hashes are always used and win.  Records whose market hash is unknown are
        left untouched (:meth:`unkeyed_families` lists them).  The record gains
        ``market_hash`` and ``family_rekeyed_from``.  Backup + atomic replace as in
        :meth:`migrate_family_names`; idempotent."""
        mapping = {**(market_of or {}), **self.known_market_hashes()}

        def new_family(rec: dict) -> tuple[str, dict[str, Any]] | None:
            res = rekeyed_family(rec, mapping)
            if res is None:
                return None
            return res[0], {"market_hash": res[1], "family_rekeyed_from": rec.get("family")}

        return self._rewrite_families(new_family, dry_run)

    def unkeyed_families(self, market_of: dict[str, str] | None = None) -> dict[str, int]:
        """``family -> raw record count`` of pipeline families still keyed on a data
        hash whose market hash is not known (neither on a record nor in ``market_of``):
        what :meth:`rekey_families` cannot move and ``merge-families`` is for."""
        mapping = {**(market_of or {}), **self.known_market_hashes()}
        out: dict[str, int] = {}
        for r in self.raw_records():
            m = _PIPELINE_FAMILY.match(str(r.get("family") or ""))
            dh = str(r.get("data_hash") or "")
            if m and dh and m.group(2) == dh[:12] and not (r.get("market_hash") or mapping.get(dh)):
                out[str(r["family"])] = out.get(str(r["family"]), 0) + 1
        return out

    def merge_families(self, into: str, sources: list[str], dry_run: bool = False) -> list[dict]:
        """Rename every record of the ``sources`` families to ``into`` (the owner's
        statement that they are one search; module docstring).  ``into`` need not
        exist yet.  Unknown source names raise ``KeyError`` (a typo must not pass
        silently).  The record keeps its hashes and gains ``family_merged_from``
        (the family it had before its *first* merge is kept when merged again).
        Backup + atomic replace as in :meth:`migrate_family_names`; idempotent."""
        into = str(into)
        if not into:
            raise ValueError("merge_families: empty target family")
        src = {str(x) for x in sources} - {into}
        have = {str(r.get("family")) for r in self.raw_records()}
        missing = sorted(src - have)
        if missing:
            raise KeyError(f"no such family in {self.path}: {missing}")

        def new_family(rec: dict) -> tuple[str, dict[str, Any]] | None:
            if str(rec.get("family")) not in src:
                return None
            return into, {"family_merged_from": rec.get("family_merged_from") or rec.get("family")}

        return self._rewrite_families(new_family, dry_run)

    def families(self) -> list[dict]:
        """One summary row per family, in order of first appearance: ``family``,
        ``n_records`` (raw lines), ``n_trials`` (identical re-runs collapsed: the
        DSR's raw count), ``n_not_counted`` (reference re-runs and non-trial records:
        unseal events, capacity / calibration runs), ``n_effective`` (Nyholt), ``data_hashes`` /
        ``market_hashes`` (distinct, in order), ``first_ts`` / ``last_ts``,
        ``best_sharpe`` (annual; ``None`` without a Sharpe)."""
        raw: dict[str, int] = {}
        for r in self.raw_records():
            raw[str(r.get("family"))] = raw.get(str(r.get("family")), 0) + 1
        rows: dict[str, dict[str, Any]] = {}
        for r in self._records(include_non_trials=True):
            fam = str(r.get("family"))
            row = rows.setdefault(
                fam,
                {
                    "family": fam,
                    "n_records": raw.get(fam, 0),
                    "n_trials": 0,
                    "n_not_counted": 0,
                    "data_hashes": [],
                    "market_hashes": [],
                    "first_ts": r.get("ts_first", r.get("ts")),
                    "last_ts": r.get("ts"),
                    "best_sharpe": None,
                },
            )
            if not is_trial_record(r):
                # reference re-runs, unseal events, capacity / calibration runs
                row["n_not_counted"] += 1
                row["last_ts"] = max(str(row["last_ts"]), str(r.get("ts")))
                continue
            row["n_trials"] += 1
            row["last_ts"] = max(str(row["last_ts"]), str(r.get("ts")))
            for key, col in (("data_hash", "data_hashes"), ("market_hash", "market_hashes")):
                v = str(r.get(key) or "")
                if v and v not in row[col]:
                    row[col].append(v)
            sr = (r.get("metrics") or {}).get("sharpe_annual")
            if isinstance(sr, (int, float)) and math.isfinite(sr):
                row["best_sharpe"] = sr if row["best_sharpe"] is None else max(row["best_sharpe"], sr)
        for fam, row in rows.items():
            row["n_effective"] = self.effective_trials_for(fam)
        return list(rows.values())

    def dsr_inputs(self, family: str, n_obs: int, periods_per_year: int = 252) -> dict[str, float]:
        """``n_trials`` / cross-trial SR variance for the DSR from a ledger family
        (the pipeline's rule, shared with ``qts ledger restate``):

        * a single trial (or none) -> ``n_trials=1``, variance 0 (PSR against 0);
        * 2 trials -> raw count and their sample variance;
        * >= 3 trials -> Nyholt :meth:`effective_trials_for` (correlation of the
          stored monthly return streams) and the family's annual SR variance.
        The variance is floored at the pure sampling-noise variance of an annual
        SR over ``n_obs`` days (``periods_per_year / n_obs``) whenever more than
        one trial exists.  ``n_trials_raw`` is the de-duplicated count."""
        n_raw = self.n_trials(family)
        if n_raw <= 1:
            return {"n_trials": 1.0, "var_trials_sr_annual": 0.0, "n_trials_raw": float(n_raw)}
        n_eff = self.effective_trials_for(family) if n_raw >= 3 else float(n_raw)
        var = self.family_sharpe_variance(family, annual=True, periods_per_year=periods_per_year)
        var = max(var, periods_per_year / max(n_obs, 2))
        return {"n_trials": float(n_eff), "var_trials_sr_annual": float(var), "n_trials_raw": float(n_raw)}

    # ------------------------------------------------------------------ queries
    def n_trials(self, family: str | None = None) -> int:
        recs = self._records()
        if family is None:
            return len(recs)
        return sum(1 for r in recs if r.get("family") == family)

    def report(self) -> pd.DataFrame:
        recs = self._records()
        if not recs:
            return pd.DataFrame(columns=["ts", "name", "family", "data_hash", "code_version", "notes"])
        rows = []
        for r in recs:
            row = {k: r.get(k) for k in ("ts", "name", "family", "data_hash", "code_version", "notes")}
            for k, v in (r.get("metrics") or {}).items():
                row[f"m.{k}"] = v
            for k, v in (r.get("config") or {}).items():
                row[f"c.{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def records(self, family: str | None = None, include_non_trials: bool = False) -> list[dict]:
        """De-duplicated records (of ``family``); trials only unless
        ``include_non_trials`` (unseal events, capacity / calibration runs and
        reference re-runs are then included, each with its ``kind`` / ``reference``)."""
        recs = self._records(include_non_trials=include_non_trials)
        return recs if family is None else [r for r in recs if r.get("family") == family]

    def trial_sharpes_annual(self, family: str | None = None, periods_per_year: int = 252) -> pd.Series:
        return self.trial_sharpes_daily(family) * math.sqrt(periods_per_year)

    def family_sharpe_variance(
        self, family: str | None = None, annual: bool = True, periods_per_year: int = 252
    ) -> float:
        """Sample variance (ddof=1) of the logged Sharpe ratios in ``family``
        (0.0 with fewer than two trials) — the ``V`` of Bailey & LdP's E[max SR]."""
        srs = (
            self.trial_sharpes_annual(family, periods_per_year)
            if annual
            else self.trial_sharpes_daily(family)
        )
        if len(srs) < 2:
            return 0.0
        v = float(srs.var(ddof=1))
        return v if math.isfinite(v) else 0.0

    def returns_matrix(self, family: str | None = None, min_overlap: int = 12) -> pd.DataFrame:
        """Month x trial matrix of the stored ``returns_monthly`` streams on
        their common months (empty if fewer than two trials have ``min_overlap``
        months in common)."""
        cols = {}
        for i, r in enumerate(self.records(family)):
            rm = r.get("returns_monthly")
            if rm:
                cols[f"{r['name']}#{i}"] = pd.Series({pd.Period(k, "M"): float(v) for k, v in rm.items()})
        if len(cols) < 2:
            return pd.DataFrame()
        M = pd.DataFrame(cols).dropna(how="any")
        return M if len(M) >= min_overlap else pd.DataFrame()

    def effective_trials_for(self, family: str | None = None, min_trials_for_correlation: int = 3) -> float:
        """Nyholt effective number of trials in ``family`` from the correlation of
        their stored monthly return streams; falls back to the raw count when
        fewer than ``min_trials_for_correlation`` trials (or no streams) exist."""
        n = self.n_trials(family)
        if n < min_trials_for_correlation:
            return float(n)
        M = self.returns_matrix(family)
        if M.shape[1] < min_trials_for_correlation:
            return float(n)
        C = np.array(M.corr().fillna(0.0).to_numpy(), dtype=float, copy=True)
        np.fill_diagonal(C, 1.0)
        n_eff = self.effective_trials(C)
        # streams may cover only a subset of the family: scale the shortfall as independent
        return float(min(n, n_eff + (n - M.shape[1])))

    def trial_sharpes_daily(self, family: str | None = None) -> pd.Series:
        recs = self._records()
        out = {}
        for i, r in enumerate(recs):
            if family is not None and r.get("family") != family:
                continue
            m = r.get("metrics") or {}
            if "sharpe_daily" in m:
                out[r["name"] + f"#{i}"] = float(m["sharpe_daily"])
            elif "sharpe_annual" in m:
                out[r["name"] + f"#{i}"] = deannualize_sharpe(float(m["sharpe_annual"]))
        return pd.Series(out, dtype=float)

    @staticmethod
    def effective_trials(correlation_matrix_of_trial_returns=None, n_trials: int | None = None) -> float:
        """Nyholt effective number of independent trials.

        If no correlation matrix is given, returns ``n_trials`` (assumes
        independence, the conservative choice).
        """
        if correlation_matrix_of_trial_returns is None:
            if n_trials is None:
                raise ValueError("need a correlation matrix or n_trials")
            return float(n_trials)
        C = np.asarray(correlation_matrix_of_trial_returns, dtype=float)
        M = C.shape[0]
        if M <= 1:
            return float(M)
        lam = np.linalg.eigvalsh((C + C.T) / 2)
        return float(1.0 + (M - 1) * (1.0 - np.var(lam, ddof=1) / M))

    def dsr_for(self, name: str, family: str | None = None, n_trials: float | None = None) -> dict:
        """DSR of the named trial using the ledger's count and cross-trial SR variance.

        ``family`` restricts the search set; ``n_trials`` overrides the count
        (e.g. with :meth:`effective_trials`).  The trial's ``metrics`` must
        hold ``n_obs`` and a Sharpe; ``skew``/``kurt`` default to 0/3.
        """
        recs = self._records()
        matches = [r for r in recs if r["name"] == name]
        if not matches:
            raise KeyError(name)
        r = matches[-1]
        fam = family if family is not None else r.get("family")
        srs = self.trial_sharpes_daily(fam)
        m = r["metrics"]
        sr_d = (
            float(m["sharpe_daily"]) if "sharpe_daily" in m else deannualize_sharpe(float(m["sharpe_annual"]))
        )
        n = int(m["n_obs"])
        skew = float(m.get("skew", 0.0))
        kurt = float(m.get("kurt", 3.0))
        N = float(n_trials) if n_trials is not None else float(len(srs))
        var_sr = float(srs.var(ddof=1)) if len(srs) > 1 else 0.0
        if not math.isfinite(var_sr):
            var_sr = 0.0
        dsr = deflated_sharpe_ratio(sr_d, max(N, 1.0), var_sr, n, skew, kurt)
        return {
            "name": name,
            "family": fam,
            "n_trials": N,
            "var_trials_sr": var_sr,
            "sr_daily": sr_d,
            "n_obs": n,
            "dsr": dsr,
        }


def _jsonable(d):
    if d is None:
        return {}
    out = {}
    for k, v in dict(d).items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (pd.Timestamp, datetime)):
            v = v.isoformat()
        out[str(k)] = v
    return out


def _jsonable_nested(d):
    """``_jsonable`` applied recursively (full dataclass configs)."""
    if d is None:
        return {}
    out = {}
    for k, v in _jsonable(d).items():
        if isinstance(v, dict):
            v = _jsonable_nested(v)
        elif isinstance(v, (list, tuple)):
            v = [x.item() if isinstance(x, (np.floating, np.integer)) else x for x in v]
        out[k] = v
    return out
