"""Run registry: the ``runs/<run_id>/`` directories written by ``qts.pipeline.save_run``
(+ ``qts.runs.artifacts.write_extra_artifacts``), parsed into the API's shapes.

Caching
-------
Each run directory is keyed by a *signature*: the ``(name, mtime_ns, size)`` of every
file in it.  The JSON files (manifest, summary, hurdles, extras, config) are parsed
once per signature and kept for every run (they are small); the Parquet frames
(``qts.pipeline.load_run``) are loaded lazily and kept in a small LRU.  A rewritten
or deleted file changes the signature, so the next request re-parses.

Older runs
----------
Runs written before ``extras.json`` existed still load: DSR moments and IC by year
are recomputed from ``returns.parquet`` / ``ic.parquet``; universe size and market
length are read from the recorded CLI command when it was ``qts.cli demo``; the
execution venue, the ex-ante vol series, factor exposures and the per-holding
alpha / specific vol / industry are ``null`` / empty (the UI shows empty states).
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from qts.pipeline import RunRecord, list_runs, load_run
from qts.runs.artifacts import EXTRAS_FILE, read_extras

from qts_api import schemas as S
from qts_api.services import series as SR

log = logging.getLogger(__name__)

FULL_CACHE_SIZE = 8
#: A run id is a directory name: starts with a letter or digit, then letters, digits,
#: ``.``, ``_`` or ``-``; at most 128 characters (``new_run_id`` gives 31).  Anything else
#: (``..``, ``_jobs``, ``.hidden``, 300 characters ...) is a 404, never a filesystem call.
RUN_ID_MAX_LEN = 128
RUN_ID_RE = re.compile(rf"^[A-Za-z0-9][A-Za-z0-9._-]{{0,{RUN_ID_MAX_LEN - 1}}}$")


def valid_run_id(run_id: object) -> bool:
    return isinstance(run_id, str) and bool(RUN_ID_RE.fullmatch(run_id)) and ".." not in run_id


GATE_NAMES = set(S.GateName.__args__)  # type: ignore[attr-defined]
DEMO_DEFAULTS = {"stocks": 300, "days": 1512}
#: Written by the API worker next to the run's artifacts (``qts_api.services.jobs``).
LAUNCH_FILE = "launch.json"
#: ``/api/runs/compare``: at most this many points on the shared date axis.
COMPARE_MAX_POINTS = 2000
#: ``/api/runs/compare``: at most this many runs per request.
COMPARE_MAX_RUNS = 8
#: What the known run-directory artifacts hold (``/api/runs/{id}/files``).
FILE_DESCRIPTIONS: dict[str, str] = {
    "manifest.json": "Run manifest: git sha, config and data hashes, library versions, timings, seeds",
    "summary.json": "Backtest summary, IC report, DSR inputs, cost sensitivity and the pipeline's notes",
    "hurdles.json": "Hurdle gates with values, thresholds and reasons",
    "config.yaml": "The SystemConfig the run used",
    "launch.json": "The API launch request that produced the run (data source, flags, base run, CLI equivalent)",
    "extras.json": "Run facts recorded at save time (label, universe size, venue, exposure roles)",
    "nav.parquet": "Net asset value by realisation date",
    "returns.parquet": "Gross and net daily returns",
    "weights.parquet": "Post-trade weights by decision date (fraction of NAV)",
    "trades.parquet": "Trades by decision date (fraction of NAV)",
    "costs.parquet": "Spread, impact, fees and borrow in USD by decision date",
    "ic.parquet": "Rank IC per formation date",
    "exante_vol.parquet": "The risk model's ex-ante portfolio volatility by decision date",
    "exposures.parquet": "Net book exposure to each style factor by decision date",
    "holdings.parquet": "Month-end positions with alpha, specific vol and industry",
    "alpha_refits.parquet": "Per-refit feature importance of every ML sleeve (long: refit date, sleeve, feature)",
    "alpha_refits.json": "The refits' training windows, blend weights and the full-sample feature IC table",
    "theme_weights.parquet": "The composite's theme weights by decision date",
    "alpha_ic_by_sleeve.parquet": "Daily rank IC of the combined, composite, ML and theme scores",
    "constraints.parquet": "Per-rebalance solve log: status, binding-constraint counts, vol-target reason, transfer coefficient",
    "report.txt": "The printed report card",
}
FILE_KINDS: dict[str, str] = {
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".parquet": "parquet",
    ".txt": "text",
    ".md": "text",
    ".log": "text",
}


def flatten_config(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """``{"portfolio": {"trade_aversion": 1.0}}`` -> ``{"portfolio.trade_aversion": 1.0}``.

    Scalars (bool / int / float / str / None) are leaves; non-finite floats become
    ``None``; lists and any other value are serialised as compact JSON so two configs
    that differ in a list still diff as one key."""
    out: dict[str, Any] = {}
    if not isinstance(cfg, dict):
        return out
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            if v:
                out.update(flatten_config(v, f"{key}."))
            else:
                out[key] = "{}"
        elif v is None or isinstance(v, (bool, int, str)):
            out[key] = v
        elif isinstance(v, float):
            out[key] = v if np.isfinite(v) else None
        else:
            try:
                out[key] = json.dumps(v, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                out[key] = str(v)
    return out


def thin_axis(n: int, max_points: int) -> list[int]:
    """Indices of at most ``max_points`` evenly spaced positions of ``range(n)``, always
    including the first and the last."""
    if n <= max_points:
        return list(range(n))
    step = (n - 1) / (max_points - 1)
    idx = sorted({round(i * step) for i in range(max_points)} | {0, n - 1})
    return [i for i in idx if 0 <= i < n]


class RunNotFound(LookupError):
    pass


class RunUnreadable(RuntimeError):
    pass


def _signature(path: Path) -> tuple[tuple[str, int, int], ...]:
    try:
        return tuple(sorted((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in path.iterdir() if f.is_file()))
    except FileNotFoundError:
        return ()


def _json(path: Path) -> Any:
    v = json.loads(path.read_text())
    return _nanify(v)


def _nanify(v: Any) -> Any:
    """``"nan"`` / ``"inf"`` strings written by ``qts.pipeline._json_safe`` -> float."""
    if isinstance(v, str) and v in ("nan", "inf", "-inf"):
        return float(v)
    if isinstance(v, dict):
        return {k: _nanify(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_nanify(x) for x in v]
    return v


def _num(v: Any) -> float:
    return SR.finite_or_nan(v)


def _opt_num(v: Any) -> float | None:
    if v is None:
        return None
    f = SR.finite_or_nan(v)
    return f if np.isfinite(f) else None


def _gate_value(v: Any) -> float | str | None:
    if v is None or (isinstance(v, str) and v not in ("nan", "inf", "-inf")):
        return v
    return _opt_num(v)


def _command_args(command: str) -> dict[str, Any]:
    """Facts recoverable from a recorded ``qts.cli demo`` command line."""
    try:
        argv = shlex.split(command or "")
    except ValueError:
        return {}
    if "demo" not in argv:
        return {"kind": "store"} if "run" in argv else {}
    out: dict[str, Any] = {"kind": "synthetic", **DEMO_DEFAULTS}
    for flag, key in (("--stocks", "stocks"), ("--days", "days"), ("--seed", "seed")):
        if flag in argv:
            i = argv.index(flag)
            try:
                out[key] = int(argv[i + 1])
            except (IndexError, ValueError):
                out.pop(key, None)
    return out


_SOURCE_NULLS: dict[str, Any] = {
    "stocks": None,
    "days": None,
    "store_path": None,
    "start": None,
    "end": None,
    "max_names": None,
    "min_mcap_ever": None,
    "max_days": None,
    "cache": False,
    "venue": None,
}


def _flag_value(argv: list[str], flag: str, cast: Any = str) -> Any:
    """Value of ``--flag V`` or ``--flag=V`` in an argv (``None`` when absent or malformed)."""
    for i, a in enumerate(argv):
        raw = None
        if a == flag and i + 1 < len(argv):
            raw = argv[i + 1]
        elif a.startswith(flag + "="):
            raw = a.split("=", 1)[1]
        if raw is not None:
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return None
    return None


def command_source(command: str) -> dict[str, Any] | None:
    """Data source and flags recoverable from a recorded ``qts demo`` / ``qts run``
    command line (``None`` when it is neither)."""
    try:
        argv = shlex.split(command or "")
    except ValueError:
        return None
    out = dict(_SOURCE_NULLS)
    out["max_days"] = _flag_value(argv, "--max-days", int)
    out["cache"] = "--cache" in argv
    if "demo" in argv:
        out.update(
            kind="synthetic",
            stocks=_flag_value(argv, "--stocks", int) or DEMO_DEFAULTS["stocks"],
            days=_flag_value(argv, "--days", int) or DEMO_DEFAULTS["days"],
        )
        return out
    if "run" in argv and _flag_value(argv, "--store") is not None:
        out.update(
            kind="store",
            store_path=_flag_value(argv, "--store"),
            start=_flag_value(argv, "--start"),
            end=_flag_value(argv, "--end"),
            max_names=_flag_value(argv, "--max-names", int),
            min_mcap_ever=_flag_value(argv, "--min-mcap-ever", float),
        )
        return out
    return None


def run_source(e: RunEntry) -> dict[str, Any]:
    """The run's :class:`~qts_api.schemas.RunSource`: ``launch.json`` (API jobs), else the
    manifest's CLI command, else ``recorded: false`` with what the run facts say."""
    command = str(e.manifest.get("command", "") or "")
    req = e.launch.get("request") if isinstance(e.launch.get("request"), dict) else None
    if req is not None:
        kind = "store" if req.get("kind") == "store" else "synthetic"
        synthetic = kind == "synthetic"
        return {
            "recorded": True,
            "origin": "api",
            "kind": kind,
            "stocks": req.get("stocks") if synthetic else None,
            "days": req.get("days") if synthetic else None,
            "store_path": None if synthetic else e.launch.get("store_path"),
            "start": None if synthetic else req.get("start"),
            "end": None if synthetic else req.get("end"),
            "max_names": None if synthetic else req.get("max_names"),
            "min_mcap_ever": None if synthetic else req.get("min_mcap_ever"),
            "max_days": req.get("max_days"),
            "cache": bool(req.get("cache", False)),
            "venue": req.get("venue") if req.get("venue") in ("routed", "auction", "continuous") else None,
            "command": str(e.launch.get("command") or command),
        }
    parsed = command_source(command)
    if parsed is not None:
        return {"recorded": True, "origin": "cli", **parsed, "command": command}
    synthetic = e.facts.get("kind", "synthetic") != "store"
    return {
        "recorded": False,
        "origin": "unknown",
        **_SOURCE_NULLS,
        "kind": "synthetic" if synthetic else "store",
        "stocks": e.facts.get("stocks") if synthetic else None,
        "days": e.facts.get("days") if synthetic else None,
        "command": command,
    }


@dataclass
class Frames:
    """The Parquet side of a run (``load_run``) plus the derived return-series frame."""

    record: RunRecord
    exante_vol: pd.Series | None
    exposures: pd.DataFrame | None
    holdings: pd.DataFrame | None
    alpha_refits: pd.DataFrame | None = None
    theme_weights: pd.DataFrame | None = None
    sleeve_ic: pd.DataFrame | None = None
    constraints: pd.DataFrame | None = None
    _series: pd.DataFrame | None = None

    @property
    def series(self) -> pd.DataFrame:
        if self._series is None:
            r = self.record
            self._series = SR.return_series_frame(r.nav, r.returns, r.trades, self.exante_vol, r.weights)
        return self._series


@dataclass
class RunEntry:
    run_id: str
    path: Path
    sig: tuple
    manifest: dict[str, Any]
    summary: dict[str, Any]
    hurdles: dict[str, Any]
    extras: dict[str, Any]
    config_yaml: str
    config: dict[str, Any]
    facts: dict[str, Any] = field(default_factory=dict)
    launch: dict[str, Any] = field(default_factory=dict)  # launch.json (API-launched runs)

    # ------------------------------------------------------------------ facts
    @property
    def backtest(self) -> dict[str, Any]:
        return self.summary.get("backtest", {}) or {}

    @property
    def use_ml(self) -> bool:
        return bool(self.summary.get("use_ml", False))

    @property
    def phase(self) -> int:
        return 2 if int(self.summary.get("phase", 1) or 1) == 2 else 1

    @property
    def signal_lag_days(self) -> int:
        return int(self.summary.get("signal_lag_days", 0) or 0)

    @property
    def seed(self) -> int:
        seeds = self.manifest.get("seeds") or {}
        if "system" in seeds:
            return int(seeds["system"])
        return int(self.facts.get("seed", self.config.get("seed", 0)) or 0)

    @property
    def horizon_days(self) -> int:
        return int((self.config.get("alpha") or {}).get("horizon_days", 20))

    @property
    def label(self) -> str:
        if self.facts.get("label"):
            return str(self.facts["label"])
        model = "Composite + ML sleeve" if self.use_ml else "Composite"
        parts = [model, f"seed {self.seed}"]
        if self.signal_lag_days == 0:
            parts.append("same-close execution (lag 0)")
        return ", ".join(parts)


class RunRegistry:
    def __init__(self, runs_root: Path):
        self.runs_root = Path(runs_root)
        self._entries: dict[str, RunEntry] = {}
        self._frames: OrderedDict[str, tuple[tuple, Frames]] = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ discovery
    def run_dirs(self) -> list[Path]:
        return [d for d in list_runs(self.runs_root) if valid_run_id(d.name)]

    def _parse(self, path: Path, sig: tuple) -> RunEntry:
        try:
            manifest = _json(path / "manifest.json")
            summary = _json(path / "summary.json")
            hurdles = _json(path / "hurdles.json")
            config_yaml = (path / "config.yaml").read_text() if (path / "config.yaml").exists() else ""
        except (OSError, ValueError) as e:
            raise RunUnreadable(f"{path.name}: {e}") from e
        try:
            config = yaml.safe_load(config_yaml) or {}
        except yaml.YAMLError:
            config = {}
        extras = read_extras(path) if (path / EXTRAS_FILE).exists() else {}
        try:
            launch = _json(path / LAUNCH_FILE) if (path / LAUNCH_FILE).exists() else {}
        except (OSError, ValueError):
            launch = {}
        facts = {**_command_args(str(manifest.get("command", ""))), **(extras.get("run") or {})}
        # The id is the directory name (unique, and what get() resolves); a copied run
        # directory keeps its manifest's run_id, which would duplicate list rows (audit 2 #12).
        if str(manifest.get("run_id", path.name)) != path.name:
            log.warning(
                "run directory %s has manifest run_id %r; listing it under the directory name",
                path.name,
                manifest.get("run_id"),
            )
        return RunEntry(
            run_id=path.name,
            path=path,
            sig=sig,
            manifest=manifest,
            summary=summary,
            hurdles=hurdles,
            extras=extras,
            config_yaml=config_yaml,
            config=config if isinstance(config, dict) else {},
            facts={k: v for k, v in facts.items() if v is not None},
            launch=launch if isinstance(launch, dict) else {},
        )

    def _entry_for(self, path: Path) -> RunEntry:
        sig = _signature(path)
        with self._lock:
            cur = self._entries.get(path.name)
            if cur is not None and cur.sig == sig:
                return cur
        entry = self._parse(path, sig)
        with self._lock:
            self._entries[path.name] = entry
            self._frames.pop(path.name, None)
        return entry

    def entries(self) -> list[RunEntry]:
        out = []
        dirs = self.run_dirs()
        names = {d.name for d in dirs}
        for d in dirs:
            try:
                out.append(self._entry_for(d))
            except RunUnreadable as e:  # half-written or damaged: skip, keep serving the rest
                log.warning("skipping run directory: %s", e)
        with self._lock:
            for gone in set(self._entries) - names:
                self._entries.pop(gone, None)
                self._frames.pop(gone, None)
        return out

    def get(self, run_id: str) -> RunEntry:
        if not valid_run_id(run_id):
            raise RunNotFound(str(run_id)[:RUN_ID_MAX_LEN])
        path = self.runs_root / run_id
        if not (path / "manifest.json").exists():
            raise RunNotFound(run_id)
        return self._entry_for(path)

    def exists(self, run_id: str) -> bool:
        try:
            self.get(run_id)
        except (RunNotFound, RunUnreadable):
            return False
        return True

    def frames(self, entry: RunEntry) -> Frames:
        with self._lock:
            hit = self._frames.get(entry.run_id)
            if hit is not None and hit[0] == entry.sig:
                self._frames.move_to_end(entry.run_id)
                return hit[1]
        fr = self._load_frames(entry)
        with self._lock:
            self._frames[entry.run_id] = (entry.sig, fr)
            self._frames.move_to_end(entry.run_id)
            while len(self._frames) > FULL_CACHE_SIZE:
                self._frames.popitem(last=False)
        return fr

    def _load_frames(self, entry: RunEntry) -> Frames:
        p = entry.path
        try:
            rec = load_run(p)
        except (KeyError, TypeError, ValueError) as e:
            # e.g. a config.yaml written by an older SystemConfig (strict from_dict): the
            # frames are still readable, only the typed config is not.
            log.info("load_run(%s) failed (%s); reading the frames directly", p.name, e)
            rec = self._load_frames_loose(entry)

        def opt(name: str) -> pd.DataFrame | None:
            f = p / name
            return pd.read_parquet(f) if f.exists() else None

        ev = opt("exante_vol.parquet")
        return Frames(
            record=rec,
            exante_vol=None if ev is None or "exante_vol" not in ev else ev["exante_vol"],
            exposures=opt("exposures.parquet"),
            holdings=opt("holdings.parquet"),
            alpha_refits=opt("alpha_refits.parquet"),
            theme_weights=opt("theme_weights.parquet"),
            sleeve_ic=opt("alpha_ic_by_sleeve.parquet"),
            constraints=opt("constraints.parquet"),
        )

    @staticmethod
    def _load_frames_loose(entry: RunEntry) -> RunRecord:
        from qts.core.config import SystemConfig
        from qts.core.manifest import RunManifest
        from qts.validation import HurdleResult

        p = entry.path

        def pq(name: str) -> pd.DataFrame:
            f = p / name
            return pd.read_parquet(f) if f.exists() else pd.DataFrame()

        nav, ic = pq("nav.parquet"), pq("ic.parquet")
        return RunRecord(
            run_dir=p,
            manifest=RunManifest.load(p),
            config=SystemConfig(),
            summary=entry.summary,
            hurdles=HurdleResult.from_dict(entry.hurdles),
            nav=nav["nav"] if "nav" in nav else pd.Series(dtype=float, name="nav"),
            returns=pq("returns.parquet"),
            weights=pq("weights.parquet"),
            trades=pq("trades.parquet"),
            costs=pq("costs.parquet"),
            ic=ic["rank_ic"] if "rank_ic" in ic else pd.Series(dtype=float, name="rank_ic"),
        )

    # ------------------------------------------------------------------ shapes
    def overall(self, e: RunEntry) -> str:
        ov = e.hurdles.get("overall", e.summary.get("overall"))
        return "PASS" if bool(ov) else "FAIL"

    def run_summary(self, e: RunEntry) -> dict[str, Any]:
        b = e.backtest
        return {
            "id": e.run_id,
            "created_at": str(e.manifest.get("created_at", "")),
            "label": e.label,
            "kind": e.facts.get("kind", "synthetic"),
            "phase": e.phase,
            "use_ml": e.use_ml,
            "config_hash": str(e.manifest.get("config_hash", e.summary.get("config_hash", ""))),
            "data_hash": str(e.manifest.get("data_hash", e.summary.get("data_hash", ""))),
            "ledger_family": str(e.manifest["ledger_family"]) if e.manifest.get("ledger_family") else None,
            "git_sha": str(e.manifest.get("git_sha", "unknown")),
            "status": "done",
            "headline": {
                "sharpe_net": _num(b.get("sharpe_net")),
                "sharpe_gross": _num(b.get("sharpe_gross")),
                "cagr_net": _num(b.get("cagr_net")),
                "max_drawdown": _num(b.get("max_drawdown_net")),
                "avg_cost_bps_per_side": _num(b.get("avg_cost_bps_per_side")),
                "dsr": _num(e.summary.get("dsr")),
                "invested_day_frac": _num(b.get("invested_day_frac", 1.0)),
                "overall_gate": self.overall(e),
            },
            "config_leaves": flatten_config(e.config),
        }

    def list_summaries(self) -> list[dict[str, Any]]:
        return [self.run_summary(e) for e in self.entries()]

    def _hurdles(self, e: RunEntry) -> dict[str, Any]:
        h = e.hurdles
        gates = []
        for g in h.get("gates", []):
            name = g.get("name")
            if name not in GATE_NAMES:
                log.warning("run %s: unknown gate %r omitted", e.run_id, name)
                continue
            status = g.get("status")
            if status not in ("PASS", "FAIL", "skipped"):
                passed = g.get("passed")
                status = "skipped" if passed is None else ("PASS" if passed else "FAIL")
            gates.append(
                {
                    "name": name,
                    "status": status,
                    "value": _gate_value(g.get("value")),
                    "threshold": _gate_value(g.get("threshold")),
                    "reason": str(g.get("reason", "")),
                    "required": bool(g.get("required", False)),
                }
            )
        return {
            "overall": self.overall(e),
            "phase": 2 if int(h.get("phase", e.phase)) == 2 else 1,
            "required": [r for r in h.get("required", []) if r in GATE_NAMES],
            "gates": gates,
        }

    def _manifest(self, e: RunEntry) -> dict[str, Any]:
        m = e.manifest
        return {
            "run_id": str(m.get("run_id", e.run_id)),
            "created_at": str(m.get("created_at", "")),
            "git_sha": str(m.get("git_sha", "unknown")),
            "git_dirty": m.get("git_dirty"),
            "config_hash": str(m.get("config_hash", "")),
            "data_hash": str(m.get("data_hash", "")),
            "ledger_family": str(m["ledger_family"]) if m.get("ledger_family") else None,
            "python": str(m.get("python_version", m.get("python", ""))),
            "platform": str(m.get("platform", "")),
            "versions": {str(k): str(v) for k, v in (m.get("versions") or {}).items()},
            "timings": {str(k): _num(v) for k, v in (m.get("timings") or {}).items()},
            "seeds": {str(k): int(v) for k, v in (m.get("seeds") or {}).items()},
        }

    def _ic_report(self, e: RunEntry) -> dict[str, Any]:
        ic = e.summary.get("ic") or {}
        by_year = e.extras.get("ic_by_year")
        if not isinstance(by_year, list) or not by_year:
            by_year = SR.ic_by_year(self.frames(e).record.ic)
        return {
            "mean_ic": _num(ic.get("mean_ic")),
            "ic_sd": _num(ic.get("ic_sd")),
            "icir": _num(ic.get("icir")),
            "t_stat": _num(ic.get("t_stat")),
            "hit_rate": _num(ic.get("hit_rate")),
            "periods_per_year": _num(ic.get("periods_per_year")),
            "n": int(ic.get("n", 0) or 0),
            "horizon_days": e.horizon_days,
            "by_year": [
                {"year": int(y["year"]), "mean_ic": _num(y.get("mean_ic")), "n": int(y.get("n", 0))} for y in by_year
            ],
        }

    def _dsr_inputs(self, e: RunEntry) -> dict[str, Any]:
        d = e.summary.get("dsr_inputs") or {}
        x = e.extras.get("dsr_inputs") or {}
        if not {"skew", "kurtosis", "n_obs"} <= set(x):
            r = self.frames(e).record.returns
            net = r["net"] if "net" in r else pd.Series(dtype=float)
            sk, ku, n = SR.moments(net)
            x = {"skew": sk, "kurtosis": ku, "n_obs": n}
        return {
            "n_trials": _num(d.get("n_trials", 1.0)),
            "var_trials_sr_annual": _num(d.get("var_trials_sr_annual", 0.0)),
            "skew": _num(x.get("skew")),
            "kurtosis": _num(x.get("kurtosis")),
            "n_obs": int(x.get("n_obs", 0) or 0),
        }

    @staticmethod
    def _fold_label(start: str, end: str) -> str:
        return f"{str(start)[:7]}–{str(end)[:7]}"

    def detail(self, run_id: str) -> dict[str, Any]:
        e = self.get(run_id)
        b = e.backtest
        summary = {k: b.get(k) for k in S.BacktestSummary.model_fields}
        for k in ("start", "end"):
            summary[k] = str(summary.get(k) or "")
        for k in ("n_days", "n_delistings", "n_failed_solves"):
            summary[k] = int(summary.get(k) or 0)
        for k, v in summary.items():
            if k not in ("start", "end", "n_days", "n_delistings", "n_failed_solves"):
                summary[k] = _num(v)
        dec = e.summary.get("decile")
        folds = e.summary.get("folds") or []
        return {
            "id": e.run_id,
            "label": e.label,
            "status": "done",
            "summary": summary,
            "manifest": self._manifest(e),
            "config": {
                "kind": e.facts.get("kind", "synthetic"),
                "use_ml": e.use_ml,
                "phase": e.phase,
                "stocks": e.facts.get("stocks"),
                "days": e.facts.get("days"),
                "seed": e.seed,
                "signal_lag_days": e.signal_lag_days,
                "config_yaml": e.config_yaml,
                "source": run_source(e),
            },
            "hurdles": self._hurdles(e),
            "ic_report": self._ic_report(e),
            "decile": None
            if not isinstance(dec, dict)
            else {"mean": _num(dec.get("mean")), "sharpe": _num(dec.get("sharpe")), "t_stat": _num(dec.get("t_stat"))},
            "dsr": _num(e.summary.get("dsr")),
            "dsr_inputs": self._dsr_inputs(e),
            "pbo": _opt_num(e.summary.get("pbo")),
            "fold_sharpes": [
                {"label": self._fold_label(f.get("start", ""), f.get("end", "")), "sharpe_net": _num(f.get("sharpe"))}
                for f in folds
            ],
            "execution_venue": e.facts.get("execution_venue"),
            "notes": [str(n) for n in e.summary.get("notes", [])],
        }

    def compare(self, run_ids: list[str], max_points: int = COMPARE_MAX_POINTS) -> dict[str, Any]:
        """Side-by-side payload: NAV of every run on the union of their dates (thinned to
        ``max_points``), fold Sharpes, hurdles and flattened configs."""
        entries = [self.get(r) for r in run_ids]
        navs: dict[str, pd.Series] = {}
        for e in entries:
            df = self.frames(e).series
            nav = df["nav"] if "nav" in df else pd.Series(dtype=float)
            navs[e.run_id] = nav[~nav.index.duplicated()].sort_index()
        union = pd.DatetimeIndex(sorted(set().union(*(set(v.index) for v in navs.values()))))
        keep = thin_axis(len(union), max_points)
        axis = union[keep]
        runs = []
        for e in entries:
            s = self.run_summary(e)
            b = e.backtest
            folds = e.summary.get("folds") or []
            runs.append(
                {
                    "id": e.run_id,
                    "label": e.label,
                    "created_at": s["created_at"],
                    "phase": e.phase,
                    "start": str(b.get("start") or ""),
                    "end": str(b.get("end") or ""),
                    "headline": s["headline"],
                    "n_trials": _num((e.summary.get("dsr_inputs") or {}).get("n_trials", 1.0)),
                    "fold_sharpes": [
                        {
                            "label": self._fold_label(f.get("start", ""), f.get("end", "")),
                            "sharpe_net": _num(f.get("sharpe")),
                        }
                        for f in folds
                    ],
                    "hurdles": self._hurdles(e),
                    "config_leaves": s["config_leaves"],
                }
            )
        return {
            "dates": SR.iso_dates(axis),
            "nav": {rid: SR.clean(v.reindex(axis)) for rid, v in navs.items()},
            "runs": runs,
            "downsampled": len(keep) < len(union),
        }

    def series(self, run_id: str, names: list[str]) -> dict[str, Any]:
        e = self.get(run_id)
        df = self.frames(e).series
        return {
            "dates": SR.iso_dates(df.index),
            "series": {n: SR.clean(df[n]) if n in df else [None] * len(df) for n in names},
        }

    def costs(self, run_id: str) -> dict[str, Any]:
        e = self.get(run_id)
        fr = self.frames(e)
        r = fr.record
        c = r.costs
        idx = pd.DatetimeIndex(c.index) if len(c) else pd.DatetimeIndex([])
        nav_pre = r.nav.reindex(idx) if len(r.nav) else pd.Series(np.nan, index=idx)
        s = fr.series
        sens = SR.cost_sensitivity_rows(
            e.summary.get("cost_sensitivity"),
            SR.gross_after_borrow(r.returns, c, nav_pre, s.get("gross")),
            s.get("turnover"),
            _num(e.backtest.get("avg_cost_bps_per_side")),
        )
        return {
            "daily": {
                "dates": SR.iso_dates(idx),
                "series": {k: SR.clean(c[k]) if k in c else [None] * len(idx) for k in S.COST_COMPONENTS},
            },
            "nav": SR.clean(nav_pre),
            "cost_sensitivity": sens,
        }

    def exposures(self, run_id: str) -> dict[str, Any]:
        e = self.get(run_id)
        x = self.frames(e).exposures
        if x is None or not len(x):
            return {"factors": [], "roles": {}, "history": {"dates": [], "series": {}}, "latest": None}
        from qts.runs.artifacts import _role

        stored = e.extras.get("exposure_roles") or {}
        factors = [str(c) for c in x.columns]
        roles = {f: stored.get(f, _role(f)) for f in factors}
        last = x.iloc[-1]
        return {
            "factors": factors,
            "roles": roles,
            "history": {"dates": SR.iso_dates(x.index), "series": {f: SR.clean(x[f]) for f in factors}},
            "latest": {
                "date": SR.iso_dates([x.index[-1]])[0],
                "rows": [
                    {"factor": f, "exposure": _num(last[f]), "role": roles[f]}
                    for f in factors
                    if np.isfinite(_num(last[f]))
                ],
            },
        }

    def holdings(self, run_id: str, date: str | None) -> dict[str, Any]:
        """Positions at a month-end decision date (default: the latest available)."""
        e = self.get(run_id)
        fr = self.frames(e)
        h = fr.holdings
        if h is not None and len(h):
            h = h.assign(date=pd.to_datetime(h["date"]))
            available = sorted(h["date"].unique())
        else:
            W = fr.record.weights
            pos = SR.month_end_positions(list(W.index))
            available = [W.index[i] for i in pos][-60:]
        avail_iso = SR.iso_dates(available)
        if not avail_iso:
            raise RunNotFound(f"run {run_id} has no positions")
        chosen = date or avail_iso[-1]
        if chosen not in avail_iso:
            raise KeyError(chosen)
        d = pd.Timestamp(chosen)
        positions: list[dict[str, Any]] = []
        if h is not None and len(h):
            rows = h[h["date"] == d]
            for r in rows.itertuples(index=False):
                positions.append(
                    {
                        "ticker": str(r.ticker),
                        "weight": float(r.weight),
                        "alpha": _opt_num(r.alpha),
                        "specific_vol": _opt_num(r.specific_vol),
                        "industry": None if pd.isna(r.industry) else str(r.industry),
                    }
                )
        else:
            w = fr.record.weights.loc[d]
            w = w[w.abs() > 1e-8]
            positions = [
                {"ticker": str(t), "weight": float(v), "alpha": None, "specific_vol": None, "industry": None}
                for t, v in w.items()
            ]
        positions.sort(key=lambda p: -p["weight"])
        return {"date": chosen, "available_dates": avail_iso, "positions": positions}

    def files(self, run_id: str) -> dict[str, Any]:
        """The regular files directly inside the run directory, by name."""
        e = self.get(run_id)
        out = []
        for f in sorted(e.path.iterdir(), key=lambda x: x.name):
            if not f.is_file() or f.name.startswith("."):
                continue
            st = f.stat()
            out.append(
                {
                    "name": f.name,
                    "size_bytes": int(st.st_size),
                    "modified_at": pd.Timestamp(st.st_mtime, unit="s", tz="UTC").isoformat(timespec="seconds"),
                    "kind": FILE_KINDS.get(f.suffix.lower(), "other"),
                    "description": FILE_DESCRIPTIONS.get(f.name, ""),
                    "download_url": f"/api/runs/{run_id}/files/{f.name}",
                }
            )
        return {"run_id": run_id, "files": out}

    def file_path(self, run_id: str, name: str) -> Path:
        """Path of one file of the run directory; ``RunNotFound`` for anything that is not a
        plain file name listed by :meth:`files` (no separators, no ``..``, no hidden files)."""
        e = self.get(run_id)
        if not name or "/" in name or "\\" in name or name.startswith(".") or ".." in name:
            raise RunNotFound(f"{run_id}/{name}")
        p = e.path / name
        if not p.is_file() or p.resolve().parent != e.path.resolve():
            raise RunNotFound(f"{run_id}/{name}")
        return p

    # ------------------------------------------------------------------ labs
    #: Sleeve rows of ``/alpha/ic-by-year`` in display order (themes follow, alphabetically).
    SLEEVE_ORDER = ("combined", "composite", "ml")
    THEME_WEIGHTS_MAX_POINTS = 2000

    def _refits_meta(self, e: RunEntry) -> dict[str, Any] | None:
        f = e.path / "alpha_refits.json"
        if not f.exists():
            return None
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            log.warning("unreadable %s", f)
            return None
        return d if isinstance(d, dict) else None

    def alpha_refits(self, run_id: str) -> dict[str, Any]:
        e = self.get(run_id)
        meta = self._refits_meta(e)
        long = self.frames(e).alpha_refits
        refits_in = list((meta or {}).get("refits") or [])
        refits = [
            {
                "refit_date": str(r.get("refit_date", "")),
                "train_start": str(r.get("train_start", "")),
                "train_end": str(r.get("train_end", "")),
                "valid_start": None if r.get("valid_start") is None else str(r["valid_start"]),
                "pred_end": str(r.get("pred_end", "")),
                "n_train": int(r.get("n_train", 0) or 0),
                "n_valid": int(r.get("n_valid", 0) or 0),
                "blend_weights": {str(k): _num(v) for k, v in (r.get("blend_weights") or {}).items()},
            }
            for r in refits_in
            if isinstance(r, dict)
        ]
        dates = [r["refit_date"] for r in refits]
        sleeves: list[str] = []
        features: list[str] = []
        importance: dict[str, dict[str, list[float | None]]] = {}
        if long is not None and len(long) and {"refit_date", "sleeve", "feature", "importance"} <= set(long.columns):
            df = long.assign(refit_date=pd.to_datetime(long["refit_date"]).dt.strftime("%Y-%m-%d"))
            if not dates:  # metadata missing: the refit dates are the table's
                dates = sorted(df["refit_date"].unique())
            present = list(dict.fromkeys(str(s) for s in df["sleeve"]))
            sleeves = [s for s in ("lgbm", "ridge", "mlp", "ensemble") if s in present] + [
                s for s in present if s not in ("lgbm", "ridge", "mlp", "ensemble")
            ]
            features = sorted({str(f) for f in df["feature"]})
            pos = {d: i for i, d in enumerate(dates)}
            for sl in sleeves:
                sub = df[df["sleeve"] == sl]
                table: dict[str, list[float | None]] = {f: [None] * len(dates) for f in features}
                for rd, feat, val in zip(sub["refit_date"], sub["feature"], sub["importance"], strict=True):
                    i = pos.get(str(rd))
                    if i is not None:
                        table[str(feat)][i] = _opt_num(val)
                importance[sl] = table
        feature_ic = [
            {
                "feature": str(f.get("feature", "")),
                "sign": _num(f.get("sign", 1.0)),
                "theme": None if not f.get("theme") else str(f["theme"]),
                "mean_ic": _opt_num(f.get("mean_ic")),
                "t_stat": _opt_num(f.get("t_stat")),
                "n": int(f.get("n", 0) or 0),
            }
            for f in ((meta or {}).get("feature_ic") or [])
            if isinstance(f, dict)
        ]
        return {
            "refits": refits,
            "sleeves": sleeves,
            "features": features,
            "importance": importance,
            "feature_ic": feature_ic,
            "embargo_days": int((meta or {}).get("embargo_days", 0) or 0),
            "refit_frequency_days": int(
                (meta or {}).get("refit_frequency_days", (e.config.get("alpha") or {}).get("refit_frequency_days", 0))
                or 0
            ),
            "notes": [str(n) for n in ((meta or {}).get("notes") or [])],
            "recorded": meta is not None,
        }

    def theme_weights(self, run_id: str) -> dict[str, Any]:
        e = self.get(run_id)
        acfg = e.config.get("alpha") or {}
        static = {str(k): _num(v) for k, v in (acfg.get("theme_weights") or {}).items()}
        dynamic = bool(acfg.get("dynamic_weights", False))
        W = self.frames(e).theme_weights
        if W is None or not len(W):
            return {"dates": [], "series": {}, "themes": [], "static": static, "dynamic": dynamic, "downsampled": False}
        keep = thin_axis(len(W), self.THEME_WEIGHTS_MAX_POINTS)
        sub = W.iloc[keep]
        themes = [str(c) for c in W.columns]
        return {
            "dates": SR.iso_dates(sub.index),
            "series": {t: SR.clean(sub[t]) for t in themes},
            "themes": themes,
            "static": static,
            "dynamic": dynamic,
            "downsampled": len(keep) < len(W),
        }

    def ic_by_year(self, run_id: str) -> dict[str, Any]:
        """Mean rank IC and t per sleeve per calendar year: from the daily per-sleeve IC
        (Newey-West t, overlap = horizon + lag) when recorded, else the sampled combined IC."""
        e = self.get(run_id)
        fr = self.frames(e)
        overlap = e.horizon_days + e.signal_lag_days
        si = fr.sleeve_ic
        if si is not None and len(si) and "combined" in si.columns:
            cols = [c for c in self.SLEEVE_ORDER if c in si.columns] + sorted(
                str(c) for c in si.columns if str(c).startswith("theme_")
            )
            frame = si[cols].astype(float)
            source = "daily_sleeves"
        else:
            frame = fr.record.ic.astype(float).to_frame("combined")
            source = "sampled_combined"
            overlap = 1
        idx = pd.DatetimeIndex(frame.index)
        years = sorted({int(y) for y in idx.year})
        rows = []
        for col in frame.columns:
            s = frame[col]
            cells = []
            for y in years:
                v = s[idx.year == y].dropna()
                n = len(v)
                cells.append(
                    {
                        "year": y,
                        "mean_ic": float(v.mean()) if n else None,
                        "t_stat": SR.ic_t_stat(v, overlap) if n else None,
                        "n": n,
                    }
                )
            rows.append({"sleeve": str(col), "cells": cells})
        return {"years": years, "rows": rows, "source": source, "overlap_days": int(overlap)}

    def constraints(self, run_id: str) -> dict[str, Any]:
        e = self.get(run_id)
        cs = self.frames(e).constraints
        empty = {
            "dates": [],
            "names": [],
            "binding": {},
            "status": [],
            "vol_target_binding": [],
            "transfer_coefficient": [],
            "n_traded": [],
            "n_solves": [],
            "share_binding": {},
            "vol_target_reasons": [],
            "n_rebalances": 0,
        }
        if cs is None or not len(cs) or "date" not in cs.columns:
            return empty
        df = cs.sort_values("date")
        n = len(df)
        bind_cols = [c for c in df.columns if str(c).startswith("binding_")]
        names = sorted(str(c)[len("binding_") :] for c in bind_cols)
        binding = {nm: [int(v) for v in df[f"binding_{nm}"].fillna(0)] for nm in names}
        share = {nm: float((df[f"binding_{nm}"].fillna(0) > 0).mean()) for nm in names}
        vtb = [str(v) if isinstance(v, str) else "" for v in df.get("vol_target_binding", pd.Series([""] * n))]
        return {
            "dates": SR.iso_dates(pd.to_datetime(df["date"])),
            "names": names,
            "binding": binding,
            "status": [str(v) for v in df.get("status", pd.Series([""] * n))],
            "vol_target_binding": vtb,
            "transfer_coefficient": SR.clean(df.get("transfer_coefficient", pd.Series([np.nan] * n))),
            "n_traded": [int(v) for v in df.get("n_traded", pd.Series([0] * n)).fillna(0)],
            "n_solves": [int(v) for v in df.get("n_solves", pd.Series([0] * n)).fillna(0)],
            "share_binding": share,
            "vol_target_reasons": sorted({v for v in vtb if v}),
            "n_rebalances": n,
        }

    def ic(self, run_id: str) -> dict[str, Any]:
        e = self.get(run_id)
        ic = self.frames(e).record.ic
        decay = e.summary.get("ic_decay") or {}
        pts = []
        for k, v in decay.items():
            try:
                pts.append({"horizon_days": int(float(k)), "mean_ic": _num(v)})
            except (TypeError, ValueError):
                continue
        pts.sort(key=lambda p: p["horizon_days"])
        return {
            "ic": {"dates": SR.iso_dates(ic.index), "series": {"rank_ic": SR.clean(ic)}},
            "decay": pts,
            "horizon_days": e.horizon_days,
        }
