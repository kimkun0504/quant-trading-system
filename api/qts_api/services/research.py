"""Read side of the research tools and the paper books (results written by the library,
the ledger, the seal), and the request -> task-parameter mapping of the launch routes.

Results are the library's own files, read as written: ``runs/_signals/*.json``,
``runs/_sweeps/*.json``, ``runs/_capacity/*.json``, ``runs/_risk/*.json``,
``runs/_checks/midcap_*.json`` and ``<run>/midcap_check.json`` (a check run on another
machine and copied over is listed the same way).  Nothing here computes a statistic the
library does not: the trial impact is :func:`qts.research.sweep.trial_impact`, the Holm
adjustment :func:`qts.research.signals.holm_adjust`, the family a store slice logs into
the cached :func:`qts.research.data.full_history_market_hash`.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from qts_api.settings import Settings

SIGNALS_DIR, SWEEPS_DIR, CAPACITY_DIR, RISK_DIR, CHECKS_DIR = "_signals", "_sweeps", "_capacity", "_risk", "_checks"
RESULT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SWEEP_ID_RE = re.compile(r"^sw_[0-9a-f]{12}$")
BOOK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
#: What a user types to force a second mid-cap check.
MIDCAP_FORCE_PHRASE = "not out-of-sample"


class NotFound(LookupError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _results(root: Path, sub: str) -> list[tuple[Path, dict[str, Any]]]:
    d = root / sub
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.json"), reverse=True):
        x = read_json(f)
        if isinstance(x, dict):
            out.append((f, x))
    return out


def result_file(root: Path, sub: str, rid: str) -> dict[str, Any]:
    if not isinstance(rid, str) or not RESULT_ID_RE.fullmatch(rid) or ".." in rid:
        raise NotFound(rid)
    x = read_json(root / sub / f"{rid}.json")
    if not isinstance(x, dict):
        raise NotFound(rid)
    return x


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ----------------------------------------------------------------------------- seal
def seal_view(settings: Settings) -> dict[str, Any]:
    from qts.research.seal import seal_status

    st = seal_status(str(settings.ledger)).to_dict()
    seal = st.get("seal")
    return {
        **st,
        "data_through": None if seal is None else seal.get("load_end"),
        "n_forced": sum(1 for u in st.get("unseals") or [] if u.get("forced")),
    }


def unseal_check(settings: Settings, config_hash: str, phase: int) -> dict[str, Any]:
    from qts.research.seal import load_seal, previous_unseals

    prev = previous_unseals(str(settings.ledger), config_hash, phase)
    keys = ("ts", "config_hash", "family", "phase", "unseal_reason", "forced", "n_previous_unseals")
    return {
        "config_hash": config_hash,
        "phase": phase,
        "sealed": load_seal() is not None,
        "previous": [{k: r.get(k) for k in keys} for r in prev],
        "requires_force": bool(prev),
    }


# ----------------------------------------------------------------------------- families
def store_spec_dict(settings: Settings, store: dict[str, Any], default_max_rank: int | None = None) -> dict[str, Any]:
    """The StoreSpec dict of a request's store slice (``QTS_STORE_PATH`` as the store)."""
    d = {k: v for k, v in store.items() if v is not None}
    if default_max_rank is not None and not any(k in d for k in ("max_names", "min_mcap_ever", "max_rank")):
        d["max_rank"] = default_max_rank
    if settings.store_path is not None:
        d["store"] = str(settings.store_path)
    return d


def resolve_family(settings: Settings, store: dict[str, Any]) -> dict[str, Any]:
    """The full-history market hash of a store slice, when cached (no store read)."""
    from qts.research.data import StoreSpec, cached_full_history_market_hash

    if settings.store_path is None:
        return {"known": False, "market_hash_full": None, "reason": "QTS_STORE_PATH is not set on the API server"}
    try:
        spec = StoreSpec.from_dict(store_spec_dict(settings, store))
    except ValueError as e:
        return {"known": False, "market_hash_full": None, "reason": str(e)}
    mh = cached_full_history_market_hash(spec, str(settings.runs_root / "_cache" / "market_hash"))
    if mh is None:
        return {
            "known": False,
            "market_hash_full": None,
            "reason": "The trial family of these load flags is set by a hash of the full-history close panel, "
            "which has not been computed for this store snapshot yet. Resolving it reads the close panel "
            "once (about a minute on the full store) and is cached afterwards.",
        }
    return {"known": True, "market_hash_full": mh, "reason": ""}


def trial_impact(settings: Settings, family: str, n_new: int, n_reference: int = 0) -> dict[str, Any]:
    from dataclasses import asdict

    from qts.research.common import json_safe
    from qts.research.sweep import trial_impact as _impact

    return json_safe(asdict(_impact(str(settings.ledger), family, int(n_new), int(n_reference))))


# ----------------------------------------------------------------------------- features
def feature_list(feature_set: str) -> dict[str, Any]:
    from qts.alpha.composite import THEMES
    from qts.alpha.feature_registry import feature_info

    rows = feature_info(feature_set)
    names = {r.name for r in rows}
    themes = [{"theme": str(t), "members": [f for f, _ in members if f in names]} for t, members in THEMES.items()]
    return {
        "feature_set": feature_set,
        "features": [r.to_dict() for r in rows],
        "themes": [t for t in themes if t["members"]],
    }


# ----------------------------------------------------------------------------- signals
def _family_tests(settings: Settings, family: str) -> list[dict[str, Any]]:
    from qts.validation import TrialLedger

    if not settings.ledger.exists():
        return []
    out = []
    for r in TrialLedger(str(settings.ledger)).records(family):
        m = r.get("metrics") or {}
        out.append(
            {
                "signal": (r.get("config") or {}).get("signal"),
                "t": _f(m.get("t_primary")),
                "ts": r.get("ts"),
                "id": r.get("signal_test_id"),
            }
        )
    return out


def family_now(settings: Settings, family: str, test_id: str | None = None, hurdle: float = 3.0) -> dict[str, Any]:
    """Holm over every test logged in ``family`` now (``qts.research.signals.holm_adjust``
    of the two-sided normal p of each test's primary t)."""
    from qts.research.signals import holm_adjust, two_sided_p

    tests = _family_tests(settings, family)
    ps = [two_sided_p(float("nan") if t["t"] is None else float(t["t"])) for t in tests]
    adj = holm_adjust(ps) if tests else []
    rows = [{**t, "p": _f(p), "p_holm": _f(a)} for t, p, a in zip(tests, ps, adj, strict=True)]
    me = next((r for r in rows if test_id is not None and r.get("id") == test_id), None)
    return {
        "family": family,
        "n_tests": len(rows),
        "hurdle": float(hurdle),
        "tests": rows,
        "p_holm": None if me is None else me["p_holm"],
    }


def signal_tests(settings: Settings) -> list[dict[str, Any]]:
    out = []
    fam_cache: dict[str, dict[str, Any]] = {}
    for _, d in _results(settings.runs_root, SIGNALS_DIR):
        tr = d.get("trial") or {}
        fam = str(tr.get("family") or "")
        if fam not in fam_cache:
            fam_cache[fam] = family_now(settings, fam, hurdle=float(tr.get("hurdle") or 3.0))
        now = fam_cache[fam]
        me = next((r for r in now["tests"] if r.get("id") == d.get("id")), None)
        sig = d.get("signal") or {}
        prim = str(d.get("primary_horizon"))
        out.append(
            {
                "id": str(d.get("id")),
                "signal": str(sig.get("name", "")),
                "theme": str(sig.get("theme", "")),
                "source": str(sig.get("source", "")),
                "primary_horizon": int(d.get("primary_horizon") or 0),
                "t": _f(tr.get("t")),
                "p": _f(tr.get("p")),
                "p_holm_at_test": _f(tr.get("p_holm")),
                "p_holm_now": None if me is None else me["p_holm"],
                "n_tests_now": int(now["n_tests"]),
                "passes_hurdle": bool(tr.get("passes_hurdle")),
                "logged": bool(tr.get("logged")),
                "family": fam,
                "window": [str(x) for x in d.get("window") or []],
                "sealed_end": (d.get("baseline") or {}).get("sealed_end"),
                "mean_ic_primary": _f(((d.get("ic_by_horizon") or {}).get(prim) or {}).get("mean_ic")),
                "sharpe_net_paper": _f((d.get("paper") or {}).get("sharpe_net")),
            }
        )
    return out


def signal_test_detail(settings: Settings, rid: str) -> dict[str, Any]:
    d = result_file(settings.runs_root, SIGNALS_DIR, rid)
    tr = d.get("trial") or {}
    return {
        "result": d,
        "family_now": family_now(settings, str(tr.get("family") or ""), d.get("id"), float(tr.get("hurdle") or 3.0)),
    }


# ----------------------------------------------------------------------------- sweeps
def sweep_summaries(settings: Settings) -> list[dict[str, Any]]:
    out = []
    for _, d in _results(settings.runs_root, SWEEPS_DIR):
        members = d.get("members") or []
        sharpes = [_f((m.get("metrics") or {}).get("sharpe_net")) for m in members]
        sharpes = [x for x in sharpes if x is not None]
        pbo = d.get("pbo") or {}
        imp = d.get("impact") or {}
        out.append(
            {
                "id": str(d.get("id")),
                "created_at": str(d.get("created_at") or ""),
                "base_config": str(d.get("base_config") or ""),
                "params": d.get("params") or [],
                "mode": str(d.get("mode") or "grid"),
                "n_members": len(members),
                "n_done": sum(1 for m in members if m.get("status") == "done"),
                "status": str(d.get("status") or ""),
                "family": str(d.get("family") or ""),
                "n_new_trials": imp.get("n_new"),
                "pbo": _f(pbo.get("pbo")),
                "partial": pbo.get("partial") if pbo else None,
                "best_sharpe_net": max(sharpes) if sharpes else None,
            }
        )
    return sorted(out, key=lambda r: r["created_at"], reverse=True)


def sweep_manifest(settings: Settings, sweep_id: str) -> dict[str, Any]:
    if not isinstance(sweep_id, str) or not SWEEP_ID_RE.fullmatch(sweep_id):
        raise NotFound(sweep_id)
    return result_file(settings.runs_root, SWEEPS_DIR, sweep_id)


def sweep_task_params(plan: dict[str, Any]) -> dict[str, Any]:
    """What a sweep's member / finalize jobs carry (for the command line and the label)."""
    st = plan.get("store") or {}
    return {
        "sweep_id": plan["id"],
        "base_config": plan.get("base_config"),
        "params": plan.get("params") or [],
        "mode": plan.get("mode"),
        "store": {k: st.get(k) for k in ("start", "end", "max_names", "min_mcap_ever", "max_rank")},
        "nav": plan.get("nav"),
        "max_days": plan.get("max_days"),
        "use_ml": plan.get("use_ml"),
    }


# ----------------------------------------------------------------------------- capacity / risk
def capacity_summaries(settings: Settings) -> list[dict[str, Any]]:
    out = []
    for _, d in _results(settings.runs_root, CAPACITY_DIR):
        out.append(
            {
                "id": str(d.get("id")),
                "config_hash": str(d.get("config_hash") or ""),
                "family": str(d.get("family") or ""),
                "navs": [float(p.get("nav")) for p in d.get("points") or [] if _f(p.get("nav")) is not None],
                "half_sharpe_nav": _f(d.get("half_sharpe_nav")),
                "reference_sharpe_net": _f(d.get("reference_sharpe_net")),
                "sealed_end": d.get("sealed_end"),
            }
        )
    return out


def risk_summaries(settings: Settings) -> list[dict[str, Any]]:
    out = []
    for _, d in _results(settings.runs_root, RISK_DIR):
        out.append(
            {
                "id": str(d.get("id")),
                "config_hash": str(d.get("config_hash") or ""),
                "nus": [float(r.get("nu_scale")) for r in d.get("results") or [] if _f(r.get("nu_scale")) is not None],
                "current_nu_scale": float(d.get("current_nu_scale") or 0.0),
                "recommended_nu_scale": _f(d.get("recommended_nu_scale")),
                "sealed_end": d.get("sealed_end"),
            }
        )
    return out


# ----------------------------------------------------------------------------- mid-cap
def midcap_results(settings: Settings) -> list[tuple[str, str, dict[str, Any]]]:
    """``(run_id, source, payload)`` of every finished check found: ``runs/_checks/
    midcap_<run_id>.json`` first, then ``<run>/midcap_check.json`` of runs not covered."""
    seen: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for f, d in _results(settings.runs_root, CHECKS_DIR):
        if not f.name.startswith("midcap_") or d.get("dry_run"):
            continue
        rid = str(((d.get("verdict") or {}).get("run_id")) or f.stem.removeprefix("midcap_"))
        seen.setdefault(rid, (rid, f"{CHECKS_DIR}/{f.name}", d))
    root = settings.runs_root
    if root.is_dir():
        for f in sorted(root.glob("*/midcap_check.json"), reverse=True):
            d = read_json(f)
            if isinstance(d, dict) and not d.get("dry_run"):
                rid = str(((d.get("verdict") or {}).get("run_id")) or f.parent.name)
                seen.setdefault(rid, (rid, f"{f.parent.name}/midcap_check.json", d))
    return sorted(seen.values(), key=lambda t: t[0], reverse=True)


def midcap_result(settings: Settings, run_id: str) -> dict[str, Any]:
    if not RESULT_ID_RE.fullmatch(run_id or ""):
        raise NotFound(run_id)
    for rid, _src, d in midcap_results(settings):
        if rid == run_id:
            return d
    raise NotFound(run_id)


def midcap_static(settings: Settings) -> dict[str, Any]:
    from qts.research.midcap import (
        PREREGISTRATION,
        REFERENCE_RUN_ID,
        criteria_table,
        midcap_config,
        preregistration_sha256,
        previous_midcap_records,
    )

    cfg = midcap_config()
    sha = preregistration_sha256()
    prev = previous_midcap_records(str(settings.ledger), cfg.hash())
    return {
        "criteria": criteria_table(),
        "config_hash": cfg.hash(),
        "config_path": "configs/baseline.yaml",
        "preregistration": PREREGISTRATION,
        "preregistration_sha256": sha,
        "previous_runs": [
            {
                "ts": r.get("ts"),
                "family": r.get("family"),
                "config_hash": r.get("config_hash"),
                "sharpe_annual": _f((r.get("metrics") or {}).get("sharpe_annual")),
                "forced": bool(r.get("midcap_forced", False)),
            }
            for r in prev
        ],
        "reference_run_id": REFERENCE_RUN_ID,
    }


def preregistration(settings: Settings) -> dict[str, Any]:
    from qts.research.common import resolve_repo_path
    from qts.research.midcap import PREREGISTRATION, preregistration_sha256

    p = resolve_repo_path(PREREGISTRATION)
    text = p.read_text() if p.exists() else ""
    return {"path": PREREGISTRATION, "exists": p.exists(), "sha256": preregistration_sha256(), "text": text}


# ----------------------------------------------------------------------------- paper
def book_name(name: str) -> str:
    if not isinstance(name, str) or not BOOK_RE.fullmatch(name):
        raise NotFound(name)
    return name


def books(settings: Settings) -> list[str]:
    root = settings.paper_root
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "book.json").exists() and BOOK_RE.fullmatch(d.name))


def book_exists(settings: Settings, name: str) -> bool:
    return (settings.paper_root / book_name(name) / "book.json").exists()


def paper_status(settings: Settings, name: str) -> dict[str, Any]:
    """``qts.paper.status`` with the series completed for the page: the live / backfill
    mode per day (from ``history``), the calibrated boundaries per day
    (``qts.paper.boundaries``), empty lists for a book without days."""
    from qts import paper

    root = str(settings.paper_root)
    st = paper.status(name, paper_root=root)
    s = st.get("series") or {}
    hist = paper.history(name, paper_root=root)
    first = (st.get("header") or {}).get("first_execution") or ""
    mode_by_date = {r["date"]: r.get("mode") for r in hist}
    dates = list(s.get("date") or [])
    keys = ("nav", "ret", "cum_log", "drawdown", "gross", "net", "turnover", "exante_vol", "realised_vol_21d", "status")
    series = {k: list(s.get(k) or []) for k in keys}
    series["date"] = dates
    series["mode"] = [mode_by_date.get(d) for d in dates]
    series["cone_analytic"] = s.get("cone_analytic")
    series["cone_bootstrap"] = s.get("cone_bootstrap")
    series["boundaries"] = None
    if dates:
        try:
            b = paper.boundaries(name, paper_root=root)
            series["boundaries"] = {k: b[k] for k in ("dd_watch", "dd_stop", "sharpe_watch", "sharpe_stop", "levels")}
        except Exception:  # an expectations table without the quantiles: show the series without bands
            series["boundaries"] = None
    status = dict(st.get("status") or {})
    for k, v in (("stop_rules", []), ("suppressed", []), ("acknowledged", False), ("trading_halted", False)):
        status.setdefault(k, v)
    status.setdefault("checks", [])
    status.setdefault("metrics", {})
    latest = st.get("latest")
    if isinstance(latest, dict):
        latest = {**latest, "warnings": [str(w) for w in latest.get("warnings") or []]}
    header = st.get("header") or {}
    return {
        "book": st.get("book"),
        "root": st.get("root"),
        "header": header,
        "n_days": int(st.get("n_days") or 0),
        "last_date": st.get("last_date"),
        "acks": st.get("acks") or [],
        "latest": latest,
        "status": status,
        "series": series,
        "stop_rules": {
            k: v
            for k, v in (header.get("stop_rules") or {}).items()
            if v is None or isinstance(v, (bool, int, float, str))
        },
        "_first_execution": first,
    }


def book_summary(settings: Settings, name: str) -> dict[str, Any]:
    from qts import paper

    st = paper.status(name, paper_root=str(settings.paper_root))
    h = st.get("header") or {}
    latest = st.get("latest") or {}
    s = st.get("status") or {}
    return {
        "name": name,
        "sizing": str(h.get("sizing", "")),
        "nav0": float(h.get("nav0") or 0.0),
        "start": str(h.get("start", "")),
        "first_execution": str(h.get("first_execution", "")),
        "use_ml": bool(h.get("use_ml")),
        "config_hash": str(h.get("config_hash", "")),
        "reference_run": h.get("reference_run"),
        "n_days": int(st.get("n_days") or 0),
        "last_date": st.get("last_date"),
        "last_mode": latest.get("mode"),
        "level": s.get("level", "OK"),
        "trading_halted": bool(s.get("trading_halted")),
        "nav": _f(latest.get("nav")),
    }


def paper_holdings(settings: Settings, name: str, date: str | None) -> dict[str, Any]:
    from qts import paper

    h = paper.holdings(name, date=date, paper_root=str(settings.paper_root))
    return {
        "date": h.get("date"),
        "nav": _f(h.get("nav")),
        "holdings": [
            {
                "ticker": str(p["ticker"]),
                "shares": float(p["shares"]),
                "px": _f(p.get("px")),
                "usd": _f(p.get("value")),
                "weight": _f(p.get("weight")),
            }
            for p in h.get("positions") or []
        ],
    }


__all__ = [
    "MIDCAP_FORCE_PHRASE",
    "NotFound",
    "book_summary",
    "books",
    "capacity_summaries",
    "family_now",
    "feature_list",
    "midcap_result",
    "midcap_results",
    "midcap_static",
    "paper_holdings",
    "paper_status",
    "preregistration",
    "resolve_family",
    "result_file",
    "risk_summaries",
    "seal_view",
    "signal_test_detail",
    "signal_tests",
    "store_spec_dict",
    "sweep_manifest",
    "sweep_summaries",
    "sweep_task_params",
    "trial_impact",
    "unseal_check",
]
