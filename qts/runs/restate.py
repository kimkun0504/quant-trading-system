"""Restate the deflated Sharpe ratio of a finished run at the ledger's current
trial count (``qts ledger restate <run_dir>``).

A run's ``summary.json["dsr"]`` is frozen at the trial count its ledger family
had when the run finished.  The family grows afterwards (later trials), and it
can also grow backwards: ``qts ledger rekey`` / ``merge-families`` pool earlier
trials on the same price history that were filed under another family
(:mod:`qts.validation.ledger`).  :func:`restate_dsr` recomputes the DSR from the
run directory and the ledger alone -- no data, no backtest:

* the run's net Sharpe (``summary.json``) and the length, skew and kurtosis of
  ``returns.parquet["net"]`` -- the pipeline's own inputs;
* the family: ``family`` argument > the family of the run's own ledger record
  (matched on config hash + data hash, the record closest in time to the run) >
  the manifest's / summary's ``ledger_family`` > the family rebuilt from the
  phase and the market (or, for runs before 2026-11-12, data) hash;
* ``n_trials`` (Nyholt effective count), the raw de-duplicated count and the
  cross-trial Sharpe variance from :meth:`TrialLedger.dsr_inputs` -- the rule
  the pipeline uses.

The result is written to ``extras.json["dsr_restated"]`` (the original
``summary.json`` is never modified: it stays the record of what the run reported
at the time) and returned.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from qts.runs.artifacts import EXTRAS_FILE, _finite, _write_json, read_extras
from qts.runs.registry import _from_json_scalar
from qts.validation.deflated_sharpe import deflated_sharpe_ratio_annual, expected_max_sharpe
from qts.validation.ledger import TrialLedger, research_family

DSR_RESTATED_KEY = "dsr_restated"


def _parse_ts(ts: object) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(ts))
    except ValueError:
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def find_run_record(ledger: TrialLedger, config_hash: str, data_hash: str, created_at: str) -> dict | None:
    """The ledger record a run wrote: same config hash and data hash, closest in
    time to the manifest's ``created_at`` (``None`` when there is none)."""
    cands = [
        r
        for r in ledger.raw_records()
        if str(r.get("config_hash")) == str(config_hash) and str(r.get("data_hash")) == str(data_hash)
    ]
    if not cands:
        return None
    t0 = _parse_ts(created_at)
    if t0 is None:
        return cands[-1]

    def dist(r: dict) -> float:
        t = _parse_ts(r.get("ts"))
        return float("inf") if t is None else abs((t - t0).total_seconds())

    return min(cands, key=dist)


def restate_dsr(
    run_dir: str | Path,
    ledger_path: str | Path,
    family: str | None = None,
    periods_per_year: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Recompute a finished run's DSR at its family's current trial count (module
    docstring).  Returns the ``dsr_restated`` payload; with ``write`` it is also
    stored in ``<run_dir>/extras.json``."""
    p = Path(run_dir)
    if not (p / "manifest.json").exists():
        raise FileNotFoundError(f"{p} is not a run directory (no manifest.json)")
    manifest = json.loads((p / "manifest.json").read_text())
    summary = _from_json_scalar(json.loads((p / "summary.json").read_text()))
    net = pd.read_parquet(p / "returns.parquet")["net"].astype(float)
    sharpe = float(summary["backtest"]["sharpe_net"])
    n_obs = len(net)
    skew, kurt = float(cast(Any, net.skew())), float(cast(Any, net.kurt())) + 3.0
    if periods_per_year is None:
        periods_per_year = _periods_per_year(p)

    ledger = TrialLedger(str(ledger_path))
    rec = find_run_record(
        ledger, str(manifest.get("config_hash")), str(manifest.get("data_hash")), str(manifest.get("created_at"))
    )
    if family is not None:
        fam, source = str(family), "argument"
    elif rec is not None:
        fam, source = str(rec.get("family")), "ledger record of the run"
    elif manifest.get("ledger_family") or summary.get("ledger_family"):
        fam, source = str(manifest.get("ledger_family") or summary.get("ledger_family")), "run manifest"
    else:
        key = str(manifest.get("market_hash") or manifest.get("data_hash") or "")
        fam, source = research_family(int(summary.get("phase", 1)), key), "rebuilt from the run's hashes"

    inputs = ledger.dsr_inputs(fam, n_obs, periods_per_year)
    dsr = deflated_sharpe_ratio_annual(
        sharpe,
        n_trials=max(float(inputs["n_trials"]), 1.0),
        var_trials_sr_annual=inputs["var_trials_sr_annual"],
        n_obs=n_obs,
        skew=skew,
        kurt=kurt,
        periods_per_year=periods_per_year,
    )
    e_max = expected_max_sharpe(max(float(inputs["n_trials"]), 1.0), inputs["var_trials_sr_annual"])
    orig_in = summary.get("dsr_inputs") or {}
    payload: dict[str, Any] = {
        "dsr": _finite(dsr),
        "n_trials": float(inputs["n_trials"]),  # Nyholt effective count (what the DSR used)
        "n_trials_raw": float(inputs["n_trials_raw"]),
        "var_trials_sr_annual": float(inputs["var_trials_sr_annual"]),
        "expected_max_sharpe_annual": float(e_max),
        "sharpe_net": sharpe,
        "n_obs": n_obs,
        "skew": _finite(skew),
        "kurtosis": _finite(kurt),
        "family": fam,
        "family_source": source,
        "run_in_family": bool(rec is not None and str(rec.get("family")) == fam),
        "ledger_path": str(ledger_path),
        "restated_at": datetime.now(timezone.utc).isoformat(),
        "original": {
            "dsr": _finite(summary.get("dsr")),
            "n_trials": _finite(orig_in.get("n_trials")),
            "n_trials_raw": _finite(orig_in.get("n_trials_raw")),
            "var_trials_sr_annual": _finite(orig_in.get("var_trials_sr_annual")),
        },
    }
    if write:
        extras = read_extras(p)
        if not extras and (p / EXTRAS_FILE).exists():
            raise ValueError(f"{p / EXTRAS_FILE} is unreadable; not overwriting it")
        extras[DSR_RESTATED_KEY] = payload
        _write_json(p / EXTRAS_FILE, extras)
    return payload


def _periods_per_year(run_dir: Path) -> int:
    """``trading_days_per_year`` of the run's own config (252 when unreadable)."""
    import yaml

    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
        return int(cfg.get("trading_days_per_year", 252))
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        return 252


def format_restated(run_id: str, r: dict[str, Any]) -> str:
    """Report-card style text of a :func:`restate_dsr` payload."""
    o = r["original"]

    def f(x: Any, spec: str) -> str:
        return "n/a" if x is None else format(x, spec)

    lines = [
        f"run {run_id}",
        f"family {r['family']}  (from: {r['family_source']})",
        f"net Sharpe {r['sharpe_net']:.3f} over {r['n_obs']} days  "
        f"(skew {f(r['skew'], '.2f')}, kurtosis {f(r['kurtosis'], '.2f')})",
        f"as reported : DSR {f(o['dsr'], '.3f')}  (n_trials {f(o['n_trials'], '.1f')}, "
        f"raw {f(o['n_trials_raw'], '.0f')}, var SR {f(o['var_trials_sr_annual'], '.3f')})",
        f"restated    : DSR {f(r['dsr'], '.3f')}  (n_trials {r['n_trials']:.1f} Nyholt N_eff, "
        f"raw {r['n_trials_raw']:.0f}, var SR {r['var_trials_sr_annual']:.3f}, "
        f"E[max SR] {r['expected_max_sharpe_annual']:.3f})",
    ]
    if not r["run_in_family"]:
        lines.append(
            "WARNING: the run's own ledger record is not in this family (not logged to this ledger, or "
            "filed elsewhere): the count above may not include the run itself"
        )
    return "\n".join(lines)
