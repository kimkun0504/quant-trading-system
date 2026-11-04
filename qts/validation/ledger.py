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

Identity / de-duplication: a record carries ``config_hash`` (hash of the full
config) and ``data_hash``.  ``log(..., replace_identical=True)`` *updates* the
existing record with the same ``(name, config_hash, data_hash)`` instead of
appending, so re-running an unchanged experiment on unchanged data does not
inflate ``n_trials`` (an identical re-run is not a new trial).  Anything that
differs in config or data is a new trial and is always appended.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from qts.validation.deflated_sharpe import deannualize_sharpe, deflated_sharpe_ratio


class TrialLedger:
    def __init__(self, path: str):
        self.path = str(path)
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(self.path):
            open(self.path, "a").close()

    # ------------------------------------------------------------------ io
    def _records(self) -> List[dict]:
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def log(self, name: str, config: Dict, metrics: Dict, data_hash: str = "", code_version: str = "",
            notes: str = "", family: Optional[str] = None, config_hash: Optional[str] = None,
            returns_monthly: Optional[pd.Series] = None, replace_identical: bool = False) -> dict:
        """Append one trial (or, with ``replace_identical=True``, update the
        record with the same ``(name, config_hash, data_hash)``).

        ``metrics`` should include ``sharpe_annual`` (or ``sharpe_daily``),
        ``n_obs`` and optionally ``skew``/``kurt``.  ``config`` should be the
        *full* configuration (``SystemConfig.to_dict()``); ``config_hash``
        defaults to a hash of its canonical JSON.  ``returns_monthly`` (a
        month-indexed Series) enables :meth:`effective_trials_for`.
        """
        cfg_json = _jsonable_nested(config)
        if config_hash is None:
            config_hash = hashlib.sha256(json.dumps(cfg_json, sort_keys=True, default=str).encode()).hexdigest()[:16]
        rec = {
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
        if returns_monthly is not None and len(returns_monthly):
            rm = pd.Series(returns_monthly, dtype=float)
            rec["returns_monthly"] = {str(pd.Timestamp(k).strftime("%Y-%m")): float(v) for k, v in rm.items()}
        if replace_identical and data_hash:
            recs = self._records()
            for i, r in enumerate(recs):
                if (r.get("name") == rec["name"] and r.get("config_hash") == rec["config_hash"]
                        and r.get("data_hash") == rec["data_hash"]):
                    rec["ts_first"] = r.get("ts_first", r.get("ts"))
                    rec["n_reruns"] = int(r.get("n_reruns", 0)) + 1
                    recs[i] = rec
                    tmp = self.path + ".tmp"
                    with open(tmp, "w") as f:
                        for rr in recs:
                            f.write(json.dumps(rr, default=str) + "\n")
                    os.replace(tmp, self.path)
                    return rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    # ------------------------------------------------------------------ queries
    def n_trials(self, family: Optional[str] = None) -> int:
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

    def records(self, family: Optional[str] = None) -> List[dict]:
        recs = self._records()
        return recs if family is None else [r for r in recs if r.get("family") == family]

    def trial_sharpes_annual(self, family: Optional[str] = None, periods_per_year: int = 252) -> pd.Series:
        return self.trial_sharpes_daily(family) * math.sqrt(periods_per_year)

    def family_sharpe_variance(self, family: Optional[str] = None, annual: bool = True,
                               periods_per_year: int = 252) -> float:
        """Sample variance (ddof=1) of the logged Sharpe ratios in ``family``
        (0.0 with fewer than two trials) — the ``V`` of Bailey & LdP's E[max SR]."""
        srs = self.trial_sharpes_annual(family, periods_per_year) if annual else self.trial_sharpes_daily(family)
        if len(srs) < 2:
            return 0.0
        v = float(srs.var(ddof=1))
        return v if math.isfinite(v) else 0.0

    def returns_matrix(self, family: Optional[str] = None, min_overlap: int = 12) -> pd.DataFrame:
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

    def effective_trials_for(self, family: Optional[str] = None, min_trials_for_correlation: int = 3) -> float:
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

    def trial_sharpes_daily(self, family: Optional[str] = None) -> pd.Series:
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
    def effective_trials(correlation_matrix_of_trial_returns=None, n_trials: Optional[int] = None) -> float:
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

    def dsr_for(self, name: str, family: Optional[str] = None, n_trials: Optional[float] = None) -> dict:
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
        sr_d = float(m["sharpe_daily"]) if "sharpe_daily" in m else deannualize_sharpe(float(m["sharpe_annual"]))
        n = int(m["n_obs"])
        skew = float(m.get("skew", 0.0))
        kurt = float(m.get("kurt", 3.0))
        N = float(n_trials) if n_trials is not None else float(len(srs))
        var_sr = float(srs.var(ddof=1)) if len(srs) > 1 else 0.0
        if not math.isfinite(var_sr):
            var_sr = 0.0
        dsr = deflated_sharpe_ratio(sr_d, max(int(round(N)), 1), var_sr, n, skew, kurt)
        return {"name": name, "family": fam, "n_trials": N, "var_trials_sr": var_sr,
                "sr_daily": sr_d, "n_obs": n, "dsr": dsr}


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
