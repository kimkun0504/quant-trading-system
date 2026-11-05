"""Trial ledger (``qts.validation.TrialLedger`` JSONL) as the API's ``Ledger`` shape,
cached by the file's mtime/size."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

from qts.validation import TrialLedger


def _f(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x


class LedgerService:
    def __init__(self, path: Path, periods_per_year: int = 252):
        self.path = Path(path)
        self.ppy = periods_per_year
        self._sig: tuple[int, int] | None = None
        self._value: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _signature(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def get(self) -> dict[str, Any]:
        sig = self._signature()
        with self._lock:
            if self._value is not None and sig == self._sig:
                return self._value
        value = self._build() if sig is not None else {"trials": [], "families": []}
        with self._lock:
            self._sig, self._value = sig, value
        return value

    def _build(self) -> dict[str, Any]:
        # TrialLedger(path) creates a missing file; the caller checked it exists.
        led = TrialLedger(str(self.path))
        recs = led.records()
        trials = []
        for i, r in enumerate(recs):
            m = r.get("metrics") or {}
            if "sharpe_annual" in m:
                sr = _f(m["sharpe_annual"])
            elif "sharpe_daily" in m:
                sr = _f(m["sharpe_daily"]) * math.sqrt(self.ppy)
            else:
                sr = float("nan")
            mic = m.get("mean_ic")
            trials.append(
                {
                    "id": f"trial-{i + 1}",
                    "ts": str(r.get("ts", "")),
                    "name": str(r.get("name", "")),
                    "family": str(r.get("family", str(r.get("name", "")).split(":")[0])),
                    "config_hash": str(r.get("config_hash", "")),
                    "data_hash": str(r.get("data_hash", "")),
                    "sharpe_annual": sr,
                    "n_obs": int(_f(m.get("n_obs"), 0.0)),
                    "mean_ic": None if mic is None or not math.isfinite(_f(mic)) else _f(mic),
                    "n_reruns": int(r.get("n_reruns", 0) or 0),
                }
            )
        families = []
        for fam in sorted({t["family"] for t in trials}):
            families.append(
                {
                    "family": fam,
                    "n_trials": led.n_trials(fam),
                    "effective_trials": float(led.effective_trials_for(fam)),
                    "var_sr_annual": float(led.family_sharpe_variance(fam, annual=True, periods_per_year=self.ppy)),
                }
            )
        return {"trials": trials, "families": families}
