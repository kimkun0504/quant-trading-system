"""Trial ledger (``qts.validation.TrialLedger`` JSONL) as the API's ``Ledger`` shape,
cached by the file's mtime/size."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

from qts.validation import TrialLedger
from qts.validation.ledger import is_trial_record, record_kind

#: Why a record does not count towards its family's N and V, by kind.
NOT_COUNTED: dict[str, str] = {
    "unseal": "An unseal event: it records that the holdout of this configuration was looked at. Not a trial.",
    "capacity": "A capacity run: the baseline re-run at another NAV. Not an alpha trial, not counted in any DSR.",
    "calibration": "A risk-calibration run: a report variant of the baseline. Not counted in any DSR.",
    "event": "A ledger event, not a trial.",
}
REFERENCE_NOTE = (
    "A reference re-run: the same configuration as an already counted trial (for example the baseline re-run "
    "under the seal). Logged, but not counted again in N."
)


def not_counted_note(r: dict[str, Any]) -> str | None:
    if is_trial_record(r):
        return None
    kind = record_kind(r)
    if kind == "trial" and r.get("reference"):
        return REFERENCE_NOTE
    return NOT_COUNTED.get(kind, f"A {kind} record, not a trial: not counted in N.")


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
        recs = led.records(include_non_trials=True)
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
                    "kind": record_kind(r),
                    "reference": bool(r.get("reference")),
                    "counted": is_trial_record(r),
                    "note": not_counted_note(r),
                    "holdout_already_used": bool(r.get("holdout_already_used")),
                    "unsealed": bool(r.get("unsealed")),
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
                    "n_not_counted": sum(1 for t in trials if t["family"] == fam and not t["counted"]),
                }
            )
        return {"trials": trials, "families": families}
