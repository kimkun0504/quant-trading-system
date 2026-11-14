"""The repo-level research holdout seal (``research/seal.yaml``).

Why
---
Every design choice of the baseline was made after seeing 2003-06..2026-11, so there
is no clean holdout in time.  From 2026-11-13 the last three years of the store
(``holdout_start: 2023-11-09``) are sealed for all *new* research: every research
command that loads the store loads data only up to the day before ``holdout_start``,
so a feature, label, risk input or backtest of new research can never touch the
sealed returns.  The seal lives outside :class:`~qts.core.config.SystemConfig` on
purpose: strategy config hashes (the ledger's identity of a strategy) do not change
when the seal is set, moved or lifted.

What is sealed
--------------
``qts run --store``, the diagnostics scripts that load the store
(``scripts/diagnose_*.py``, ``attribute_run.py``, ``find_outlier_returns.py``),
``qts signal test``, ``qts sweep``, ``qts capacity`` and ``qts risk calibrate``.  The
load is cut at ``holdout_start - 1 day`` *at load time* (the peak-cap prefilter, every
panel, the dated reference tables, delisting dates / reasons after the cut:
``MarketData.slice_until``), so labels near the boundary are NaN like at the end of
any panel instead of reading sealed prices.  Not sealed: ``qts demo`` (synthetic
data), ``qts ledger rekey --store`` (hashes only), ``qts check midcap`` (the
pre-registered one-shot confirmation: full history by design, logged with
``seal_exempt: midcap``) and paper trading (forward data: ``qts.paper`` calls
:func:`is_sealed` / :func:`load_seal`).

Unsealing
---------
``qts run ... --unseal "<reason>"`` loads the full history once per configuration:
it writes an ``unseal`` record to the trial ledger (``kind: "unseal"``: config hash,
family, reason, time) *before* the run starts (looking is what uses the holdout,
whether or not the run finishes), prints a banner, and the run's record, summary and
manifest carry ``unsealed: true``.  A second unseal of the same configuration
(:func:`previous_unseals`: same normalised config hash and phase) is refused unless
``--force``; the forced run is flagged "holdout already used" on the report card, in
``summary.json`` and on its ledger record.

Ledger family under the seal
----------------------------
A sealed panel has its own market hash, which would start a new trial family with
``N = 1`` although the earlier trials saw this very history.  A sealed run therefore
logs into the family of the *full-history* panel of the same load
(``pipeline_phase{p}:{market_hash_full[:12]}``; :mod:`qts.research.data` computes
that hash from the close panel alone and caches it), storing both hashes on the
record (``market_hash`` = sealed, ``market_hash_full``).  With the reference load
flags (``--start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8``) and an unchanged
store that is ``pipeline_phase2:c1da477b63d2`` with its 9 trials.  ``seal.yaml``'s
``inherit_family`` lists the families new research is expected to inherit; a run
that resolves to another family of the same phase prints a warning (a refreshed store
or other load flags: merge the families by hand if they are one search).

API for other packages
----------------------
:func:`load_seal` -> :class:`Seal` or ``None``; :func:`is_sealed(date)`;
:func:`seal_status` -> :class:`SealStatus` (``qts seal status``); :func:`apply_seal`.
The file is found at ``$QTS_SEAL_FILE`` (empty / ``none`` = no seal; tests) or at
``<repo>/research/seal.yaml``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from qts.research.common import json_safe, repo_root

if TYPE_CHECKING:
    from qts.core.types import MarketData

#: Environment variable naming the seal file (``""`` / ``"none"``: no seal).
SEAL_FILE_ENV = "QTS_SEAL_FILE"
#: Default location, relative to the repo root.
DEFAULT_SEAL_FILE = "research/seal.yaml"
#: Ledger family of the unseal events.
UNSEAL_KIND = "unseal"


class SealError(RuntimeError):
    """A seal file is malformed, or an unseal is refused."""


@dataclass(frozen=True)
class Seal:
    """The seal in force (``research/seal.yaml``)."""

    holdout_start: pd.Timestamp
    reason: str = ""
    sealed_by: str = ""
    sealed_at: str = ""
    store_end: str | None = None
    inherit_family: tuple[str, ...] = ()
    path: str = ""

    @property
    def holdout_start_iso(self) -> str:
        return str(self.holdout_start.date())

    @property
    def load_end(self) -> pd.Timestamp:
        """Last calendar day a sealed load may contain (the day before ``holdout_start``)."""
        return self.holdout_start - pd.Timedelta(days=1)

    @property
    def load_end_iso(self) -> str:
        return str(self.load_end.date())

    def is_sealed(self, date: Any) -> bool:
        """True when ``date`` falls in the sealed holdout (``>= holdout_start``)."""
        return bool(pd.Timestamp(date).normalize() >= self.holdout_start)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "holdout_start": self.holdout_start_iso,
                "load_end": self.load_end_iso,
                "reason": self.reason,
                "sealed_by": self.sealed_by,
                "sealed_at": self.sealed_at,
                "store_end": self.store_end,
                "inherit_family": list(self.inherit_family),
                "path": self.path,
            }
        )


def seal_path() -> Path | None:
    """The seal file to read (``None``: sealing disabled through ``$QTS_SEAL_FILE``)."""
    env = os.environ.get(SEAL_FILE_ENV)
    if env is not None:
        if env.strip().lower() in ("", "none", "off"):
            return None
        return Path(env)
    return repo_root() / DEFAULT_SEAL_FILE


def load_seal(path: str | os.PathLike[str] | None = None) -> Seal | None:
    """The seal at ``path`` (default :func:`seal_path`), or ``None`` when there is no
    seal file.  Raises :class:`SealError` on a malformed file (a typo must not
    silently lift the seal)."""
    import yaml

    p = Path(path) if path is not None else seal_path()
    if p is None or not p.exists():
        return None
    try:
        d = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise SealError(f"{p}: not valid YAML: {e}") from e
    if not isinstance(d, Mapping) or "holdout_start" not in d:
        raise SealError(f"{p}: needs a mapping with holdout_start")
    known = {"holdout_start", "reason", "sealed_by", "sealed_at", "store_end", "inherit_family"}
    unknown = sorted(set(d) - known)
    if unknown:
        raise SealError(f"{p}: unknown key(s) {unknown}; known: {sorted(known)}")
    try:
        start = pd.Timestamp(d["holdout_start"]).normalize()
    except (TypeError, ValueError) as e:
        raise SealError(f"{p}: holdout_start {d['holdout_start']!r} is not a date") from e
    inh = d.get("inherit_family") or ()
    if isinstance(inh, str):
        inh = (inh,)
    return Seal(
        holdout_start=start,
        reason=str(d.get("reason") or "").strip(),
        sealed_by=str(d.get("sealed_by") or ""),
        sealed_at=str(d.get("sealed_at") or ""),
        store_end=None if d.get("store_end") is None else str(d.get("store_end")),
        inherit_family=tuple(str(x) for x in inh),
        path=str(p),
    )


def is_sealed(date: Any, seal: Seal | None = None) -> bool:
    """True when ``date`` lies in the holdout of ``seal`` (default: the seal in force);
    False without a seal.  For paper trading: a forward date is always ``>=
    holdout_start``, which is fine -- the seal restricts research, not live use."""
    s = seal if seal is not None else load_seal()
    return False if s is None else s.is_sealed(date)


def apply_seal(md: MarketData, seal: Seal) -> MarketData:
    """``md`` cut at ``seal.load_end`` (``MarketData.slice_until``: every date-indexed
    panel, the dated reference tables, delisting dates / reasons after the cut).
    Asserts that nothing on or after ``holdout_start`` survives."""
    out = md.slice_until(seal.load_end)
    if len(out.dates) and out.dates[-1] >= seal.holdout_start:
        raise SealError(f"seal: data through {out.dates[-1].date()} survived the cut at {seal.load_end_iso}")
    for name in ("earnings_dates", "earnings_fundamentals"):
        tbl = getattr(out, name, None)
        col = out.DATED_TABLES.get(name)
        if (
            tbl is not None
            and col is not None
            and len(tbl)
            and pd.to_datetime(tbl[col]).max() >= seal.holdout_start
        ):
            raise SealError(f"seal: {name} has rows on or after {seal.holdout_start_iso}")
    if out.delisting_dates is not None:
        dd = pd.to_datetime(out.delisting_dates)
        if (dd >= seal.holdout_start).any():
            raise SealError("seal: a delisting date on or after the holdout start survived the cut")
    return out


# ----------------------------------------------------------------------------- unseal
def unseal_records(ledger_path: str | None) -> list[dict]:
    """Every ``unseal`` event of the ledger, in order."""
    if not ledger_path or not Path(ledger_path).exists():
        return []
    from qts.validation import TrialLedger

    return [r for r in TrialLedger(ledger_path).raw_records() if r.get("kind") == UNSEAL_KIND]


def previous_unseals(ledger_path: str | None, config_hash: str, phase: int) -> list[dict]:
    """Earlier unseals of the same strategy: same phase and the same configuration
    (stored hash, or the stored config re-hashed by the current ``SystemConfig``:
    :func:`qts.validation.ledger.normalized_config_hash`)."""
    from qts.validation.ledger import normalized_config_hash

    out = []
    for r in unseal_records(ledger_path):
        if int(r.get("phase", -1)) != int(phase):
            continue
        if r.get("config_hash") == config_hash or normalized_config_hash(r) == config_hash:
            out.append(r)
    return out


def record_unseal(
    ledger_path: str,
    *,
    seal: Seal,
    config: Mapping[str, Any],
    config_hash: str,
    family: str,
    phase: int,
    reason: str,
    forced: bool,
    n_previous: int,
    market_hash: str = "",
    command: str = "",
) -> dict:
    """Append the ``unseal`` event (``kind: "unseal"``; never a trial)."""
    from qts.validation import TrialLedger

    if not str(reason or "").strip():
        raise SealError('an unseal needs a reason (--unseal "<reason>")')
    return TrialLedger(ledger_path).log(
        name=f"unseal:{family}",
        family=family,
        config=dict(config),
        config_hash=config_hash,
        metrics={},
        market_hash=market_hash,
        notes=f"unseal: {reason}",
        extra={
            "kind": UNSEAL_KIND,
            "phase": int(phase),
            "unseal_reason": str(reason),
            "holdout_start": seal.holdout_start_iso,
            "forced": bool(forced),
            "n_previous_unseals": int(n_previous),
            "command": command,
        },
    )


UNSEAL_BANNER = """\
################################################################################
#  UNSEALED RUN: the holdout from {start} is loaded.
#  Reason: {reason}
#  Recorded in the trial ledger as an unseal of config {chash} ({family}).
#  This configuration's holdout is now used: a second unseal is refused without
#  --force and is then flagged "holdout already used".
################################################################################"""


def unseal_banner(seal: Seal, reason: str, config_hash: str, family: str) -> str:
    return UNSEAL_BANNER.format(start=seal.holdout_start_iso, reason=reason, chash=config_hash, family=family)


# ----------------------------------------------------------------------------- status
@dataclass
class SealStatus:
    """``qts seal status``: the seal, every unseal so far, and the research runs made
    under the seal (ledger records carrying its ``holdout_start`` and ``unsealed:
    false``), by family and kind; and the trial count of each inherited family."""

    seal: Seal | None
    unseals: list[dict[str, Any]] = field(default_factory=list)
    runs_under_seal: int = 0
    by_family: dict[str, dict[str, int]] = field(default_factory=dict)
    inherited: dict[str, int | None] = field(default_factory=dict)
    exempt: list[dict[str, Any]] = field(default_factory=list)
    ledger_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "seal": None if self.seal is None else self.seal.to_dict(),
                "unseals": self.unseals,
                "runs_under_seal": self.runs_under_seal,
                "by_family": self.by_family,
                "inherited_families": self.inherited,
                "exempt_runs": self.exempt,
                "ledger_path": self.ledger_path,
            }
        )

    def format(self) -> str:
        if self.seal is None:
            return f"no seal in force ({SEAL_FILE_ENV} or {DEFAULT_SEAL_FILE} not found)"
        s = self.seal
        lines = [
            f"seal          {s.path}",
            f"holdout       from {s.holdout_start_iso} (research loads data through {s.load_end_iso})",
            f"reason        {s.reason}",
            f"sealed by     {s.sealed_by}  at {s.sealed_at}"
            + (f"  (store end {s.store_end})" if s.store_end else ""),
        ]
        for fam, n in self.inherited.items():
            lines.append(
                f"inherits      {fam}: "
                + ("not in this ledger" if n is None else f"{n} trial(s) counted by every sealed run")
            )
        lines.append(f"\nresearch runs under the seal: {self.runs_under_seal}")
        for fam, kinds in sorted(self.by_family.items()):
            lines.append(f"  {fam:40s} " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
        lines.append(f"\nunseals: {len(self.unseals)}")
        for u in self.unseals:
            lines.append(
                f"  {str(u.get('ts'))[:19]}  config {u.get('config_hash')}  {u.get('family')}  "
                f"{'FORCED ' if u.get('forced') else ''}reason: {u.get('unseal_reason')}"
            )
        if self.exempt:
            lines.append(f"\nexempt full-history runs: {len(self.exempt)}")
            for e in self.exempt:
                lines.append(f"  {str(e.get('ts'))[:19]}  {e.get('family')}  ({e.get('seal_exempt')})")
        return "\n".join(lines)


def seal_status(ledger_path: str | None, seal: Seal | None = None) -> SealStatus:
    """Collect :class:`SealStatus` from the ledger (default seal: the one in force)."""
    from qts.validation import TrialLedger
    from qts.validation.ledger import record_kind

    s = seal if seal is not None else load_seal()
    st = SealStatus(seal=s, ledger_path=ledger_path)
    if s is None or not ledger_path or not Path(ledger_path).exists():
        return st
    led = TrialLedger(ledger_path)
    raw = led.raw_records()
    for r in raw:
        if r.get("holdout_start") != s.holdout_start_iso:
            continue
        if r.get("kind") == UNSEAL_KIND:
            st.unseals.append(
                {
                    k: r.get(k)
                    for k in (
                        "ts",
                        "config_hash",
                        "family",
                        "phase",
                        "unseal_reason",
                        "forced",
                        "n_previous_unseals",
                    )
                }
            )
            continue
        if r.get("seal_exempt"):
            st.exempt.append({k: r.get(k) for k in ("ts", "family", "seal_exempt", "config_hash")})
            continue
        if r.get("unsealed"):
            continue
        st.runs_under_seal += 1
        kinds = st.by_family.setdefault(str(r.get("family")), {})
        kind = "reference" if r.get("reference") else record_kind(r)
        kinds[kind] = kinds.get(kind, 0) + 1
    for fam in s.inherit_family:
        st.inherited[fam] = led.n_trials(fam) if any(r.get("family") == fam for r in raw) else None
    return st


def family_note(seal: Seal | None, family: str) -> str | None:
    """A warning when a sealed run of phase ``p`` resolves to a family other than the
    seal's inherited family of that phase (``None`` when all is as expected)."""
    if seal is None or not seal.inherit_family:
        return None
    prefix = family.rsplit(":", 1)[0] + ":"
    same_phase = [f for f in seal.inherit_family if f.startswith(prefix)]
    if not same_phase or family in same_phase:
        return None
    return (
        f"seal: this load's full-history family is {family}, not the seal's inherited {', '.join(same_phase)}: "
        "the trials counted there are not counted here.  If the store was refreshed or other load flags were "
        f"used for the same research, pool them: qts ledger merge-families --into {family} {same_phase[0]}"
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DEFAULT_SEAL_FILE",
    "SEAL_FILE_ENV",
    "Seal",
    "SealError",
    "SealStatus",
    "apply_seal",
    "family_note",
    "is_sealed",
    "load_seal",
    "previous_unseals",
    "record_unseal",
    "seal_path",
    "seal_status",
    "unseal_banner",
    "unseal_records",
]
