"""Research context of one pipeline run: the holdout seal, the trial-ledger family
and the kind of ledger record (:class:`RunContext`).

Nothing here changes a number.  A :class:`~qts.pipeline.ResearchPipeline` built
with a context

* logs its ledger record under ``context.family`` instead of the default research
  family ``pipeline_phase{p}:{market_hash[:12]}`` (a run under the seal logs into the
  family of the *full-history* price panel, so it counts the trials that already saw
  that history; ``qts.research.seal``);
* adds ``context.ledger_fields()`` to the record: ``kind`` (only ``"trial"`` records
  count towards the deflated Sharpe ratio's ``N`` and ``V``), ``reference`` (a
  re-run of a strategy that is already counted, e.g. the baseline re-run under the
  seal: excluded from ``N``), the seal fields and any ``record_extra``;
* writes ``sealed_end`` / ``unsealed`` / ``holdout_start`` / ``holdout_already_used``
  (and the rest of :meth:`summary_fields`) into ``summary.json`` and the
  :class:`~qts.core.manifest.RunManifest`, and the report card prints a banner.

The default context (``RunContext()``) is the behaviour before the seal existed:
default family, a counted trial, no seal fields set (``sealed_end`` ``None``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: Ledger record kinds.  Only ``"trial"`` records enter the DSR's trial count and
#: cross-trial variance (:class:`qts.validation.ledger.TrialLedger`).
RECORD_KINDS: tuple[str, ...] = ("trial", "unseal", "capacity", "calibration", "event")


@dataclass(frozen=True)
class RunContext:
    """See the module docstring.  All fields are JSON-serialisable."""

    #: first sealed date of the seal in force (ISO ``YYYY-MM-DD``); ``None`` = no seal
    holdout_start: str | None = None
    #: last date the run loaded when the seal was applied (ISO); ``None`` = full history
    sealed_end: str | None = None
    #: a seal is in force but this run loaded the full history (``--unseal``)
    unsealed: bool = False
    unseal_reason: str = ""
    #: an earlier unseal of the same configuration already looked at the holdout
    holdout_already_used: bool = False
    #: why a full-history run is exempt from the seal (``"midcap"``), else ``""``
    seal_exempt: str = ""
    #: ledger family override (``None``: the pipeline's default research family)
    family: str | None = None
    #: market hash of the full-history panel of the same load (sealed runs)
    market_hash_full: str = ""
    #: ledger record kind (:data:`RECORD_KINDS`)
    record_kind: str = "trial"
    #: a re-run of an already counted strategy: logged, never counted in ``N``
    reference: bool = False
    #: extra JSON fields stored on the ledger record (never overriding a standard key)
    record_extra: Mapping[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def __post_init__(self) -> None:
        if self.record_kind not in RECORD_KINDS:
            raise ValueError(
                f"RunContext: record_kind must be one of {RECORD_KINDS}, got {self.record_kind!r}"
            )

    @property
    def counts_as_trial(self) -> bool:
        """True when the run's ledger record enters the DSR's ``N`` and ``V``."""
        return self.record_kind == "trial" and not self.reference

    def ledger_fields(self) -> dict[str, Any]:
        """Fields added to the run's trial-ledger record (empty for the default context,
        so records of runs without a context are unchanged)."""
        out: dict[str, Any] = {}
        if self.record_kind != "trial":
            out["kind"] = self.record_kind
        if self.reference:
            out["reference"] = True
        if self.holdout_start is not None:
            out["holdout_start"] = self.holdout_start
            out["sealed_end"] = self.sealed_end
            out["unsealed"] = bool(self.unsealed)
        if self.unseal_reason:
            out["unseal_reason"] = self.unseal_reason
        if self.holdout_already_used:
            out["holdout_already_used"] = True
        if self.seal_exempt:
            out["seal_exempt"] = self.seal_exempt
        if self.market_hash_full:
            out["market_hash_full"] = self.market_hash_full
        for k, v in dict(self.record_extra).items():
            out.setdefault(str(k), v)
        return out

    def summary_fields(self) -> dict[str, Any]:
        """Keys written to ``summary.json`` (always present, ``None`` / ``False`` when
        not applicable, so the API can rely on them)."""
        return {
            "holdout_start": self.holdout_start,
            "sealed_end": self.sealed_end,
            "unsealed": bool(self.unsealed),
            "unseal_reason": self.unseal_reason or None,
            "holdout_already_used": bool(self.holdout_already_used),
            "seal_exempt": self.seal_exempt or None,
            "record_kind": self.record_kind,
            "reference": bool(self.reference),
            "market_hash_full": self.market_hash_full or None,
        }

    def banner(self) -> list[str]:
        """Report-card lines (empty for a run without a seal in force)."""
        lines: list[str] = []
        if self.holdout_already_used:
            lines.append(
                "!!! HOLDOUT ALREADY USED: an earlier unseal of this configuration looked at the sealed "
                "period; this result is NOT out-of-sample !!!"
            )
        if self.unsealed and self.seal_exempt:
            lines.append(
                f"seal: exempt ({self.seal_exempt}): full history loaded by design; holdout from "
                f"{self.holdout_start} included"
            )
        elif self.unsealed:
            lines.append(
                f"!!! UNSEALED: full history including the holdout from {self.holdout_start} "
                f"(reason: {self.unseal_reason or 'n/a'}) !!!"
            )
        elif self.holdout_start is not None:
            lines.append(
                f"seal: data through {self.sealed_end} (holdout from {self.holdout_start} not loaded)"
            )
        if self.reference:
            lines.append("reference re-run of an already counted strategy: logged, not counted as a trial")
        if self.record_kind != "trial":
            lines.append(
                f"ledger record kind {self.record_kind!r}: not an alpha trial (not counted in the DSR)"
            )
        return lines


__all__ = ["RECORD_KINDS", "RunContext"]
