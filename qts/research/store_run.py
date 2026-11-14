"""``qts run --store`` under the research seal: load, unseal bookkeeping, reference
re-runs (:func:`prepare_store_run` -> :class:`StoreRun`; :meth:`StoreRun.finalize`).

Two steps because two facts are known at different times:

1. before the load (:func:`prepare_store_run`): whether an ``--unseal`` is allowed
   (refused when the same configuration was unsealed before, unless ``force``), and
   the sealed or full-history load itself;
2. after the CLI has fixed the final configuration (``universe.top_n_by_mcap`` clamped
   to the loaded ids, as ``qts run`` always did), :meth:`StoreRun.finalize` writes the
   ``unseal`` event to the ledger *before* the run starts (looking is what uses the
   holdout), prints the banner, and for a sealed run checks whether the configuration
   is already a counted trial of the inherited family -- then the run is the same
   strategy on a subset of the same history: logged with ``reference: true`` and not
   counted in the DSR's ``N`` (handbook 10 "The holdout seal").
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from qts.research.data import Printer, ResearchLoad, StoreSpec, _quiet, load_research_market
from qts.research.seal import (
    SealError,
    family_note,
    load_seal,
    previous_unseals,
    record_unseal,
    unseal_banner,
)

if TYPE_CHECKING:
    from qts.core.config import SystemConfig
    from qts.runs.context import RunContext


def reference_trial(ledger_path: str | None, family: str, config_hash: str, phase: int) -> dict | None:
    """The latest counted trial of ``family`` with the same strategy: same phase and the
    same configuration (stored hash, or re-hashed with the current ``SystemConfig``).
    ``None`` when the configuration has not been tried in that family."""
    from pathlib import Path

    from qts.validation import TrialLedger
    from qts.validation.ledger import normalized_config_hash

    if not ledger_path or not Path(ledger_path).exists():
        return None
    hit = None
    for r in TrialLedger(ledger_path).records(family):
        m = r.get("metrics") or {}
        ph = m.get("phase")
        if ph is not None and int(ph) != int(phase):
            continue
        if r.get("config_hash") == config_hash or normalized_config_hash(r) == config_hash:
            hit = r
    return hit


@dataclass
class StoreRun:
    """A prepared store run (:func:`prepare_store_run`)."""

    load: ResearchLoad
    phase: int
    unseal_reason: str | None = None
    previous_unseals: list[dict] = field(default_factory=list)
    force: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def md(self) -> Any:
        return self.load.md

    def finalize(
        self, cfg: SystemConfig, ledger_path: str | None, log: Printer = _quiet, command: str | None = None
    ) -> RunContext:
        """The run's :class:`~qts.runs.context.RunContext` (see the module docstring)."""
        from qts.core.manifest import hash_market
        from qts.validation.ledger import research_family

        ld = self.load
        chash = cfg.hash()
        if ld.unsealed and not ld.exempt and ld.seal is not None:
            family = research_family(self.phase, hash_market(ld.md))
            prev = previous_unseals(ledger_path, chash, self.phase) or self.previous_unseals
            if prev and not self.force:  # e.g. the clamp changed nothing, the pre-check was by-passed
                raise SealError(_refusal(prev))
            assert ledger_path is not None
            record_unseal(
                ledger_path,
                seal=ld.seal,
                config=cfg.to_dict(),
                config_hash=chash,
                family=family,
                phase=self.phase,
                reason=str(self.unseal_reason or ""),
                forced=bool(prev),
                n_previous=len(prev),
                market_hash=hash_market(ld.md),
                command=command if command is not None else " ".join(sys.argv),
            )
            log(unseal_banner(ld.seal, str(self.unseal_reason), chash, family))
            if prev:
                log("!!! --force: this configuration's holdout was already used; the result is flagged !!!")
            return ld.context(
                self.phase, unseal_reason=str(self.unseal_reason or ""), holdout_already_used=bool(prev)
            )
        if ld.sealed:
            family = ld.family_full(self.phase) or ""
            note = family_note(ld.seal, family)
            if note:
                log(f"WARNING: {note}")
                self.notes.append(note)
            ref = reference_trial(ledger_path, family, chash, self.phase)
            if ref is not None:
                log(
                    f"seal: config {chash} is already a counted trial of {family} ({str(ref.get('ts'))[:19]}): "
                    "this sealed re-run is logged as a reference (reference: true), not a new trial"
                )
                return ld.context(
                    self.phase,
                    reference=True,
                    record_extra={
                        "reference_of": {"ts": ref.get("ts"), "config_hash": ref.get("config_hash")}
                    },
                )
            return ld.context(self.phase)
        return ld.context(self.phase)


def _refusal(prev: list[dict]) -> str:
    when = ", ".join(str(r.get("ts"))[:19] for r in prev)
    return (
        f"this configuration was already unsealed ({when}): its holdout is used.  --force runs it again; "
        "the result is then flagged 'holdout already used'"
    )


def prepare_store_run(
    spec: StoreSpec,
    cfg: SystemConfig,
    *,
    use_ml: bool,
    ledger_path: str | None,
    unseal_reason: str | None = None,
    force: bool = False,
    hash_cache_dir: str | None = "runs/_cache/market_hash",
    log: Printer = _quiet,
    allow_partial_coverage: bool = False,
) -> StoreRun:
    """Check an ``--unseal`` (refused without a reason, without a ledger, or when this
    configuration was unsealed before and ``force`` is off), then load under the seal
    (:func:`qts.research.data.load_research_market`)."""
    phase = 2 if use_ml else 1
    seal = load_seal()
    prev: list[dict] = []
    unseal = unseal_reason is not None
    if unseal:
        if not str(unseal_reason).strip():
            raise SealError('--unseal needs a reason: --unseal "<why the holdout must be looked at>"')
        if seal is None:
            log("no seal in force: --unseal has nothing to lift (ignored)")
            unseal = False
        else:
            if not ledger_path:
                raise SealError("an unseal must be logged: give a trial ledger (--ledger)")
            prev = previous_unseals(ledger_path, cfg.hash(), phase)
            if prev and not force:
                raise SealError(_refusal(prev))
    load = load_research_market(
        spec,
        seal=seal,
        unseal=unseal,
        hash_cache_dir=hash_cache_dir,
        log=log,
        universe=cfg,
        allow_partial_coverage=allow_partial_coverage,
    )
    return StoreRun(
        load=load,
        phase=phase,
        unseal_reason=unseal_reason if unseal else None,
        previous_unseals=prev,
        force=force,
    )


__all__ = ["StoreRun", "prepare_store_run", "reference_trial"]
