"""One code path for ``POST /api/config/validate`` and ``POST /api/runs``: turn a launch
request into the ``SystemConfig`` the worker will run, or the list of issues.

Order (the same as ``qts demo`` / ``qts run``): base config (``SystemConfig()``, or the
``config.yaml`` of ``base_run_id``: the run builder's "start from a run") -> dotted
overrides through ``SystemConfig.with_overrides`` (strictly checked first, see
:mod:`qts_api.services.config_schema`) -> the request's own ``seed`` and
``signal_lag_days`` -> for a synthetic market, ``universe.top_n_by_mcap`` clamped to the
number of stocks (what the pipeline does on load, so the hash shown before launch is
the hash of the run).  The stage cache is operational and outside the hash: a base
run's ``cache`` section is reset and the request's ``cache`` flag is applied in the
worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qts_api import schemas as S
from qts_api.services.config_schema import ConfigError, validate_overrides
from qts_api.services.registry import RunNotFound, RunRegistry, RunUnreadable
from qts_api.settings import Settings

Issue = dict[str, Any]


def _issue(loc: list[str | int], msg: str, type_: str = "value_error") -> Issue:
    return {"loc": loc, "msg": msg, "type": type_}


@dataclass
class Prepared:
    cfg: Any | None  # SystemConfig, None when there are errors
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    base_config_hash: str | None = None
    warmup: dict[str, Any] | None = None

    @property
    def config_hash(self) -> str | None:
        return None if self.cfg is None or self.errors else self.cfg.hash()


def warmup_info(cfg: Any, ml: bool) -> dict[str, Any]:
    """``qts.pipeline.estimate_warmup_days`` as the :class:`~qts_api.schemas.Warmup` body."""
    from qts.pipeline import estimate_warmup_days

    est = estimate_warmup_days(cfg, use_ml=ml)
    tdy = int(cfg.trading_days_per_year)
    return {
        "ml": bool(ml),
        "signal_lag_days": est.signal_lag_days,
        "warmup_days": est.days,
        "min_days": est.days + tdy,
        "risk_days": est.risk_days,
        "ml_days": est.ml_days,
        "exact": est.exact,
        "trading_days_per_year": tdy,
    }


def _base_config(body: S.LaunchRunRequest, registry: RunRegistry, errors: list[Issue]) -> Any | None:
    from qts.core.config import SystemConfig

    if not body.base_run_id:
        return SystemConfig()
    loc: list[str | int] = ["body", "base_run_id"]
    try:
        entry = registry.get(body.base_run_id)
    except (RunNotFound, RunUnreadable):
        errors.append(_issue(loc, f"no finished run with id {body.base_run_id[:64]} to start from"))
        return None
    try:
        cfg = SystemConfig.from_dict(entry.config)
    except (KeyError, TypeError, ValueError) as e:
        errors.append(
            _issue(
                loc,
                f"the config of run {entry.run_id} no longer loads with this version of SystemConfig "
                f"({str(e)[:200]}); start from the defaults instead",
            )
        )
        return None
    # operational section: never inherited (the request's `cache` flag decides)
    return cfg.with_overrides({"cache.enabled": False, "cache.root": ""})


def prepare_launch(body: S.LaunchRunRequest, settings: Settings, registry: RunRegistry) -> Prepared:
    errors: list[Issue] = []
    warnings: list[Issue] = []
    if body.phase == 2 and not body.ml:
        errors.append(_issue(["body", "ml"], "phase 2 tests the ML sleeve: set ml=true or phase=1"))
    if body.kind == "store":
        if settings.store_path is None:
            errors.append(_issue(["body", "kind"], "store runs need QTS_STORE_PATH set on the API server"))
        if body.start and body.end and body.start > body.end:
            errors.append(_issue(["body", "end"], f"end ({body.end}) is before start ({body.start})"))
    base = _base_config(body, registry, errors)
    base_hash = base.hash() if base is not None else None
    cfg = None
    if base is not None:
        try:
            cfg = validate_overrides(body.config_overrides, base)
        except ConfigError as e:
            errors += [_issue(loc, msg, t) for loc, msg, t in e.issues]
    if cfg is None or errors:
        return Prepared(None, errors, warnings, base_hash)

    try:
        cfg = cfg.with_overrides({"seed": body.seed, "execution.signal_lag_days": body.signal_lag_days})
        if body.kind == "synthetic":
            cfg = cfg.with_overrides({"universe.top_n_by_mcap": min(cfg.universe.top_n_by_mcap, body.stocks)})
        cfg.validate()
    except (KeyError, TypeError, ValueError) as e:
        return Prepared(None, [_issue(["body", "config_overrides"], str(e), "value_error.config")], warnings, base_hash)
    wu = warmup_info(cfg, body.ml)
    if body.kind == "synthetic" and body.days < wu["min_days"]:
        # P7: the floor is the run's own effective warm-up plus one year
        errors.append(
            _issue(
                ["body", "days"],
                f"days must be at least {wu['min_days']} ({wu['warmup_days']} warm-up days "
                f"for this configuration plus one year)",
            )
        )
        return Prepared(None, errors, warnings, base_hash, wu)

    # ---- warnings: launchable, but worth knowing before a trial is spent
    if body.signal_lag_days == 0:
        warnings.append(
            _issue(
                ["body", "signal_lag_days"],
                "same-close execution (lag 0) trades at the price that produced the signal: optimistic, "
                "and the run will be flagged suspect",
                "warning.lag0",
            )
        )
    if not body.ml:
        ml_keys = sorted(k for k in body.config_overrides if k.startswith(("alpha.lgbm.", "alpha.mlp.")))
        if ml_keys:
            warnings.append(
                _issue(
                    ["body", "config_overrides", ml_keys[0]],
                    f"{', '.join(ml_keys)} only affect{'s' if len(ml_keys) == 1 else ''} the ML sleeve, "
                    "which is off: the result will not change",
                    "warning.ml_off",
                )
            )
    h = cfg.hash()
    same = [e for e in registry.entries() if str(e.manifest.get("config_hash", "")) == h]
    if same:
        newest = max(same, key=lambda e: str(e.manifest.get("created_at", "")))
        label = str((newest.extras.get("run") or {}).get("label") or newest.run_id)
        warnings.append(
            _issue(
                ["body", "config_overrides"],
                f"same configuration as run {newest.run_id} ({label}): on the same data and clean code "
                "this is a re-run, not a new trial",
                "warning.same_config",
            )
        )
    return Prepared(cfg, errors, warnings, base_hash, wu)
