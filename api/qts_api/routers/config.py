"""``/api/config/schema``: JSON Schema of SystemConfig (drives the run builder's editor);
``/api/config/warmup``: the effective warm-up of the default config;
``/api/config/validate``: dry run of a launch (errors, warnings, config hash, warm-up)."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from qts_api import schemas as S
from qts_api.deps import get_registry, get_settings
from qts_api.routers._common import problems
from qts_api.services.config_schema import config_json_schema
from qts_api.services.launch import prepare_launch, warmup_info
from qts_api.services.registry import RunRegistry
from qts_api.settings import Settings

__all__ = ["router", "warmup_info"]

router = APIRouter(prefix="/api/config", tags=["config"])


@lru_cache(maxsize=1)
def _schema() -> S.JsonSchemaObject:
    return S.JsonSchemaObject.model_validate(config_json_schema())


@router.get("/schema", response_model=S.JsonSchemaObject, response_model_exclude_none=True)
def get_config_schema() -> S.JsonSchemaObject:
    return _schema()


@router.get("/warmup", response_model=S.Warmup, summary="Effective warm-up for the default config")
def get_warmup(
    ml: Annotated[bool, Query(description="include the walk-forward ML sleeve")] = False,
    signal_lag_days: Annotated[int, Query(ge=0, le=1, description="execution delay in days (D4)")] = 1,
) -> dict[str, Any]:
    """Derived from ``risk.min_history_days``, ``alpha.train_min_days`` (with ML),
    ``alpha.horizon_days`` and the signal lag of the default ``SystemConfig``.  A
    launch with ``config_overrides`` is checked against its own config."""
    from qts.core.config import SystemConfig

    cfg = SystemConfig().with_overrides({"execution.signal_lag_days": int(signal_lag_days)})
    return warmup_info(cfg, ml)


@router.post(
    "/validate",
    response_model=S.ConfigValidation,
    responses=problems(422),
    summary="Dry run of a launch: errors, warnings, the resulting config hash and warm-up",
)
def validate_config(
    body: S.LaunchRunRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    registry: Annotated[RunRegistry, Depends(get_registry)],
) -> dict[str, Any]:
    """Takes the body of ``POST /api/runs`` and prepares it the same way (base config of
    ``base_run_id`` or the defaults, overrides through ``SystemConfig.with_overrides``,
    seed, signal lag), without queueing anything.  Config problems come back as
    ``errors`` with a 200 (``valid: false``); only a malformed body is a 422."""
    p = prepare_launch(body, settings, registry)
    wu = p.warmup or {}
    return {
        "valid": not p.errors,
        "errors": p.errors,
        "warnings": p.warnings,
        "config_hash": p.config_hash,
        "base_config_hash": p.base_config_hash,
        "warmup_days": wu.get("warmup_days"),
        "min_days": wu.get("min_days"),
        "overrides_applied": len(body.config_overrides),
    }
