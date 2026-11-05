"""``/api/config/schema``: JSON Schema of SystemConfig for the launch form;
``/api/config/warmup``: the effective warm-up the launch form validates ``days`` against."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Query

from qts_api import schemas as S
from qts_api.services.config_schema import config_json_schema

router = APIRouter(prefix="/api/config", tags=["config"])


@lru_cache(maxsize=1)
def _schema() -> S.JsonSchemaObject:
    return S.JsonSchemaObject.model_validate(config_json_schema())


@router.get("/schema", response_model=S.JsonSchemaObject, response_model_exclude_none=True)
def get_config_schema() -> S.JsonSchemaObject:
    return _schema()


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


@router.get("/warmup", response_model=S.Warmup, summary="Effective warm-up for the default config")
def get_warmup(
    ml: Annotated[bool, Query(description="include the walk-forward ML sleeve")] = False,
    signal_lag_days: Annotated[int, Query(ge=0, le=1, description="execution delay in days (D4)")] = 1,
) -> dict[str, Any]:
    """Derived from ``risk.min_history_days``, ``alpha.train_min_days`` (with ML),
    ``alpha.horizon_days`` and the signal lag of the default ``SystemConfig``.  A
    launch with ``config_overrides`` is checked against its own config."""
    from qts.core.config import SystemConfig

    cfg = SystemConfig()
    cfg.execution.signal_lag_days = int(signal_lag_days)
    return warmup_info(cfg, ml)
