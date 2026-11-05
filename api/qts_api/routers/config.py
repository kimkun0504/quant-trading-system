"""``/api/config/schema``: JSON Schema of SystemConfig for the launch form."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter

from qts_api import schemas as S
from qts_api.services.config_schema import config_json_schema

router = APIRouter(prefix="/api/config", tags=["config"])


@lru_cache(maxsize=1)
def _schema() -> S.JsonSchemaObject:
    return S.JsonSchemaObject.model_validate(config_json_schema())


@router.get("/schema", response_model=S.JsonSchemaObject, response_model_exclude_none=True)
def get_config_schema() -> S.JsonSchemaObject:
    return _schema()
