"""``/api/ledger``: the trial ledger (input to every run's deflated Sharpe)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from qts_api import schemas as S
from qts_api.deps import get_ledger
from qts_api.services.ledger import LedgerService

router = APIRouter(prefix="/api/ledger", tags=["ledger"])


@router.get("", response_model=S.Ledger)
def get_ledger_(ledger: Annotated[LedgerService, Depends(get_ledger)]) -> dict[str, Any]:
    return ledger.get()
