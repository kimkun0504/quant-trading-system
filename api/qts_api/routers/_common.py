from __future__ import annotations

from typing import Any

PROBLEM = {
    "content": {"application/problem+json": {"schema": {"$ref": "#/components/schemas/ApiErrorBody"}}},
}


def problems(*codes: int) -> dict[int | str, dict[str, Any]]:
    desc = {
        404: "Not found",
        409: "The run is still running or failed before writing results",
        422: "Invalid request",
        429: "Too many runs are already queued",
        500: "Internal error",
    }
    return {c: {**PROBLEM, "description": desc.get(c, "Error")} for c in codes}
