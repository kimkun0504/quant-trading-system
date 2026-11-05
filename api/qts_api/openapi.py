"""Export the OpenAPI document: ``python -m qts_api.openapi [out.json]`` (stdout by default)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def openapi_document() -> dict:
    from qts_api.main import create_app
    from qts_api.settings import Settings

    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Settings(runs_root=Path(tmp)), start_jobs=False)
        return app.openapi()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    text = json.dumps(openapi_document(), indent=2, sort_keys=False) + "\n"
    if argv:
        Path(argv[0]).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
