"""Serve the built UI and the API over a runs root that holds the fixture run.

Used by ``ui/playwright.real.config.ts``::

    python api/scripts/e2e_server.py --port 8765 [--runs-root DIR]

The runs root (default ``$TMPDIR/qts-e2e-runs``) is reused across invocations; the
fixture run (~20-40 s) is built only when no run directory is there yet.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--runs-root", default=str(Path(tempfile.gettempdir()) / "qts-e2e-runs"))
    a = ap.parse_args()
    root = Path(a.runs_root)
    sys.path.insert(0, str(REPO / "api"))
    from qts.pipeline import list_runs

    if not list_runs(root):
        from qts_api.fixture import build_fixture_run

        print(f"building the fixture run under {root} ...", flush=True)
        print(build_fixture_run(root), flush=True)
    os.environ.update(
        QTS_RUNS_ROOT=str(root),
        QTS_SERVE_UI="1",
        QTS_UI_DIST=str(REPO / "ui" / "dist"),
        QTS_CORS_ORIGINS=f"http://localhost:{a.port}",
    )
    import uvicorn

    uvicorn.run("qts_api.main:app", host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
