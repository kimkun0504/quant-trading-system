"""Shared fixtures: one tiny real run (built once per session) + an "old" copy of it.

``fixture_root`` holds:
* the fixture run, written by ``ResearchPipeline(runs_root=...)`` (so by ``save_run`` and
  the ``write_extra_artifacts`` hook), and the trial ledger it logged to;
* ``OLD_RUN_ID``: the same run with the extra artifacts removed and the manifest's command
  set to a ``qts.cli demo`` line, i.e. a run directory written before ``extras.json`` existed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from qts_api.fixture import build_fixture_run
from qts_api.main import create_app
from qts_api.settings import Settings

OLD_RUN_ID = "20200101T000000000000Z_0ldrun00"
EXTRA_FILES = ("extras.json", "exante_vol.parquet", "exposures.parquet", "holdings.parquet")


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("runs")
    run_dir = build_fixture_run(root)
    old = root / OLD_RUN_ID
    shutil.copytree(run_dir, old)
    for f in EXTRA_FILES:
        (old / f).unlink()
    m = json.loads((old / "manifest.json").read_text())
    m["run_id"] = OLD_RUN_ID
    m["created_at"] = "2020-01-01T00:00:00+00:00"
    m["command"] = "qts/cli.py demo --stocks 60 --days 764 --seed 5"
    (old / "manifest.json").write_text(json.dumps(m))
    return root


@pytest.fixture(scope="session")
def run_id(fixture_root) -> str:
    return next(d.name for d in fixture_root.iterdir() if (d / "extras.json").exists())


@pytest.fixture(scope="session")
def run_dir(fixture_root, run_id) -> Path:
    return fixture_root / run_id


@pytest.fixture(scope="session")
def summary(run_dir) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


@pytest.fixture(scope="session")
def returns(run_dir) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "returns.parquet")


@pytest.fixture()
def client(fixture_root) -> TestClient:
    """App over the session's runs root; the job dispatcher is off (no worker processes)."""
    app = create_app(Settings(runs_root=fixture_root), start_jobs=False)
    with TestClient(app) as c:
        yield c
    for f in (fixture_root / "_jobs").glob("*.json"):
        f.unlink()
