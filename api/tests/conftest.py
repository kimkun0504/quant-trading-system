"""Shared fixtures: one tiny real run (built once per session) + an "old" copy of it.

``fixture_root`` holds:
* the fixture run, written by ``ResearchPipeline(runs_root=...)`` (so by ``save_run`` and
  the ``write_extra_artifacts`` hook), and the trial ledger it logged to;
* ``OLD_RUN_ID``: the same run with the extra artifacts removed and the manifest's command
  set to a ``qts.cli demo`` line, i.e. a run directory written before ``extras.json`` existed.

The research holdout seal (``research/seal.yaml``) is off in these tests, as in the
library's (``tests/conftest.py``): ``QTS_SEAL_FILE=none`` is set before anything runs
(worker processes inherit it), and a test that needs a seal sets it to its own file
(``world`` below: :func:`qts_api.research_fixture.build_research_world`).
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


def pytest_configure(config):
    import os

    # never the repo's real seal: synthetic stores end in 2020; a test turns a seal on itself
    os.environ["QTS_SEAL_FILE"] = "none"
    # the store load-coverage guard (qts.research.coverage) refuses the small synthetic
    # prefilters; off here as in the library's suite (a test turns it on with monkeypatch)
    os.environ.setdefault("QTS_LOAD_COVERAGE", "off")


#: TestClient base URL: the app only accepts Host headers on its allow-list (localhost, ...).
LOCAL = "http://localhost"
EXTRA_FILES = (
    "extras.json",
    "exante_vol.parquet",
    "exposures.parquet",
    "holdings.parquet",
    "alpha_refits.json",
    "theme_weights.parquet",
    "alpha_ic_by_sleeve.parquet",
    "constraints.parquet",
)


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("runs")
    run_dir = build_fixture_run(root)
    old = root / OLD_RUN_ID
    shutil.copytree(run_dir, old)
    for f in EXTRA_FILES:
        (old / f).unlink(missing_ok=True)
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
    with TestClient(app, base_url=LOCAL) as c:
        yield c
    for f in (fixture_root / "_jobs").glob("*.json"):
        f.unlink()


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """Every research tool and a paper book run once on a synthetic store under a seal
    (about a minute; :mod:`qts_api.research_fixture`).  The seal stays on only inside
    the tests that ask for ``sealed`` below."""
    import os

    from qts_api.research_fixture import build_research_world

    root = tmp_path_factory.mktemp("world")
    before = os.environ.get("QTS_SEAL_FILE")
    try:
        w = build_research_world(root)
    finally:
        os.environ["QTS_SEAL_FILE"] = before if before is not None else "none"
    return w


@pytest.fixture()
def sealed(world, monkeypatch):
    monkeypatch.setenv("QTS_SEAL_FILE", str(world.seal_file))
    return world


@pytest.fixture()
def wclient(sealed, tmp_path):
    """App over the world (its runs root, store, paper books); job dispatcher off, jobs
    written to a fresh directory so tests do not see each other's queued jobs."""
    s = Settings(runs_root=sealed.runs_root, store_path=sealed.store, paper_root=sealed.paper_root)
    app = create_app(s, start_jobs=False)
    jobs = app.state.jobs
    jobs.dir = tmp_path / "_jobs"
    jobs.dir.mkdir()
    with TestClient(app, base_url=LOCAL) as c:
        yield c
