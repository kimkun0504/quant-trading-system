"""RunManifest and the all-panel data hash (handbook fix #37)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qts.core.manifest import RunManifest, git_info, hash_market_data, library_versions, new_run_id
from qts.data.synthetic import SyntheticSpec, generate


@pytest.fixture(scope="module")
def md():
    return generate(SyntheticSpec(n_stocks=30, n_days=120, seed=3))


def _copy(md):
    return dataclasses.replace(md, extra=dict(md.extra))


def test_data_hash_deterministic(md):
    assert hash_market_data(md) == hash_market_data(_copy(md))
    assert len(hash_market_data(md)) == 16


@pytest.mark.parametrize(
    "panel",
    ["close", "volume", "market_cap", "spread_bps", "borrow_fee", "short_interest_ratio", "book_to_price"],
)
def test_data_hash_covers_every_panel(md, panel):
    base = hash_market_data(md)
    v = getattr(md, panel)
    if v is None:
        pytest.skip(f"synthetic market has no {panel}")
    v2 = v.copy()
    v2.iloc[-1, 0] = (v2.iloc[-1, 0] if np.isfinite(v2.iloc[-1, 0]) else 1.0) * 1.0001 + 1e-9
    assert hash_market_data(dataclasses.replace(md, **{panel: v2})) != base


def test_data_hash_covers_optional_extra_industry_and_eligibility(md):
    base = hash_market_data(md)
    # an optional panel removed (None) differs from present
    assert hash_market_data(dataclasses.replace(md, borrow_fee=None)) != base
    # extra panels
    m2 = _copy(md)
    m2.extra["new_panel"] = md.close * 0 + 1.0
    assert hash_market_data(m2) != base
    # industry map
    ind = md.industry.copy()
    ind.iloc[0] = "SOMETHING_ELSE"
    assert hash_market_data(dataclasses.replace(md, industry=ind)) != base
    # eligibility panel (PIT universe)
    elig = md.close.notna()
    elig.iloc[-1, :5] = False
    assert hash_market_data(dataclasses.replace(md, eligible=elig)) != base


def test_manifest_round_trip(tmp_path):
    m = RunManifest.create(
        config_hash="abc123", data_hash="def456", timings={"fit": 1.5}, seeds={"system": 0}
    )
    assert m.run_id.endswith("_abc123") and m.created_at
    assert set(library_versions()) <= set(m.versions)
    for lib in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "cvxpy", "clarabel"):
        assert lib in m.versions
    assert m.git_sha and m.python_version and m.platform
    p = m.save(tmp_path / "run")
    assert p.name == "manifest.json"
    back = RunManifest.load(tmp_path / "run")
    assert back == m


def test_git_info_outside_repo(tmp_path):
    g = git_info(cwd=tmp_path)
    assert g["git_sha"] == "unknown" and g["git_dirty"] is None


def test_run_id_sortable():
    a, b = new_run_id("aaaaaaaa1"), new_run_id("bbbbbbbb2")
    assert a[:8].isdigit() and a.endswith("_aaaaaaaa") and b.endswith("_bbbbbbbb")
