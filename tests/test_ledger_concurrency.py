"""Trial ledger identity and concurrency (audit 2, findings 2, 3, 14, 17)."""

from __future__ import annotations

import functools
import json
import math
import multiprocessing as mp
import subprocess

import numpy as np
import pytest
from scipy.stats import norm

from qts.core.manifest import code_version, git_info
from qts.validation.deflated_sharpe import (
    EULER_GAMMA,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from qts.validation.ledger import TrialLedger, callable_identity


def _log(path: str, k: int, **kw) -> None:
    TrialLedger(path).log(
        f"fam:{k}",
        {"k": k},
        {"sharpe_annual": 0.1 * (k % 7), "n_obs": 500},
        data_hash="d",
        family="fam",
        code_version="abc",
        **kw,
    )


def _worker(path: str, ks: list[int]) -> None:
    for k in ks:
        _log(path, k)


def _rerunner(path: str, n: int) -> None:
    led = TrialLedger(path)
    for _ in range(n):
        led.log(
            "rerun",
            {"k": -1},
            {"sharpe_annual": 1.0, "n_obs": 500},
            data_hash="d",
            family="fam",
            config_hash="same",
            code_version="abc",
            replace_identical=True,
        )


def test_concurrent_appends_all_present(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    TrialLedger(path)
    ctx = mp.get_context("fork")
    chunks = [list(range(i, 400, 8)) for i in range(8)]
    procs = [ctx.Process(target=_worker, args=(path, c)) for c in chunks]
    procs.append(ctx.Process(target=_rerunner, args=(path, 50)))
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    led = TrialLedger(path)
    raw = led.raw_records()
    assert len(raw) == 450  # every line intact (each parses as JSON)
    names = {r["name"] for r in raw}
    assert {f"fam:{k}" for k in range(400)} <= names
    # 400 distinct trials + the 50 identical re-runs counted once
    assert led.n_trials("fam") == 401
    rerun = [r for r in led.records("fam") if r["name"] == "rerun"]
    assert len(rerun) == 1 and rerun[0]["n_reruns"] == 49
    with open(path) as f:
        for line in f:
            json.loads(line)


def _kw(**over):
    kw = {
        "name": "p",
        "config": {"a": 1},
        "metrics": {"sharpe_annual": 1.0, "n_obs": 300},
        "data_hash": "d1",
        "config_hash": "c1",
        "code_version": "sha1",
        "replace_identical": True,
        "run_spec": {"use_ml": False, "venue": "routed"},
    }
    kw.update(over)
    return kw


def test_ledger_key_includes_code_version_and_run_spec(tmp_path):
    led = TrialLedger(tmp_path / "l.jsonl")
    led.log(**_kw())
    led.log(**_kw())  # identical clean re-run -> same trial
    assert led.n_trials("p") == 1
    led.log(**_kw(code_version="sha2"))  # new code -> new trial
    assert led.n_trials("p") == 2
    led.log(**_kw(run_spec={"use_ml": True, "venue": "routed"}))  # knob outside the config
    assert led.n_trials("p") == 3
    led.log(**_kw(run_spec={"use_ml": False, "venue": "close_auction"}))
    assert led.n_trials("p") == 4
    # dirty / unknown / empty code versions never de-duplicate (audit repro: was 1 trial)
    for cv in ("sha_B+dirty", "sha_B+dirty", "unknown", "unknown", ""):
        led.log(**_kw(code_version=cv))
    assert led.n_trials("p") == 9
    # the surviving clean record is the latest one, at the first one's position
    led.log(**_kw(metrics={"sharpe_annual": 1.6, "n_obs": 300}))
    recs = led.records("p")
    assert len(recs) == 9 and recs[0]["metrics"]["sharpe_annual"] == 1.6 and recs[0]["n_reruns"] == 2
    assert len(led.raw_records()) == 11


def test_callable_identity_distinguishes_factories():
    def make(alpha):
        return lambda: alpha

    f1, f2 = make(1.0), make(2.0)
    assert callable_identity(None) is None
    assert callable_identity(f1) != callable_identity(f2)  # same qualname, different closure
    assert callable_identity(f1) == callable_identity(make(1.0))  # stable across instances
    p1 = functools.partial(dict, a=1)
    assert callable_identity(p1) != callable_identity(functools.partial(dict, a=2))
    assert "RidgeModel" in callable_identity(__import__("qts.alpha.models", fromlist=["x"]).RidgeModel)


def test_expected_max_sharpe_uses_fractional_trials():
    # hand computation of Bailey & LdP eq. 8 at N = 2.4, V = 1
    n = 2.4
    z1, z2 = norm.ppf(1 - 1 / n), norm.ppf(1 - 1 / (n * math.e))
    want = (1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2
    assert expected_max_sharpe(2.4, 1.0) == pytest.approx(want, rel=1e-12)
    assert expected_max_sharpe(2.4, 1.0) > expected_max_sharpe(2, 1.0)  # no rounding down
    assert expected_max_sharpe(1.0, 1.0) == 0.0


def test_expected_max_sharpe_never_negative_for_small_effective_n():
    # The closed form alone gives -0.79 at N = 1.01 and is negative up to N ~ 1.28,
    # which would make the DSR benchmark negative and inflate the DSR.  A
    # near-duplicate family (Nyholt N_eff just above 1) must deflate at least as
    # hard as a single trial, never less.
    n2 = expected_max_sharpe(2.0, 1.0)
    assert n2 == pytest.approx(0.52, abs=0.01)
    prev = 0.0
    for n in np.linspace(1.0, 3.0, 201):
        v = expected_max_sharpe(float(n), 1.0)
        assert v >= 0.0
        assert v >= prev - 1e-12  # non-decreasing in N
        prev = v
    assert expected_max_sharpe(1.5, 1.0) == pytest.approx(0.5 * n2)
    assert expected_max_sharpe(2.0 - 1e-9, 1.0) == pytest.approx(n2, abs=1e-6)  # continuous at 2
    sr_d = 0.55 / math.sqrt(252)
    assert deflated_sharpe_ratio(sr_d, 1.0096, 0.251 / 252, 1005) <= deflated_sharpe_ratio(
        sr_d, 1.0, 0.0, 1005
    )


def test_git_dirty_counts_untracked_files(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    try:
        git("init", "-q")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    assert git_info(tmp_path)["git_dirty"] is False
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "out.txt").write_text("run output\n")
    assert git_info(tmp_path)["git_dirty"] is False  # .gitignore respected
    (tmp_path / "new_module.py").write_text("y = 2\n")
    info = git_info(tmp_path)
    assert info["git_dirty"] is True  # untracked module the run may import
    assert code_version(info).endswith("+dirty")
    assert code_version({"git_sha": "unknown", "git_dirty": None}) == "unknown"
    assert code_version({"git_sha": "abc", "git_dirty": None}) == "abc+dirty"
    assert code_version({"git_sha": "abc", "git_dirty": False}) == "abc"
