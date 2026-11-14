"""Research routes over a real research world (``conftest.world``: a sealed store, the
research tools run once, a paper book): every read route against the library's own
files and functions, every launch route's validation and queueing (dispatcher off), and
the trial previews (family, count, DSR haircut) against the library."""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from qts_api import schemas as S
from qts_api.main import create_app
from qts_api.services.research import MIDCAP_FORCE_PHRASE
from qts_api.settings import Settings

from .conftest import LOCAL

STORE = {"max_names": 50}


def queued(c: TestClient) -> list[dict]:
    return [j for j in c.get("/api/jobs").json()["jobs"] if j["status"] == "queued"]


# --------------------------------------------------------------------------- seal
def test_seal_status_and_unseal_check(wclient, sealed):
    st = S.SealStatus.model_validate(wclient.get("/api/seal").json())
    assert st.seal is not None and st.seal.holdout_start == sealed.holdout
    assert st.data_through == str((pd.Timestamp(sealed.holdout) - pd.Timedelta(days=1)).date())
    assert len(st.unseals) == 1 and st.n_forced == 0 and st.unseals[0].unseal_reason == "fixture: final check"
    from qts.validation import TrialLedger

    fam = st.seal.inherit_family[0]
    assert st.inherited_families[fam] == TrialLedger(str(sealed.ledger)).n_trials(fam) == 5
    # under the seal: the sealed run, the variant, two sweep members, and the reference re-run
    assert st.by_family[fam] == {"trial": 4, "reference": 1}
    assert st.exempt_runs and st.exempt_runs[0].seal_exempt == "midcap"
    unsealed_hash = st.unseals[0].config_hash
    u = S.UnsealCheck.model_validate(wclient.get(f"/api/seal/unseals?config_hash={unsealed_hash}&phase=1").json())
    assert u.requires_force and u.sealed and len(u.previous) == 1
    other = wclient.get("/api/seal/unseals?config_hash=0123456789abcdef&phase=1").json()
    assert other["requires_force"] is False and other["previous"] == []
    assert wclient.get("/api/seal/unseals?config_hash=../x").status_code == 422


def test_runs_carry_their_seal_state(wclient, sealed):
    runs = {r["id"]: r for r in wclient.get("/api/runs").json()}
    ids = sealed.run_ids
    assert (
        runs[ids["sealed"]]["seal"]["state"] == "sealed"
        and runs[ids["sealed"]]["seal"]["sealed_end"] == sealed.sealed_end
    )
    assert runs[ids["reference"]]["seal"]["reference"] is True
    un = runs[ids["unsealed"]]["seal"]
    assert (
        un["state"] == "unsealed" and un["unseal_reason"] == "fixture: final check" and not un["holdout_already_used"]
    )
    assert runs[sealed.ids["midcap"]]["seal"]["state"] == "exempt"
    caps = [r for r in runs.values() if r["seal"]["record_kind"] == "capacity"]
    assert len(caps) == 4
    d = wclient.get(f"/api/runs/{ids['unsealed']}").json()
    assert d["seal"]["state"] == "unsealed"


def test_ledger_marks_records_that_are_not_counted(wclient, sealed):
    led = wclient.get("/api/ledger").json()  # (an unseal event has no Sharpe: null)
    by_kind: dict[str, list] = {}
    for t in led["trials"]:
        by_kind.setdefault(t["kind"], []).append(t)
    assert {"trial", "unseal", "capacity", "calibration"} <= set(by_kind)
    for k in ("unseal", "capacity", "calibration"):
        assert all(not t["counted"] and t["note"] for t in by_kind[k])
    refs = [t for t in led["trials"] if t["reference"]]
    assert len(refs) == 1 and not refs[0]["counted"] and "reference" in (refs[0]["note"] or "")
    assert all(t["counted"] and t["note"] is None for t in by_kind["trial"] if not t["reference"])
    assert [t["unsealed"] for t in by_kind["unseal"]] == [False]  # the event itself; its run is unsealed
    fam = {f["family"]: f for f in led["families"]}
    seal_fam = wclient.get("/api/seal").json()["seal"]["inherit_family"][0]
    assert fam[seal_fam]["n_not_counted"] == 2 and fam[seal_fam]["n_trials"] == 5  # the reference, the unseal event
    assert any(k.startswith("capacity:") and f["n_trials"] == 0 and f["n_not_counted"] == 4 for k, f in fam.items())


# --------------------------------------------------------------------------- impact / features
def test_trial_impact_is_the_library_function(wclient, sealed):
    from dataclasses import asdict

    from qts.research.sweep import trial_impact

    fam = wclient.get("/api/seal").json()["seal"]["inherit_family"][0]
    got = wclient.get(f"/api/ledger/impact?family={fam}&n_new=3").json()
    want = asdict(trial_impact(str(sealed.ledger), fam, 3, 0))
    assert got["n_trials_raw"] == want["n_trials_raw"] == 5 and got["n_new"] == 3
    assert got["dsr_best_after"] == pytest.approx(want["dsr_best_after"])
    assert got["dsr_best_after"] <= got["dsr_best_before"]  # more trials: a haircut, never a bonus
    assert wclient.get("/api/ledger/impact?family=x").status_code == 422


def test_features(wclient):
    v1 = S.FeatureList.model_validate(wclient.get("/api/features?feature_set=v1").json())
    v2 = S.FeatureList.model_validate(wclient.get("/api/features?feature_set=v2").json())
    assert len(v1.features) == 18 and len(v2.features) == 26
    assert all(f.description for f in v2.features)
    mom = next(t for t in v1.themes if "mom_12_1" in t.members)
    assert {f.name for f in v1.features} >= set(mom.members)
    assert wclient.get("/api/features?feature_set=v9").status_code == 422


# --------------------------------------------------------------------------- signals
def test_signal_list_tests_and_holm_now(wclient, sealed):
    sigs = [S.SignalDef.model_validate(x) for x in wclient.get("/api/signals").json()]
    assert {"mom_12_1", "high_52w"} <= {s.name for s in sigs} and all(s.description for s in sigs)
    tests = S.SignalTestList.model_validate(wclient.get("/api/signals/tests").json()).tests
    assert [t.signal for t in tests] == ["strev_21", "mom_12_1"]  # newest first
    assert all(t.n_tests_now == 2 for t in tests)
    first = tests[1]
    assert first.p_holm_now is not None and first.p_holm_at_test is not None
    assert first.p_holm_now >= first.p_holm_at_test - 1e-12  # a later test can only raise it
    d = S.SignalTestDetail.model_validate(wclient.get(f"/api/signals/tests/{first.id}").json())
    raw = json.loads((sealed.runs_root / "_signals" / f"{first.id}.json").read_text())
    assert d.result.trial.t == pytest.approx(raw["trial"]["t"]) and d.family_now.n_tests == 2
    assert d.family_now.p_holm == pytest.approx(first.p_holm_now)
    assert set(d.result.ic_by_horizon) == {"5", "20", "60"}
    assert wclient.get("/api/signals/tests/nope").status_code == 404
    assert wclient.get("/api/signals/tests/..%2F..%2Fetc").status_code == 404


def test_signal_preview_and_launch(wclient, sealed):
    body = {"feature": "high_52w", "store": STORE, "horizons": [60, 5, 20]}
    p = S.SignalTestPreview.model_validate(wclient.post("/api/signals/tests/preview", json=body).json())
    assert p.family and p.family.startswith("signal:") and p.n_tests == 2 and p.n_tests_after == 3 and p.adds == 1
    assert p.command.startswith("qts signal test high_52w --store ") and "--horizons 5,20,60" in p.command
    nolog = wclient.post("/api/signals/tests/preview", json={**body, "log_trial": False}).json()
    assert nolog["adds"] == 0 and nolog["n_tests_after"] == 2 and "--no-log" in nolog["command"]
    other = wclient.post("/api/signals/tests/preview", json={**body, "store": {"max_names": 40}}).json()
    assert (
        other["family"] is None and not other["resolution"]["known"] and "full-history" in other["resolution"]["reason"]
    )
    r = wclient.post("/api/signals/tests", json=body)
    assert r.status_code == 202, r.text
    out = S.TaskLaunchResponse.model_validate(r.json())
    job = S.Job.model_validate(wclient.get(f"/api/jobs/{out.job_id}").json())
    assert job.task == "signal_test" and job.subject == "high_52w" and job.command == out.command == p.command
    assert job.status == "queued" and job.run_id == ""
    # not a run: the runs list does not show task jobs
    assert all(rr["id"] for rr in wclient.get("/api/runs").json())
    for bad in (
        {**body, "feature": "nope"},
        {**body, "horizons": []},
        {**body, "horizons": [5, 5]},
        {**body, "horizons": [500]},
        {**body, "baseline": "/etc/passwd"},
        {**body, "store": {"max_names": "50"}},
    ):
        assert wclient.post("/api/signals/tests", json=bad).status_code == 422, bad


def test_research_launches_need_a_store(tmp_path, sealed):
    app = create_app(Settings(runs_root=tmp_path), start_jobs=False)
    with TestClient(app, base_url=LOCAL) as c:
        r = c.post("/api/signals/tests", json={"feature": "mom_12_1", "horizons": [5]})
        assert r.status_code == 422 and "QTS_STORE_PATH" in r.json()["detail"]
        assert c.post("/api/capacity", json={"navs": [5e6]}).status_code == 422
        assert c.get("/api/signals/tests").json() == {"tests": []}
        assert c.get("/api/sweeps").json() == {"sweeps": []}
        assert c.get("/api/paper/books").json()["books"] == []


# --------------------------------------------------------------------------- sweeps
SWEEP = {
    "params": [{"key": "portfolio.trade_aversion", "values": [0.5, 1.0, 2.0]}],
    "base_config": "configs/baseline.yaml",
    "store": STORE,
    "use_ml": False,
    "max_days": 60,
}


def test_sweep_list_and_detail(wclient, sealed):
    lst = S.SweepList.model_validate(wclient.get("/api/sweeps").json()).sweeps
    assert [s.id for s in lst] == [sealed.ids["sweep"]] and lst[0].n_done == 2 and lst[0].status == "done"
    d = S.SweepDetail.model_validate(wclient.get(f"/api/sweeps/{sealed.ids['sweep']}").json())
    assert d.sweep.pbo is not None and d.sweep.pbo.partial is False and d.resumable == 0 and d.active == 0
    assert all(m.metrics.dsr is not None for m in d.sweep.members)
    assert d.command.startswith("qts sweep --config ") and "--param portfolio.trade_aversion=0.5,1.5" in d.command
    assert wclient.get("/api/sweeps/sw_000000000000").status_code == 404
    assert wclient.get("/api/sweeps/notanid").status_code == 404


def test_sweep_plan_confirm_create_and_queue(wclient, sealed):
    from qts.research.sweep import trial_impact

    v = S.SweepPlanView.model_validate(wclient.post("/api/sweeps/plan", json=SWEEP).json())
    plan = v.plan
    assert len(plan.members) == 3 and not v.exists and v.n_pending == 3
    assert plan.family == wclient.get("/api/seal").json()["seal"]["inherit_family"][0]
    imp = trial_impact(str(sealed.ledger), plan.family, 3, 0)
    assert plan.impact is not None and plan.impact.n_new == 3 and plan.impact.n_trials_raw == imp.n_trials_raw
    assert plan.impact.dsr_best_after == pytest.approx(imp.dsr_best_after)
    assert not (sealed.runs_root / "_sweeps" / f"{plan.id}.json").exists()  # nothing written by a plan
    # create: refused when the confirmed plan is not the plan
    assert wclient.post("/api/sweeps", json={**SWEEP, "expected_id": "sw_000000000000"}).status_code == 409
    r = wclient.post("/api/sweeps", json={**SWEEP, "expected_id": plan.id})
    assert r.status_code == 202, r.text
    out = S.SweepLaunchResponse.model_validate(r.json())
    assert len(out.job_ids) == 3 and out.finalize_job_id and out.n_queued == 4
    q = queued(wclient)
    assert [j["task"] for j in q] == ["sweep_member"] * 3 + ["sweep_finalize"]
    assert [j["subject"] for j in q] == [plan.id] * 4 and "member 1 of 3" in q[0]["label"]
    assert (sealed.runs_root / "_sweeps" / f"{plan.id}.json").exists()
    # resume while queued: nothing to add
    assert wclient.post(f"/api/sweeps/{plan.id}/resume").status_code == 409
    d = wclient.get(f"/api/sweeps/{plan.id}").json()
    assert d["active"] == 4 and d["resumable"] == 0
    # an interrupted sweep: one member's job cancelled -> resume queues it (and PBO) once
    wclient.delete(f"/api/jobs/{out.job_ids[1]}")
    wclient.delete(f"/api/jobs/{out.finalize_job_id}")
    r = wclient.post(f"/api/sweeps/{plan.id}/resume")
    assert r.status_code == 202 and len(r.json()["job_ids"]) == 1 and r.json()["finalize_job_id"]
    assert sum(1 for j in queued(wclient) if j["task"] == "sweep_member") == 3
    (sealed.runs_root / "_sweeps" / f"{plan.id}.json").unlink(missing_ok=True)


def test_sweep_validation_and_unknown_family(wclient):
    bad = [
        {**SWEEP, "params": []},
        {**SWEEP, "params": [{"key": "portfolio.trade_aversion", "values": []}]},
        {**SWEEP, "params": [{"key": "Bad Key", "values": [1]}]},
        {**SWEEP, "max_runs": 13},
    ]
    for b in bad:
        assert wclient.post("/api/sweeps/plan", json=b).status_code == 422, b
    big = {**SWEEP, "params": [{"key": "portfolio.trade_aversion", "values": [0.1 * i for i in range(1, 5)]},
                               {"key": "portfolio.hold_aversion", "values": [0.5, 1.0, 2.0, 3.0]}]}  # fmt: skip
    r = wclient.post("/api/sweeps/plan", json=big)
    assert r.status_code == 422 and "max-runs" in r.json()["detail"]
    unknown = wclient.post("/api/sweeps/plan", json={**SWEEP, "params": [{"key": "portfolio.nope", "values": [1]}]})
    assert unknown.status_code == 422
    r = wclient.post("/api/sweeps/plan", json={**SWEEP, "store": {"max_names": 40}})
    assert r.status_code == 409 and r.json()["type"].endswith("family-unknown")
    # resolving the family is a job
    j = wclient.post("/api/families/resolve/jobs", json={"max_names": 40})
    assert j.status_code == 202 and "--max-names 40" in j.json()["command"]
    assert wclient.post("/api/families/resolve", json={"max_names": 50}).json()["known"] is True


def test_sweep_queue_capacity_is_checked_before_queueing(sealed, tmp_path):
    s = Settings(runs_root=sealed.runs_root, store_path=sealed.store, paper_root=sealed.paper_root, max_queued=2)
    app = create_app(s, start_jobs=False)
    app.state.jobs.dir = tmp_path / "_jobs"
    app.state.jobs.dir.mkdir()
    with TestClient(app, base_url=LOCAL) as c:
        plan = c.post("/api/sweeps/plan", json=SWEEP).json()["plan"]
        r = c.post("/api/sweeps", json={**SWEEP, "expected_id": plan["id"]})
        assert r.status_code == 429 and queued(c) == []
    (sealed.runs_root / "_sweeps" / f"{plan['id']}.json").unlink(missing_ok=True)


# --------------------------------------------------------------------------- capacity / risk
def test_capacity_routes(wclient, sealed):
    lst = S.CapacityList.model_validate(wclient.get("/api/capacity").json()).results
    assert lst[0].id == sealed.ids["capacity"] and lst[0].navs == [5e6, 5e7, 2.5e8, 1e9]
    d = S.CapacityResult.model_validate(wclient.get(f"/api/capacity/{sealed.ids['capacity']}").json())
    assert len(d.points) == 4 and d.family.startswith("capacity:")
    assert wclient.get("/api/capacity/nope").status_code == 404
    r = wclient.post("/api/capacity", json={"store": STORE, "navs": [5e7, 5e6], "max_days": 120})
    assert r.status_code == 202
    assert r.json()["command"].endswith("--max-names 50 --navs 5e+06,5e+07 --max-days 120")
    for bad in ({"navs": []}, {"navs": [5e6, 5e6]}, {"navs": [-1]}, {"navs": [5e6], "max_days": 3}):
        assert wclient.post("/api/capacity", json={"store": STORE, **bad}).status_code == 422, bad


def test_risk_routes(wclient, sealed):
    lst = S.RiskCalibrationList.model_validate(wclient.get("/api/risk-calibrations").json()).results
    assert lst[0].nus == [0.0, 0.05, 0.25, 1.0]
    d = S.RiskCalibrationResult.model_validate(wclient.get(f"/api/risk-calibrations/{sealed.ids['risk']}").json())
    assert len(d.results) == 4 and d.rule and d.family.startswith("calibration:")
    assert all(r.by_year for r in d.results)
    r = wclient.post("/api/risk-calibrations", json={"store": STORE, "nu": [0, 0.25]})
    assert r.status_code == 202 and "--nu 0,0.25" in r.json()["command"]
    assert wclient.post("/api/risk-calibrations", json={"store": STORE, "nu": [-1]}).status_code == 422


# --------------------------------------------------------------------------- mid-cap
def test_midcap_status_results_and_preregistration(wclient, sealed):
    st = S.MidcapStatus.model_validate(wclient.get("/api/checks/midcap").json())
    assert [c.id for c in st.criteria][:3] == ["C1", "C2", "C3"] and st.preregistration_sha256
    assert st.force_phrase == MIDCAP_FORCE_PHRASE and st.store_configured
    # the world's check is a scaled-down one (another config hash): listed, but not "this check ran"
    assert [r.run_id for r in st.results] == [sealed.ids["midcap"]] and st.can_launch and not st.needs_force
    res = st.results[0]
    assert res.outcome in ("PASS", "PASS, cost-fragile", "FAIL", "INCONCLUSIVE") and res.sha_matches
    d = S.MidcapResultDetail.model_validate(wclient.get(f"/api/checks/midcap/results/{res.run_id}").json())
    assert d.run is not None and d.run.seal.state == "exempt"
    assert d.reference is not None and d.reference.id == sealed.run_ids["sealed"]
    assert {c.id for c in d.result.verdict.criteria} == {"C1", "C2", "C3"}
    assert {c.id for c in d.result.verdict.validity} == {"V1", "V2", "V3"}
    assert d.result.verdict.side_by_side and d.sha_matches
    pre = S.MidcapPreregistration.model_validate(wclient.get("/api/checks/midcap/preregistration").json())
    assert pre.exists and pre.sha256 == st.preregistration_sha256 and "## Result" not in pre.text.split("\n")[0]
    assert wclient.get("/api/checks/midcap/results/nope").status_code == 404


def test_midcap_result_found_in_a_run_directory_only(wclient, sealed, tmp_path):
    """A check run on another machine: only <run>/midcap_check.json was copied over."""
    src = sealed.runs_root / "_checks" / f"midcap_{sealed.ids['midcap']}.json"
    moved = tmp_path / src.name
    shutil.move(src, moved)
    try:
        st = wclient.get("/api/checks/midcap").json()
        assert [r["source"] for r in st["results"]] == [f"{sealed.ids['midcap']}/midcap_check.json"]
    finally:
        shutil.move(moved, src)


def test_midcap_launch_is_once_only(wclient, sealed, monkeypatch):
    r = wclient.post("/api/checks/midcap", json={"dry_run": True})
    assert r.status_code == 202 and r.json()["command"].endswith("--max-rank 2750 --cache --dry-run")
    assert wclient.post("/api/checks/midcap", json={"dry_run": True}).status_code == 409  # one job at a time
    for j in queued(wclient):
        wclient.delete(f"/api/jobs/{j['job_id']}")
    # pretend the real check of this configuration already ran
    from qts_api.services import research as R

    real = R.midcap_static

    def ran(settings):
        st = real(settings)
        return {**st, "previous_runs": [{"ts": "2026-11-13T10:00:00", "family": "confirm:midcap:x", "forced": False}]}

    monkeypatch.setattr(R, "midcap_static", ran)
    st = wclient.get("/api/checks/midcap").json()
    assert not st["can_launch"] and st["needs_force"] and "once-only" in st["blocked_reason"]
    assert wclient.post("/api/checks/midcap", json={}).status_code == 409
    r = wclient.post("/api/checks/midcap", json={"force": True, "confirm": "yes"})
    assert r.status_code == 422 and MIDCAP_FORCE_PHRASE in r.json()["detail"]
    r = wclient.post("/api/checks/midcap", json={"force": True, "confirm": MIDCAP_FORCE_PHRASE})
    assert r.status_code == 202 and r.json()["command"].endswith("--force")
    job = wclient.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert "forced" in job["label"]
    # without the pre-registration the check never runs
    monkeypatch.setattr(R, "midcap_static", lambda s: {**real(s), "preregistration_sha256": ""})
    for j in queued(wclient):
        wclient.delete(f"/api/jobs/{j['job_id']}")
    assert wclient.post("/api/checks/midcap", json={"dry_run": True}).status_code == 409


# --------------------------------------------------------------------------- paper
def test_paper_read_routes(wclient, sealed):
    books = S.PaperBooks.model_validate(wclient.get("/api/paper/books").json()).books
    assert [b.name for b in books] == ["check"] and books[0].sizing == "replica"
    st = S.PaperStatus.model_validate(wclient.get("/api/paper/books/check/status").json())
    s = st.series
    n = len(s.date)
    assert n > 20 and all(
        len(getattr(s, k)) == n for k in ("nav", "drawdown", "mode", "exante_vol", "realised_vol_21d")
    )
    assert s.boundaries is not None and len(s.boundaries.dd_stop) == n
    assert s.cone_bootstrap is not None and len(s.cone_bootstrap.lower) == n
    assert set(s.mode) == {"backfill"}  # replica sizing is always backfill
    assert st.status.level in ("OK", "WATCH", "STOP") and st.status.checks
    assert {c.rule for c in st.status.checks} >= {"drawdown", "sharpe", "vol_ratio"}
    assert st.stop_rules and st.latest is not None
    h = S.PaperHistory.model_validate(wclient.get("/api/paper/books/check/history").json())
    assert h.rows[-1].date == st.last_date
    o = S.PaperOrders.model_validate(wclient.get("/api/paper/books/check/orders").json())
    assert o.date == st.last_date
    hold = S.PaperHoldings.model_validate(wclient.get("/api/paper/books/check/holdings").json())
    assert hold.date == st.last_date and hold.holdings
    assert sum(abs(x.weight or 0) for x in hold.holdings) == pytest.approx(st.latest.gross, rel=1e-6)
    sch = S.PaperSchedule.model_validate(wclient.get("/api/paper/books/check/schedule").json())
    assert "paper run --name check" in sch.crontab and "<plist" in sch.plist
    assert wclient.get("/api/paper/books/nope/status").status_code == 404
    assert wclient.get("/api/paper/books/..%2Fx/status").status_code == 404
    assert wclient.get("/api/paper/books/check/orders?date=1999-01-04").status_code == 409


def test_paper_ack_run_and_reconcile_launches(sealed, tmp_path):
    # ack on a copy of the book (an ack is appended to the book)
    root = tmp_path / "paper"
    shutil.copytree(sealed.paper_root, root)
    app = create_app(Settings(runs_root=sealed.runs_root, store_path=sealed.store, paper_root=root), start_jobs=False)
    app.state.jobs.dir = tmp_path / "_jobs"
    app.state.jobs.dir.mkdir()
    with TestClient(app, base_url=LOCAL) as c:
        assert c.post("/api/paper/books/check/ack", json={"note": ""}).status_code == 422
        assert c.post("/api/paper/books/check/run", json={"until": "2020-1-1"}).status_code == 422
        assert c.post("/api/paper/books/check/run", json={"signal_mode": "weekly"}).status_code == 422
        a = S.PaperAck.model_validate(
            c.post("/api/paper/books/check/ack", json={"note": "checked the drawdown"}).json()
        )
        assert a.note == "checked the drawdown"
        st = c.get("/api/paper/books/check/status").json()
        assert st["status"]["acknowledged"] is True and st["acks"][-1]["note"] == "checked the drawdown"
        r = c.post("/api/paper/books/check/run", json={"no_sync": True})
        assert r.status_code == 202 and r.json()["command"].endswith(
            "--name check --paper-root " + str(root) + " --no-sync"
        )
        assert c.post("/api/paper/books/check/run", json={}).status_code == 409  # one job per book
        assert c.post("/api/paper/books/check/reconcile").status_code == 409
        assert c.post("/api/paper/books/check/ack", json={"note": "again"}).status_code == 409
        jobs = c.get("/api/paper/books/check/status").json()["jobs"]
        assert jobs[0]["task"] == "paper_run" and jobs[0]["status"] == "queued"
        c.delete(f"/api/jobs/{jobs[0]['job_id']}")
        r = c.post("/api/paper/books/check/reconcile")
        assert r.status_code == 202 and "paper reconcile --name check" in r.json()["command"]
        assert c.post("/api/paper/books/nope/run", json={}).status_code == 404
