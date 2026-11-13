"""Ledger, config schema, health, CORS, serving the built UI, and the OpenAPI document."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from qts.core.config import SystemConfig

from qts_api import schemas as S
from qts_api.main import create_app
from qts_api.services.config_schema import RESERVED, leaf_index, validate_overrides
from qts_api.settings import Settings

LOCAL = "http://localhost"  # Host allow-list (see conftest.LOCAL)


def test_ledger(client, fixture_root, summary, run_id):
    led = S.Ledger.model_validate(client.get("/api/ledger").json())
    recs = [json.loads(x) for x in (fixture_root / "trial_ledger.jsonl").read_text().splitlines() if x.strip()]
    assert len(led.trials) == len(recs) >= 1
    t = led.trials[-1]
    # DSR family = the one the pipeline counted in (ResearchPipeline.ledger_family; keyed by the
    # market hash), which the manifest records and the run detail serves
    family = S.RunDetail.model_validate(client.get(f"/api/runs/{run_id}").json()).manifest.ledger_family
    assert family and family.startswith("pipeline_phase1:")
    assert t.family == family and t.config_hash == summary["config_hash"]
    assert t.sharpe_annual == summary["backtest"]["sharpe_net"]
    fam = {f.family: f for f in led.families}[family]
    assert fam.n_trials == len(recs) and fam.effective_trials >= 1


def test_ledger_missing_file(tmp_path):
    with TestClient(create_app(Settings(runs_root=tmp_path), start_jobs=False), base_url=LOCAL) as c:
        assert c.get("/api/ledger").json() == {"trials": [], "families": []}
        assert c.get("/api/runs").json() == []
        assert not (tmp_path / "trial_ledger.jsonl").exists()


def test_config_schema(client):
    r = client.get("/api/config/schema")
    body = r.json()
    schema = S.JsonSchemaObject.model_validate(body)
    assert schema.title == "SystemConfig"
    assert ": null" not in json.dumps(body)  # optional keys are omitted, never null
    port = body["properties"]["portfolio"]
    assert port["title"] == "Portfolio construction"
    ta = port["properties"]["trade_aversion"]
    assert ta == {**ta, "type": "number", "default": 1.0, "minimum": 0, "maximum": 10}
    assert port["properties"]["book"]["enum"] == ["long_short", "long_only", "130_30"]
    assert port["properties"]["vol_target_mode"]["enum"] == ["projection", "constraint"]
    assert port["properties"]["beta_neutral"]["type"] == "boolean"
    assert body["properties"]["validation"]["properties"]["n_splits"]["type"] == "integer"
    assert body["properties"]["alpha"]["properties"]["lgbm"]["type"] == "object"
    # descriptions come from the field comments in qts/core/config.py
    assert "signal" in body["properties"]["execution"]["properties"]["signal_lag_days"]["description"]
    # every scalar default round-trips through the strict validator
    defaults = {k: v["default"] for k, v in leaf_index().items() if "default" in v and k not in RESERVED}
    assert validate_overrides(defaults).hash() == SystemConfig().hash()
    assert "styles" not in body["properties"]["risk"]["properties"]  # lists are not form fields


def test_health(client):
    h = S.Health.model_validate(client.get("/api/health").json())
    assert h.status == "ok" and h.n_runs == 2 and h.runs_root_exists


def test_cors(client):
    r = client.options("/api/runs", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    r = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_serve_ui(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (dist / "favicon.svg").write_text("<svg/>")
    s = Settings(runs_root=tmp_path / "runs", serve_ui=True, ui_dist=dist)
    with TestClient(create_app(s, start_jobs=False), base_url=LOCAL) as c:
        assert "id=root" in c.get("/").text
        assert "id=root" in c.get("/runs/abc/risk").text  # SPA fallback
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/favicon.svg").text == "<svg/>"
        assert "id=root" in c.get("/../../etc/passwd").text
        r = c.get("/api/missing")
        assert r.status_code == 404 and r.headers["content-type"].startswith("application/problem+json")
        assert c.get("/api/health").json()["status"] == "ok"


def test_openapi_document():
    from qts_api.openapi import openapi_document

    doc = openapi_document()
    names = set(doc["components"]["schemas"])
    for n in (
        "RunSummary",
        "RunDetail",
        "ColumnarSeries",
        "RunCosts",
        "RunExposures",
        "RunHoldings",
        "RunIc",
        "LaunchRunRequest",
        "LaunchRunResponse",
        "Job",
        "Ledger",
        "JsonSchemaObject",
        "ApiErrorBody",
    ):
        assert n in names, n
    assert not any(n.endswith(("-Input", "-Output")) for n in names)
    paths = doc["paths"]
    for p in (
        "/api/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/series",
        "/api/runs/{run_id}/costs",
        "/api/runs/{run_id}/exposures",
        "/api/runs/{run_id}/holdings",
        "/api/runs/{run_id}/ic",
        "/api/jobs/{job_id}",
        "/api/ledger",
        "/api/config/schema",
        "/api/health",
    ):
        assert p in paths, p
    assert set(paths["/api/runs"]) == {"get", "post"} and set(paths["/api/jobs/{job_id}"]) == {"get", "delete"}
