"""A small but real research world: every research tool and a paper book run once on a
synthetic data store, for the API tests and the UI's mock fixtures.

``build_research_world(root)`` writes under ``root``:

* ``store/``: a ParquetStore of a synthetic market (1,000 business days from 2018-01-02);
* ``seal.yaml``: a holdout seal from 2021-03-01 (set ``QTS_SEAL_FILE`` to it);
* ``runs/``: a sealed run, a sealed reference re-run, a sealed variant, an unsealed run
  (logged unseal) and the trial ledger; two signal tests (``_signals/``), a two-member
  sweep with PBO (``_sweeps/``), a capacity curve (``_capacity/``), a risk calibration
  report (``_risk/``) and a scaled-down mid-cap check (``_checks/``);
* ``paper/check``: a paper book initialised from the sealed run and run day by day.

Everything is produced by the library entry points the API's jobs call, so the JSON is
exactly what the real tools write; only the scale differs (60 names, a few months of
backtest).  ``python -m qts_api.research_fixture OUT_DIR`` builds it; about two minutes.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOLDOUT = "2021-03-01"
MAX_DAYS = 60
N_DAYS = 1000


@dataclass
class ResearchWorld:
    root: Path
    store: Path
    seal_file: Path
    runs_root: Path
    ledger: Path
    paper_root: Path
    config: Path
    holdout: str = HOLDOUT
    sealed_end: str = ""
    run_ids: dict[str, str] = field(default_factory=dict)
    ids: dict[str, Any] = field(default_factory=dict)


def _quiet() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", UserWarning)


def build_research_world(root: str | Path, *, paper: bool = True, midcap: bool = True) -> ResearchWorld:
    import yaml
    from qts.cli import main as qts_main
    from qts.core.config import SystemConfig
    from qts.data.store import ParquetStore
    from qts.data.synthetic import SyntheticSpec, generate
    from qts.pipeline import list_runs
    from qts.research.capacity import run_capacity
    from qts.research.data import StoreSpec
    from qts.research.risk_calibration import run_risk_calibration
    from qts.research.signals import run_signal_test
    from qts.research.sweep import create_sweep, finalize_sweep, plan_sweep, run_sweep_member

    _quiet()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    w = ResearchWorld(
        root=root,
        store=root / "store",
        seal_file=root / "seal.yaml",
        runs_root=root / "runs",
        ledger=root / "runs" / "trial_ledger.jsonl",
        paper_root=root / "paper",
        config=root / "baseline.yaml",
    )
    md = generate(SyntheticSpec(n_stocks=60, n_days=N_DAYS, seed=5))
    import pandas as pd

    w.sealed_end = str(md.dates[md.dates < pd.Timestamp(HOLDOUT)][-1].date())
    ParquetStore(str(w.store)).write_market_data(md)
    w.seal_file.write_text(
        yaml.safe_dump(
            {
                "holdout_start": HOLDOUT,
                "reason": "fixture: the last months of the synthetic store are the holdout",
                "sealed_by": "fixture",
                "sealed_at": "2026-11-13",
                "store_end": str(md.dates[-1].date()),
            }
        )
    )
    os.environ["QTS_SEAL_FILE"] = str(w.seal_file)
    # a 60-name synthetic store: the load-coverage guard would refuse its small prefilters
    os.environ.setdefault("QTS_LOAD_COVERAGE", "off")
    cfg = SystemConfig.from_yaml(
        "configs/baseline.yaml" if Path("configs/baseline.yaml").exists() else _repo("configs/baseline.yaml")
    )
    cfg = cfg.with_overrides({"universe.top_n_by_mcap": 40})
    w.config.write_text(cfg.to_yaml())
    spec = StoreSpec(store=str(w.store), max_names=50)
    runs, led = str(w.runs_root), str(w.ledger)
    base_argv = [
        "run", "--store", str(w.store), "--max-names", "50", "--config", str(w.config),
        "--max-days", str(MAX_DAYS), "--ledger", led, "--runs-root", runs, "--cache",
    ]  # fmt: skip

    def run(*extra: str) -> str:
        assert qts_main([*base_argv, *extra]) == 0
        return list_runs(runs)[0].name

    # sealed baseline, its reference re-run, a sealed variant, an unsealed run
    w.run_ids["sealed"] = run()
    # the seal inherits the sealed run's family (what research/seal.yaml does for the baseline's)
    fam = json.loads((w.runs_root / w.run_ids["sealed"] / "summary.json").read_text()).get("ledger_family")
    seal = yaml.safe_load(w.seal_file.read_text())
    w.seal_file.write_text(yaml.safe_dump({**seal, "inherit_family": [fam] if fam else []}))
    w.run_ids["reference"] = run("--nav", "2e6")
    w.run_ids["variant"] = run("--set", "portfolio.trade_aversion=2.0")
    w.run_ids["unsealed"] = run("--set", "portfolio.trade_aversion=2.0", "--unseal", "fixture: final check")
    # signal tests: two tests in one family (Holm over both)
    for feat in ("mom_12_1", "strev_21"):
        r = run_signal_test(
            feat, spec, horizons=[5, 20, 60], baseline=str(w.config), use_ml=False, ledger_path=led,
            runs_root=runs, cost_bps=None,
        )  # fmt: skip
        w.ids.setdefault("signals", []).append(r.id)
    # sweep: two members, PBO over them
    plan = plan_sweep(
        [("portfolio.trade_aversion", [0.5, 1.5])], base_config=str(w.config), spec=spec, use_ml=False,
        max_days=MAX_DAYS, ledger_path=led, runs_root=runs,
    )  # fmt: skip
    create_sweep(plan)
    for m in plan.members:
        run_sweep_member(plan.id, m.index, runs_root=runs, ledger_path=led)
    finalize_sweep(plan.id, runs)
    w.ids["sweep"] = plan.id
    cap = run_capacity(
        spec, navs=[5e6, 5e7, 2.5e8, 1e9], baseline=str(w.config), use_ml=False, ledger_path=led, runs_root=runs,
        max_days=MAX_DAYS,
    )  # fmt: skip
    w.ids["capacity"] = cap.id
    risk = run_risk_calibration(
        spec, nu_grid=[0.0, 0.05, 0.25, 1.0], baseline=str(w.config), use_ml=False, ledger_path=led,
        runs_root=runs, max_days=120,
    )  # fmt: skip
    w.ids["risk"] = risk.id
    if midcap:
        from qts.research.midcap import run_midcap_check

        mc = run_midcap_check(
            StoreSpec(store=str(w.store), max_rank=55), config_path=str(_scaled_midcap_config(w)), ledger_path=led,
            runs_root=runs, use_ml=False, rank_offset=10, coverage_rank=40, max_days=MAX_DAYS,
            reference_run=str(w.runs_root / w.run_ids["sealed"]),
        )  # fmt: skip
        w.ids["midcap"] = mc.verdict.run_id if mc.verdict is not None else None
    if paper:
        _paper_book(w, md)
    return w


def _scaled_midcap_config(w: ResearchWorld) -> Path:
    from qts.core.config import SystemConfig

    p = w.root / "midcap_base.yaml"
    p.write_text(SystemConfig.from_yaml(str(w.config)).with_overrides({"universe.top_n_by_mcap": 30}).to_yaml())
    return p


def _paper_book(w: ResearchWorld, md: Any) -> None:
    """A store book (composite only) from the sealed run: init on a session two months
    before the end of the store, then the daily job day by day through the end."""
    import pandas as pd
    from qts.paper import init_book, run_book
    from qts.paper.book import DataSpec
    from qts.paper.data import NYSECalendar

    cal = NYSECalendar()
    dates = [d for d in md.dates[-45:] if cal.is_session(d)]
    start = str(pd.Timestamp(dates[0]).date())
    data = DataSpec(kind="store", store=str(w.store), start=None, max_names=50, min_mcap_ever=None)
    init_book(
        "check", str(w.config), nav=5e6, start=start, reference_run=str(w.runs_root / w.run_ids["sealed"]),
        paper_root=str(w.paper_root), data=data, sizing="replica", use_ml=False,
    )  # fmt: skip
    run_book("check", until=str(md.dates[-1].date()), no_sync=True, paper_root=str(w.paper_root), signal_mode="single")
    w.ids["paper"] = "check"


def _repo(rel: str) -> str:
    return str(Path(__file__).resolve().parents[2] / rel)


def export_payloads(w: ResearchWorld, out: Path) -> dict[str, Path]:
    """The JSON payloads of the world as the API serves them, written to ``out``
    (the UI's mock fixtures)."""
    from fastapi.testclient import TestClient

    from qts_api.main import create_app
    from qts_api.settings import Settings

    out.mkdir(parents=True, exist_ok=True)
    app = create_app(Settings(runs_root=w.runs_root, store_path=w.store, paper_root=w.paper_root), start_jobs=False)
    written: dict[str, Path] = {}

    def dump(name: str, url: str) -> None:
        with TestClient(app, base_url="http://localhost") as c:
            r = c.get(url)
            assert r.status_code == 200, (url, r.status_code, r.text[:500])
            p = out / f"{name}.json"
            p.write_text(json.dumps(r.json(), indent=1, sort_keys=False) + "\n")
            written[name] = p

    dump("seal", "/api/seal")
    dump("ledger", "/api/ledger")
    dump("features_v1", "/api/features?feature_set=v1")
    dump("features_v2", "/api/features?feature_set=v2")
    dump("signals", "/api/signals")
    dump("signal_tests", "/api/signals/tests")
    dump("signal_test", f"/api/signals/tests/{w.ids['signals'][-1]}")
    dump("sweeps", "/api/sweeps")
    dump("sweep", f"/api/sweeps/{w.ids['sweep']}")
    dump("capacity_list", "/api/capacity")
    dump("capacity", f"/api/capacity/{w.ids['capacity']}")
    dump("risk_list", "/api/risk-calibrations")
    dump("risk", f"/api/risk-calibrations/{w.ids['risk']}")
    dump("midcap", "/api/checks/midcap")
    if w.ids.get("midcap"):
        dump("midcap_result", f"/api/checks/midcap/results/{w.ids['midcap']}")
    dump("midcap_preregistration", "/api/checks/midcap/preregistration")
    if w.ids.get("paper"):
        dump("paper_books", "/api/paper/books")
        dump("paper_status", "/api/paper/books/check/status")
        dump("paper_history", "/api/paper/books/check/history")
        dump("paper_orders", "/api/paper/books/check/orders")
        dump("paper_schedule", "/api/paper/books/check/schedule")
    dump("runs", "/api/runs")
    return written


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import time

    ap = argparse.ArgumentParser(description="build the research world and export the API payloads")
    ap.add_argument("root", help="where to build it (a new directory)")
    ap.add_argument("--export", default=None, help="write the API payloads (JSON) to this directory")
    a = ap.parse_args()
    t0 = time.perf_counter()
    world = build_research_world(a.root)
    print(f"built {world.root} in {time.perf_counter() - t0:.0f}s: {world.run_ids} {world.ids}")
    if a.export:
        for k, v in export_payloads(world, Path(a.export)).items():
            print(f"  {k}: {v}")
