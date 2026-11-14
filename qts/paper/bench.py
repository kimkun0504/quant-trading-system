"""Runtime benchmark of the paper book's daily job on a synthetic ParquetStore.

    python -m qts.paper.bench --stocks 1000 --days 1150 --out /tmp/paper_bench

Writes a synthetic market into a store, initialises a ``moc`` book on the baseline
config (the real LightGBM + MLP + ridge ensemble, three seeds) with its first job day
``--start-offset`` sessions before the end, and runs the job day by day, printing the
wall time of each day and the stage timings of every fit: the first day trains every
walk-forward fold, a normal rebalance day refits risk + alpha with the frozen folds,
a hold day fits nothing, and a refit day (the data cross the next scheduled refit)
trains exactly one new fold.  ``--refit-every`` shortens the refit schedule so a short
run crosses a refit.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--stocks", type=int, default=1000)
    ap.add_argument("--days", type=int, default=1150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument(
        "--start-offset", type=int, default=12, help="first job day = this many sessions before the end"
    )
    ap.add_argument("--refit-every", type=int, default=None, help="override alpha.refit_frequency_days")
    ap.add_argument("--out", default="/tmp/paper_bench")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    import pandas as pd

    from qts.core.config import SystemConfig
    from qts.data.store import ParquetStore
    from qts.data.synthetic import SyntheticSpec, generate
    from qts.paper import expectations as X
    from qts.paper import runner
    from qts.paper.book import DataSpec, read_jsonl

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    md = generate(SyntheticSpec(n_stocks=a.stocks, n_days=a.days, seed=a.seed))
    md = replace(md, extra={})
    store = out / "store"
    if not (store / "prices_close").exists():
        ParquetStore(str(store)).write_market_data(md)
    t_store = time.perf_counter() - t0
    cfg = SystemConfig.from_yaml(a.config)
    if a.refit_every:
        cfg = cfg.with_overrides({"alpha.refit_frequency_days": int(a.refit_every)})
    cfg_path = out / "cfg.yaml"
    cfg_path.write_text(cfg.to_yaml())
    # a stand-in yardstick (the benchmark measures time, not the rules)
    rng = __import__("numpy").random.default_rng(0)
    r = pd.Series(rng.normal(0.0002, 0.003, 2000), index=pd.bdate_range("2000-01-03", periods=2000))
    exp = X.Expectations(X.summary_from_returns(r), X.block_bootstrap_table(r.to_numpy()), r)
    start = md.dates[-a.start_offset - 1]
    name = f"bench_{a.stocks}x{a.days}"
    root = out / "paper"
    if not (root / name / "book.json").exists():
        runner.init_book(
            name,
            str(cfg_path),
            nav=5e6,
            start=str(start.date()),
            paper_root=root,
            data=DataSpec(kind="store", store=str(store), start=None, max_names=None, min_mcap_ever=None),
            sizing="moc",
            calendar="index",
            expectations=exp,
            source=__import__("qts.paper.data", fromlist=["StoreSource"]).StoreSource(
                DataSpec(kind="store", store=str(store), start=None, max_names=None, min_mcap_ever=None)
            ),
        )
    rows = []
    later = datetime(2100, 1, 1, tzinfo=timezone.utc)
    for d in md.dates[md.dates >= start]:
        t1 = time.perf_counter()
        rep = runner.run_book(
            name, paper_root=root, date=str(d.date()), no_sync=True, now=later, progress=lambda m: None
        )
        sec = time.perf_counter() - t1
        recs = runner.history(name, paper_root=root)
        day = rep["days"][-1] if rep.get("days") else {}
        raw = read_jsonl(root / name / "journal.jsonl")[0]
        models = raw[-1].get("models", {}) if raw else {}
        fits = models.get("fits", [])
        row = {
            "date": str(d.date()),
            "seconds": round(sec, 1),
            "rebalance_decision": bool(raw[-1].get("rebalance_decision")) if raw else None,
            "fit_seconds": [f["seconds"] for f in fits],
            "folds_trained": [
                (f["train_end"], round(f["seconds"], 1)) for f in models.get("folds_trained", [])
            ],
            "n_orders": day.get("n_orders"),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del recs
    summary = {"stocks": a.stocks, "days": a.days, "store_write_seconds": round(t_store, 1), "rows": rows}
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
