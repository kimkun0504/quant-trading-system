"""``qts paper ...`` subcommands (registered by :func:`qts.cli.build_parser` through
:func:`add_paper_parser`).

Exit codes: 0 success; 1 error (message on stderr); 2 bad arguments; 3 stale data
(the vendor has not published the expected session yet: re-run later); 4 the book
is locked by another job.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

DEFAULT_NAME = "baseline"
DEFAULT_ROOT = "paper"
DEFAULT_REFERENCE_RUN = "runs/20261113T023313333450Z_e4667d90"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", default=DEFAULT_NAME, help="book name (directory under --paper-root)")
    p.add_argument("--paper-root", default=DEFAULT_ROOT, help="where paper books live (default: paper)")


def add_paper_parser(sub: Any) -> None:
    """Add the ``paper`` command and its subcommands to the top-level subparsers."""
    pp = sub.add_parser("paper", help="internal paper-trading book of a frozen config (handbook 14)")
    ps = pp.add_subparsers(dest="paper_cmd", required=True)

    i = ps.add_parser("init", help="create a paper book: freeze config, expectations and stop rules")
    _common(i)
    i.add_argument("--config", required=True, help="strategy config to freeze (e.g. configs/baseline.yaml)")
    i.add_argument("--nav", type=float, default=5e6, help="starting NAV in USD (default 5e6)")
    i.add_argument(
        "--start",
        required=True,
        help="first job day YYYY-MM-DD: its close produces the first orders, filled at the next session's close",
    )
    i.add_argument(
        "--reference-run",
        default=DEFAULT_REFERENCE_RUN,
        help=f"run directory the expectations come from (default {DEFAULT_REFERENCE_RUN})",
    )
    i.add_argument(
        "--sizing",
        choices=("moc", "replica"),
        default="moc",
        help="moc (live-tradable, default) | replica (batch-identical pipeline check)",
    )
    i.add_argument(
        "--no-ml", action="store_true", help="composite only (phase 1); the baseline uses the ML sleeve"
    )
    i.add_argument("--store", default="data/store", help="ParquetStore root")
    i.add_argument(
        "--data-start",
        default="2000-01-01",
        help="first data date loaded (fixes the refit anchor; as `qts run --start`)",
    )
    i.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="screened-rank store prefilter as `qts run --max-rank` (default 1500 unless --max-names / "
        "--min-mcap-ever is given)",
    )
    i.add_argument(
        "--max-names",
        type=int,
        default=None,
        help="peak-cap store prefilter as `qts run --max-names` (can miss universe names: refused unless it "
        "covers; prefer --max-rank)",
    )
    i.add_argument(
        "--min-mcap-ever",
        type=float,
        default=None,
        help="peak-cap store prefilter as `qts run --min-mcap-ever`",
    )

    r = ps.add_parser("run", help="the daily job: refresh data, fill yesterday's orders, generate today's")
    _common(r)
    r.add_argument(
        "--date", default=None, help="process through this day (default: the latest session in the store)"
    )
    r.add_argument(
        "--until", default=None, help="backfill: process day by day through this date (no freshness check)"
    )
    r.add_argument("--no-sync", action="store_true", help="skip the Sharadar incremental sync")
    r.add_argument(
        "--dry-run", action="store_true", help="compute and print, write nothing (no sync, no journal)"
    )
    r.add_argument(
        "--redo-last", action="store_true", help="re-process the last journal day (supersedes its record)"
    )
    r.add_argument(
        "--signal-mode",
        choices=("daily", "single"),
        default="daily",
        help="daily: fit on data through each signal date (default); single: one fit for a long catch-up",
    )
    r.add_argument("-v", "--verbose", action="store_true")

    s = ps.add_parser("status", help="status (OK / WATCH / STOP), every check and the latest day")
    _common(s)
    s.add_argument("--json", action="store_true", help="print the full JSON (the API payload)")

    h = ps.add_parser("history", help="one line per journal day")
    _common(h)
    h.add_argument("--json", action="store_true")
    h.add_argument("--tail", type=int, default=None, help="last N days only")

    o = ps.add_parser("orders", help="orders generated on a day (for the next close) and that day's fills")
    _common(o)
    o.add_argument("--date", default=None, help="job day (default: the latest)")
    o.add_argument("--json", action="store_true")
    o.add_argument("--csv", default=None, help="also write the orders to this CSV file")

    a = ps.add_parser("ack", help="acknowledge the latest status (a STOP needs it to resume trading)")
    _common(a)
    a.add_argument("--note", required=True, help="why it is safe to continue")

    rp = ps.add_parser("replay", help="validate the journal and rebuild every derived file from it")
    _common(rp)

    sc = ps.add_parser(
        "schedule", help="print a launchd plist and a crontab line for the daily job (installs nothing)"
    )
    _common(sc)
    sc.add_argument(
        "--print", dest="print_", action="store_true", help="print (the only mode; nothing is installed)"
    )

    rc = ps.add_parser(
        "reconcile", help="paper vs the batch simulator over the book's dates (backfill check)"
    )
    _common(rc)
    rc.add_argument("--json", action="store_true")
    pp.set_defaults(func=cmd_paper)


def _fmt_pct(x: Any, nd: int = 2) -> str:
    try:
        return f"{100 * float(x):.{nd}f}%"
    except (TypeError, ValueError):
        return "-"


def cmd_paper(a: argparse.Namespace) -> int:
    from qts import paper
    from qts.paper.book import BookLockedError, DataSpec, PaperError, StaleDataError

    try:
        return _dispatch(a, paper, DataSpec)
    except StaleDataError as e:
        print(f"stale data: {e}", file=sys.stderr)
        return 3
    except BookLockedError as e:
        print(f"locked: {e}", file=sys.stderr)
        return 4
    except (PaperError, FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _dispatch(a: argparse.Namespace, paper: Any, DataSpec: Any) -> int:
    c = a.paper_cmd
    if c == "init":
        peak = a.max_names is not None or a.min_mcap_ever is not None
        if peak and a.max_rank is not None:
            raise paper.PaperError("--max-rank cannot be combined with --max-names / --min-mcap-ever")
        data = DataSpec(
            kind="store",
            store=a.store,
            start=a.data_start,
            max_names=a.max_names,
            min_mcap_ever=a.min_mcap_ever,
            max_rank=None if peak else (a.max_rank if a.max_rank is not None else 1500),
        )
        out = paper.init_book(
            a.name,
            a.config,
            nav=a.nav,
            start=a.start,
            reference_run=a.reference_run,
            paper_root=a.paper_root,
            data=data,
            sizing=a.sizing,
            use_ml=not a.no_ml,
        )
        h = out["header"]
        print(f"paper book {out['root']} created")
        print(f"  config {h['config_source']} (hash {h['config_hash']}, frozen copy {h['config_file']})")
        print(f"  NAV {h['nav0']:,.0f}  sizing {h['sizing']}  ML {h['use_ml']}")
        print(f"  first job day {h['start']} -> first fills at the close of {h['first_execution']}")
        print(f"  reference run {h['reference_run']}  expectations {h['expectations_sha256'][:16]}")
        cal = (out.get("expectations") or {}).get("calibration") or {}
        if cal:
            ach, fx = cal.get("achieved", {}), cal.get("fixed_quantile_rates", {})
            print(
                "  stop-rule boundaries calibrated on the reference run: P(false STOP in year 1) "
                f"drawdown {100 * ach.get('dd_stop', 0):.1f}%, Sharpe {100 * ach.get('sharpe_stop', 0):.1f}%, "
                f"either {100 * ach.get('any_return_stop', 0):.1f}% (fixed per-horizon quantiles would give "
                f"{100 * fx.get('dd_stop', 0):.1f}% / {100 * fx.get('sharpe_stop', 0):.1f}%)"
            )
        for n in h.get("notes", []):
            print(f"  NOTE: {n}")
        print("  next: qts paper run --name " + h["name"] + "   (schedule: qts paper schedule --print)")
        return 0
    if c == "run":
        if a.verbose:
            logging.getLogger("qts").setLevel(logging.INFO)
        rep = paper.run_book(
            a.name,
            date=a.date,
            until=a.until,
            no_sync=a.no_sync,
            dry_run=a.dry_run,
            redo_last=a.redo_last,
            paper_root=a.paper_root,
            signal_mode=a.signal_mode,
            progress=lambda m: print(m, flush=True),
        )
        if rep.get("message"):
            print(rep["message"])
        st = rep.get("status") or {}
        if st:
            print(f"status {st.get('level')}" + ("".join(f"\n  - {x}" for x in st.get("reasons", []))))
        if a.dry_run:
            last = (rep.get("days") or [{}])[-1]
            for x in last.get("orders", [])[:50]:
                print(
                    f"  {x['side']:4s} {abs(x['shares']):>10,.0f} {x['ticker']:12s} @ {x['px']:10.2f}  est {x['est_cost_bps']:.1f} bp"
                )
            if len(last.get("orders", [])) > 50:
                print(f"  ... {len(last['orders']) - 50} more")
            print(
                "(dry run: no sync, nothing written to the journal or the day views; a newly trained fold model is still cached)"
            )
        return 0
    if c == "status":
        st = paper.status(a.name, paper_root=a.paper_root)
        if a.json:
            print(json.dumps(st, indent=1))
            return 0
        s = st["status"]
        lat = st.get("latest") or {}
        print(f"book {st['book']}  last day {st['last_date']}  ({st['n_days']} days)")
        if lat:
            print(
                f"  NAV {lat['nav']:,.0f}  gross {lat['gross']:.2f}  net {lat['net']:+.3f}  "
                f"{lat['n_long']}L/{lat['n_short']}S  orders for {lat['orders_for']}: {lat['n_orders']}  mode {lat['mode']}"
            )
        m = s.get("metrics", {})
        print(
            f"  cum {_fmt_pct(m.get('cum_return'))}  max DD {_fmt_pct(m.get('max_drawdown'))}  "
            f"Sharpe {m.get('sharpe') if m.get('sharpe') is None else round(m['sharpe'], 2)}  vol ratio {m.get('vol_ratio')}"
        )
        print(
            f"STATUS {s['level']}"
            + ("  (trading halted until `qts paper ack`)" if s.get("trading_halted") else "")
        )
        for ch in s.get("checks", []):
            print(f"  [{ch['level']:5s}] {ch['rule']:18s} {ch['message']}")
        return 0
    if c == "history":
        rows = paper.history(a.name, paper_root=a.paper_root)
        if a.tail:
            rows = rows[-a.tail :]
        if a.json:
            print(json.dumps(rows, indent=1))
            return 0
        print(
            f"{'date':10s} {'mode':8s} {'nav':>14s} {'ret':>8s} {'gross':>6s} {'fills':>5s} {'orders':>6s} {'cost$':>9s} {'status':>6s}"
        )
        for r in rows:
            print(
                f"{r['date']:10s} {r['mode']!s:8s} {r['nav']:14,.0f} {_fmt_pct(r['ret']):>8s} "
                f"{(r['gross'] or 0):6.2f} {r['n_fills'] or 0:5d} {r['n_orders']:6d} {(r['trading_cost'] or 0):9,.0f} {r['status']!s:>6s}"
            )
        return 0
    if c == "orders":
        o = paper.orders(a.name, date=a.date, paper_root=a.paper_root)
        if a.csv:
            import pandas as pd

            pd.DataFrame(o["orders"]).to_csv(a.csv, index=False)
        if a.json:
            print(json.dumps(o, indent=1))
            return 0
        print(
            f"job day {o['date']}: {o['n_orders']} orders for the close of {o['orders_for']} "
            f"(buy {o['buy_usd']:,.0f}, sell {o['sell_usd']:,.0f}, est. cost {o['est_cost_usd']:,.0f}) [{o['mode']}]"
        )
        for x in o["orders"]:
            print(
                f"  {x['side']:4s} {abs(x['shares']):>10,.0f} {x['ticker']:12s} @ {x['px']:10.2f}  {x['kind']:10s} est {x['est_cost_bps']:.1f} bp"
            )
        if o.get("cancelled_by_stop"):
            print(f"  ({o['cancelled_by_stop']} orders cancelled by a STOP)")
        return 0
    if c == "ack":
        r = paper.ack(a.name, a.note, paper_root=a.paper_root)
        print(f"acknowledged {r['date']} (status {r['level']}, rules {r['rules']}): {r['note']}")
        return 0
    if c == "replay":
        r = paper.replay(a.name, paper_root=a.paper_root)
        print(json.dumps(r, indent=1))
        return 0 if r.get("ok") else 1
    if c == "schedule":
        t = paper.schedule_text(a.name, paper_root=a.paper_root)
        print(f"# launchd: ~/Library/LaunchAgents/{t['label']}.plist\n{t['plist']}")
        print(f"# crontab (crontab -e):\n{t['crontab']}\n")
        print(t["install"])
        return 0
    if c == "reconcile":
        r = paper.reconcile(a.name, paper_root=a.paper_root)
        if a.json:
            print(json.dumps(r, indent=1))
            return 0
        for k in (
            "sizing",
            "dates",
            "max_abs_weight_diff",
            "max_rel_nav_diff",
            "cum_return_paper",
            "cum_return_batch",
            "daily_return_corr",
            "tracking_error_ann",
            "note",
        ):
            print(f"  {k:24s} {r[k]}")
        return 0
    return 2
