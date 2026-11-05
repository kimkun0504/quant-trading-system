"""Command-line interface: ``python -m qts.cli <command>`` (or ``qts <command>``
once ``[project.scripts] qts = "qts.cli:main"`` is installed).

Commands
--------
``demo``             synthetic end-to-end run (``--ml``, ``--stocks``, ``--days``, ``--seed``,
                     ``--signal-lag N``, ``--compare-lag`` prints lag-0 vs lag-1 Sharpe)
``run``              real data: ``--store PATH`` (ParquetStore) or ``--yfinance T1 T2 ...``
``config dump``      print the default (or ``--config``) SystemConfig as YAML;
                     ``configs/default.yaml`` is ``python -m qts.cli config dump``
``config validate``  strict-load a YAML file (unknown keys fail), print its hash
``runs list``        run directories under ``--root`` (default ``runs``), newest first
``runs show ID``     manifest + headline numbers + hurdles of one run
``ledger show``      trial ledger table (``--path``, ``--family``)
``ledger migrate``   one-off rename of pre-Nov-2026 bare ``pipeline_phaseN`` families
                     (``--dry-run``; see ``qts.validation.ledger``)

Every pipeline command writes a run directory (``--runs-root``, default
``runs``; ``--no-save`` disables) and logs to the trial ledger (``--ledger``).
Only the standard library is used for argument parsing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

DEFAULT_RUNS_ROOT = "runs"
DEFAULT_LEDGER = "runs/trial_ledger.jsonl"


# ----------------------------------------------------------------------------- helpers
def _load_config(path: str | None):
    from qts.core.config import SystemConfig

    return SystemConfig.from_yaml(path) if path else SystemConfig()


def _set_signal_lag(cfg, lag: int | None) -> str | None:
    """Set ``cfg.execution.signal_lag_days`` if the field exists; returns a
    warning text when it does not (older ExecutionConfig)."""
    if lag is None:
        return None
    if not hasattr(cfg.execution, "signal_lag_days"):
        return "ExecutionConfig has no signal_lag_days field yet: --signal-lag ignored"
    cfg.execution.signal_lag_days = int(lag)
    return None


def _add_pipeline_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=None, help="SystemConfig YAML (default: built-in defaults)")
    ap.add_argument("--ml", action="store_true", help="include the walk-forward ML sleeve (phase 2)")
    ap.add_argument(
        "--signal-lag",
        type=int,
        default=None,
        help="execution delay in days (ExecutionConfig.signal_lag_days; design decision D4)",
    )
    ap.add_argument(
        "--compare-lag",
        action="store_true",
        help="run the backtest with signal lag 0 and 1 and print both Sharpe ratios",
    )
    ap.add_argument("--rebalance-every", type=int, default=1)
    ap.add_argument("--nav", type=float, default=5e6)
    ap.add_argument("--ledger", default=DEFAULT_LEDGER, help="trial ledger JSONL ('' disables)")
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT, help="where run directories are written")
    ap.add_argument("--no-save", action="store_true", help="do not write a run directory")
    ap.add_argument("--max-days", type=int, default=None, help="backtest only the last N out-of-sample days")
    ap.add_argument("-v", "--verbose", action="store_true")


def _run_pipeline(md, cfg, a: argparse.Namespace, out=None) -> int:
    from qts.pipeline import ResearchPipeline, format_result

    out = out if out is not None else sys.stdout  # resolved at call time (capture-friendly)
    cfg.universe.top_n_by_mcap = min(cfg.universe.top_n_by_mcap, md.close.shape[1])
    warn = _set_signal_lag(cfg, a.signal_lag)
    if warn:
        print(f"WARNING: {warn}", file=out)
    pipe = ResearchPipeline(
        cfg,
        use_ml=a.ml,
        ledger_path=a.ledger or None,
        rebalance_every=a.rebalance_every,
        nav0=a.nav,
        runs_root=None if a.no_save else a.runs_root,
    )
    if not a.compare_lag:
        res = pipe.run(md, progress=a.verbose, max_days=a.max_days)
        print(format_result(res), file=out)
        return 0
    if not hasattr(cfg.execution, "signal_lag_days"):
        print(
            "WARNING: ExecutionConfig has no signal_lag_days field yet: --compare-lag runs lag 0 only",
            file=out,
        )
        res = pipe.run(md, progress=a.verbose, max_days=a.max_days)
        print(format_result(res), file=out)
        return 0
    # headline = the configured lag (production default 1, or --signal-lag); lag 0 and 1
    # are always run for the D4 comparison.  Each lag is a full fit + backtest:
    # ResearchPipeline.run refits whenever cfg.execution.signal_lag_days differs from
    # the lag the alpha was fitted with (labels, purge and IC windows depend on it),
    # so nothing is reused across lags.
    base = int(cfg.execution.signal_lag_days)
    results = {}
    for lag in sorted({0, 1, base}):
        cfg.execution.signal_lag_days = lag
        results[lag] = pipe.run(md, progress=a.verbose, max_days=a.max_days)
    cfg.execution.signal_lag_days = base
    print(format_result(results[base]), file=out)
    print("", file=out)
    print(compare_lag_report(results[0], results[1]), file=out)
    return 0


def compare_lag_report(r0, r1) -> str:
    """D4: bound the close-as-signal bias by the Sharpe difference between
    executing at the signal close (lag 0) and one day later (lag 1)."""
    s0, s1 = r0.summary, r1.summary
    lines = [
        "=== Execution-delay check (D4: signal lag 0 vs 1 day) ===",
        f"{'':18s}{'lag 0':>10s}{'lag 1':>10s}{'diff':>10s}",
    ]
    for k in ("sharpe_gross", "sharpe_net", "cagr_net", "avg_daily_turnover"):
        a, b = float(s0[k]), float(s1[k])
        lines.append(f"{k:18s}{a:10.3f}{b:10.3f}{b - a:+10.3f}")
    if r1.signal_lag_days != 1:
        lines.append(
            "NOTE: the simulator did not apply the lag (no signal_lag_days support); both runs are lag 0"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- commands
def cmd_demo(a: argparse.Namespace) -> int:
    from qts.data.synthetic import SyntheticSpec, generate

    t0 = time.time()
    md = generate(SyntheticSpec(n_stocks=a.stocks, n_days=a.days, seed=a.seed))
    print(
        f"synthetic market: {md.close.shape[1]} stocks x {md.close.shape[0]} days  ({time.time() - t0:.1f}s)"
    )
    cfg = _load_config(a.config)
    cfg.seed = a.seed
    return _run_pipeline(md, cfg, a)


def cmd_run(a: argparse.Namespace) -> int:
    if a.store:
        from qts.data.store import ParquetStore

        md = ParquetStore(a.store).as_market_data(start=a.start, end=a.end)
    elif a.yfinance:
        from qts.data.loaders import yfinance_to_market_data

        md = yfinance_to_market_data(a.yfinance, start=a.start or "2015-01-01", end=a.end)
    else:
        print("error: give --store or --yfinance", file=sys.stderr)
        return 2
    from qts.data.universe import survivorship_check

    survivorship_check(md)
    return _run_pipeline(md, _load_config(a.config), a)


def cmd_config(a: argparse.Namespace) -> int:
    from qts.core.config import SystemConfig, dump_default_yaml

    if a.config_cmd == "dump":
        if a.config:
            sys.stdout.write(SystemConfig.from_yaml(a.config).to_yaml())
        else:
            sys.stdout.write(dump_default_yaml())
        return 0
    if a.config_cmd == "validate":
        try:
            cfg = SystemConfig.from_yaml(a.path).validate()
        except (KeyError, TypeError, ValueError) as e:
            print(f"INVALID {a.path}: {e}", file=sys.stderr)
            return 1
        print(f"OK {a.path}  hash {cfg.hash()}")
        return 0
    return 2


def cmd_runs(a: argparse.Namespace) -> int:
    from qts.pipeline import list_runs, load_run

    if a.runs_cmd == "list":
        runs = list_runs(a.root)
        if not runs:
            print(f"(no runs under {a.root})")
            return 0
        print(
            f"{'run_id':40s} {'sharpe_net':>10s} {'overall':>8s} {'ml':>3s} {'lag':>3s}  config / data / git"
        )
        for d in runs[: a.limit]:
            try:
                s = json.loads((d / "summary.json").read_text())
                m = json.loads((d / "manifest.json").read_text())
            except (OSError, ValueError):
                print(f"{d.name:40s} (unreadable)")
                continue
            sr = s.get("backtest", {}).get("sharpe_net", float("nan"))
            print(
                f"{d.name:40s} {float(sr):10.2f} {('PASS' if s.get('overall') else 'FAIL'):>8s} "
                f"{('y' if s.get('use_ml') else 'n'):>3s} {s.get('signal_lag_days', 0):>3d}  "
                f"{m.get('config_hash')} / {m.get('data_hash')} / {str(m.get('git_sha'))[:10]}"
            )
        return 0
    if a.runs_cmd == "show":
        p = Path(a.run_id)
        if not p.exists():
            p = Path(a.root) / a.run_id
        rec = load_run(p)
        m = rec.manifest
        print(f"run {m.run_id}  created {m.created_at}")
        print(
            f"config {m.config_hash}  data {m.data_hash}  git {m.git_sha}{' (dirty)' if m.git_dirty else ''}"
        )
        print("versions: " + ", ".join(f"{k} {v}" for k, v in m.versions.items()))
        b = rec.summary.get("backtest", {})
        for k in (
            "start",
            "end",
            "n_days",
            "sharpe_gross",
            "sharpe_net",
            "cagr_net",
            "max_drawdown_net",
            "avg_daily_turnover",
            "avg_cost_bps_per_side",
        ):
            if k in b:
                print(f"  {k:24s} {b[k]}")
        ic = rec.summary.get("ic", {})
        print(
            f"  mean IC {ic.get('mean_ic')}  t {ic.get('t_stat')}  NW t {ic.get('t_stat_nw')}  n {ic.get('n')}"
        )
        print(rec.hurdles.format())
        return 0
    return 2


def cmd_ledger(a: argparse.Namespace) -> int:
    import pandas as pd

    from qts.validation import TrialLedger

    if not Path(a.path).exists():
        print(f"(no ledger at {a.path})")
        return 0
    if a.ledger_cmd == "migrate":
        changes = TrialLedger(a.path).migrate_family_names(dry_run=a.dry_run)
        if not changes:
            print("nothing to migrate: every family already has the current format")
            return 0
        for c in changes:
            print(f"line {c['line']:>4d}  {str(c['ts'])[:19]}  {c['old']} -> {c['new']}")
        verb = "would rename" if a.dry_run else "renamed"
        print(
            f"{verb} {len(changes)} record(s)"
            + ("" if a.dry_run else f"; backup at {a.path}.bak-<timestamp>")
        )
        return 0
    recs = TrialLedger(a.path).records(a.family)
    if not recs:
        print("(empty)")
        return 0
    rows = []
    for r in recs:
        mt = r.get("metrics", {})
        rows.append(
            {
                "ts": str(r.get("ts", ""))[:19],
                "family": r.get("family"),
                "sharpe": mt.get("sharpe_annual"),
                "n_obs": mt.get("n_obs"),
                "mean_ic": mt.get("mean_ic"),
                "config_hash": r.get("config_hash", ""),
                "data_hash": r.get("data_hash", ""),
                "code": str(r.get("code_version", ""))[:12],
                "reruns": r.get("n_reruns", 0),
            }
        )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qts", description="Quant trading system research CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="end-to-end run on the synthetic market")
    d.add_argument("--stocks", type=int, default=300)
    d.add_argument("--days", type=int, default=1512)
    d.add_argument("--seed", type=int, default=0)
    _add_pipeline_args(d)
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="run on real data (ParquetStore or yfinance)")
    r.add_argument("--store", default=None, help="ParquetStore root")
    r.add_argument("--yfinance", nargs="*", default=None, help="tickers for a yfinance prototype run")
    r.add_argument("--start", default=None)
    r.add_argument("--end", default=None)
    _add_pipeline_args(r)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("config", help="dump / validate SystemConfig YAML")
    csub = c.add_subparsers(dest="config_cmd", required=True)
    cd = csub.add_parser("dump", help="print YAML (defaults, or --config file normalised)")
    cd.add_argument("--config", default=None)
    cv = csub.add_parser("validate", help="strict-load a YAML config")
    cv.add_argument("path")
    c.set_defaults(func=cmd_config)

    rn = sub.add_parser("runs", help="inspect run directories")
    rsub = rn.add_subparsers(dest="runs_cmd", required=True)
    rl = rsub.add_parser("list")
    rl.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    rl.add_argument("--limit", type=int, default=50)
    rs = rsub.add_parser("show")
    rs.add_argument("run_id", help="run id (under --root) or a path to a run directory")
    rs.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    rn.set_defaults(func=cmd_runs)

    lg = sub.add_parser("ledger", help="trial ledger")
    lsub = lg.add_subparsers(dest="ledger_cmd", required=True)
    ls = lsub.add_parser("show")
    ls.add_argument("--path", default=DEFAULT_LEDGER)
    ls.add_argument("--family", default=None)
    lm = lsub.add_parser(
        "migrate",
        help="one-off: rename legacy 'pipeline_phaseN' families to 'pipeline_phaseN:<data_hash[:12]>' "
        "(records without a data hash go to the 'legacy' family); writes a backup first",
    )
    lm.add_argument("--path", default=DEFAULT_LEDGER)
    lm.add_argument("--dry-run", action="store_true", help="print the renames, change nothing")
    lg.set_defaults(func=cmd_ledger)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.INFO if getattr(a, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(a.func(a) or 0)


def _entry(argv: list[str] | None = None) -> None:  # pragma: no cover
    sys.exit(main(argv))


if __name__ == "__main__":  # pragma: no cover
    _entry()
