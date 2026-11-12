"""Command-line interface: ``python -m qts.cli <command>`` (or ``qts <command>``
once ``[project.scripts] qts = "qts.cli:main"`` is installed).

Commands
--------
``demo``             synthetic end-to-end run (``--ml``, ``--stocks``, ``--days``, ``--seed``,
                     ``--signal-lag N``, ``--compare-lag`` prints lag-0 vs lag-1 Sharpe)
``run``              real data: ``--store PATH`` (ParquetStore) or ``--yfinance T1 T2 ...``;
                     ``--min-mcap-ever USD`` / ``--max-names N`` pre-filter a large store by
                     peak market cap before anything is pivoted
``--set KEY=VALUE``  (``demo`` and ``run``, repeatable) dotted-path config override applied
                     after ``--config`` is loaded, e.g. ``--set portfolio.risk_aversion=5
                     --set portfolio.risk_aversion_mode=target_vol``; VALUE is parsed as a
                     YAML scalar (``0``, ``0.0``, ``true``, ``null``, bare words are strings);
                     unknown keys are an error (exit 2)
``data sharadar sync``      fill a ParquetStore from the Sharadar bundle (``--store``,
                     ``--raw``, ``--tables``, ``--since``, ``--from-files DIR``, ``--dry-run``)
``data sharadar derive``    rebuild ``delisting_reasons`` from the stored ACTIONS / TICKERS
                            tables (no download) and print the counts by reason and year
``data sharadar validate``  sanity checks of a Sharadar-filled store (PASS/WARN per check)
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
``--cache`` turns on the stage cache (``cfg.cache``, :mod:`qts.runs.cache`):
fitted risk / alpha models are stored under ``<runs-root>/_cache`` and reused
when data, config sections and code are unchanged (same numbers, same hashes).
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


class ConfigOverrideError(ValueError):
    """A ``--set KEY=VALUE`` could not be parsed or applied."""


def parse_set_overrides(items: Sequence[str] | None) -> dict[str, object]:
    """``["a.b=1", "c=target_vol"]`` -> ``{"a.b": 1, "c": "target_vol"}``.

    Values are parsed as YAML scalars (``yaml.safe_load``): ``0`` -> int, ``0.0``
    -> float, ``true``/``false`` -> bool, ``null``/``~``/empty -> None, anything
    else a string.  Quote a value to force a string (``--set x='"1"'``).
    Raises :class:`ConfigOverrideError` on a missing ``=`` or an empty key."""
    import yaml

    out: dict[str, object] = {}
    for item in items or ():
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigOverrideError(f"--set expects KEY=VALUE, got {item!r}")
        try:
            val = yaml.safe_load(raw.strip()) if raw.strip() else None
        except yaml.YAMLError as e:
            raise ConfigOverrideError(f"--set {key}: cannot parse value {raw!r} as YAML: {e}") from e
        if isinstance(val, (dict, list)):
            raise ConfigOverrideError(f"--set {key}: value must be a scalar, got {type(val).__name__}")
        out[key] = val
    return out


def _apply_set_overrides(cfg, items: Sequence[str] | None):
    """``cfg.with_overrides(parse_set_overrides(items))`` with unknown keys and
    section validation failures re-raised as :class:`ConfigOverrideError`."""
    ov = parse_set_overrides(items)
    if not ov:
        return cfg
    try:
        return cfg.with_overrides(ov)
    except KeyError as e:
        raise ConfigOverrideError(f"--set: {e.args[0] if e.args else e}") from e
    except (TypeError, ValueError) as e:
        raise ConfigOverrideError(f"--set: invalid value: {e}") from e


def _set_signal_lag(cfg, lag: int | None):
    """``(cfg with execution.signal_lag_days = lag, warning)``; the warning is a
    text when the field does not exist (older ExecutionConfig), else ``None``."""
    if lag is None:
        return cfg, None
    if not hasattr(cfg.execution, "signal_lag_days"):
        return cfg, "ExecutionConfig has no signal_lag_days field yet: --signal-lag ignored"
    return cfg.with_overrides({"execution.signal_lag_days": int(lag)}), None


def _add_pipeline_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=None, help="SystemConfig YAML (default: built-in defaults)")
    ap.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        metavar="KEY=VALUE",
        default=None,
        help="dotted config override applied after --config (repeatable), e.g. "
        "--set portfolio.risk_aversion=5; VALUE is a YAML scalar; unknown keys are an error",
    )
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
    ap.add_argument(
        "--rebalance-every",
        type=int,
        default=None,
        help="solve every k-th day (default: portfolio.rebalance_every_days of the config)",
    )
    ap.add_argument("--nav", type=float, default=5e6)
    ap.add_argument("--ledger", default=DEFAULT_LEDGER, help="trial ledger JSONL ('' disables)")
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT, help="where run directories are written")
    ap.add_argument("--no-save", action="store_true", help="do not write a run directory")
    ap.add_argument("--max-days", type=int, default=None, help="backtest only the last N out-of-sample days")
    ap.add_argument(
        "--cache",
        action="store_true",
        help="reuse fitted risk / alpha models from the stage cache (cache.enabled; "
        "default dir <runs-root>/_cache); results and config hash are unchanged",
    )
    ap.add_argument("-v", "--verbose", action="store_true")


def _run_pipeline(md, cfg, a: argparse.Namespace, out=None) -> int:
    from qts.pipeline import ResearchPipeline, format_result

    out = out if out is not None else sys.stdout  # resolved at call time (capture-friendly)
    cfg = cfg.with_overrides({"universe.top_n_by_mcap": min(cfg.universe.top_n_by_mcap, md.close.shape[1])})
    if getattr(a, "cache", False):
        root = cfg.cache.root or f"{a.runs_root}/_cache"  # also with --no-save
        cfg = cfg.with_overrides({"cache.enabled": True, "cache.root": root})
    cfg, warn = _set_signal_lag(cfg, a.signal_lag)
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
        pipe.cfg = cfg.with_overrides({"execution.signal_lag_days": lag})
        results[lag] = pipe.run(md, progress=a.verbose, max_days=a.max_days)
    pipe.cfg = cfg
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
    try:
        cfg = _apply_set_overrides(_load_config(a.config).with_overrides({"seed": a.seed}), a.set_overrides)
    except ConfigOverrideError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return _run_pipeline(md, cfg, a)


def cmd_run(a: argparse.Namespace) -> int:
    try:  # fail before any data is loaded
        cfg = _apply_set_overrides(_load_config(a.config), a.set_overrides)
    except ConfigOverrideError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.store:
        from qts.data.store import ParquetStore

        store = ParquetStore(a.store)
        tickers = None
        if a.min_mcap_ever is not None or a.max_names is not None:
            # pre-filter by peak market cap (reads only the long market_cap table), so a
            # ten-thousand-name store is never pivoted whole; the pipeline's own PIT
            # universe rules (top_n_by_mcap, min price, ADV) still apply on top
            tickers = store.tickers_by_market_cap(
                a.start, a.end, min_mcap_ever=a.min_mcap_ever, max_names=a.max_names
            )
            print(f"prefilter: {len(tickers)} candidate ids by peak market cap")
            if not tickers:
                print("error: the prefilter left no ids", file=sys.stderr)
                return 2
        md = store.as_market_data(universe_tickers=tickers, start=a.start, end=a.end)
        print(
            f"store: {md.close.shape[1]} ids x {md.close.shape[0]} days ({md.dates[0].date()}..{md.dates[-1].date()})"
        )
    elif a.yfinance:
        from qts.data.loaders import yfinance_to_market_data

        md = yfinance_to_market_data(a.yfinance, start=a.start or "2015-01-01", end=a.end)
    else:
        print("error: give --store or --yfinance", file=sys.stderr)
        return 2
    from qts.data.universe import survivorship_check

    survivorship_check(md)
    return _run_pipeline(md, cfg, a)


def cmd_data(a: argparse.Namespace) -> int:
    if a.data_cmd != "sharadar":
        return 2
    from qts.data import sharadar

    if a.sharadar_cmd == "sync":
        tables = [t.strip().upper() for t in a.tables.split(",") if t.strip()]
        try:
            if a.from_files:
                print(sharadar.format_source_map(sharadar.resolve_sources(a.from_files, tables)))
            rep = sharadar.sync(
                store_root=a.store,
                raw_dir=a.raw,
                tables=tables,
                since=a.since,
                from_files=a.from_files,
                api_key=a.api_key,
                chunksize=a.chunksize,
                dry_run=a.dry_run,
                force=a.force,
            )
        except sharadar.SharadarError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(rep.format())
        return 0
    if a.sharadar_cmd == "derive":
        try:
            drep = sharadar.derive_from_store(a.store, window_days=a.window)
        except sharadar.SharadarError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(drep.format(top=a.top))
        print(f"written: {a.store}/delisting_reasons ({drep.n_delisted} rows)")
        if a.json:
            Path(a.json).write_text(json.dumps(drep.to_dict(), indent=1, default=str))
            print(f"written {a.json}")
        return 0
    if a.sharadar_cmd == "validate":
        rep = sharadar.validate(a.store, sample_size=a.sample)
        print(rep.format())
        if a.json:
            Path(a.json).write_text(json.dumps(rep.to_dict(), indent=1, default=str))
            print(f"written {a.json}")
        return 0 if all(c.status != "WARN" or c.name != "required_datasets" for c in rep.checks) else 1
    return 2


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
    r.add_argument(
        "--min-mcap-ever",
        type=float,
        default=None,
        help="store prefilter: load only ids whose peak market cap in [start, end] is >= this (USD)",
    )
    r.add_argument(
        "--max-names",
        type=int,
        default=None,
        help="store prefilter: load only the N ids with the largest peak market cap in [start, end]",
    )
    _add_pipeline_args(r)
    r.set_defaults(func=cmd_run)

    dt = sub.add_parser("data", help="load vendor data into a ParquetStore")
    dsub = dt.add_subparsers(dest="data_cmd", required=True)
    sh = dsub.add_parser("sharadar", help="Sharadar Core US Equities Bundle (Nasdaq Data Link)")
    shsub = sh.add_subparsers(dest="sharadar_cmd", required=True)
    ss = shsub.add_parser("sync", help="download (or read --from-files) -> transform -> write the store")
    ss.add_argument("--store", default="data/store", help="ParquetStore root")
    ss.add_argument(
        "--raw", default="data/raw/sharadar", help="raw zip cache (<raw>/<TABLE>/<date>_<full|since-..>.zip)"
    )
    ss.add_argument(
        "--tables",
        default="SEP,TICKERS,DAILY,SF1,ACTIONS,SP500",
        help="comma-separated tables; optional extras: METRICS,EVENTS,SFP,INDICATORS",
    )
    ss.add_argument("--since", default=None, help="incremental: rows with lastupdated >= YYYY-MM-DD (upsert)")
    ss.add_argument(
        "--from-files", default=None, help="read already-downloaded zips/CSVs from DIR (no network)"
    )
    ss.add_argument(
        "--api-key", default=None, help="Nasdaq Data Link key (default: $NASDAQ_DATA_LINK_API_KEY or .env)"
    )
    ss.add_argument("--chunksize", type=int, default=500_000, help="CSV rows per streamed chunk")
    ss.add_argument("--dry-run", action="store_true", help="show what would be downloaded / written")
    ss.add_argument("--force", action="store_true", help="ignore the per-table resume state")
    ss.add_argument("-v", "--verbose", action="store_true", help="log progress per chunk / year")
    sd = shsub.add_parser(
        "derive",
        help="rebuild derived reference tables (delisting_reasons) from the stored tables; no download",
    )
    sd.add_argument("--store", default="data/store", help="ParquetStore root")
    sd.add_argument(
        "--window",
        type=int,
        default=5,
        help="business days around the delisting date in which an ACTIONS row counts (default 5)",
    )
    sd.add_argument("--top", type=int, default=15, help="action codes listed per table")
    sd.add_argument("--json", default=None, help="also write the count report as JSON to this path")
    sv = shsub.add_parser("validate", help="sanity checks of a Sharadar-filled store")
    sv.add_argument("--store", default="data/store")
    sv.add_argument("--sample", type=int, default=50, help="ids sampled for the per-name checks")
    sv.add_argument("--json", default=None, help="also write the report as JSON to this path")
    dt.set_defaults(func=cmd_data)

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
