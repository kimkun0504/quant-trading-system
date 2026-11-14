"""Argparse wiring of the research commands (registered by :func:`qts.cli.build_parser`).

Each command is a thin layer over a library entry point of :mod:`qts.research`:

``check midcap``    :func:`qts.research.midcap.run_midcap_check`
``seal status``     :func:`qts.research.seal.seal_status`
``signal list``     :func:`qts.research.signals.list_signals`
``signal test``     :func:`qts.research.signals.run_signal_test`
``features list``   :func:`qts.alpha.feature_registry.feature_info`
``sweep``           :func:`qts.research.sweep.plan_sweep` / ``run_sweep``
``capacity``        :func:`qts.research.capacity.run_capacity`
``risk calibrate``  :func:`qts.research.risk_calibration.run_risk_calibration`
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

DEFAULT_RUNS_ROOT = "runs"
DEFAULT_LEDGER = "runs/trial_ledger.jsonl"


def _print(msg: str) -> None:
    print(msg, flush=True)


def _store_args(
    ap: argparse.ArgumentParser,
    max_names: int | None = None,
    required: bool = True,
    max_rank: int | None = None,
) -> None:
    ap.add_argument("--store", required=required, default=None, help="ParquetStore root")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--min-mcap-ever",
        type=float,
        default=None,
        help="store prefilter: load only ids whose peak market cap in [start, end] is >= this (USD)",
    )
    ap.add_argument(
        "--max-names",
        type=int,
        default=max_names,
        help="store prefilter: load only the N ids with the largest peak market cap in [start, end]"
        + (f" (default {max_names})" if max_names is not None else ""),
    )
    ap.add_argument(
        "--max-rank",
        type=int,
        default=max_rank,
        help="store prefilter: load the ids that rank <= K by market cap, among the names passing the "
        "universe's price / ADV screens, on at least one date in [start, end] (not with --max-names / "
        "--min-mcap-ever)" + (f"; default {max_rank}" if max_rank is not None else ""),
    )


def _common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ledger", default=DEFAULT_LEDGER, help="trial ledger JSONL")
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT, help="where run directories are written")
    ap.add_argument(
        "--cache",
        action="store_true",
        help="reuse fitted risk / alpha models from the stage cache (<runs-root>/_cache)",
    )
    ap.add_argument("--json", default=None, help="also write the result as JSON to this path")
    ap.add_argument("-v", "--verbose", action="store_true")


# ----------------------------------------------------------------------------- check midcap
def cmd_check(a: argparse.Namespace) -> int:
    if a.check_cmd != "midcap":
        return 2
    from qts.research.common import write_json
    from qts.research.data import StoreLoadError, StoreSpec
    from qts.research.midcap import MIDCAP_DEFAULT_MAX_RANK, MidcapCheckError, run_midcap_check

    if (a.max_names is not None or a.min_mcap_ever is not None) and a.max_rank == MIDCAP_DEFAULT_MAX_RANK:
        a.max_rank = None  # an explicit peak-cap prefilter replaces the default rank prefilter
    try:
        r = run_midcap_check(
            StoreSpec.from_args(a),
            config_path=a.config,
            ledger_path=a.ledger or None,
            runs_root=a.runs_root,
            cache_root=f"{a.runs_root}/_cache" if a.cache else None,
            nav=a.nav,
            force=a.force,
            dry_run=a.dry_run,
            reference_run=a.reference_run,
            progress=a.verbose,
            log=_print,
        )
    except (MidcapCheckError, StoreLoadError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        write_json(a.json, r.to_dict())
        print(f"written {a.json}")
    if r.json_path:
        print(f"written {r.json_path}")
    return 0


def _add_check(sub: Any) -> None:
    from qts.research.midcap import MIDCAP_DEFAULT_MAX_RANK, PREREGISTRATION

    ck = sub.add_parser("check", help="pre-registered one-shot confirmations")
    csub = ck.add_subparsers(dest="check_cmd", required=True)
    m = csub.add_parser(
        "midcap",
        help="run the frozen baseline once on stocks ranked 1001-2000 by market cap (full history, "
        f"exempt from the seal; criteria pre-registered in {PREREGISTRATION})",
    )
    m.add_argument("--config", default="configs/baseline.yaml", help="frozen baseline (default %(default)s)")
    _store_args(m, max_rank=MIDCAP_DEFAULT_MAX_RANK)
    _common_args(m)
    m.add_argument("--nav", type=float, default=5e6)
    m.add_argument(
        "--force",
        action="store_true",
        help="run although this configuration's mid-cap check already ran (the repeat is NOT "
        "out-of-sample and is flagged on its ledger record)",
    )
    m.add_argument(
        "--dry-run",
        action="store_true",
        help="print the load-coverage table and the plan; load and log nothing",
    )
    m.add_argument(
        "--reference-run",
        default=None,
        help="large-cap run directory for the side-by-side (default runs/20261113T023313333450Z_e4667d90)",
    )
    ck.set_defaults(func=cmd_check)


# ----------------------------------------------------------------------------- seal
def cmd_seal(a: argparse.Namespace) -> int:
    from qts.research.common import write_json
    from qts.research.seal import SealError, seal_status

    if a.seal_cmd != "status":
        return 2
    try:
        st = seal_status(a.ledger or None)
    except SealError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(st.format())
    if a.json:
        write_json(a.json, st.to_dict())
        print(f"written {a.json}")
    return 0


def _add_seal(sub: Any) -> None:
    sl = sub.add_parser("seal", help="the research holdout seal (research/seal.yaml)")
    ssub = sl.add_subparsers(dest="seal_cmd", required=True)
    st = ssub.add_parser("status", help="the seal, every unseal so far and the research runs made under it")
    st.add_argument("--ledger", default=DEFAULT_LEDGER, help="trial ledger JSONL")
    st.add_argument("--json", default=None, help="also write the status as JSON to this path")
    sl.set_defaults(func=cmd_seal)


# ----------------------------------------------------------------------------- signal
def _parse_ints(text: str) -> list[int]:
    try:
        out = [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from e
    if not out or any(v <= 0 for v in out):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return out


def cmd_signal(a: argparse.Namespace) -> int:
    from qts.research.common import write_json

    if a.signal_cmd == "list":
        from qts.research.signals import list_signals

        sigs = list_signals(a.signal_module)
        print(f"{'name':20s} {'source':8s} {'set':6s} {'theme':15s} {'sign':>4s} {'role':16s} description")
        for s_ in sigs:
            print(
                f"{s_.name:20s} {s_.source:8s} {s_.feature_set:6s} {s_.theme:15s} {s_.sign:+4.0f} {s_.role:16s} "
                f"{s_.description}"
            )
        if a.json:
            write_json(a.json, [s_.to_dict() for s_ in sigs])
            print(f"written {a.json}")
        return 0
    if a.signal_cmd != "test":
        return 2
    from qts.research.data import StoreLoadError, StoreSpec
    from qts.research.seal import SealError
    from qts.research.signals import run_signal_test

    try:
        r = run_signal_test(
            a.feature,
            StoreSpec.from_args(a),
            horizons=a.horizons,
            baseline=a.baseline,
            step=a.step,
            cost_bps=a.cost_bps,
            ledger_path=a.ledger or None,
            runs_root=a.runs_root,
            cache_root=None if not a.no_cache else "",
            use_ml=not a.no_ml,
            signal_module=a.signal_module,
            log_trial=not a.no_log,
            log=_print,
        )
    except (KeyError, ValueError, TypeError, SealError, StoreLoadError) as e:
        print(f"error: {e.args[0] if isinstance(e, KeyError) and e.args else e}", file=sys.stderr)
        return 2
    print(r.format())
    if a.json:
        write_json(a.json, r.to_dict())
        print(f"written {a.json}")
    return 0


def _add_signal(sub: Any) -> None:
    from qts.research.signals import DEFAULT_HORIZONS, DEFAULT_STEP

    sg = sub.add_parser("signal", help="single-signal research (pre-seal period)")
    ssub = sg.add_subparsers(dest="signal_cmd", required=True)
    sl = ssub.add_parser("list", help="every testable signal: v1 + v2 features and registered custom signals")
    sl.add_argument(
        "--signal-module", default=None, help="also import custom signals from this .py file / module"
    )
    sl.add_argument("--json", default=None)
    st = ssub.add_parser(
        "test",
        help="IC by horizon / year / fold, decay, neutral paper book, correlations and incremental value "
        "over the baseline of one signal; logged as a trial in family signal:<market_hash12>",
    )
    st.add_argument("feature", help="signal name (`qts signal list`)")
    st.add_argument(
        "--horizons",
        type=_parse_ints,
        default=list(DEFAULT_HORIZONS),
        help="label horizons in trading days, comma separated (default %(default)s)",
    )
    st.add_argument(
        "--baseline", default="configs/baseline.yaml", help="baseline config (default %(default)s)"
    )
    _store_args(st)
    _common_args(st)
    st.add_argument(
        "--step", type=int, default=DEFAULT_STEP, help="sampling step / paper-book holding (days)"
    )
    st.add_argument(
        "--cost-bps", type=float, default=None, help="cost per side of the paper book (default: baseline run)"
    )
    st.add_argument("--signal-module", default=None, help="import custom signals from this .py file / module")
    st.add_argument(
        "--no-log", action="store_true", help="do not log the test in the trial ledger (not counted!)"
    )
    st.add_argument("--no-cache", action="store_true", help="do not use the stage cache for the baseline fit")
    st.add_argument("--no-ml", action="store_true", help="baseline without the ML sleeve (phase 1 fit)")
    sg.set_defaults(func=cmd_signal)


# ----------------------------------------------------------------------------- features
def cmd_features(a: argparse.Namespace) -> int:
    from qts.alpha.feature_registry import feature_info
    from qts.cli import ConfigOverrideError, _apply_set_overrides, _load_config
    from qts.research.common import write_json

    if a.features_cmd != "list":
        return 2
    try:
        cfg = _apply_set_overrides(_load_config(a.config), a.set_overrides)
    except ConfigOverrideError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rows = feature_info(cfg.alpha.feature_set)
    off = set(cfg.alpha.disabled_features)
    print(f"feature set {cfg.alpha.feature_set}: {len(rows)} features, {len(off)} disabled")
    print(f"{'name':20s} {'theme':15s} {'sign':>4s} {'role':16s} {'set':4s} {'on':3s} description")
    for r in rows:
        print(
            f"{r.name:20s} {r.theme:15s} {r.sign:+4.0f} {r.role:16s} {r.feature_set:4s} "
            f"{'off' if r.name in off else 'on':3s} {r.description}"
        )
    if a.json:
        write_json(a.json, [{**r.to_dict(), "enabled": r.name not in off} for r in rows])
        print(f"written {a.json}")
    return 0


def _add_features(sub: Any) -> None:
    fe = sub.add_parser("features", help="alpha features of a configuration")
    fsub = fe.add_subparsers(dest="features_cmd", required=True)
    fl = fsub.add_parser(
        "list", help="name, theme, sign, role, set, enabled and description of every feature"
    )
    fl.add_argument("--config", default=None)
    fl.add_argument("--set", dest="set_overrides", action="append", metavar="KEY=VALUE", default=None)
    fl.add_argument("--json", default=None)
    fe.set_defaults(func=cmd_features)


# ----------------------------------------------------------------------------- sweep
def _confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        print(f"{question} -- not a terminal: pass --yes to run without asking", file=sys.stderr)
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def cmd_sweep(a: argparse.Namespace) -> int:
    from qts.research.data import StoreLoadError, StoreSpec
    from qts.research.seal import SealError
    from qts.research.sweep import SweepError, parse_param, plan_sweep, run_sweep

    try:
        params = [parse_param(p) for p in a.param or []]
        spec = StoreSpec.from_args(a)
        plan = plan_sweep(
            params,
            base_config=a.config,
            mode="zip" if a.zip else "grid",
            spec=spec,
            use_ml=not a.no_ml,
            nav=a.nav,
            max_runs=a.max_runs,
            max_days=a.max_days,
            ledger_path=a.ledger or None,
            runs_root=a.runs_root,
            sweep_name=a.id,
            log=_print,
        )
    except (SweepError, SealError, StoreLoadError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(plan.format())
    todo = [m for m in plan.members if m.status != "done"]
    if a.dry_run:
        print("\ndry run: nothing run, nothing logged")
        return 0
    if not todo:
        print("\nevery member is done")
    elif not a.yes and not _confirm(
        f"\nrun {len(todo)} member(s) (each a logged trial unless marked reference)?"
    ):
        print("not run")
        return 1
    try:
        res = run_sweep(plan, ledger_path=a.ledger or None, progress=a.verbose, log=_print)
    except (SweepError, SealError, StoreLoadError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(res.format())
    if a.json:
        from qts.research.common import write_json

        write_json(a.json, res.to_dict())
        print(f"written {a.json}")
    return 0


def _add_sweep(sub: Any) -> None:
    from qts.research.sweep import DEFAULT_MAX_RUNS

    sw = sub.add_parser(
        "sweep",
        help="run a parameter grid under the seal with trial accounting, then PBO (CSCV) across its members",
    )
    sw.add_argument("--config", default="configs/baseline.yaml", help="base config (default %(default)s)")
    sw.add_argument(
        "--param",
        action="append",
        metavar="KEY=V1,V2,...",
        help="a swept field and its values (repeatable), e.g. portfolio.trade_aversion=0.5,1,2",
    )
    g = sw.add_mutually_exclusive_group()
    g.add_argument("--grid", action="store_true", help="cartesian product of the --param values (default)")
    g.add_argument("--zip", action="store_true", help="element-wise combination (equal lengths)")
    _store_args(sw)
    _common_args(sw)
    sw.add_argument("--nav", type=float, default=5e6)
    sw.add_argument(
        "--max-runs", type=int, default=DEFAULT_MAX_RUNS, help="refuse larger sweeps (default %(default)s)"
    )
    sw.add_argument("--max-days", type=int, default=None, help="backtest only the last N days of each member")
    sw.add_argument(
        "--id", default=None, help="sweep id (default: a hash of the sweep; re-running resumes it)"
    )
    sw.add_argument("--dry-run", action="store_true", help="print the plan and the trial impact only")
    sw.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    sw.add_argument("--no-ml", action="store_true", help="members without the ML sleeve (phase 1)")
    sw.set_defaults(func=cmd_sweep)


# ----------------------------------------------------------------------------- capacity / risk
def _parse_floats(text: str) -> list[float]:
    try:
        out = [float(x) for x in str(text).split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from e
    if not out:
        raise argparse.ArgumentTypeError("expected at least one number")
    return out


def cmd_capacity(a: argparse.Namespace) -> int:
    from qts.research.capacity import run_capacity
    from qts.research.common import write_json
    from qts.research.data import StoreLoadError, StoreSpec
    from qts.research.seal import SealError

    try:
        r = run_capacity(
            StoreSpec.from_args(a),
            navs=a.navs,
            baseline=a.config,
            use_ml=not a.no_ml,
            ledger_path=a.ledger or None,
            runs_root=a.runs_root,
            max_days=a.max_days,
            progress=a.verbose,
            log=_print,
        )
    except (ValueError, SealError, StoreLoadError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(r.format())
    if a.json:
        write_json(a.json, r.to_dict())
        print(f"written {a.json}")
    return 0


def cmd_risk(a: argparse.Namespace) -> int:
    from qts.research.common import write_json
    from qts.research.data import StoreLoadError, StoreSpec
    from qts.research.risk_calibration import run_risk_calibration
    from qts.research.seal import SealError

    if a.risk_cmd != "calibrate":
        return 2
    try:
        r = run_risk_calibration(
            StoreSpec.from_args(a),
            nu_grid=a.nu,
            baseline=a.config,
            use_ml=not a.no_ml,
            nav=a.nav,
            ledger_path=a.ledger or None,
            runs_root=a.runs_root,
            max_days=a.max_days,
            progress=a.verbose,
            log=_print,
        )
    except (ValueError, SealError, StoreLoadError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(r.format())
    if a.json:
        write_json(a.json, r.to_dict())
        print(f"written {a.json}")
    return 0


def _add_capacity_and_risk(sub: Any) -> None:
    from qts.research.capacity import DEFAULT_NAVS
    from qts.research.risk_calibration import DEFAULT_NU_GRID

    cp = sub.add_parser(
        "capacity",
        help="re-run only the baseline backtest at several NAVs (fits cached; under the seal; not trials)",
    )
    cp.add_argument("--config", default="configs/baseline.yaml")
    cp.add_argument(
        "--navs",
        type=_parse_floats,
        default=list(DEFAULT_NAVS),
        help="NAVs in USD, comma separated (default %(default)s)",
    )
    _store_args(cp)
    _common_args(cp)
    cp.add_argument("--max-days", type=int, default=None, help="backtest only the last N days (quicker read)")
    cp.add_argument("--no-ml", action="store_true", help="baseline without the ML sleeve (phase 1)")
    cp.set_defaults(func=cmd_capacity)

    rk = sub.add_parser("risk", help="risk-model reports")
    rsub = rk.add_subparsers(dest="risk_cmd", required=True)
    rc = rsub.add_parser(
        "calibrate",
        help="realised / ex-ante vol of the baseline book by year for a grid of risk.alignment_nu_scale "
        "(backtest re-runs only; report, no default change)",
    )
    rc.add_argument("--config", default="configs/baseline.yaml")
    rc.add_argument(
        "--nu", type=_parse_floats, default=list(DEFAULT_NU_GRID), help="nu scales (default %(default)s)"
    )
    rc.add_argument("--nav", type=float, default=5e6)
    _store_args(rc)
    _common_args(rc)
    rc.add_argument("--max-days", type=int, default=None, help="backtest only the last N days (quicker read)")
    rc.add_argument("--no-ml", action="store_true", help="baseline without the ML sleeve (phase 1)")
    rk.set_defaults(func=cmd_risk)


def register_research_commands(sub: Any) -> None:
    """Add the research sub-commands to ``qts``'s sub-parser action."""
    _add_check(sub)
    _add_seal(sub)
    _add_signal(sub)
    _add_features(sub)
    _add_sweep(sub)
    _add_capacity_and_risk(sub)


__all__ = ["register_research_commands"]
