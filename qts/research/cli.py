"""Argparse wiring of the research commands (registered by :func:`qts.cli.build_parser`).

Each command is a thin layer over a library entry point of :mod:`qts.research`:

``check midcap``    :func:`qts.research.midcap.run_midcap_check`
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


def register_research_commands(sub: Any) -> None:
    """Add the research sub-commands to ``qts``'s sub-parser action."""
    _add_check(sub)


__all__ = ["register_research_commands"]
