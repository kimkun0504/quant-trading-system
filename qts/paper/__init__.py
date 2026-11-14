"""Internal paper-trading book for a frozen strategy config (handbook 14).

Library entry points (all return JSON-serialisable dicts; shapes in
``docs/handbook/14_paper_trading.md``)::

    from qts import paper

    paper.init_book(
        "baseline",
        "configs/baseline.yaml",
        nav=5e6,
        start="2026-11-13",
        reference_run="runs/20261113T023313333450Z_e4667d90",
    )
    paper.run_book("baseline")  # the daily job (sync, freshness, catch-up)
    paper.status("baseline")  # status, checks, metrics, series
    paper.history("baseline")
    paper.orders("baseline")
    paper.ack("baseline", note="...")
    paper.replay("baseline")
    paper.schedule_text("baseline")
    paper.reconcile("baseline")

Modules: :mod:`~qts.paper.book` (storage), :mod:`~qts.paper.data` (sources, sync,
freshness, calendars), :mod:`~qts.paper.serving` (frozen walk-forward models,
day-``s`` signals), :mod:`~qts.paper.engine` (the daily step), :mod:`~qts.paper.monitor`
(stop rules), :mod:`~qts.paper.expectations` (backtest yardstick),
:mod:`~qts.paper.runner` (entry points).
"""

from qts.paper.book import BookHeader, BookLockedError, DataSpec, PaperError, StaleDataError
from qts.paper.monitor import StopRules
from qts.paper.runner import (
    ack,
    history,
    init_book,
    orders,
    reconcile,
    replay,
    run_book,
    schedule_text,
    status,
)

__all__ = [
    "BookHeader",
    "BookLockedError",
    "DataSpec",
    "PaperError",
    "StaleDataError",
    "StopRules",
    "ack",
    "history",
    "init_book",
    "orders",
    "reconcile",
    "replay",
    "run_book",
    "schedule_text",
    "status",
]
