"""``/api/paper``: the paper books under ``QTS_PAPER_ROOT`` (``qts paper ...``).

Reads go straight to the library (:mod:`qts.paper`: they read the committed journal and
take no lock).  ``run`` and ``reconcile`` are long (minutes) and run as background jobs;
``ack`` is quick and takes the book's lock (409 when another job holds it).  Nothing is
ever installed: ``schedule`` returns the launchd plist and crontab text to copy.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response

from qts_api import schemas as S
from qts_api.deps import get_jobs, get_settings
from qts_api.problems import Problem
from qts_api.routers._common import problems
from qts_api.routers.research import ACTIVE, _queue, _views
from qts_api.services import research as R
from qts_api.services.jobs import JobManager
from qts_api.services.tasks import _paper_root_flag
from qts_api.settings import Settings

router = APIRouter(prefix="/api/paper", tags=["paper"])
Jobs = Annotated[JobManager, Depends(get_jobs)]
Cfg = Annotated[Settings, Depends(get_settings)]


def _book(settings: Settings, name: str) -> str:
    try:
        if R.book_exists(settings, name):
            return name
    except R.NotFound:
        pass
    raise Problem(404, f"No paper book {name[:64]!r} under {settings.paper_root}", type_="book-not-found")


def _paper_error(e: Exception) -> Problem:
    from qts.paper.book import BookLockedError

    if isinstance(e, BookLockedError):
        return Problem(409, f"{e}. Wait for the other job to finish, then try again.", type_="book-locked")
    return Problem(409, str(e), type_="paper-error")


@router.get("/books", response_model=S.PaperBooks, summary="The paper books, with their current status level")
def list_books(settings: Cfg) -> dict[str, Any]:
    from qts.paper.book import PaperError

    out = []
    for name in R.books(settings):
        try:
            out.append(R.book_summary(settings, name))
        except (PaperError, OSError, ValueError, KeyError):  # a damaged book is skipped, the rest served
            continue
    flag = _paper_root_flag(str(settings.paper_root))
    return {"paper_root": str(settings.paper_root), "paper_root_flag": flag[1] if flag else None, "books": out}


@router.get(
    "/books/{name}/status",
    response_model=S.PaperStatus,
    responses=problems(404, 409),
    summary="Status level, checks, metrics and series of a book (qts paper status)",
)
def get_status(name: str, settings: Cfg, jobs: Jobs) -> dict[str, Any]:
    from qts.paper.book import PaperError

    _book(settings, name)
    try:
        st = R.paper_status(settings, name)
    except (PaperError, OSError, ValueError) as e:
        raise _paper_error(e) from e
    st.pop("_first_execution", None)
    views = _views(jobs, "paper_run", name, 8) + _views(jobs, "paper_reconcile", name, 4)
    views.sort(key=lambda j: j["created_at"], reverse=True)
    st["jobs"] = views
    st["last_reconcile"] = None
    for j in jobs.tasks("paper_reconcile", name):
        if j.get("status") == "done" and isinstance(j.get("result_payload"), dict):
            st["last_reconcile"] = j["result_payload"]
            break
    return st


@router.get("/books/{name}/history", response_model=S.PaperHistory, responses=problems(404, 409))
def get_history(name: str, settings: Cfg) -> dict[str, Any]:
    from qts import paper
    from qts.paper.book import PaperError

    _book(settings, name)
    try:
        rows = paper.history(name, paper_root=str(settings.paper_root))
    except (PaperError, OSError, ValueError) as e:
        raise _paper_error(e) from e
    return {"book": name, "rows": rows}


@router.get(
    "/books/{name}/orders",
    response_model=S.PaperOrders,
    responses=problems(404, 409),
    summary="Orders generated on a job day (for the next close) with estimated costs, and that day's fills",
)
def get_orders(
    name: str, settings: Cfg, date: Annotated[str | None, Query(pattern=S.DATE_RE)] = None
) -> dict[str, Any]:
    from qts import paper
    from qts.paper.book import PaperError

    _book(settings, name)
    try:
        return paper.orders(name, date=date, paper_root=str(settings.paper_root))
    except (PaperError, OSError, ValueError) as e:
        raise _paper_error(e) from e


@router.get("/books/{name}/holdings", response_model=S.PaperHoldings, responses=problems(404, 409))
def get_holdings(
    name: str, settings: Cfg, date: Annotated[str | None, Query(pattern=S.DATE_RE)] = None
) -> dict[str, Any]:
    from qts.paper.book import PaperError

    _book(settings, name)
    try:
        return R.paper_holdings(settings, name, date)
    except (PaperError, OSError, ValueError) as e:
        raise _paper_error(e) from e


@router.get(
    "/books/{name}/schedule",
    response_model=S.PaperSchedule,
    responses=problems(404),
    summary="launchd plist and crontab line for the daily job (printed, never installed)",
)
def get_schedule(name: str, settings: Cfg) -> dict[str, Any]:
    from qts import paper

    _book(settings, name)
    return paper.schedule_text(name, paper_root=_rel_root(settings))


def _rel_root(settings: Settings) -> str:
    """The paper root as the CLI would be given it: relative to the working directory
    when it lies under it (the plist / crontab run from the repo)."""
    from pathlib import Path

    try:
        return str(settings.paper_root.relative_to(Path.cwd()))
    except ValueError:
        return str(settings.paper_root)


@router.post(
    "/books/{name}/ack",
    response_model=S.PaperAck,
    responses=problems(404, 409, 422),
    summary="Acknowledge the latest status (a STOP needs it before trading resumes)",
)
def post_ack(name: str, body: S.PaperAckRequest, settings: Cfg, jobs: Jobs) -> dict[str, Any]:
    from qts import paper
    from qts.paper.book import PaperError

    _book(settings, name)
    if any(j.get("status") in ACTIVE for j in jobs.tasks("paper_run", name)):
        raise Problem(409, "A job of this book is queued or running; acknowledge after it finishes.", type_="book-busy")
    try:
        return paper.ack(name, body.note, paper_root=str(settings.paper_root))
    except (PaperError, OSError, ValueError) as e:
        raise _paper_error(e) from e


def _busy(jobs: JobManager, name: str) -> None:
    for task in ("paper_run", "paper_reconcile"):
        if any(j.get("status") in ACTIVE for j in jobs.tasks(task, name)):
            raise Problem(409, "A job of this book is already queued or running.", type_="book-busy")


@router.post(
    "/books/{name}/run",
    response_model=S.TaskLaunchResponse,
    status_code=202,
    responses=problems(404, 409, 422, 429),
    summary="The daily job (qts paper run) as a background job",
)
def post_run(name: str, body: S.PaperRunRequest, settings: Cfg, jobs: Jobs, response: Response) -> dict[str, Any]:
    _book(settings, name)
    _busy(jobs, name)
    what = "dry run" if body.dry_run else ("backfill" if body.until else "run today")
    return _queue(
        jobs, response, task="paper_run", params={"book": name, **body.model_dump()}, subject=name,
        label=f"Paper book {name}: {what}",
    )  # fmt: skip


@router.post(
    "/books/{name}/reconcile",
    response_model=S.TaskLaunchResponse,
    status_code=202,
    responses=problems(404, 409, 429),
    summary="Paper vs the batch simulator over the book's dates (a background job)",
)
def post_reconcile(name: str, settings: Cfg, jobs: Jobs, response: Response) -> dict[str, Any]:
    _book(settings, name)
    _busy(jobs, name)
    return _queue(
        jobs, response, task="paper_reconcile", params={"book": name}, subject=name,
        label=f"Paper book {name}: reconcile with the batch simulator",
    )  # fmt: skip
