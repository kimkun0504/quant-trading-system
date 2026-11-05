"""Settings from the environment (``QTS_*``); no extra dependency.

=====================  =====================================  ==========================
variable               meaning                                default
=====================  =====================================  ==========================
QTS_RUNS_ROOT          run directories (``runs/<run_id>/``)   ``./runs``
QTS_LEDGER_PATH        trial ledger JSONL                     ``<runs root>/trial_ledger.jsonl``
QTS_CORS_ORIGINS       comma-separated allowed origins        ``http://localhost:5173``
QTS_SERVE_UI           serve the built UI at ``/`` (1/true)   off
QTS_UI_DIST            the built UI                           ``<repo>/ui/dist``
QTS_MAX_WORKERS        concurrent pipeline jobs               ``1``
QTS_STORE_PATH         ParquetStore root for ``kind=store``   unset (store runs refused)
QTS_JOB_START_METHOD   multiprocessing start method           ``spawn``
QTS_ALLOWED_HOSTS      extra Host header names, comma-        (none; localhost, 127.0.0.1
                       separated (``*.example.com`` allowed)  and ::1 are always allowed)
QTS_MAX_QUEUED         jobs waiting for a worker; more -> 429 ``4``
=====================  =====================================  ==========================

The Host allow-list (``TrustedHostMiddleware``) is the DNS-rebinding defence: a web page
that rebinds its own domain to 127.0.0.1 still sends ``Host: <its domain>``, which gets
a 400.  Add a name to ``QTS_ALLOWED_HOSTS`` only for a deliberate remote setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Host header values always accepted (``[::1]`` is how an IPv6 literal arrives with a port).
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1", "[::1]")


def _flag(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    runs_root: Path = Path("runs")
    ledger_path: Path | None = None
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    serve_ui: bool = False
    ui_dist: Path = REPO_ROOT / "ui" / "dist"
    max_workers: int = 1
    store_path: Path | None = None
    job_start_method: str = "spawn"
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    max_queued: int = 4
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def ledger(self) -> Path:
        return self.ledger_path or (self.runs_root / "trial_ledger.jsonl")

    @property
    def jobs_dir(self) -> Path:
        return self.runs_root / "_jobs"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = dict(os.environ if env is None else env)
        origins = tuple(o.strip() for o in e.get("QTS_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip())
        ledger = e.get("QTS_LEDGER_PATH")
        store = e.get("QTS_STORE_PATH")
        extra_hosts = tuple(h.strip() for h in e.get("QTS_ALLOWED_HOSTS", "").split(",") if h.strip())
        return cls(
            runs_root=Path(e.get("QTS_RUNS_ROOT", "runs")).expanduser().resolve(),
            ledger_path=Path(ledger).expanduser().resolve() if ledger else None,
            cors_origins=origins,
            serve_ui=_flag(e.get("QTS_SERVE_UI")),
            ui_dist=Path(e.get("QTS_UI_DIST", str(REPO_ROOT / "ui" / "dist"))).expanduser().resolve(),
            max_workers=max(1, int(e.get("QTS_MAX_WORKERS", "1"))),
            store_path=Path(store).expanduser().resolve() if store else None,
            job_start_method=e.get("QTS_JOB_START_METHOD", "spawn"),
            allowed_hosts=DEFAULT_ALLOWED_HOSTS + tuple(h for h in extra_hosts if h not in DEFAULT_ALLOWED_HOSTS),
            max_queued=max(0, int(e.get("QTS_MAX_QUEUED", "4"))),
        )
