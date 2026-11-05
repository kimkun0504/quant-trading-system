"""Run identity and reproducibility: :class:`RunManifest` and data hashing.

A manifest is written next to every pipeline run (``runs/<run_id>/manifest.json``,
see :mod:`qts.pipeline` for the full run-directory layout).  It answers "what
exactly produced these numbers?":

* ``config_hash``  — ``SystemConfig.hash()`` (SHA-256 of the canonical JSON);
* ``data_hash``    — :func:`hash_market_data`: every ``MarketData`` panel
  (required *and* optional, the ``extra`` dict, the industry map) plus the
  eligibility panel — handbook fix #37 (the old hash covered ``close`` only);
* ``git_sha`` / ``git_dirty`` — code version (``git rev-parse HEAD``;
  ``"unknown"`` outside a git checkout); a dirty tree (modified tracked files
  *or* untracked, non-ignored files) means the sha alone does not reproduce
  the run;
* library versions (numerics change across numpy / pandas / scipy / sklearn /
  LightGBM / cvxpy / Clarabel releases), python version, platform;
* ``timings`` (seconds per stage) and ``seeds``.

Nothing here imports the heavy layers, so the manifest can be read without them.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qts.core.types import MarketData

MANIFEST_FILE = "manifest.json"
MANIFEST_VERSION = 1

#: Libraries whose versions change numerical results.
TRACKED_LIBRARIES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "lightgbm",
    "cvxpy",
    "clarabel",
    "pyarrow",
    "yaml",
)


# ----------------------------------------------------------------------------- hashing
#: Optional fields added after the hash format was fixed: when ``None`` they
#: contribute nothing, so data hashes of existing ``MarketData`` (and the trial
#: ledger keys built on them) do not change; when present they are hashed.
_SKIP_IF_NONE = frozenset({"delisting_dates"})


def _hash_frame(h: Any, name: str, obj: Any) -> None:
    h.update(f"<{name}:{type(obj).__name__}:{obj.shape}>".encode())
    h.update("|".join(map(str, obj.index)).encode())
    if isinstance(obj, pd.DataFrame):
        h.update("|".join(map(str, obj.columns)).encode())
        vals = obj.to_numpy()
    else:
        vals = obj.to_numpy()
    if vals.dtype.kind in "biuf":
        h.update(np.ascontiguousarray(vals.astype(float)).tobytes())
    else:  # object / string / categorical panels
        h.update("|".join(map(str, vals.ravel().tolist())).encode())


def hash_market_data(md: MarketData, n_chars: int = 16) -> str:
    """Short SHA-256 over **every** panel of ``md``.

    Covers each dataclass field that is a DataFrame / Series (``close``,
    ``volume``, ``market_cap``, ``industry`` and all optional panels, including
    fields added later), every entry of ``md.extra`` (sorted by key), and
    ``md.eligibility()`` (the PIT universe actually used).  Absent optional
    panels contribute a ``<name:None>`` marker so "missing" differs from "empty".
    Index and column labels are hashed too, so a re-labelled panel changes the
    hash.  Deterministic across processes for equal data.
    """
    h = hashlib.sha256()
    for f in fields(md):
        v = getattr(md, f.name)
        if f.name == "extra":
            continue
        if v is None:
            if f.name in _SKIP_IF_NONE:
                continue
            h.update(f"<{f.name}:None>".encode())
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            _hash_frame(h, f.name, v)
        else:
            h.update(f"<{f.name}:{v!r}>".encode())
    for k in sorted(md.extra):
        v: Any = md.extra[k]
        if isinstance(v, (pd.DataFrame, pd.Series)):
            _hash_frame(h, f"extra.{k}", v)
        else:
            h.update(f"<extra.{k}:{v!r}>".encode())
    _hash_frame(h, "eligibility()", md.eligibility())
    return h.hexdigest()[:n_chars]


# ----------------------------------------------------------------------------- environment
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_info(cwd: str | Path | None = None) -> dict[str, Any]:
    """``{"git_sha": str, "git_dirty": bool | None}``; sha ``"unknown"`` (dirty
    ``None``) when git or a repository is unavailable."""
    cwd = str(cwd or _repo_root())
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        st = subprocess.run(
            # untracked files count (a new module the run imports changes the code);
            # ``--untracked-files=normal`` respects .gitignore (runs/, data, caches)
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return {"git_sha": sha or "unknown", "git_dirty": bool(st.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": "unknown", "git_dirty": None}


def code_version(info: dict[str, Any] | None = None) -> str:
    """Code identity for the trial ledger: ``<sha>`` for a clean checkout,
    ``<sha>+dirty`` when ``git_dirty`` is True or unknown (``None``), and
    ``"unknown"`` outside a git checkout.  The ledger never de-duplicates a
    record whose code version is dirty or unknown."""
    g = git_info() if info is None else info
    sha = str(g.get("git_sha") or "unknown")
    if sha == "unknown":
        return "unknown"
    return sha if g.get("git_dirty") is False else f"{sha}+dirty"


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_LIBRARIES:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "not installed"
    return out


def new_run_id(config_hash: str, when: datetime | None = None) -> str:
    """``YYYYmmddTHHMMSSffffffZ_<config_hash[:8]>`` — sortable by creation time."""
    when = when or datetime.now(timezone.utc)
    return f"{when.strftime('%Y%m%dT%H%M%S%fZ')}_{config_hash[:8]}"


# ----------------------------------------------------------------------------- manifest
@dataclass
class RunManifest:
    run_id: str
    created_at: str  # ISO-8601 UTC
    config_hash: str
    data_hash: str
    git_sha: str = "unknown"
    git_dirty: bool | None = None
    versions: dict[str, str] = field(default_factory=lambda: cast(dict[str, str], {}))
    python_version: str = ""
    platform: str = ""
    timings: dict[str, float] = field(default_factory=lambda: cast(dict[str, float], {}))
    seeds: dict[str, int] = field(default_factory=lambda: cast(dict[str, int], {}))
    command: str = ""
    notes: str = ""
    manifest_version: int = MANIFEST_VERSION

    @classmethod
    def create(
        cls,
        config_hash: str,
        data_hash: str,
        run_id: str | None = None,
        timings: dict[str, float] | None = None,
        seeds: dict[str, int] | None = None,
        command: str | None = None,
        notes: str = "",
    ) -> RunManifest:
        now = datetime.now(timezone.utc)
        g = git_info()
        return cls(
            run_id=run_id or new_run_id(config_hash, now),
            created_at=now.isoformat(),
            config_hash=config_hash,
            data_hash=data_hash,
            git_sha=g["git_sha"],
            git_dirty=g["git_dirty"],
            versions=library_versions(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            timings=dict(timings or {}),
            seeds=dict(seeds or {}),
            command=command if command is not None else " ".join(sys.argv),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunManifest:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise KeyError(f"RunManifest: unknown key(s) {unknown}")
        return cls(**d)

    def save(self, run_dir: str | Path) -> Path:
        p = Path(run_dir)
        p.mkdir(parents=True, exist_ok=True)
        out = p / MANIFEST_FILE
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        os.replace(tmp, out)
        return out

    @classmethod
    def load(cls, run_dir: str | Path) -> RunManifest:
        p = Path(run_dir)
        if p.is_dir():
            p = p / MANIFEST_FILE
        return cls.from_dict(json.loads(p.read_text()))
