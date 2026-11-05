"""A tiny but real run directory, for tests and the real-API end-to-end smoke.

``build_fixture_run(runs_root)`` generates a small synthetic market, runs the
research pipeline on it (rank composite, last ``max_days`` out-of-sample days)
with ``runs_root`` and a trial ledger set, so the run directory is written by the
pipeline's own writer (``save_run`` + ``write_extra_artifacts``).  About 20 s.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixtureSpec:
    n_stocks: int = 60
    n_days: int = 764
    seed: int = 5
    ic_true: float = 0.08
    max_days: int = 120
    signal_lag_days: int = 1


def build_fixture_run(runs_root: str | Path, spec: FixtureSpec | None = None) -> Path:
    from qts.core.config import SystemConfig
    from qts.data.synthetic import SyntheticSpec, generate
    from qts.pipeline import ResearchPipeline

    spec = spec or FixtureSpec()
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    md = generate(SyntheticSpec(n_stocks=spec.n_stocks, n_days=spec.n_days, seed=spec.seed, ic_true=spec.ic_true))
    cfg = SystemConfig().with_overrides(
        {
            "seed": spec.seed,
            "universe.top_n_by_mcap": min(SystemConfig().universe.top_n_by_mcap, spec.n_stocks),
            "execution.signal_lag_days": spec.signal_lag_days,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipe = ResearchPipeline(
            cfg,
            use_ml=False,
            ledger_path=str(root / "trial_ledger.jsonl"),
            nav0=5e6,
            runs_root=str(root),
        )
        res = pipe.run(md, max_days=spec.max_days)
    assert res.run_dir is not None
    return Path(res.run_dir)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="write a tiny real run directory")
    ap.add_argument("runs_root")
    a = ap.parse_args()
    print(build_fixture_run(a.runs_root))
