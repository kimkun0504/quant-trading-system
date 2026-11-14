"""Researcher tools on top of the pipeline (2026-11-13; docs/notes/research_tools_2026-11-13.md).

Library entry points (each returns a dataclass with ``to_dict()`` giving a
JSON-serialisable dict, so the API can call them directly; the CLI in
:mod:`qts.research.cli` is a thin layer over them):

* :mod:`qts.research.data`     -- ``StoreSpec``, ``load_store_market`` (the one store
  loader: prefilter, holdout seal, full-history market identity);
* :mod:`qts.research.seal`     -- the repo-level holdout seal (``load_seal``,
  ``is_sealed``, unseal bookkeeping, ``seal_status``);
* :mod:`qts.research.midcap`   -- the pre-registered mid-cap confirmation
  (``run_midcap_check``, ``prefilter_coverage``, ``evaluate_midcap``);
* :mod:`qts.research.signals`  -- single-feature tests (``run_signal_test``,
  ``list_signals``);
* :mod:`qts.research.sweep`    -- parameter sweeps with trial accounting and PBO
  (``plan_sweep``, ``run_sweep``, ``run_sweep_member``, ``finalize_sweep``);
* :mod:`qts.research.capacity` -- capacity curve (``run_capacity``);
* :mod:`qts.research.risk_calibration` -- realised / ex-ante vol by year for a grid of
  ``risk.alignment_nu_scale`` (``run_risk_calibration``).

Importing this package imports nothing heavy; import the submodule you need.
"""
