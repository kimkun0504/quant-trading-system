"""Trial-ledger record kinds, reference re-runs and extra fields (2026-11-13):
only ordinary trials enter the DSR inputs; a non-trial record never replaces a trial
in the de-duplicated view; ``normalized_config_hash`` survives a new config field."""

from __future__ import annotations

import pandas as pd
import pytest

from qts.core.config import SystemConfig
from qts.runs.context import RunContext
from qts.validation import TrialLedger
from qts.validation.ledger import is_trial_record, normalized_config_hash, record_kind


def _log(led, sr, fam="F", **kw):
    months = pd.date_range("2020-01-31", periods=24, freq="ME")
    rm = pd.Series([0.01 * ((i * 7 + int(sr * 10)) % 5 - 2) for i in range(24)], index=months)
    return led.log(
        name=fam,
        family=fam,
        config={"x": sr},
        metrics={"sharpe_annual": sr, "n_obs": 500},
        data_hash="d" * 16,
        code_version="abc123",
        returns_monthly=rm,
        replace_identical=True,
        run_spec={"k": 1},
        **kw,
    )


def test_non_trials_and_references_are_logged_but_not_counted(tmp_path):
    led = TrialLedger(str(tmp_path / "l.jsonl"))
    _log(led, 0.5)
    _log(led, 0.7, config_hash=None)
    base = led.dsr_inputs("F", 1000)
    _log(led, 3.0, extra={"reference": True})
    _log(led, 2.0, extra={"kind": "capacity"})
    led.log(name="F", family="F", config={}, metrics={}, extra={"kind": "unseal", "unseal_reason": "x"})
    assert led.n_trials("F") == 2
    assert led.dsr_inputs("F", 1000) == base
    assert len(led.records("F")) == 2 and len(led.records("F", include_non_trials=True)) == 5
    assert led.trial_sharpes_annual("F").max() == pytest.approx(0.7)
    row = next(r for r in led.families() if r["family"] == "F")
    assert row["n_trials"] == 2 and row["n_not_counted"] == 3 and row["n_records"] == 5
    assert row["best_sharpe"] == pytest.approx(0.7)
    kinds = [record_kind(r) for r in led.raw_records()]
    assert kinds == ["trial", "trial", "trial", "capacity", "unseal"]
    assert [is_trial_record(r) for r in led.raw_records()] == [True, True, False, False, False]


def test_a_reference_rerun_never_hides_the_trial_it_repeats(tmp_path):
    led = TrialLedger(str(tmp_path / "l.jsonl"))
    a = _log(led, 0.5)
    b = _log(led, 0.5, extra={"reference": True})  # same name / config / data / code / spec
    assert a["dedup_key"] != b["dedup_key"]
    assert led.n_trials("F") == 1
    assert led.trial_sharpes_annual("F").iloc[0] == pytest.approx(0.5)
    c = _log(led, 0.5)  # an identical re-run of the trial still collapses
    assert c["dedup_key"] == a["dedup_key"] and led.n_trials("F") == 1


def test_extra_fields_cannot_overwrite_standard_ones(tmp_path):
    led = TrialLedger(str(tmp_path / "l.jsonl"))
    with pytest.raises(ValueError, match="collide"):
        led.log(name="F", config={}, metrics={}, extra={"family": "other"})
    rec = led.log(name="F", config={}, metrics={}, extra={"sealed_end": "2023-11-08", "kind": "trial"})
    assert rec["sealed_end"] == "2023-11-08"


def test_normalized_config_hash_is_stable_when_a_field_is_added():
    cfg = SystemConfig().with_overrides({"alpha.horizon_days": 60})
    d = cfg.to_dict()
    d["universe"].pop("rank_offset")
    d["alpha"].pop("disabled_features")  # a record written before both fields existed
    assert normalized_config_hash({"config": d}) == cfg.hash()
    assert normalized_config_hash({"config": {"universe": {"no_such": 1}}}) is None
    assert normalized_config_hash({}) is None


def test_run_context_fields():
    assert RunContext().ledger_fields() == {}
    assert RunContext().counts_as_trial
    ctx = RunContext(
        holdout_start="2023-11-09", sealed_end="2023-11-08", reference=True, market_hash_full="m" * 16
    )
    f = ctx.ledger_fields()
    assert f["reference"] is True and f["sealed_end"] == "2023-11-08" and f["unsealed"] is False
    assert not ctx.counts_as_trial
    s = ctx.summary_fields()
    assert s["sealed_end"] == "2023-11-08" and s["holdout_already_used"] is False and s["reference"] is True
    assert any("reference" in ln for ln in ctx.banner())
    with pytest.raises(ValueError, match="record_kind"):
        RunContext(record_kind="bogus")
    used = RunContext(holdout_start="2023-11-09", unsealed=True, holdout_already_used=True, unseal_reason="r")
    assert "HOLDOUT ALREADY USED" in used.banner()[0]
