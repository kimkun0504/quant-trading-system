# D6: Grinold IC estimate blended with the prior; floor above zero

**Found:** 2026-11-09, first real-data run (Sharadar, top-1000 US names, 2000–2026). The backtest
log showed `gross=0.0000` and a flat NAV for years at a time. `scripts/diagnose_signal.py`
showed why: `IC_used = max(0, 0.5 × trailing 252-day OOS IC)` was **exactly zero on 49% of
dates** (2004–2008, 2011, 2014–2015, 2017, 2019–2020), because the composite's trailing one-year
IC was negative. A zero forecast for every name makes the optimizer unwind the whole book (any
risk is pure cost) and hold cash until the trailing IC turns positive. Even on the synthetic
market (planted, positive signal) IC_used was zero on 21% of sampled dates: a one-year trailing
IC of a weak signal is far too noisy to switch the book on and off.

**Real-data diagnosis (composite, equal theme weights, ~1,000 names/year):**

| | full-sample rank IC | note |
|---|---|---|
| combined | +0.0045 | very weak after costs (~10 bp/side) |
| quality | +0.0164 | positive every year but three |
| low risk | +0.0074 | |
| momentum | +0.0028 | |
| investment | −0.0014 | |
| value (B/P) | −0.0133 | negative 2004–2020, positive 2000–01, 2009, 2016, 2021–22, 2025–26 = HML's own history |

Industries (48 FF labels, 0.5% Other) and the universe (~1,000 eligible names/year) are fine. The
value sign is the market, not a data bug; the lesson is that equal theme weights fixed in 2000 are
a poor forecast, and that is what the walk-forward ML sleeve is for. Theme weights were **not**
hand-tuned on this table (that would be in-sample selection; the trial ledger counts runs).

**Change (`AlphaConfig`, `AlphaModel`):**

```
IC_hat  = (n · IC_trailing + k · default_ic) / (n + k)     k = ic_prior_obs (252), n := 0 below ic_min_obs
IC_used = max(ic_floor_frac · ic_shrink · default_ic, ic_shrink · IC_hat)   floor = 0.25 × 0.015 = 0.00375
ic_window: 252 → 756 (three years)
```

A negative trailing IC now shrinks positions toward a quarter of their prior size instead of
liquidating; a strong trailing IC is pulled toward the prior in proportion to how many realised
dates support it. With no realised IC the behaviour is unchanged (prior 0.03 shrunk to 0.015).

**Golden:** regenerated (the 60-day golden run moves slightly: net Sharpe 2.161 → 2.168, turnover
0.106 → 0.096; 32 of 70 fields). Every config hash changes (two new fields, one new default).
**Tests:** `tests/test_alpha.py::test_ic_used_blends_toward_prior_and_never_empties_the_book`,
updated `test_default_ic_is_shrunk_like_estimated_ic`.
**Docs:** handbook page 04 §8 (D6), config table; this note.
