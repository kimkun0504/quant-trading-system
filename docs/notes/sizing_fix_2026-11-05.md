# Portfolio sizing fix (audit-2 finding 1) — evaluation record

Source: the risk/portfolio fix-wave report (2026-11-05). Synthetic market, 3 seeds (0, 1, 2),
last 400 out-of-sample days per seed, `signal_lag_days=1`, routed venue, default caps
(gross ≤ 2.0, 2 % per name, 5 % ADV).

**Baseline:** fixed γ = 10 + vol-target projection (with the new tracking-variance units) —
the closest current-code equivalent of the old default.
**New default:** `risk_aversion_mode="target_vol"` (γ searched per solve to hit the 10 % target).

| 3-seed mean | baseline: fixed γ=10 + projection | new: target_vol γ search |
|---|---|---|
| net Sharpe | 0.605 | 0.603 |
| gross Sharpe | 1.005 | 0.940 |
| realised vol | 7.5 % | 7.8 % |
| ex-ante vol | 7.8 % | 8.1 % |
| average gross | 2.00 | 1.77 |
| turnover / day | 0.118 | 0.115 |
| cost drag / year | 3.0 % | 2.7 % |
| transfer coefficient | 0.735 | 0.793 |
| realised / ex-ante vol | 0.97 | 0.92 |
| solves per date | 2.0 | 2.75 |

- Paired net-Sharpe differences (new − baseline) by seed: −0.24 / +0.22 / +0.02 — inside noise
  (SE of a 400-day Sharpe ≈ 0.8 per seed).
- Median searched γ by seed: 0.94 / 2.51 / 1.71 — consistent with the audit's closed-form
  estimate γ ≈ IR/(2σ_target) ≈ 2, which is the new fixed-mode default.
- Vol-target outcome: caps bind on 66 % / 100 % / 99.5 % of days; the 10 % target is not
  reachable under gross ≤ 2.0.
- Seed 0 has 29 % zero-alpha days (trailing IC ≤ 0 → expected return 0); the new mode holds
  the book on those days, while the old projection levered it back to target alpha-blind.
- Same-load solve time: N=300 0.53–0.56 vs 0.33–0.34 s/day; N=1000 0.65–0.68 vs 0.56–0.58 s/day.

Interpretation: the fix changes *why* the book has its size (alpha-aware objective instead of
an alpha-blind rescale), raises the transfer coefficient, and lowers cost drag; it does not
change net performance on synthetic data.
