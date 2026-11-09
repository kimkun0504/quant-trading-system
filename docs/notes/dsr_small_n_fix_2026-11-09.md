# DSR: expected-maximum Sharpe went negative for 1 < N_eff < 1.28

**Found:** 2026-11-09, by the owner's first local run. A fresh ledger (one trial, N = 1) gave
DSR 0.864 for the seed-0 composite demo; the stored report `runs/demo_audit2.txt` for the
same run said 0.958 and `[PASS] dsr`.

**Cause.** Bailey & López de Prado's closed form for E[max SR] of N noise trials,
`sqrt(V) * ((1-γ) Φ⁻¹(1-1/N) + γ Φ⁻¹(1-1/(Ne)))`, is a large-N (extreme-value) approximation.
Φ⁻¹(1-1/N) → -∞ as N → 1⁺, so the formula is negative for N below ≈ 1.28 (-0.79·sd at N = 1.01).
The stored run's ledger family held five near-identical re-runs: Nyholt N_eff = 1.0096 and
V = 0.251, giving E[max] = -0.40 annualised. A negative benchmark Sharpe inflated the PSR from
0.864 (benchmark 0) to 0.958 and flipped the gate to PASS. The true expected maximum of N
zero-mean draws is never negative and is exactly 0 at N = 1.

**Fix** (`qts/validation/deflated_sharpe.py::expected_max_sharpe`): for 1 < N < 2, interpolate
linearly between 0 at N = 1 and the closed form at N = 2 (0.52·sd; exact for two standard
normals is 0.564·sd). Non-negative, continuous, non-decreasing in N; unchanged for N ≥ 2.

| N_eff | before | after |
|---|---|---|
| 1.0 | 0 | 0 |
| 1.0096 | -0.791·sd | 0.005·sd |
| 1.2 | -0.117·sd | 0.104·sd |
| 1.5 | 0.216·sd | 0.260·sd |
| 2.0 | 0.520·sd | 0.520·sd |
| 2.4 | 0.679·sd | 0.679·sd |

Seed-0 composite demo: DSR 0.958 → 0.863 with the five-re-run ledger (0.864 with a fresh
ledger). The gate reading in `runs/demo_audit2.txt` was wrong; the overall verdict for that
run was already FAIL (decile t).

**Golden:** unchanged (the golden run has no ledger, N = 1).
**Test:** `tests/test_ledger_concurrency.py::test_expected_max_sharpe_never_negative_for_small_effective_n`.
**Docs:** handbook page 10 (code block, "Small effective N", pitfall; the audit hand-check at
N_eff = 1.667 now reads slightly lower).
