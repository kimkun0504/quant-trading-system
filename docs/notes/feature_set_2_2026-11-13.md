# Feature set 2: pre-registration (2026-11-13)

## The experiment

One trial. Fixed on 2026-11-13, before any run with the new features.

| Item | Value |
|---|---|
| Baseline | real-data ML run, `alpha.horizon_days=60`, `alpha.sample_weight_halflife_years=8`: net Sharpe 0.68 |
| Change | `alpha.feature_set=v2` (the ML sleeve gets eight more features); nothing else |
| Trials | **one**: the command in "Commands" step 4, run once and logged in the trial ledger |
| Not allowed | dropping, adding or re-defining a feature after seeing the result; a second run with other windows, another event code or another horizon. Each would be a further trial of the same family and must be logged as one |

The eight features (definitions, references, signs, point-in-time rules: handbook 04
"Feature set 2"; code: `qts/alpha/features_v2.py`):

1. Earnings reaction: `ear_3d`, `sue`, `days_since_earnings`.
2. Abnormal volume: `abn_volume_5d`, `abn_volume_earn`.
3. Accruals and issuance: `accruals`, `net_issuance`.
4. `high_52w`.

Short interest is **not** in the set. FINRA history would cover only part of
2003–2026; deferred.

## Decisions taken while building (all before any v2 run)

These are the points the specification left open. They are fixed with the set.

| Point | Decision |
|---|---|
| Announcement date | every `EVENTS` row with code `22` (8-K item 2.02) is an announcement; a quarterly filing date is one only when no such event falls in the 45 days before it. Every item 2.02 event is kept, a pre-announcement included: choosing one per quarter would need the later filing date |
| Event code | `22` is an assumption (8-K item number without the dot), not checked against the vendor's documentation. `derive-earnings` prints the table that confirms or refutes it. Changing the code after the trial is a new trial; changing it before, on that table, is not |
| Visibility | announcement features from the close of the second trading day after the announcement day; `sue` and `accruals` from the first trading day on or after `datekey` + 1 calendar day |
| Expiry | `ear_3d`, `abn_volume_earn` and `sue` turn missing 126 trading days after the announcement (after the filing became visible, for `sue`); `accruals` is carried until the next filing, like the existing fundamentals |
| `sue` EPS | basic `eps` of the `ARQ` rows (already in every store), times `split_cum` (splits dated on or before the filing). Standard deviation: sample, over the trailing eight seasonal differences including the current one, at least four |
| `accruals` | literal definition, not sign-flipped: a negative IC is the published effect |
| `net_issuance` | `-ln(S_t / S_{t-252})` with `S = market_cap / adjusted close`. The adjusted close reinvests dividends, so this is Daniel–Titman composite issuance: a cash dividend counts as negative issuance |
| Windows | abnormal volume: base 252 days (at least 126), five-day mean at least 3 days, announcement mean at least 2 of 3 days; `high_52w` at least 126 days |
| Processing | seven features: winsorise → industry demean → rank-gauss → missing = 0. `days_since_earnings`: divided by 126 only; missing = 1.0. No missing-value indicators (the existing features have none) |
| Composite | unchanged: no theme reads the new features. Only the ML sleeve differs between `v1` and `v2` |

One known tension, stated here so the result is read with it: the build plan excluded
post-earnings drift on evidence (t = 1.43 outside microcaps, 2001–2024). `ear_3d` and
`sue` are that effect, included as inputs the models can condition on.

## What the owner's store needs

- `sharadar_sf1` with `ncfo`. SF1 was synced before `ncfo` was a kept field, so the
  stored table lacks it and `accruals` cannot be built. `derive-earnings` warns; the fix
  is step 2a below. A v2 run without it stops before any fit.
- `sharadar_events` (optional). Without it every announcement date is a filing date and
  `ear_3d` / `abn_volume_earn` measure the wrong three days for most quarters.
- `corporate_actions` (already there): split dates and ratios for `split_cum`.

## Commands

```bash
# 1
git pull

# 2  derive the earnings tables (no download); paste the output
qts data sharadar derive-earnings --store data/store

# 2a only if step 2 warns about ncfo and/or says EVENTS is not in the store
qts data sharadar sync --store data/store --tables SF1,EVENTS --from-files <bulk folder>
qts data sharadar derive-earnings --store data/store      # paste this output instead

# 3  coverage per year, no fit (minutes)
python scripts/diagnose_signal.py --store data/store --start 2000-01-01 --max-names 3000 \
    --min-mcap-ever 3e8 --set alpha.horizon_days=60 --set alpha.sample_weight_halflife_years=8 \
    --set alpha.feature_set=v2 --coverage-only

# 4  the single trial
qts run --store data/store --start 2000-01-01 --max-names 3000 --min-mcap-ever 3e8 --cache -v --ml \
    --set alpha.horizon_days=60 --set alpha.sample_weight_halflife_years=8 \
    --set alpha.feature_set=v2 2>&1 | tee runs/real_ml_h60_hl8_fs2.log
```

What to read in step 2 before going on:

- the code table: code `22` should show about four events per name-year and a high
  share dated in the 45 days before a quarterly filing. If another code does, rerun
  with `--codes <code>` and say so;
- the by-year table: the share of filings with an event (zero before the vendor's
  EVENTS history starts; 8-K item 2.02 exists since August 2004);
- the splits line: nearly all rows "as given". Many "as the inverse" or "neither" is a
  finding to report before the run;
- no `ncfo` warning.

What to read in step 3: per-year coverage of each of the eight features. Expected: high
for `abn_volume_5d`, `net_issuance`, `high_52w` after the first year of each name;
`sue` lower in a name's first three years (it needs twelve quarters); `accruals` lower
in the first year. A feature under 50 % over the sample is flagged.

## Stage cache

`feature_set` is part of the alpha-fit key only. This change edits `qts/core` and
`qts/data`, which are in the code hash of both stages, so the next run refits the risk
model and the alpha model once; step 4's risk fit is then reused by later runs.

## Not verified

- The `EVENTS` code and column name (`eventcodes`) on the real table.
- That `ACTIONS` split `value` is the ratio (2 for a 2-for-1). The fixture assumes it;
  the derive checks every row against the prices and uses the confirmed orientation.
- Nothing here was run on real data.
