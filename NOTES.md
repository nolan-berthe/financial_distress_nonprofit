# Phase 1 Notes — Problems Hit & Fixes

Running log for the progress report. Newest entries at the top.

## Phase 4

See `notebooks/04_explanations_and_fairness.ipynb` and
`reports/phase4_results.txt` for the full run; this is the log of problems
hit and fixes.

### Setup: `shap` installed cleanly on Python 3.14

Same situation as Phase 3's `xgboost` install — `pip install shap` resolved
`shap==0.54.1` (a `cp312-abi3` wheel, so it works unmodified on 3.14 via the
stable ABI) with no pin or workaround needed. Added `shap>=0.44` to
`requirements.txt`.

### Problem: `shap.TreeExplainer` returns a 3D array for `RandomForestClassifier`, not 2D

`shap.TreeExplainer(rf)(X)` on the random forest returns an `Explanation`
shaped `(n_rows, n_features, n_classes)` — one full attribution per class —
whereas the same call on the XGBoost model returns `(n_rows, n_features)`
directly, because XGBoost's binary classifier has a single raw (margin)
output while scikit-learn's `RandomForestClassifier` reports both class
probabilities. Indexed down to the distressed class only (`rf_shap[:, :,
1]`) before doing anything else with it (beeswarm plot, mean-|SHAP| ranking) —
not a bug, just an API difference worth documenting since passing the raw
3D object into a plot produces a confusing error otherwise.

### Problem: `shap.LinearExplainer`/`TreeExplainer` output scales differ by model, not just by magnitude

Logistic regression's `LinearExplainer` and XGBoost's `TreeExplainer` both
explain the **log-odds** (pre-sigmoid) output by default, but a random
forest's trees output class probabilities directly (no link function), so
its `TreeExplainer` values are already in **probability** units. Comparing
raw SHAP magnitudes across all three models would silently compare
apples to oranges. **Not treated as a bug to fix** (converting one model's
scale to match another's isn't well-defined in general) — instead, Part 1's
cross-model comparison only uses *direction* (sign of corr(feature,
SHAP)) and *rank* (order by mean |SHAP| within each model's own scale),
both of which are invariant to a monotonic rescaling, and the notebook/report
call out the scale difference explicitly rather than plotting all three on
one shared axis.

### Problem: SHAP's own waterfall/beeswarm plots show the *standardized* feature values for logistic regression

`shap.LinearExplainer` necessarily operates on the same standardized inputs
`fit_logistic`'s pipeline trains on, so by default its `Explanation.data`
holds z-scored values (e.g. "-0.53" instead of Oakwood's actual
`tuition_dependence` of 0.324) — unreadable in a waterfall plot meant for a
human case-study reader.

**Fix:** after computing the `Explanation` object, overwrite its `.data`
attribute with the raw (clipped+imputed, un-standardized) feature values
pulled from the same rows, before plotting — shape-compatible and purely
cosmetic (the SHAP `.values`/`.base_values` driving the plot's bars are
untouched, only the axis labels change). Confirmed this doesn't silently
break anything by checking `shap.plots.waterfall` renders and saves
normally with the substituted `.data`.

### Design decisions (not bugs, but worth recording)

- **Oakwood's FY2022 row has no train/test split membership** (its
  `distressed_next_year` label is NaN — see Phase 1/2 notes on Oakwood's
  missing FY2023 composite score, which is what FY2022's forward label would
  need) — so it is absent from `labeled`/`test_df` entirely. The Oakwood case
  study pulls features for FY2020-2023 straight from the full `feat` frame
  (applying the official train-fit clip/impute bounds), not from `test_df`,
  specifically so FY2022 and FY2023 are still included despite having no
  label.
- **Fairness bootstrap**: resampling is done **within each group separately**
  (HBCU rows resampled from HBCU rows only, same for "other"), not from the
  pooled test set, since the question is the uncertainty of each group's own
  rate, not of the test set as a whole. A resample that happens to draw zero
  actual positives (possible for the 128-row HBCU group, which has only 9
  truly distressed rows to begin with) makes that resample's recall
  undefined; those draws are excluded from that metric's CI rather than
  counted as 0 (`fairness.py:_percentile_ci`), and the count of valid draws
  per metric is computed so a CI built from suspiciously few valid draws
  would be visible (in this run, all 2,000 draws were valid for every metric
  in both groups).
- **COVID-period diagnostic refit** (Part 4) intentionally re-fits its own
  clip bounds and median imputer on the 2020-2022 rows rather than reusing
  the official train-fit (2014-2019) ones — the question being asked is
  "what does the feature-distress relationship look like *using only
  pandemic-era data*," which requires pandemic-era preprocessing too; reusing
  the 2014-2019 bounds would have been a reasonable alternative design but
  answers a subtly different question ("how do pandemic-era rows score under
  pre-pandemic assumptions about typical/extreme values"). This refit is
  explicitly diagnostic only — it is never evaluated on held-out data or
  treated as a candidate model.
- **Headline findings** (full numbers in `reports/phase4_results.txt`):
  SHAP direction agrees across all three models for 6/7 features;
  `contribution_share` is the one exception (logistic regression says higher
  contribution_share lowers risk, both tree models say the opposite — likely
  a weak/noisy feature either way, given it ranks 4th-5th of 7 in importance
  for all three). The final model's recall degrades noticeably across the
  test window (0.708 in FY2020 to 0.412 in FY2022) even though ROC-AUC stays
  high throughout, consistent with the actual distress rate itself dipping
  during 2020-2021 (COVID relief funding plausibly masking distress
  temporarily) before climbing back up in 2022. Oakwood's FY2021->FY2022 risk
  jump is driven almost entirely by `leverage` (SHAP contribution -0.60 ->
  +0.19) and `operating_margin` turning negative (-0.18 -> +0.46), not by
  revenue or enrollment changes.

## Phase 3

See `notebooks/03_model_comparison.ipynb` and `reports/phase3_results.txt`
for the full run; this is the log of problems hit and fixes.

### Setup: `xgboost` was not installed and the project targets a very new Python

The dev environment is Python 3.14.3 (pre-release-adjacent, very recent). `pip
install xgboost` resolved and installed cleanly (`xgboost==3.4.1`, a pure
`py3-none-win_amd64` wheel with no C-extension/ABI concerns), so no version
pin or workaround was needed beyond adding `xgboost>=2.0` to
`requirements.txt`.

### Problem: median-imputing a CV fold whose training window is shorter than a feature's own lag

`features.fit_median_imputer`/`fit_clip_bounds` are refit **per rolling-CV
fold** on that fold's own training years only (so no fold's validation year
leaks into its own preprocessing — the same reasoning as Phase 2's train/test
split, just applied at the fold level too). The first CV fold trains on
fiscal years 2014-2016 only. `enrollment_change_3yr` needs an enrollment
value from 3 years earlier for the *same institution*, and the panel itself
starts at fiscal year 2014 — so for every row in a 2014-2016-only training
window, `enrollment_change_3yr` is NaN (no institution has data from
2011-2013 in this panel). The column's **median over an all-NaN fold is
itself NaN**, so `fillna(NaN)` left `NaN` in the feature and
`LogisticRegression.fit()` raised `ValueError: Input X contains NaN`
(scikit-learn's tree models tolerated it silently — see note below).

**Fix:** `fit_median_imputer` now falls back to `0.0` for any feature whose
training-fold median itself comes back NaN (a neutral "no known change" value
when there simply isn't enough history yet) instead of propagating NaN.
`fit_clip_bounds` needed no equivalent fix — pandas' `.clip(lower=nan,
upper=nan)` is already a no-op per side, confirmed directly
(`pd.Series([...]).clip(np.nan, np.nan)` returns the series unchanged).

### Checked, not a bug: scikit-learn's tree ensembles accept NaN silently

Before finding the fix above, `RandomForestClassifier.fit()` on the same
NaN-containing fold did **not** raise — scikit-learn 1.9 added native missing-
value support to its tree-based estimators (NaN rows get routed to whichever
child minimizes impurity), so the bug was masked for random forest and would
have stayed invisible until the logistic-regression fold hit the same data
and raised explicitly. Fixed the root cause (the imputer) rather than relying
on this behavior, since XGBoost's own native NaN handling and scikit-learn's
are not guaranteed to agree, and Phase 3 explicitly wants identical
preprocessing across all three models for a fair comparison.

### Design decisions (not bugs, but worth recording)

- **Rolling-origin CV folds**: expanding-window, 3 folds within the training
  years (2014-2019) only — (train 2014-2016 → val 2017), (train 2014-2017 →
  val 2018), (train 2014-2018 → val 2019). `min_train_years=3` was chosen so
  every fold's training window is at least as long as the longest lag feature
  (`enrollment_change_3yr`); a shorter minimum would have hit the NaN-median
  problem above even after the fallback-to-0.0 fix, just with more of the
  column silently zeroed out instead of genuinely imputed.
- **Model selection metric**: hyperparameters are chosen by mean validation
  **PR-AUC** across folds (threshold-independent, appropriate for a ~5-7%
  positive class) rather than accuracy or ROC-AUC, which can look good on
  this data even when precision at any usable recall is poor.
- **Thresholding**: each model's cutoff is the threshold that maximizes F1 on
  that model's **pooled out-of-fold validation predictions** (refit per fold
  with the already-chosen best hyperparameters, predict on that fold's held-
  out year, concatenate across folds) — never on the test set. Phase 2's
  logistic baseline implicitly used scikit-learn's default 0.5 cutoff, which
  had never actually been tuned; Phase 3 reruns that same model through the
  same thresholding procedure as the two new models so all three are
  compared on equal footing.
- **XGBoost class imbalance**: `scale_pos_weight` (negatives/positives in the
  training rows) is computed **fresh inside each fit call** — per CV fold
  during tuning, then again on the full training set for the final model —
  rather than tuned as a grid parameter, so it never leaks information from
  outside that call's own training rows.
- **Result**: logistic regression, not either tree ensemble, wins on test-set
  PR-AUC (0.298 vs. 0.284 random forest / 0.272 XGBoost) and has by far the
  best recall (0.519 vs. 0.393/0.383) on the distressed class. Plausible
  reason: with only 7 engineered features and ~650 positive training rows,
  there's limited room for the trees to learn genuine interactions beyond
  what the already-informative ratio features (`log_total_revenue`,
  `leverage`) capture linearly. Full comparison table and confusion matrices
  in `reports/phase3_results.txt`.

## Phase 2

See `notebooks/02_features_and_baseline.ipynb` and `reports/phase2_results.txt`
for the full run; this is the log of problems hit and fixes.

### Fix: Phase 1's NOTES.md had the wrong Oakwood missing-years claim

Phase 1's note below ("Oakwood spot-check") said Oakwood had no composite
score for fiscal years 2018 and 2022. Re-checking the panel directly showed
those two years *do* have a score (2.7 and 2.2) — the actual gap is **2019
and 2023**. Confirmed by grepping the raw `composite_scores_2019.xls` /
`composite_scores_2023.xls` files for Oakwood's OPEID (00103300): it is not
present in either file at all (a text search for "Oakwood" only hits an
unrelated institution, "Appalachian College of Pharmacy," in the town of
Oakwood, VA). This is a genuine gap in the FSA source data for those two
years, not a `fetch_composite_scores.py` parsing bug — no fix made.

### Checked, not a bug: Form 990 `tax_prd_yr` fiscal-year alignment

Worried that ProPublica's `tax_prd_yr` might label a filing by the year its
fiscal period *starts* rather than *ends*, which would make every Form 990
field in the panel off by one year relative to `fiscal_year`. Checked two
ways:
1. Oakwood's own raw filings: `tax_prd` is `YYYYMM` (e.g. `201406` = fiscal
   year ending June 2014), and `tax_prd_yr` always equals `tax_prd`'s own
   ending year (`tax_prd_yr=2014` for `tax_prd=201406`) for every one of its
   filings — i.e. already an ending-year label, same convention as `fiscal_year`.
2. Panel-wide: compared Form 990 `f990_total_assets_end` against IPEDS
   `total_assets` for the same institution at shift=-1/0/+1 year. The current
   (shift=0) match agrees within 5% for 79.3% of rows, vs. 45.5% (shift=-1)
   and 46.0% (shift=+1) — shift=0 is clearly correct, not shift=-1 or +1.

No fix needed; `build_panel.py`'s existing `tax_prd_yr == ipeds_year` match is correct.

### Profile: IPEDS institutions that never match a composite score

230 of 1,901 (12.1%) private-nonprofit-4yr institutions in the panel never
get a composite score in any year. Two overlapping causes:
- **Size/geography**: never-matched institutions are much smaller (median
  enrollment 293 vs. 1,096 for matched institutions) and concentrated in CA,
  FL, NY, and Puerto Rico — consistent with smaller schools being below some
  FSA reporting threshold, exempt, or simply not posted in this window.
- **OPEID is shared by multiple IPEDS UNITIDs (36 institutions)**: the FSA
  composite score is reported once per OPEID, but IPEDS tracks each branch
  campus of a multi-campus corporate entity (e.g. "Trine University-Regional/
  Non-Traditional Campuses", the various "EDP University of Puerto Rico-*"
  branches) under its own UNITID sharing the parent's OPEID. For these, a
  sibling UNITID under the same OPEID *does* get the score — only one branch
  can ever match by construction of the opeid+year join. Not a bug to fix
  (the composite score genuinely applies at the OPEID/corporate-entity level),
  just a real limitation worth knowing before interpreting per-branch-campus
  missingness as a data problem.

### New IPEDS F2 fields: tuition/fees revenue and gifts/contributions

Verified variable names against the FY2013-14 (`F1314_F2_Dict.zip`) and
FY2021-22 (`F2122_F2_Dict.zip`) data dictionaries from nces.ed.gov (`varlist`
+ `Description` sheets), not guessed:
- `F2D01` — "Tuition and fees - Total", described as "net of any allowances
  applied in the general purpose financial statements" → `tuition_fees_revenue`.
- `F2D08` — "Private gifts, grants, and contracts - Total" → `contribution_revenue`.
  Note this is IPEDS's standard published revenue-by-source category and is
  slightly broader than pure gifts (also includes private grants/contracts);
  a narrower `F2D08A` ("Private gifts - Total") also exists if a purer gifts-only
  figure is ever needed.

Both confirmed present in every year's raw F2 CSV, 2014-2023. Added to
`config.py:F2_FIELDS` and the panel rebuilt (`fetch_ipeds.py`, `build_panel.py`
required no changes — field selection is driven entirely by `F2_FIELDS`).

### Feature construction and missing-data handling

Seven features built in `scripts/features.py` for year *t* using only that
year's own filings (IPEDS primary, Form 990 fallback for revenue/expenses/
assets/liabilities — no 990 fallback exists for tuition/gifts since the 990
doesn't break revenue into those categories). Missingness ranges from 1.7%
(`operating_margin`) to 34.4% (`enrollment_change_3yr`, since it needs an
enrollment value from 3 years earlier for the same institution). Rule: after
combining sources, each feature is winsorized at the 1st/99th percentile (a
handful of ratio features have multi-hundred-percent outliers from a near-zero
denominator in a given year) and median-imputed — both the clip bounds and
the median are fit on the **training split only** and then applied to test,
so no information about the held-out years leaks into preprocessing.

### Baseline model results (fiscal years 2014-2019 train, 2020-2022 test)

Standardized logistic regression, `class_weight="balanced"`: ROC-AUC 0.848,
PR-AUC 0.298, precision/recall on the distressed class 0.173/0.706 — far above
the naive "always predict not distressed" baseline (0.000/0.000 precision/
recall, 94.7% accuracy only because the test set is 94.7% not-distressed).
Largest-magnitude standardized coefficients: `log_total_revenue` (-1.13,
larger institutions less likely to be distressed), `leverage` (+0.99, more
debt relative to assets more likely), `tuition_dependence` (-0.48). Full
numbers in `reports/phase2_results.txt`.

## Scope decision

Targeting fiscal years **2014–2023** (10 years) rather than full history
(FSA composite scores go back to 2006–2007; IPEDS F2 further still). Full
history is a much larger download/parse job (dozens of IPEDS zips, 17 composite
files) — see `scripts/config.py:YEARS` to widen it; `COMPOSITE_SCORE_URLS`
already has entries back to 2009.

## Problem: studentaid.gov composite-score page has no stable per-year URL pattern

The download links on https://studentaid.gov/data-center/school/composite-scores
don't follow one naming convention — they're a mix of styles accumulated over
years of site redesigns, e.g.:
- `ay-22-23-composite-scores.xls` (2022–2023)
- `20152016CompositeScores.xls` (2015–2016)
- `1213CompositeScores.xls` (2012–2013)

**Fix:** scraped the actual `<a href>` list from the live page (via browser JS)
rather than guessing a pattern, and hardcoded the year → URL mapping in
`scripts/config.py:COMPOSITE_SCORE_URLS`. If studentaid.gov reorganizes the page
again, this mapping will need to be re-scraped.

## Problem: plain `requests`/`curl` calls to studentaid.gov hang or fail silently

A bare GET (no User-Agent header) to a `studentaid.gov` file URL either hung
past a 60s timeout or returned an error, but the same request with a normal
browser `User-Agent` succeeded immediately.

**Fix:** `scripts/config.py:HTTP_HEADERS` sets a Chrome-like User-Agent, and
`fetch_composite_scores.py` sends it on every request. (nces.ed.gov and the
ProPublica API didn't need this.)

## Problem: composite-score spreadsheet layout is not consistent across years

Checked the header row (row index 1, after a title row) across all 10 files and
found real differences, not just cosmetic ones:

| Year | OPEID column | Name column | FY column | Extra column |
|---|---|---|---|---|
| 2014–2015 | `OPE ID` | `Institutional Name` / `Institution Name` | `Institution Fiscal Year` (no "End") | — |
| 2016–2019 | `OPE ID` | `Institution Name` | `Institution Fiscal Year End` | `School Group Name` |
| 2020–2021 | `OPEID` | `Institution Name` | `Institution Fiscal Year End` | `School Group Name` (different column order) |
| 2022–2023 | `OPEID` | `Institution Name` | `Institution Fiscal Year End` | — |

A first pass that hardcoded column names (`raw["OPEID"]`, `raw["Institution Name"]`)
worked for 2020–2023 and 2016–2017 but threw `KeyError: 'OPEID'` on 2014, 2015,
2018, 2019 (2018/2019 actually broke for a different, non-obvious reason — the
columns render fine but something upstream of the KeyError pointed at a stale
cached read; re-checking row-by-row headers was what surfaced the 2014/2015 vs.
2016+ naming split).

**Fix:** rewrote `parse_one()` in `fetch_composite_scores.py` to match columns by
a normalized name (lowercase, collapse whitespace/newlines) and a predicate
(e.g. "contains 'name' and 'institution'") instead of an exact string or fixed
position. See `_find_column()` / `_normalize()`.

## Decision: fiscal_year is derived per-institution, not per-file

Each composite-score file covers one award year (e.g. "2022–2023"), but it also
reports each institution's own `Institution Fiscal Year End` date, which can fall
anywhere in that window (most are 6/30, but not all). Rather than labeling every
row in the 2022–2023 file as fiscal_year=2023, `fiscal_year` is set to the
calendar year of that institution's own fiscal-year-end date (falling back to the
file's award-year-end when the date is missing). This gives a more precise
per-institution year key to merge against IPEDS, at the cost of occasionally
pulling a row's "true" year across a file boundary (e.g. a 2013 fiscal-year-end
row that nonetheless appears in the "2014" file alongside a wave of 2014
year-end institutions).

## IPEDS field verification (done before writing the fetch code, not guessed)

Downloaded one sample year of each component (HD2022, EF2022A, F2122_F2) plus
their data dictionaries and confirmed:

- `HD{year}.zip` directly contains `EIN` and `OPEID` (8-digit, matches the
  composite-score file's OPEID format) alongside `UNITID`, `STABBR`, `SECTOR`,
  `CONTROL`, `ICLEVEL`, `HBCU`. `SECTOR == 2` is "Private not-for-profit, 4-year
  or above" — used as the filter instead of combining `CONTROL`+`ICLEVEL`.
- `EF{year}A.zip` is long-format (one row per `UNITID` × `EFALEVEL` demographic
  breakdown). Confirmed via the EF data dictionary's `Frequencies` sheet that
  `EFALEVEL == 1` is "All students total" and `EFTOTLT` is the total headcount.
- `F{yy}{yy+1}_F2.zip` (private nonprofit/FASB finance survey) variable names —
  `F2A02` (total assets), `F2A03` (total liabilities), `F2A06` (total net
  assets), `F2B01` (total revenues and investment return), `F2B02` (total
  expenses), `F2B07` (net assets, end of year) — were checked against both the
  FY2022 and FY2014 data dictionaries and are stable across that span.

## ProPublica Nonprofit Explorer API

Confirmed the v2 endpoint is `GET /nonprofits/api/v2/organizations/{ein}.json`
(EIN as a 9-digit number, no dash) by querying it live for Oakwood University's
EIN (630366652) before writing `fetch_form990.py`. One call returns *all* of an
organization's filings (`filings_with_data`), each with a `tax_prd_yr` — matched
against the panel's `fiscal_year` to pick the right filing per row.

**Politeness:** `fetch_form990.py` caches every response to
`data/raw/form990_cache/{ein}.json` permanently (a past fiscal year's 990 never
changes) and sleeps `PROPUBLICA_RATE_LIMIT_SECONDS` (0.6s) between *live*
requests only — a re-run against an already-cached EIN costs nothing.

## Problem: IPEDS file encoding is not consistent across release years

`fetch_ipeds.py` read every component with a hardcoded `encoding="latin-1"`,
which worked for FY2014–FY2022 but threw `KeyError: 'UNITID'` on `HD2023.csv`.
Tracing it down: the file opened fine under latin-1, but its first column name
came back as `'﻿UNITID'`, not `'UNITID'` — the raw bytes at the start of
the file are `EF BB BF` (a UTF-8 byte-order-mark), which HD2023's provisional
release includes and earlier years' files don't.

**Fix:** `fetch_component()` now tries `utf-8-sig` first (strips a BOM if
present, otherwise behaves like plain utf-8) and falls back to `latin-1` only
on a `UnicodeDecodeError`.

## Problem: `jupyter`/`nbconvert` not on PATH after pip install

`pip install jupyter` put `jupyter-nbconvert.exe` etc. in a user Scripts
directory that isn't on `PATH` on this machine, so both `jupyter nbconvert ...`
and the bare module path (`pip install nbformat` alone, no `nbconvert` package)
failed. **Fix:** installed `nbconvert` and `ipykernel` explicitly, registered a
kernel (`python3 -m ipykernel install --user --name python3`), and invoked
execution as `python3 -m nbconvert --to notebook --execute --inplace
--ExecutePreprocessor.timeout=3000 --ExecutePreprocessor.kernel_name=python3 ...`
instead of relying on the `jupyter` console entry point.

## Phase 1 run results (2014–2023, 10 years)

Full pipeline executed end-to-end via the notebook. Headline numbers from the
data quality report:

- **16,813** institution-year rows across **1,901** distinct institutions (UNITID), 10 fiscal years (2014–2023).
- **86.9%** of panel rows matched to a composite score for that institution+year; of **1,874** distinct OPEIDs in the IPEDS private-nonprofit-4yr universe, **1,658** (88.5%) appear at least once in the composite-score files.
- **83.6%** of rows with a non-null EIN matched a Form 990 filing for that exact fiscal year via the ProPublica API (1,789 unique EINs fetched, all cached under `data/raw/form990_cache/`).
- **77.5%** of rows have a usable `distressed_next_year` label (a score observed for that institution in year t+1); among those, **6.7%** are labeled distressed (next-year score < 1.5) — a realistic class imbalance to design around in later phases.
- Missingness is concentrated exactly where expected: `composite_score`/`next_year_score` (institutions not matched or not posted that year), the Form 990 fields (EINs with no 990 filed for that year — common for very small institutions below the IRS e-filing threshold, or filing lag), and the F2 finance fields (institutions that didn't submit a FASB finance survey that year).

**Oakwood University (EIN 63-0366652) spot-check:** panel values line up exactly
with its own raw Form 990 filings pulled independently in the same notebook —
e.g. FY2014 `total_assets` = 90,161,674 and `total_liabilities` = 20,654,626 in
the IPEDS-derived panel match `totassetsend`/`totliabend` for `tax_prd_yr=2014`
in the ProPublica record precisely. Oakwood's composite score trended down
from 3.0 (FY2014) to 2.2 (FY2022) but stayed well clear of the 1.5 threshold
throughout, so every `distressed_next_year` label for it is 0 (not 1 — none of
its observed years dip below 1.5). Two fiscal years (2018-end, 2022-end per the
file's own `Institution Fiscal Year End` date) have no matching composite-score
row at all even though adjacent years do — worth spot-checking against the raw
.xls files directly in Phase 2 rather than assuming it's a pipeline bug.
