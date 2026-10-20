# Phase 1 Notes — Problems Hit & Fixes

Running log for the progress report. Newest entries at the top.

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
