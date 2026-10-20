# Financial Distress in Private Nonprofit 4-Year Colleges — Phase 1

Phase 1 of a research project predicting financial distress in U.S. private nonprofit
4-year colleges. This phase collects and merges the raw data; later phases will handle
feature engineering and modeling.

## Data sources

1. **FSA Financial Responsibility Composite Scores** [studentaid.gov/data-center/school/composite-scores](https://studentaid.gov/data-center/school/composite-scores).
   One `.xls` file per award year; each row is an institution's composite score
   (-1.0 to 3.0; <1.5 is not fully "financially responsible").
2. **IPEDS complete data files** [nces.ed.gov/ipeds/datacenter](https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx).
   HD (directory/characteristics), EF (fall enrollment), F2 (finance survey for
   private non-profit/FASB institutions), filtered to private nonprofit 4-year
   institutions (`SECTOR == 2`).
3. **Form 990 financials** — [ProPublica Nonprofit Explorer API](https://projects.propublica.org/nonprofits/api/), looked up by EIN, cached to disk.

## Scope

`scripts/config.py:YEARS` currently covers fiscal years **2014–2023** (the 10 most
recent years with both composite scores and IPEDS data available as of 2026-10-19).
Widen that list (and `COMPOSITE_SCORE_URLS`, which has entries back to 2009) to pull
more history.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Running Phase 1

Open `notebooks/01_data_collection_and_merging.ipynb` and run all cells, or run the
pipeline directly:

```bash
cd scripts
python fetch_composite_scores.py
python fetch_ipeds.py
python -c "
import pandas as pd
from build_panel import build_full_panel
from quality_report import print_report

ipeds = pd.read_csv('../data/raw/ipeds/ipeds_combined.csv', dtype={'opeid': str, 'ein': str})
composite = pd.read_csv('../data/raw/composite_scores/composite_scores_combined.csv', dtype={'opeid': str})
panel = build_full_panel(ipeds, composite)
panel.to_csv('../data/processed/panel.csv', index=False)
print_report(panel, ipeds, composite)
"
```

The Form 990 fetch step is rate-limited (see `NOTES.md`) and caches every response
under `data/raw/form990_cache/`, so re-running the pipeline after the first full run
is fast.

## Project layout

```
scripts/                   importable modules, one per pipeline stage
  config.py                 paths, year range, source URLs
  fetch_composite_scores.py
  fetch_ipeds.py
  fetch_form990.py
  build_panel.py
  quality_report.py
notebooks/
  01_data_collection_and_merging.ipynb
data/
  raw/                      downloaded source files + Form 990 cache (gitignored)
  processed/                merged panel.csv
NOTES.md                    running log of problems hit and fixes, for progress reports
```

## Known limitations (Phase 1)

See `NOTES.md` for the full log. Headline caveats:

- Composite-score "fiscal year" is approximated from each institution's own
  `Institution Fiscal Year End` date in the FSA file, not an independently audited value.
- `distressed_next_year` requires the *same* OPEID to have a posted score in year t+1;
  institutions that stop appearing in the FSA disclosure (closed, merged, left Title IV,
  or simply not yet posted) get a missing (not a 0) label for that row.
- Form 990 matching is by EIN + calendar year the filer's fiscal period ended in
  (`tax_prd_yr`), which can be offset by a few months from the IPEDS/FSA fiscal year
  definition for institutions with non-June fiscal year ends.
