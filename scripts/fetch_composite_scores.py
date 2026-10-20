"""Download and parse FSA Financial Responsibility Composite Scores (studentaid.gov).

Each year's file is a .xls workbook with a two-row header (title row, then column
names) starting at row index 1 (0-indexed) and columns:
    Item#, OPEID, Institution Name, City, State, Zip Code,
    Institution Fiscal Year End, Institution Type,
    "Composite Score for Institution's Fiscal Year Ending Between ..."
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from config import COMPOSITE_RAW_DIR, COMPOSITE_SCORE_URLS, HTTP_HEADERS, YEARS

logger = logging.getLogger(__name__)


def download_one(year: int, force: bool = False) -> Path:
    url = COMPOSITE_SCORE_URLS[year]
    dest = COMPOSITE_RAW_DIR / f"composite_scores_{year}.xls"
    if dest.exists() and not force:
        logger.info("Composite scores %s already cached at %s", year, dest)
        return dest

    resp = requests.get(url, headers=HTTP_HEADERS, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("Downloaded composite scores %s -> %s (%d bytes)", year, dest, len(resp.content))
    time.sleep(0.3)
    return dest


def download_all(years=YEARS, force: bool = False) -> list:
    paths = []
    for year in years:
        if year not in COMPOSITE_SCORE_URLS:
            logger.warning("No composite score URL configured for year %s, skipping", year)
            continue
        paths.append(download_one(year, force=force))
    return paths


def _normalize(col) -> str:
    return " ".join(str(col).replace("\n", " ").lower().split())


def _find_column(columns, normalized_columns, predicate) -> str:
    """Return the original column name whose normalized form satisfies `predicate`.

    The studentaid.gov files use a different header layout almost every year (see
    NOTES.md: 'OPE ID' vs 'OPEID', 'Institutional Name' vs 'Institution Name',
    'Institution Fiscal Year' vs '... Year End', column order shuffled, an added/
    dropped 'School Group Name' column, etc.), so columns are matched by normalized
    name/prefix rather than by position or exact string.
    """
    for orig, norm in zip(columns, normalized_columns):
        if predicate(norm):
            return orig
    raise KeyError(f"No column matched among: {list(columns)}")


def parse_one(year: int) -> pd.DataFrame:
    path = COMPOSITE_RAW_DIR / f"composite_scores_{year}.xls"
    raw = pd.read_excel(path, header=1)
    raw.columns = [str(c).strip() for c in raw.columns]
    normalized = [_normalize(c) for c in raw.columns]

    opeid_col = _find_column(raw.columns, normalized, lambda n: n in ("opeid", "ope id"))
    name_col = _find_column(raw.columns, normalized, lambda n: "name" in n and "institution" in n)
    type_col = _find_column(raw.columns, normalized, lambda n: n == "institution type")
    fy_col = _find_column(raw.columns, normalized, lambda n: n.startswith("institution fiscal year"))
    score_col = _find_column(raw.columns, normalized, lambda n: n.startswith("composite score"))

    df = pd.DataFrame(
        {
            "opeid": raw[opeid_col].astype(str).str.strip().str.zfill(8),
            "institution_name": raw[name_col].astype(str).str.strip(),
            "institution_type": raw[type_col],
            "institution_fiscal_year_end": pd.to_datetime(raw[fy_col], errors="coerce"),
            "composite_score": pd.to_numeric(raw[score_col], errors="coerce"),
            "score_year": year,  # ending year of the AY bucket this file represents
        }
    )
    # Prefer the institution's own fiscal year end (more precise than the AY bucket)
    # for the fiscal_year merge key; fall back to the file's score_year when missing.
    df["fiscal_year"] = df["institution_fiscal_year_end"].dt.year.fillna(df["score_year"]).astype(int)
    return df


def parse_all(years=YEARS) -> pd.DataFrame:
    frames = []
    for year in years:
        if year not in COMPOSITE_SCORE_URLS:
            continue
        try:
            frames.append(parse_one(year))
        except Exception:
            logger.exception("Failed to parse composite scores for %s", year)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(COMPOSITE_RAW_DIR / "composite_scores_combined.csv", index=False)
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all()
    df = parse_all()
    print(df.shape)
    print(df.head())
