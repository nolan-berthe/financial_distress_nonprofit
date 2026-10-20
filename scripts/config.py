"""Shared configuration: paths, year range, and source URLs for Phase 1 data collection."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
COMPOSITE_RAW_DIR = RAW_DIR / "composite_scores"
IPEDS_RAW_DIR = RAW_DIR / "ipeds"
FORM990_CACHE_DIR = RAW_DIR / "form990_cache"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

for _d in (COMPOSITE_RAW_DIR, IPEDS_RAW_DIR, FORM990_CACHE_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Phase 1 scope: ~10 most recent years with both FSA composite scores and IPEDS data
# available at the time of writing (2026-10-19). Year = the ENDING year of the
# award/fiscal year (e.g. AY 2013-2014 -> 2014). Widen YEARS to go further back;
# every other module keys off this one list.
YEARS = list(range(2014, 2024))  # 2014..2023 inclusive

# studentaid.gov blocks requests without a browser-like User-Agent (plain curl/requests
# UA hangs or gets refused) -- see NOTES.md.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Direct download links scraped from https://studentaid.gov/data-center/school/composite-scores
# (the page renders the list client-side; there is no stable per-year URL pattern, so this
# mapping was built by hand from the page's <a href> list rather than guessed).
COMPOSITE_SCORE_URLS = {
    2023: "https://studentaid.gov/sites/default/files/ay-22-23-composite-scores.xls",
    2022: "https://studentaid.gov/sites/default/files/ay-21-22-composite-scores.xls",
    2021: "https://studentaid.gov/sites/default/files/ay-20-21-composite-scores.xls",
    2020: "https://studentaid.gov/sites/default/files/ay-19-20-composite-scores.xls",
    2019: "https://studentaid.gov/sites/default/files/ay-18-19-composite-scores.xls",
    2018: "https://studentaid.gov/sites/default/files/ay17-18-composite-scores.xls",
    2017: "https://studentaid.gov/sites/default/files/ay16-17-composite-scores.xls",
    2016: "https://studentaid.gov/sites/default/files/fsawg/datacenter/library/20152016CompositeScores.xls",
    2015: "https://studentaid.gov/sites/default/files/fsawg/datacenter/library/20142015CompositeScores.xls",
    2014: "https://studentaid.gov/sites/default/files/20132014CompositeScores.xls",
    # Older years available if YEARS is widened:
    2013: "https://studentaid.gov/sites/default/files/1213CompositeScores.xls",
    2012: "https://studentaid.gov/sites/default/files/1112CompositeScores.xls",
    2011: "https://studentaid.gov/sites/default/files/1011CompositeScores.xls",
    2010: "https://studentaid.gov/sites/default/files/0910CompositeScores.xls",
    2009: "https://studentaid.gov/sites/default/files/0809CompositeScores.xls",
}

IPEDS_BASE = "https://nces.ed.gov/ipeds/datacenter/data"


def ipeds_hd_url(year: int) -> str:
    """Institutional characteristics (directory) file for the given IPEDS year."""
    return f"{IPEDS_BASE}/HD{year}.zip"


def ipeds_ef_url(year: int) -> str:
    """Fall enrollment, part A (headcount by level/race/sex) for the given IPEDS year."""
    return f"{IPEDS_BASE}/EF{year}A.zip"


def ipeds_f2_url(year: int) -> str:
    """Finance survey, private not-for-profit / FASB institutions, for the given IPEDS year.

    IPEDS names this file by the two-digit academic-year pair ending in `year`,
    e.g. year=2022 -> F2122_F2.zip (FY 2021-22).
    """
    yy_end = year % 100
    yy_start = (year - 1) % 100
    return f"{IPEDS_BASE}/F{yy_start:02d}{yy_end:02d}_F2.zip"


PROPUBLICA_ORG_URL = "https://projects.propublica.org/nonprofits/api/v2/organizations/{ein}.json"

# Be polite to ProPublica's API -- see NOTES.md for why this matters.
PROPUBLICA_RATE_LIMIT_SECONDS = 0.6

# IPEDS field selections kept for every year (verified against the FY2022 and FY2014
# data dictionaries -- see NOTES.md). F2 variable names have been stable since at
# least FY2014.
HD_FIELDS = ["UNITID", "INSTNM", "OPEID", "EIN", "STABBR", "SECTOR", "CONTROL", "ICLEVEL", "HBCU"]
PRIVATE_NONPROFIT_4YR_SECTOR = 2  # SECTOR code: "Private not-for-profit, 4-year or above"

F2_FIELDS = {
    "F2A02": "total_assets",
    "F2A03": "total_liabilities",
    "F2A06": "total_net_assets",
    "F2B01": "total_revenues",
    "F2B02": "total_expenses",
    "F2B07": "net_assets_end_of_year",
}

OAKWOOD_EIN = "630366652"
