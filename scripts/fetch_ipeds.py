"""Download and parse IPEDS complete data files: HD (directory), EF (fall enrollment),
and F2 (private non-profit / FASB finance survey), for the configured YEARS, and build
a single private-nonprofit-4-year institution-year panel.
"""

import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from config import (
    F2_FIELDS,
    HD_FIELDS,
    IPEDS_RAW_DIR,
    PRIVATE_NONPROFIT_4YR_SECTOR,
    YEARS,
    ipeds_ef_url,
    ipeds_f2_url,
    ipeds_hd_url,
)

logger = logging.getLogger(__name__)

# IPEDS complete data files are not consistently encoded across release years --
# most are plain latin-1, but at least the HD2023 provisional release is
# UTF-8-with-BOM (see NOTES.md). Try utf-8-sig (handles a BOM if present, and
# is otherwise equivalent to utf-8) before falling back to latin-1.
_ENCODINGS_TO_TRY = ("utf-8-sig", "latin-1")


def _download_zip(url: str, dest_dir: Path, force: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_name = url.rsplit("/", 1)[-1]
    zip_path = dest_dir / zip_name
    if zip_path.exists() and not force:
        return zip_path
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    time.sleep(0.3)
    return zip_path


def _extract_main_csv(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the zip and return the path to the primary (non '_rv') CSV inside it."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        zf.extractall(dest_dir)
    primary = [n for n in names if "_rv" not in n.lower()]
    chosen = primary[0] if primary else names[0]
    return dest_dir / chosen


def fetch_component(year: int, component: str, url_fn, force: bool = False) -> pd.DataFrame:
    year_dir = IPEDS_RAW_DIR / str(year)
    zip_path = _download_zip(url_fn(year), year_dir, force=force)
    csv_path = _extract_main_csv(zip_path, year_dir)

    df = None
    for encoding in _ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"Could not decode {csv_path} with any of {_ENCODINGS_TO_TRY}")

    df.columns = [c.strip().upper() for c in df.columns]
    logger.info("Loaded %s %s: %s rows, %s cols", component, year, len(df), df.shape[1])
    return df


def load_hd(year: int, force: bool = False) -> pd.DataFrame:
    df = fetch_component(year, "HD", ipeds_hd_url, force=force)
    keep = [c for c in HD_FIELDS if c in df.columns]
    missing = set(HD_FIELDS) - set(keep)
    if missing:
        logger.warning("HD%s missing expected fields: %s", year, missing)
    out = df[keep].copy()
    out["UNITID"] = out["UNITID"].astype(int)
    return out


def load_enrollment(year: int, force: bool = False) -> pd.DataFrame:
    df = fetch_component(year, "EF", ipeds_ef_url, force=force)
    # EFALEVEL == 1 -> "All students total" (see NOTES.md for how this was confirmed
    # against the EF data dictionary's Frequencies sheet).
    total = df[df["EFALEVEL"] == 1][["UNITID", "EFTOTLT"]].rename(
        columns={"EFTOTLT": "total_enrollment"}
    )
    total["UNITID"] = total["UNITID"].astype(int)
    return total


def load_finance(year: int, force: bool = False) -> pd.DataFrame:
    df = fetch_component(year, "F2", ipeds_f2_url, force=force)
    keep_cols = ["UNITID"] + [c for c in F2_FIELDS if c in df.columns]
    missing = set(F2_FIELDS) - set(keep_cols)
    if missing:
        logger.warning("F2 %s missing expected fields: %s", year, missing)
    out = df[keep_cols].rename(columns=F2_FIELDS).copy()
    out["UNITID"] = out["UNITID"].astype(int)
    return out


def build_year_panel(year: int, force: bool = False) -> pd.DataFrame:
    hd = load_hd(year, force=force)
    nonprofit_4yr = hd[hd["SECTOR"] == PRIVATE_NONPROFIT_4YR_SECTOR].copy()

    enrollment = load_enrollment(year, force=force)
    finance = load_finance(year, force=force)

    merged = nonprofit_4yr.merge(enrollment, on="UNITID", how="left")
    merged = merged.merge(finance, on="UNITID", how="left")
    merged["ipeds_year"] = year
    merged = merged.rename(
        columns={
            "UNITID": "unitid",
            "INSTNM": "institution_name_ipeds",
            "OPEID": "opeid",
            "EIN": "ein",
            "STABBR": "state",
            "HBCU": "hbcu",
        }
    )
    merged["opeid"] = merged["opeid"].astype(str).str.strip().str.zfill(8)
    merged["ein"] = merged["ein"].astype(str).str.strip().str.zfill(9)
    return merged


def build_panel(years=YEARS, force: bool = False) -> pd.DataFrame:
    frames = []
    for year in years:
        try:
            frames.append(build_year_panel(year, force=force))
        except Exception:
            logger.exception("Failed to build IPEDS panel for year %s", year)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(IPEDS_RAW_DIR / "ipeds_combined.csv", index=False)
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_panel()
    print(df.shape)
    print(df.head())
