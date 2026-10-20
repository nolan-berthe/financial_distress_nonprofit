"""Merge composite scores + IPEDS + Form 990 into one institution-year panel,
and attach the distressed_next_year label.
"""

import logging

import numpy as np
import pandas as pd

from fetch_form990 import extract_filing_for_year, fetch_many

logger = logging.getLogger(__name__)


def _dedupe_composite(composite_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (opeid, fiscal_year); large multi-campus corporate
    chains can post more than one score file entry for the same locator OPEID+year,
    so duplicates are resolved by averaging (see NOTES.md)."""
    cols = ["opeid", "fiscal_year", "composite_score", "institution_name"]
    small = composite_df[cols].copy()
    agg = small.groupby(["opeid", "fiscal_year"], as_index=False).agg(
        composite_score=("composite_score", "mean"),
        institution_name_fsa=("institution_name", "first"),
    )
    return agg


def attach_composite_scores(ipeds_df: pd.DataFrame, composite_df: pd.DataFrame) -> pd.DataFrame:
    composite_small = _dedupe_composite(composite_df)

    panel = ipeds_df.merge(
        composite_small,
        left_on=["opeid", "ipeds_year"],
        right_on=["opeid", "fiscal_year"],
        how="left",
    )
    # `fiscal_year` came in from composite_small purely as a join key and is NaN for
    # any IPEDS row with no matching composite-score row; ipeds_year is the
    # authoritative year column for every row, so drop the redundant one here.
    panel = panel.drop(columns=["fiscal_year"])

    next_year_scores = composite_small.rename(
        columns={"fiscal_year": "next_year_lookup", "composite_score": "next_year_score"}
    )[["opeid", "next_year_lookup", "next_year_score"]]
    panel["next_year_lookup"] = panel["ipeds_year"] + 1
    panel = panel.merge(next_year_scores, on=["opeid", "next_year_lookup"], how="left")
    panel = panel.drop(columns=["next_year_lookup"])

    panel["distressed_next_year"] = np.where(
        panel["next_year_score"].notna(),
        (panel["next_year_score"] < 1.5).astype(float),
        np.nan,
    )
    return panel


def attach_form990(panel: pd.DataFrame, force_refetch: bool = False) -> pd.DataFrame:
    eins = panel["ein"].dropna().unique()
    org_records = fetch_many(eins, force=force_refetch)

    def _row_990(row):
        rec = org_records.get(str(row["ein"]).strip().zfill(9))
        return extract_filing_for_year(rec, int(row["ipeds_year"]))

    f990 = panel.apply(_row_990, axis=1, result_type="expand")
    return pd.concat([panel.reset_index(drop=True), f990.reset_index(drop=True)], axis=1)


def build_full_panel(ipeds_df: pd.DataFrame, composite_df: pd.DataFrame, fetch_990: bool = True) -> pd.DataFrame:
    panel = attach_composite_scores(ipeds_df, composite_df)
    if fetch_990:
        panel = attach_form990(panel)
    panel = panel.rename(columns={"ipeds_year": "fiscal_year"})
    return panel
