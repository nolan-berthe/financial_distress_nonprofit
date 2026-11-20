"""Phase 2 feature engineering: build year-t predictors from the Phase 1 panel,
using only information available by the end of year t (the same fiscal year's
IPEDS/Form 990 filings -- never a future year).

IPEDS is the primary source for every financial figure; Form 990 fills a value
only when the matching IPEDS field is missing for that institution-year (see
NOTES.md "Phase 2" for why: IPEDS F2 is the purpose-built finance survey with
audited totals, Form 990 is a general-purpose tax filing with coarser categories
and no tuition/gifts breakdown, so it's used as a gap-filler, not a primary source).
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "operating_margin",
    "leverage",
    "tuition_dependence",
    "contribution_share",
    "enrollment_change_1yr",
    "enrollment_change_3yr",
    "log_total_revenue",
]


def _combine_ipeds_990(panel: pd.DataFrame, ipeds_col: str, f990_col: str) -> pd.Series:
    """IPEDS value where present, else the Form 990 value for the same row."""
    return panel[ipeds_col].combine_first(panel[f990_col])


def add_combined_financials(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach *_filled columns: IPEDS primary, Form 990 fallback, for the four
    fields both sources report (revenue, expenses, assets, liabilities)."""
    out = panel.copy()
    out["total_revenues_filled"] = _combine_ipeds_990(out, "total_revenues", "f990_total_revenue")
    out["total_expenses_filled"] = _combine_ipeds_990(out, "total_expenses", "f990_total_expenses")
    out["total_assets_filled"] = _combine_ipeds_990(out, "total_assets", "f990_total_assets_end")
    out["total_liabilities_filled"] = _combine_ipeds_990(out, "total_liabilities", "f990_total_liabilities_end")
    return out


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return `panel` with one column per feature in FEATURE_COLS added.

    Tuition dependence and contribution share use IPEDS F2 fields only
    (tuition_fees_revenue, contribution_revenue) -- Form 990 does not break
    revenue into those categories, so there is no fallback source for the
    numerator and these two features are NaN wherever IPEDS F2 is missing.
    """
    out = add_combined_financials(panel)
    out = out.sort_values(["unitid", "fiscal_year"]).reset_index(drop=True)

    revenue = out["total_revenues_filled"].replace(0, np.nan)
    assets = out["total_assets_filled"].replace(0, np.nan)

    out["operating_margin"] = (revenue - out["total_expenses_filled"]) / revenue
    out["leverage"] = out["total_liabilities_filled"] / assets
    out["tuition_dependence"] = out["tuition_fees_revenue"] / revenue
    out["contribution_share"] = out["contribution_revenue"] / revenue
    out["log_total_revenue"] = np.log(out["total_revenues_filled"].where(out["total_revenues_filled"] > 0))

    grouped = out.groupby("unitid")["total_enrollment"]
    enroll_lag1 = grouped.shift(1)
    enroll_lag3 = grouped.shift(3)
    out["enrollment_change_1yr"] = (out["total_enrollment"] - enroll_lag1) / enroll_lag1
    out["enrollment_change_3yr"] = (out["total_enrollment"] - enroll_lag3) / enroll_lag3

    return out


def missingness_report(df: pd.DataFrame, cols=FEATURE_COLS) -> pd.DataFrame:
    """Missing count/pct per feature, for the progress report."""
    n = len(df)
    rows = []
    for col in cols:
        missing = df[col].isna().sum()
        rows.append({"feature": col, "missing_count": missing, "missing_pct": round(100 * missing / n, 1)})
    return pd.DataFrame(rows).set_index("feature")


def fit_clip_bounds(train_df: pd.DataFrame, cols=FEATURE_COLS, lower_q=0.01, upper_q=0.99) -> dict:
    """1st/99th percentile of each feature on the training rows only.

    Several ratio features (operating margin, leverage, tuition/contribution
    share, enrollment change) have a long tail driven by institutions with a
    near-zero denominator in a given year (e.g. enrollment of 3 dropping to 1)
    -- a handful of these multi-hundred-percent outliers would otherwise
    dominate a standardized linear model. Winsorizing at the 1st/99th
    percentile keeps every row but bounds its influence; bounds are fit on
    train only for the same leakage reason as the imputer below.

    quantile() on an entirely-missing column (possible in a short Phase 3 CV
    fold, see `fit_median_imputer`) returns NaN; pandas' `.clip()` already
    treats a NaN bound as "no bound on that side," so no extra guard is
    needed here -- a fold with no values yet simply clips nothing.
    """
    return {col: (train_df[col].quantile(lower_q), train_df[col].quantile(upper_q)) for col in cols}


def apply_clip(df: pd.DataFrame, clip_bounds: dict, cols=FEATURE_COLS) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        lower, upper = clip_bounds[col]
        out[col] = out[col].clip(lower, upper)
    return out


def fit_median_imputer(train_df: pd.DataFrame, cols=FEATURE_COLS) -> dict:
    """Median of each feature computed on the training rows only -- the rule
    used to fill missing feature values, documented here so test-set rows never
    influence the fill value (would leak information about the held-out period).

    Phase 3's rolling-origin CV folds can have a training window too short for
    a lagged feature to ever be non-null (e.g. `enrollment_change_3yr` needs
    3 prior years of history, so a fold training on the panel's first 3 years
    has it missing for every single row, making the median itself NaN). In
    that case fall back to 0.0 -- a neutral "no known change" value -- rather
    than propagating a NaN into the model.
    """
    medians = {}
    for col in cols:
        median = train_df[col].median()
        medians[col] = median if pd.notna(median) else 0.0
    return medians


def apply_imputer(df: pd.DataFrame, impute_values: dict, cols=FEATURE_COLS) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].fillna(impute_values[col])
    return out
