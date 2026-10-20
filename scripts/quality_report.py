"""Print a data quality report for the merged panel: coverage, match rates between
sources, label balance, missingness, and summary statistics.
"""

import pandas as pd


def print_report(panel: pd.DataFrame, ipeds_df: pd.DataFrame, composite_df: pd.DataFrame) -> None:
    n_institutions = panel["unitid"].nunique()
    n_years = panel["fiscal_year"].nunique()
    n_rows = len(panel)

    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)
    print(f"Panel rows (institution-year observations): {n_rows}")
    print(f"Distinct institutions (UNITID):              {n_institutions}")
    print(f"Distinct fiscal years:                        {n_years}")
    print(f"Fiscal year range:                            {panel['fiscal_year'].min()}"
          f"-{panel['fiscal_year'].max()}")

    print("\n--- Match rates between sources ---")
    ipeds_rows = len(ipeds_df)
    matched_composite = panel["composite_score"].notna().sum()
    print(f"IPEDS private-nonprofit-4yr rows:                     {ipeds_rows}")
    print(f"  ... matched to a composite score (same opeid+year): "
          f"{matched_composite} ({matched_composite / n_rows:.1%} of panel rows)")

    composite_opeids = set(composite_df["opeid"])
    ipeds_opeids = set(panel["opeid"].dropna())
    print(f"Distinct OPEIDs in composite-score files:             {len(composite_opeids)}")
    print(f"Distinct OPEIDs in IPEDS private-nonprofit-4yr universe: {len(ipeds_opeids)}")
    print(f"  ... overlap:                                        "
          f"{len(composite_opeids & ipeds_opeids)}")

    if "f990_total_revenue" in panel.columns:
        matched_990 = panel["f990_total_revenue"].notna().sum()
        has_ein = panel["ein"].notna().sum()
        print(f"\nRows with a non-null EIN:                             {has_ein}")
        print(f"  ... matched to a Form 990 filing for that fiscal year: "
              f"{matched_990} ({matched_990 / max(has_ein, 1):.1%} of rows with an EIN)")

    print("\n--- Label balance (distressed_next_year) ---")
    labeled = panel["distressed_next_year"].notna()
    print(f"Rows with a usable label (score observed in year t+1): {labeled.sum()} "
          f"({labeled.mean():.1%} of panel)")
    if labeled.sum() > 0:
        rate = panel.loc[labeled, "distressed_next_year"].mean()
        print(f"Share distressed (score < 1.5 next year) among labeled rows: {rate:.1%}")

    print("\n--- Missing values per column ---")
    missing = panel.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / n_rows * 100).round(1)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print(report[report["missing_count"] > 0].to_string())

    print("\n--- Summary statistics (numeric columns) ---")
    numeric_cols = panel.select_dtypes("number").columns
    print(panel[numeric_cols].describe().transpose().to_string())
