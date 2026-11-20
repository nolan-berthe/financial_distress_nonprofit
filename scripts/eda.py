"""Phase 2 exploratory analysis: distressed vs. non-distressed feature comparison
and a feature correlation matrix, with charts saved to reports/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")

LABEL_COL = "distressed_next_year"


def feature_comparison_table(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Median of each feature among distressed vs. non-distressed rows (labeled
    rows only), plus a Mann-Whitney U test p-value for whether the two groups'
    distributions differ (robust to the long-tailed tails noted in NOTES.md,
    unlike a t-test on raw, unclipped ratios)."""
    labeled = df[df[LABEL_COL].notna()]
    rows = []
    for col in feature_cols:
        g0 = labeled.loc[labeled[LABEL_COL] == 0, col].dropna()
        g1 = labeled.loc[labeled[LABEL_COL] == 1, col].dropna()
        _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        rows.append(
            {
                "feature": col,
                "median_not_distressed": g0.median(),
                "median_distressed": g1.median(),
                "mannwhitney_p": p,
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def plot_feature_distributions(df: pd.DataFrame, feature_cols, save_path: Path) -> None:
    labeled = df[df[LABEL_COL].notna()].copy()
    labeled["status"] = labeled[LABEL_COL].map({0: "Not distressed", 1: "Distressed (next yr)"})

    n = len(feature_cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, col in zip(axes, feature_cols):
        sns.boxplot(
            data=labeled, x="status", y=col, hue="status", ax=ax, showfliers=False, legend=False
        )
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Feature distributions: distressed vs. not distressed (outliers hidden)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, feature_cols, save_path: Path) -> None:
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(curves: dict, test_base_rate: float, save_path: Path) -> None:
    """Phase 3: one precision-recall curve per model, `curves` mapping model
    name -> (recall, precision) arrays from `model.pr_curve_data`. The dashed
    line is a random model's PR-AUC -- equal to the test-set distress rate."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (recall, precision) in curves.items():
        ax.plot(recall, precision, label=name)
    ax.axhline(test_base_rate, color="gray", linestyle="--", label=f"Random (base rate={test_base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(curves: dict, save_path: Path) -> None:
    """One ROC curve per model, `curves` mapping model name -> (fpr, tpr)
    arrays from `model.roc_curve_data`."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random (AUC=0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
