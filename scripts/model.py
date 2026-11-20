"""Phase 2 baseline model: time-based train/test split, standardized logistic
regression with class weights, and evaluation against a naive baseline.

Phase 3 adds random forest / XGBoost fitting, rolling-origin time-based CV
for hyperparameter search within the training years only, and a
threshold-selection step that also uses validation folds only (never the
test set) -- see NOTES.md "Phase 3".
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

LABEL_COL = "distressed_next_year"


def time_based_split(df: pd.DataFrame, train_years, test_years):
    """Split a labeled feature frame by fiscal_year -- train on `train_years`,
    test on `test_years` (no shuffling across time, so the model is always
    evaluated on years strictly later than anything it was fit on)."""
    train = df[df["fiscal_year"].isin(train_years)].copy()
    test = df[df["fiscal_year"].isin(test_years)].copy()
    return train, test


def fit_baseline(train_df: pd.DataFrame, feature_cols):
    """Standardize features (scaler fit on train only) and fit a logistic
    regression with class_weight='balanced' to offset the ~7% distressed rate."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    y_train = train_df[LABEL_COL].astype(int)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    return clf, scaler


def evaluate(clf, scaler, test_df: pd.DataFrame, feature_cols) -> dict:
    X_test = scaler.transform(test_df[feature_cols])
    y_test = test_df[LABEL_COL].astype(int)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    return {
        "n_test": len(y_test),
        "base_rate": y_test.mean(),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "y_test": y_test.values,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def naive_baseline(test_df: pd.DataFrame) -> dict:
    """'Always predict not distressed' -- the comparison point for the logistic
    model. Precision/recall on the distressed class are 0 by construction
    (the naive rule never predicts positive), so only accuracy is informative."""
    y_test = test_df[LABEL_COL].astype(int)
    y_pred = np.zeros(len(y_test), dtype=int)
    return {
        "n_test": len(y_test),
        "accuracy": (y_pred == y_test).mean(),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }


def coefficients_table(clf, feature_cols) -> pd.DataFrame:
    return pd.DataFrame({"feature": feature_cols, "coefficient": clf.coef_[0]}).sort_values(
        "coefficient", key=abs, ascending=False
    )


# --------------------------------------------------------------------------
# Phase 3: random forest / XGBoost, rolling-origin CV, thresholding
# --------------------------------------------------------------------------


def rolling_cv_folds(train_years, min_train_years=3):
    """Expanding-window, time-based CV folds within the training years only.

    Fold k trains on the first `min_train_years + k` years and validates on
    the single year immediately after -- e.g. for train_years=2014..2019 and
    min_train_years=3: (train=[2014-2016], val=2017), (train=[2014-2017],
    val=2018), (train=[2014-2018], val=2019). Never touches the test years.
    """
    years = sorted(train_years)
    folds = []
    for i in range(min_train_years, len(years)):
        fold_train = years[:i]
        fold_val = [years[i]]
        folds.append((fold_train, fold_val))
    return folds


def fit_logistic(train_df: pd.DataFrame, feature_cols, **params):
    """Standardized logistic regression as a single fitted Pipeline (scaler +
    classifier), so it can be fit/predicted through the same `fit_fn`/
    `predict_proba` interface as the tree models below -- unlike Phase 2's
    `fit_baseline`, which returns the scaler separately."""
    defaults = dict(class_weight="balanced", max_iter=1000)
    defaults.update(params)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(**defaults))
    pipe.fit(train_df[feature_cols], train_df[LABEL_COL].astype(int))
    return pipe


def fit_random_forest(train_df: pd.DataFrame, feature_cols, **params) -> RandomForestClassifier:
    """Random forest with class_weight='balanced' to offset the rare-class
    imbalance (same rationale as the logistic baseline's class_weight)."""
    defaults = dict(class_weight="balanced", random_state=0, n_jobs=-1)
    defaults.update(params)
    clf = RandomForestClassifier(**defaults)
    clf.fit(train_df[feature_cols], train_df[LABEL_COL].astype(int))
    return clf


def fit_xgboost(train_df: pd.DataFrame, feature_cols, **params) -> XGBClassifier:
    """XGBoost with scale_pos_weight set from this training set's own class
    ratio (neg/pos), the standard way to offset imbalance for XGBoost since
    it has no class_weight='balanced' option. Computed fresh per fold/call so
    it never leaks information from outside that call's own training rows."""
    y_train = train_df[LABEL_COL].astype(int)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    defaults = dict(
        scale_pos_weight=scale_pos_weight,
        random_state=0,
        eval_metric="logloss",
        n_jobs=-1,
    )
    defaults.update(params)
    clf = XGBClassifier(**defaults)
    clf.fit(train_df[feature_cols], y_train)
    return clf


def predict_proba(model, df: pd.DataFrame, feature_cols) -> np.ndarray:
    """Predicted P(distressed) for any of the three model types. All three
    take the (clipped + imputed) feature columns directly -- the logistic
    model's own Pipeline (see `fit_logistic`) handles its scaling internally."""
    return model.predict_proba(df[feature_cols])[:, 1]


def _preprocess_fold(fold_train: pd.DataFrame, fold_other: pd.DataFrame, feature_cols):
    """Winsorize + median-impute, fit on `fold_train` only and applied to
    both -- the same two-step rule as the Phase 2 baseline (see
    `features.fit_clip_bounds`/`fit_median_imputer`), redone per fold so no
    fold's validation year ever influences its own preprocessing."""
    from features import apply_clip, apply_imputer, fit_clip_bounds, fit_median_imputer

    clip_bounds = fit_clip_bounds(fold_train, feature_cols)
    train_clipped = apply_clip(fold_train, clip_bounds, feature_cols)
    other_clipped = apply_clip(fold_other, clip_bounds, feature_cols)

    impute_values = fit_median_imputer(train_clipped, feature_cols)
    train_ready = apply_imputer(train_clipped, impute_values, feature_cols)
    other_ready = apply_imputer(other_clipped, impute_values, feature_cols)
    return train_ready, other_ready


def grid_search_time_cv(fit_fn, train_df: pd.DataFrame, feature_cols, param_grid: list,
                         cv_folds) -> pd.DataFrame:
    """Try every param dict in `param_grid` across the rolling-origin CV folds
    and score each by mean validation PR-AUC (threshold-independent, and the
    right metric to optimize for a rare positive class -- see NOTES.md).

    Each fold winsorizes/imputes on its own training years only (see
    `_preprocess_fold`). `fit_fn(fold_train_ready, feature_cols, **params) ->
    model`. Returns one row per param combo, sorted best-first.
    """
    rows = []
    for params in param_grid:
        fold_scores = []
        for fold_train_years, fold_val_years in cv_folds:
            fold_train = train_df[train_df["fiscal_year"].isin(fold_train_years)]
            fold_val = train_df[train_df["fiscal_year"].isin(fold_val_years)]
            train_ready, val_ready = _preprocess_fold(fold_train, fold_val, feature_cols)

            model = fit_fn(train_ready, feature_cols, **params)
            y_prob = predict_proba(model, val_ready, feature_cols)
            fold_scores.append(average_precision_score(val_ready[LABEL_COL].astype(int), y_prob))

        rows.append({**params, "mean_val_pr_auc": np.mean(fold_scores), "fold_val_pr_auc": fold_scores})
    return pd.DataFrame(rows).sort_values("mean_val_pr_auc", ascending=False).reset_index(drop=True)


def oof_validation_predictions(fit_fn, train_df: pd.DataFrame, feature_cols, best_params: dict,
                                cv_folds) -> pd.DataFrame:
    """Refit with `best_params` on each fold's training years (with that
    fold's own clip/impute, same as `grid_search_time_cv`) and predict on
    that fold's validation year, pooling the out-of-fold predictions across
    all folds. This pooled set (still entirely within the training years) is
    what the decision threshold is chosen on -- the test set is never used
    for either hyperparameter selection or thresholding."""
    frames = []
    for fold_train_years, fold_val_years in cv_folds:
        fold_train = train_df[train_df["fiscal_year"].isin(fold_train_years)]
        fold_val = train_df[train_df["fiscal_year"].isin(fold_val_years)]
        train_ready, val_ready = _preprocess_fold(fold_train, fold_val, feature_cols)
        val_ready = val_ready.copy()

        model = fit_fn(train_ready, feature_cols, **best_params)
        val_ready["y_prob"] = predict_proba(model, val_ready, feature_cols)
        frames.append(val_ready[["fiscal_year", LABEL_COL, "y_prob"]])
    return pd.concat(frames, ignore_index=True)


def best_threshold_by_f1(y_true, y_prob) -> float:
    """Decision cutoff maximizing F1 on the distressed class, swept over the
    thresholds precision_recall_curve actually produces (every distinct
    predicted probability), computed on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
    # precision_recall_curve returns one more precision/recall point than threshold
    # (the last point is recall=0, precision=1 with no corresponding threshold)
    best_idx = np.argmax(f1[:-1])
    return thresholds[best_idx]


def evaluate_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """Metrics for a model whose probabilities are already computed, using a
    cutoff chosen beforehand (on validation data) rather than sklearn's
    default 0.5 -- the threshold never depends on `y_true`/`y_prob` here."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n": len(y_true),
        "base_rate": y_true.mean(),
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def pr_curve_data(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return recall, precision


def roc_curve_data(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr


def feature_importance_table(model, feature_cols) -> pd.DataFrame:
    """Built-in impurity/gain-based importances for a random forest or
    XGBoost model (a preview only -- SHAP values come in Phase 4)."""
    return pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
