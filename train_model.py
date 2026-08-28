"""
train_model.py

Trains and tunes a RandomForestClassifier on the feature table produced by
build_feature_table.py. Splits by a grouping column (e.g. year, or spatial
tile id) rather than random rows, to avoid optimistic leakage between
nearby/adjacent cells.

Reports both a single held-out ROC-AUC and a grouped k-fold cross-
validated ROC-AUC (mean +/- std across folds) for a more defensible
estimate than either alone, and writes both -- along with the chosen
hyperparameters and feature importances -- to a metrics JSON file next to
the saved model, so whatever a given run measures is reproducible.

Usage:
    python train_model.py --data data/processed/feature_table.csv \
        --group-col row --out model.joblib
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV

FEATURES = ["ndvi", "elevation", "slope", "aspect", "tmmx", "rmin", "vs", "pr"]
# Populated by build_feature_table.py's temporal dryness features when the
# weather input has a date column; used automatically when present.
OPTIONAL_FEATURES = ["pr_7day_sum", "days_since_rain"]
TARGET = "ignition"


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    features = FEATURES + [c for c in OPTIONAL_FEATURES if c in df.columns]
    return df, features


def spatial_train_test_split(df, group_col, test_size=0.2, seed=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx], df.iloc[test_idx]


def tune_model(X_train, y_train, groups, n_iter=40, cv_folds=5, seed=42):
    param_dist = {
        "n_estimators": [200, 300, 500, 800, 1200],
        "max_depth": [None, 8, 12, 20, 30],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 4, 8, 16],
        "max_features": ["sqrt", "log2", 0.5, 0.7],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    base = RandomForestClassifier(random_state=seed, n_jobs=-1)
    cv_splits = list(GroupKFold(n_splits=cv_folds).split(X_train, y_train, groups=groups))
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv_splits,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print(f"Best CV ROC-AUC during tuning: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


def grouped_cv_auc(model, X, y, groups, n_splits=5):
    """Mean/std ROC-AUC across grouped folds of the full dataset -- a more
    defensible estimate than a single held-out split, since it isn't
    sensitive to which particular groups happened to land in the holdout.
    """
    cv = GroupKFold(n_splits=n_splits)
    scores = []
    skipped = 0
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        # A fold can end up with only one class present -- e.g. a handful of
        # ignitions split unluckily across groups -- in which case the model
        # can't be fit meaningfully or ROC-AUC isn't defined. Skip it rather
        # than crash (predict_proba only has one column when trained on a
        # single class) or report a misleading score.
        if y_train_fold.nunique() < 2 or y_test_fold.nunique() < 2:
            skipped += 1
            continue
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y_train_fold)
        proba = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
        scores.append(roc_auc_score(y_test_fold, proba))

    if skipped:
        print(f"Skipped {skipped}/{n_splits} CV fold(s) with only one class present "
              f"(too few ignitions relative to fold count)")
    if not scores:
        return float("nan"), float("nan"), []
    return float(np.mean(scores)), float(np.std(scores)), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--group-col", default="row",
                         help="Column to group-split on (spatial row/tile, or year)")
    parser.add_argument("--n-iter", type=int, default=40, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv-folds", type=int, default=5, help="Grouped folds for tuning + reporting")
    parser.add_argument("--out", default="model.joblib")
    args = parser.parse_args()

    df, features = load_data(args.data)
    train_df, test_df = spatial_train_test_split(df, args.group_col)

    X_train, y_train = train_df[features], train_df[TARGET]
    X_test, y_test = test_df[features], test_df[TARGET]

    model, best_params, best_cv_auc = tune_model(
        X_train, y_train, train_df[args.group_col].values,
        n_iter=args.n_iter, cv_folds=args.cv_folds,
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    holdout_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nHeld-out ROC-AUC: {holdout_auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    cv_mean_auc, cv_std_auc, cv_scores = grouped_cv_auc(
        model, df[features], df[TARGET], df[args.group_col].values, n_splits=args.cv_folds
    )
    print(f"\n{args.cv_folds}-fold grouped CV ROC-AUC: {cv_mean_auc:.4f} +/- {cv_std_auc:.4f}")

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    joblib.dump(model, args.out)
    print(f"\nSaved model to {args.out}")

    metrics_path = os.path.splitext(args.out)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "n_rows": len(df),
            "ignition_rate": float(df[TARGET].mean()),
            "features": features,
            "best_params": best_params,
            "cv_auc_during_tuning": best_cv_auc,
            "holdout_auc": float(holdout_auc),
            "grouped_cv_auc_mean": cv_mean_auc,
            "grouped_cv_auc_std": cv_std_auc,
            "grouped_cv_auc_folds": cv_scores,
            "feature_importances": importances.to_dict(),
        }, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
