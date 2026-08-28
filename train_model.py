"""
train_model.py

Trains and tunes a RandomForestClassifier on the feature table produced by
build_feature_table.py, reporting ROC-AUC. Splits by a grouping column
(e.g. year, or spatial tile id) rather than random rows, to avoid
optimistic leakage between nearby/adjacent cells.

Usage:
    python train_model.py --data data/processed/feature_table.csv \
        --group-col row --out model.joblib
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV

FEATURES = ["ndvi", "elevation", "slope", "aspect", "tmmx", "rmin", "vs", "pr"]
TARGET = "ignition"


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    return df


def spatial_train_test_split(df, group_col, test_size=0.2, seed=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx], df.iloc[test_idx]


def tune_model(X_train, y_train, seed=42):
    param_dist = {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 8, 12, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    base = RandomForestClassifier(random_state=seed, n_jobs=-1)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=25,
        scoring="roc_auc",
        cv=3,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    return search.best_estimator_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--group-col", default="row",
                         help="Column to group-split on (spatial row/tile, or year)")
    parser.add_argument("--out", default="model.joblib")
    args = parser.parse_args()

    df = load_data(args.data)
    train_df, test_df = spatial_train_test_split(df, args.group_col)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    model = tune_model(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nHeld-out ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    joblib.dump(model, args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
