"""
predict_risk.py

Bridges train_model.py's output to render_risk_map.py's input: scores a
feature table with a trained model and writes a lon/lat/risk_score CSV.

The feature table is one row per (cell, date), but a risk map wants one
score per location -- so this takes each cell's most recent date's row
before scoring.

Usage:
    python predict_risk.py --data scaled_feature_table_year.csv \
        --model scaled_model_year.joblib --out predictions.csv
"""
import argparse

import joblib

from train_model import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Feature table CSV (same format used for training)")
    parser.add_argument("--model", required=True, help="Trained model .joblib from train_model.py")
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()

    df, features = load_data(args.data)
    latest = df.sort_values("date").groupby("cell_id", as_index=False).last()

    model = joblib.load(args.model)
    latest["risk_score"] = model.predict_proba(latest[features])[:, 1]

    latest[["cell_id", "lon", "lat", "risk_score"]].to_csv(args.out, index=False)
    print(f"Wrote {len(latest)} cell risk scores to {args.out}")


if __name__ == "__main__":
    main()
