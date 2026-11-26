"""
src/mp152_cinema/predict.py
---------------------------------------------------------------------------
Stage 4 (final) — Load @champion model and predict holdout occupancy_pct.

Saves predictions/holdout_predictions.csv with two columns:
    record_id, prediction

Usage:
    python src/mp152_cinema/predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"

with open(PARAMS_PATH) as fh:
    PARAMS = yaml.safe_load(fh)

HOLDOUT_PATH = ROOT / PARAMS["paths"]["holdout_raw"]
PREDICTIONS_OUT = ROOT / PARAMS["paths"]["predictions_out"]
MLFLOW_URI = PARAMS["paths"]["mlflow_db"]
MODEL_NAME = PARAMS["mlflow"]["registered_model_name"]
CHAMPION_ALIAS = PARAMS["mlflow"]["champion_alias"]

FEATURE_COLS = [
    "ticket_price_rs",
    "days_since_release",
    "star_rating",
    "show_hour",
    "screen_seats",
    "distance_to_metro_km",
    "language",
    "location",
]

LEAKING_COL = PARAMS["data"]["leaking_column"]


def clean_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same cleaning as train (without target)."""
    df = df.copy()
    if LEAKING_COL and LEAKING_COL in df.columns:
        df = df.drop(columns=[LEAKING_COL])
    # Normalize language
    abbrev_map = {"eng.": "english", "reg.": "regional", "hin.": "hindi"}
    df["language"] = df["language"].str.strip().str.lower().replace(abbrev_map)
    return df


def run_predictions():
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Load holdout data
    df_holdout = pd.read_csv(HOLDOUT_PATH)
    df_holdout = clean_holdout(df_holdout)

    record_ids = df_holdout["record_id"]
    X = df_holdout[FEATURE_COLS]

    print(f"Holdout shape: {X.shape}")

    # Load @champion model
    model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # Predict
    preds = model.predict(X)

    # Clip to valid range [0, 100]
    preds = preds.clip(0, 100)

    # Save results
    out_df = pd.DataFrame({
        "record_id": record_ids.values,
        "prediction": preds.round(4),
    })

    PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PREDICTIONS_OUT, index=False)
    print(f"\nPredictions saved to: {PREDICTIONS_OUT}")
    print(f"Sample:\n{out_df.head()}")

    return out_df


if __name__ == "__main__":
    run_predictions()
    sys.exit(0)
