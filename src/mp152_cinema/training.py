"""
src/mp152_cinema/training.py
---------------------------------------------------------------------------
Stage 2 — Model Training & MLflow Tracking for MP152 Cinema Show Occupancy.

Trains three RandomForestRegressor configurations on cleaned train.csv,
logs parameters and MAE to MLflow, registers the best model as version 1,
and assigns the @champion alias.

Usage:
    python src/mp152_cinema/training.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"

with open(PARAMS_PATH) as fh:
    PARAMS = yaml.safe_load(fh)

CLEAN_TRAIN = ROOT / PARAMS["paths"]["train_clean"]
MLFLOW_URI = PARAMS["paths"]["mlflow_db"]
EXPERIMENT = PARAMS["mlflow"]["experiment_name"]
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
TARGET_COL = "occupancy_pct"

CAT_FEATURES = ["language", "location"]
NUM_FEATURES = [c for c in FEATURE_COLS if c not in CAT_FEATURES]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pipeline(rf_params: dict) -> Pipeline:
    """Build a sklearn Pipeline with ordinal encoding + RandomForest."""
    cat_idx = [FEATURE_COLS.index(c) for c in CAT_FEATURES]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )

    # We'll apply OrdinalEncoder only to cat columns via column transformer
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, CAT_FEATURES),
            ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        ]
    )

    model = RandomForestRegressor(**rf_params)

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def load_clean_data(path: Path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def run_training():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    # SHA of raw train (original, unmodified)
    raw_train = ROOT / PARAMS["paths"]["train_raw"]
    data_sha256 = sha256_file(raw_train)

    X, y = load_clean_data(CLEAN_TRAIN)
    print(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")

    configs = [
        {
            "name": "v1_rf_200",
            "params": PARAMS["model"]["v1"],
        },
        {
            "name": "v1_rf_100",
            "params": PARAMS["model"]["v1b"],
        },
        {
            "name": "v1_rf_300",
            "params": PARAMS["model"]["v1c"],
        },
    ]

    best_mae = float("inf")
    best_run_id = None
    best_pipeline = None

    for cfg in configs:
        name = cfg["name"]
        rf_params = cfg["params"]

        with mlflow.start_run(run_name=name) as run:
            # Log parameters
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("config_name", name)
            for k, v in rf_params.items():
                mlflow.log_param(k, v)

            pipe = build_pipeline(rf_params)

            # 5-fold cross-validated MAE
            cv_scores = cross_val_score(
                pipe, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
            )
            cv_mae = float(-cv_scores.mean())

            # Fit on full training data
            pipe.fit(X, y)
            train_preds = pipe.predict(X)
            train_mae = float(mean_absolute_error(y, train_preds))

            mlflow.log_metric("mae", cv_mae)
            mlflow.log_metric("mae_train", train_mae)
            mlflow.log_metric("mae_cv", cv_mae)

            # Log model
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
                registered_model_name=MODEL_NAME,
            )

            run_id = run.info.run_id
            print(
                f"  {name}: cv_mae={cv_mae:.4f}, train_mae={train_mae:.4f}, run_id={run_id}"
            )

            if cv_mae < best_mae:
                best_mae = cv_mae
                best_run_id = run_id
                best_pipeline = pipe

    # Assign @champion to the best version
    client = MlflowClient()

    # Find the model version for the best run
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    best_version = None
    for mv in all_versions:
        if mv.run_id == best_run_id:
            best_version = mv.version
            break

    if best_version is None:
        # Fallback: use the latest version
        all_versions_sorted = sorted(all_versions, key=lambda v: int(v.version))
        best_version = all_versions_sorted[-1].version

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, best_version)
    print(f"\nBest model: version={best_version}, mae={best_mae:.4f}, run_id={best_run_id}")
    print(f"Alias '@{CHAMPION_ALIAS}' set on version {best_version}")

    # Save champion model to models/model.joblib for FastAPI service
    model_joblib_path = ROOT / PARAMS["paths"]["model_joblib"]
    model_joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_joblib_path)
    print(f"Saved champion model to {model_joblib_path}")

    return best_run_id, best_mae, best_pipeline


if __name__ == "__main__":
    run_id, mae, _ = run_training()
    print(f"\nTraining complete. Best run_id: {run_id}, best MAE: {mae:.4f}")
    sys.exit(0)
