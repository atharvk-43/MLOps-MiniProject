"""
src/mp152_cinema/evaluation.py
---------------------------------------------------------------------------
Stage 4 — Monitoring, Drift Detection (PSI), and Retraining for MP152.

Steps:
  1. Compute PSI (10 bins) for every numeric column between cleaned
     train.csv and later.csv.
  2. Compute v1 MAE on later.csv per location group.
  3. Retrain v2 on cleaned train + first 70% of later.csv.
  4. Compare v1 vs v2 on the last 30% of later.csv.
  5. Apply promotion rule: promote v2 to @champion if its MAE is at
     least 10% lower than v1's.
  6. Update findings.yaml with all monitoring results.

Usage:
    python src/mp152_cinema/evaluation.py
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
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"

with open(PARAMS_PATH) as fh:
    PARAMS = yaml.safe_load(fh)

CLEAN_TRAIN = ROOT / PARAMS["paths"]["train_clean"]
LATER_RAW = ROOT / PARAMS["paths"]["later_raw"]
MLFLOW_URI = PARAMS["paths"]["mlflow_db"]
MODEL_NAME = PARAMS["mlflow"]["registered_model_name"]
CHAMPION_ALIAS = PARAMS["mlflow"]["champion_alias"]
EXPERIMENT = PARAMS["mlflow"]["experiment_name"]
FINDINGS_PATH = ROOT / PARAMS["paths"]["findings"]

PSI_BINS = PARAMS["monitoring"]["psi_bins"]
RETRAIN_CUTOFF = PARAMS["monitoring"]["retrain_cutoff"]
PROMO_THRESHOLD = PARAMS["monitoring"]["promotion_threshold"]

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
NUM_FEATURE_COLS = [
    "ticket_price_rs",
    "days_since_release",
    "star_rating",
    "show_hour",
    "screen_seats",
    "distance_to_metro_km",
]
TARGET_COL = "occupancy_pct"
LOC_COL = "location"


# ---------------------------------------------------------------------------
# PSI helpers
# ---------------------------------------------------------------------------

def compute_psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between reference and comparison.
    Uses reference-data edges to define bins (consistent with lecture P13).
    """
    ref = reference[~np.isnan(reference)]
    cmp = comparison[~np.isnan(comparison)]

    # Build bin edges from reference
    _, edges = np.histogram(ref, bins=bins)
    edges[0] -= 1e-9   # include minimum
    edges[-1] += 1e-9  # include maximum

    ref_hist, _ = np.histogram(ref, bins=edges)
    cmp_hist, _ = np.histogram(cmp, bins=edges)

    # Convert to proportions (avoid division by zero)
    ref_pct = ref_hist / len(ref)
    cmp_pct = cmp_hist / len(cmp)

    # Replace zeros with small value to avoid log(0)
    eps = 1e-6
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cmp_pct = np.where(cmp_pct == 0, eps, cmp_pct)

    psi = np.sum((ref_pct - cmp_pct) * np.log(ref_pct / cmp_pct))
    return float(psi)


def prepare_later(df_later: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare later.csv for evaluation:
    - later.csv is stated to be clean; drop leaking column only if configured.
    """
    df = df_later.copy()
    leaking = PARAMS["data"]["leaking_column"]
    if leaking and leaking in df.columns:
        df = df.drop(columns=[leaking])
    return df


def clean_later_language(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize language if needed (later.csv is stated clean)."""
    # later.csv should already be clean; apply same normalization just in case
    abbrev_map = {"eng.": "english", "reg.": "regional", "hin.": "hindi"}
    df = df.copy()
    df["language"] = df["language"].str.strip().str.lower().replace(abbrev_map)
    return df


# ---------------------------------------------------------------------------
# Load model helpers
# ---------------------------------------------------------------------------

def load_champion_model():
    """Load the current @champion model from MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def get_champion_version() -> str:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    return mv.version


def get_champion_run_id() -> str:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    return mv.run_id


# ---------------------------------------------------------------------------
# Retraining v2
# ---------------------------------------------------------------------------

def train_v2(X_train, y_train) -> object:
    """Retrain a v2 model using the same architecture as the best v1."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.pipeline import Pipeline

    cat_features = ["language", "location"]
    num_features = [c for c in FEATURE_COLS if c not in cat_features]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_features),
            ("num", SimpleImputer(strategy="median"), num_features),
        ]
    )

    v2_params = PARAMS["model"]["v1"]  # same hyperparams as best v1
    rf = RandomForestRegressor(**v2_params)
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", rf)])
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Load data
    df_train = pd.read_csv(CLEAN_TRAIN)
    df_later_raw = pd.read_csv(LATER_RAW)
    df_later_raw["date"] = pd.to_datetime(df_later_raw["date"])

    df_later = prepare_later(df_later_raw)
    df_later = clean_later_language(df_later)

    # -----------------------------------------------------------------------
    # 1. PSI for every numeric column
    # -----------------------------------------------------------------------
    print("\n=== PSI (cleaned train vs later.csv) ===")
    psi_results = {}
    for col in NUM_FEATURE_COLS:
        ref_vals = df_train[col].values if col in df_train.columns else np.array([])
        cmp_vals = df_later[col].values if col in df_later.columns else np.array([])
        if len(ref_vals) == 0 or len(cmp_vals) == 0:
            psi_results[col] = None
            print(f"  {col}: N/A (column missing)")
        else:
            psi_val = compute_psi(ref_vals, cmp_vals, bins=PSI_BINS)
            psi_results[col] = round(psi_val, 4)
            print(f"  {col}: PSI = {psi_val:.4f}")

    most_drifted = max(
        (k for k, v in psi_results.items() if v is not None),
        key=lambda k: psi_results[k]
    )
    print(f"  Most drifted: {most_drifted} (PSI={psi_results[most_drifted]})")

    # -----------------------------------------------------------------------
    # 2. Load v1 champion model
    # -----------------------------------------------------------------------
    print("\n=== Loading v1 champion model ===")
    v1_model = load_champion_model()
    champion_version = get_champion_version()
    champion_run_id = get_champion_run_id()
    print(f"  Champion version: {champion_version}, run_id: {champion_run_id}")

    # -----------------------------------------------------------------------
    # 3. MAE by location on later.csv
    # -----------------------------------------------------------------------
    print("\n=== v1 MAE by location on later.csv ===")
    X_later = df_later[FEATURE_COLS]
    y_later = df_later[TARGET_COL]

    preds_v1_later = v1_model.predict(X_later)
    mae_by_group = {}
    for loc in ["City Centre", "Mall", "Highway"]:
        mask = df_later[LOC_COL] == loc
        if mask.sum() == 0:
            mae_by_group[loc] = None
        else:
            mae_loc = float(mean_absolute_error(y_later[mask], preds_v1_later[mask]))
            mae_by_group[loc] = round(mae_loc, 4)
            print(f"  {loc}: MAE = {mae_loc:.4f}")

    worst_group = max(
        (k for k, v in mae_by_group.items() if v is not None),
        key=lambda k: mae_by_group[k]
    )
    print(f"  Worst served group: {worst_group}")

    # -----------------------------------------------------------------------
    # 4. Retrain v2 on train + first 70% of later.csv
    # -----------------------------------------------------------------------
    print("\n=== Retraining v2 ===")
    df_later_sorted = df_later.sort_values("date").reset_index(drop=True)
    n_later = len(df_later_sorted)
    n_train_later = int(n_later * RETRAIN_CUTOFF)

    df_later_train = df_later_sorted.iloc[:n_train_later]
    df_later_check = df_later_sorted.iloc[n_train_later:]

    print(f"  later.csv split: {n_train_later} rows for retraining, {len(df_later_check)} rows for check")

    # Combine cleaned train + first 70% of later
    df_combined = pd.concat([df_train, df_later_train], ignore_index=True)
    X_combined = df_combined[FEATURE_COLS]
    y_combined = df_combined[TARGET_COL]

    # Check set
    X_check = df_later_check[FEATURE_COLS]
    y_check = df_later_check[TARGET_COL]

    # v1 on check set
    preds_v1_check = v1_model.predict(X_check)
    mae_v1_check = float(mean_absolute_error(y_check, preds_v1_check))
    print(f"  v1 MAE on last 30%: {mae_v1_check:.4f}")

    # Train v2
    raw_train = ROOT / PARAMS["paths"]["train_raw"]
    h = hashlib.sha256()
    with open(raw_train, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    data_sha256 = h.hexdigest()

    with mlflow.start_run(run_name="v2_retrained") as run:
        v2_model = train_v2(X_combined, y_combined)

        preds_v2_check = v2_model.predict(X_check)
        mae_v2_check = float(mean_absolute_error(y_check, preds_v2_check))
        print(f"  v2 MAE on last 30%: {mae_v2_check:.4f}")

        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_version", "v2")
        mlflow.log_param("retrain_data", "train_clean+later_70pct")
        mlflow.log_metric("mae_later_check", mae_v2_check)
        mlflow.log_metric("mae_v1_later_check", mae_v1_check)

        mlflow.sklearn.log_model(
            v2_model,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name=MODEL_NAME,
        )

        v2_run_id = run.info.run_id

    # -----------------------------------------------------------------------
    # 5. Promotion decision
    # -----------------------------------------------------------------------
    improvement = (mae_v1_check - mae_v2_check) / mae_v1_check
    promote = improvement >= PROMO_THRESHOLD
    decision = "promote" if promote else "keep"

    print(f"\n  Improvement: {improvement:.2%}")
    print(f"  Decision: {decision} (threshold={PROMO_THRESHOLD:.0%})")

    if promote:
        client = MlflowClient()
        # Get v2 version number
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        v2_version = None
        for mv in all_versions:
            if mv.run_id == v2_run_id:
                v2_version = mv.version
                break
        if v2_version is None:
            all_versions_sorted = sorted(all_versions, key=lambda v: int(v.version))
            v2_version = all_versions_sorted[-1].version

        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, v2_version)
        champion_version = v2_version
        model_joblib_path = ROOT / PARAMS["paths"]["model_joblib"]
        model_joblib_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(v2_model, model_joblib_path)
        print(f"  v2 promoted to @champion (version {v2_version}) and saved to {model_joblib_path}")
    else:
        print(f"  v1 (version {champion_version}) remains @champion")

    return {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
        "mae_v1_on_check": round(mae_v1_check, 4),
        "mae_v2_on_check": round(mae_v2_check, 4),
        "decision": decision,
        "champion_version": champion_version,
        "champion_run_id": champion_run_id,
    }


# ---------------------------------------------------------------------------
# Update findings.yaml
# ---------------------------------------------------------------------------

def update_findings(results: dict):
    """Write monitoring results back into findings.yaml."""
    with open(FINDINGS_PATH) as fh:
        findings = yaml.safe_load(fh)

    findings["monitoring"]["psi"] = results["psi"]
    findings["monitoring"]["most_drifted_feature"] = results["most_drifted_feature"]
    findings["monitoring"]["mae_by_group"] = results["mae_by_group"]
    findings["monitoring"]["worst_served_group"] = results["worst_served_group"]
    findings["monitoring"]["mae_v1_on_check"] = results["mae_v1_on_check"]
    findings["monitoring"]["mae_v2_on_check"] = results["mae_v2_on_check"]
    findings["monitoring"]["decision"] = results["decision"]

    findings["tracking"]["champion_version"] = results["champion_version"]

    with open(FINDINGS_PATH, "w") as fh:
        yaml.dump(findings, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nfindings.yaml updated at {FINDINGS_PATH}")


if __name__ == "__main__":
    results = run_evaluation()
    update_findings(results)
    print("\nEvaluation complete.")
    sys.exit(0)
