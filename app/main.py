"""
app/main.py
---------------------------------------------------------------------------
Stage 3 — FastAPI Prediction Service for MP152 Cinema Show Occupancy.

Endpoints:
  GET  /health   → {"status": "ok"}
  POST /predict  → {"prediction": <float>}

Client rules (enforced in Pydantic schema, returns 422 on violation):
  - distance_to_metro_km <= 8.5
  - ticket_price_rs >= 50

JSON logging: one JSON line per prediction to stdout using Python's logging
module with a JSON formatter. No personal data is logged.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# JSON logging setup
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        return json.dumps(log_obj)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("mp152_api")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# MLflow model loading
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"

import yaml
with open(PARAMS_PATH) as fh:
    PARAMS = yaml.safe_load(fh)

MLFLOW_URI = PARAMS["paths"]["mlflow_db"]
MODEL_NAME = PARAMS["mlflow"]["registered_model_name"]
CHAMPION_ALIAS = PARAMS["mlflow"]["champion_alias"]
LEAKING_COL = PARAMS["data"]["leaking_column"]

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

_model = None


def get_model():
    global _model
    if _model is None:
        joblib_path = ROOT / PARAMS["paths"]["model_joblib"]
        if joblib_path.exists():
            _model = joblib.load(joblib_path)
            logger.info("Model loaded", extra={"extra": {"model_path": str(joblib_path)}})
        else:
            mlflow.set_tracking_uri(MLFLOW_URI)
            model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
            _model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded", extra={"extra": {"model_uri": model_uri}})
    return _model


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    ticket_price_rs: float = Field(
        ...,
        ge=50,
        description="Ticket price in rupees. Must be >= 50.",
    )
    days_since_release: int = Field(..., ge=0)
    star_rating: float = Field(..., ge=1.0, le=5.0)
    show_hour: int = Field(..., ge=0, le=23)
    screen_seats: int = Field(..., gt=0)
    distance_to_metro_km: float = Field(
        ...,
        le=8.5,
        description="Distance to nearest metro in km. Must be <= 8.5.",
    )
    language: str
    location: str

    model_config = {"json_schema_extra": {
        "example": {
            "ticket_price_rs": 350,
            "days_since_release": 14,
            "star_rating": 3.5,
            "show_hour": 18,
            "screen_seats": 250,
            "distance_to_metro_km": 2.0,
            "language": "hindi",
            "location": "Mall",
        }
    }}


class PredictResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MP152 Cinema Occupancy API",
    description="Predicts cinema show occupancy percentage.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    """Return service health status."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse, summary="Predict occupancy")
def predict(req: PredictRequest):
    """
    Predict occupancy_pct for a cinema show.

    Business rules (enforced in schema):
    - distance_to_metro_km must be <= 8.5
    - ticket_price_rs must be >= 50
    """
    model = get_model()

    # Build feature DataFrame
    row = {
        "ticket_price_rs": [req.ticket_price_rs],
        "days_since_release": [req.days_since_release],
        "star_rating": [req.star_rating],
        "show_hour": [req.show_hour],
        "screen_seats": [req.screen_seats],
        "distance_to_metro_km": [req.distance_to_metro_km],
        "language": [req.language.strip().lower()],
        "location": [req.location],
    }
    X = pd.DataFrame(row)[FEATURE_COLS]

    prediction = float(np.clip(model.predict(X)[0], 0, 100))

    # JSON log — no personal data
    logger.info(
        "prediction_made",
        extra={
            "extra": {
                "ticket_price_rs": req.ticket_price_rs,
                "days_since_release": req.days_since_release,
                "star_rating": req.star_rating,
                "show_hour": req.show_hour,
                "screen_seats": req.screen_seats,
                "distance_to_metro_km": req.distance_to_metro_km,
                "language": req.language,
                "location": req.location,
                "prediction": round(prediction, 4),
            }
        },
    )

    return PredictResponse(prediction=round(prediction, 4))
