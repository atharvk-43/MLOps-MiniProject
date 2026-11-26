"""
tests/test_api.py
---------------------------------------------------------------------------
Tests for the FastAPI prediction service.

Uses FastAPI TestClient — does NOT require CSV files or a live MLflow server.
The model is mocked so tests are self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Patch the model loading before importing app to avoid MLflow dependency
# ---------------------------------------------------------------------------

MOCK_MODEL = MagicMock()
MOCK_MODEL.predict.return_value = np.array([63.5])


@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    """Patch mlflow.sklearn.load_model so no MLflow server is needed."""
    with patch("app.main.get_model", return_value=MOCK_MODEL):
        yield


@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    # Reset cached model
    import app.main as main_module
    main_module._model = MOCK_MODEL

    from app.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_ok(client):
    """GET /health must return 200 with status 'ok'."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Valid prediction
# ---------------------------------------------------------------------------

def test_predict_valid(client):
    """POST /predict with valid payload must return 200 with a prediction."""
    payload = {
        "ticket_price_rs": 350,
        "days_since_release": 14,
        "star_rating": 3.5,
        "show_hour": 18,
        "screen_seats": 250,
        "distance_to_metro_km": 2.0,
        "language": "hindi",
        "location": "Mall",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
    assert 0.0 <= body["prediction"] <= 100.0


# ---------------------------------------------------------------------------
# Validation rules — 422 on violation
# ---------------------------------------------------------------------------

def test_predict_rejects_distance_too_high(client):
    """distance_to_metro_km > 8.5 must return 422."""
    payload = {
        "ticket_price_rs": 350,
        "days_since_release": 14,
        "star_rating": 3.5,
        "show_hour": 18,
        "screen_seats": 250,
        "distance_to_metro_km": 9.0,  # violates <= 8.5
        "language": "hindi",
        "location": "Mall",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_price_too_low(client):
    """ticket_price_rs < 50 must return 422."""
    payload = {
        "ticket_price_rs": 30,  # violates >= 50
        "days_since_release": 14,
        "star_rating": 3.5,
        "show_hour": 18,
        "screen_seats": 250,
        "distance_to_metro_km": 2.0,
        "language": "hindi",
        "location": "Mall",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    """Missing required field must return 422."""
    payload = {
        "ticket_price_rs": 350,
        # days_since_release missing
        "star_rating": 3.5,
        "show_hour": 18,
        "screen_seats": 250,
        "distance_to_metro_km": 2.0,
        "language": "hindi",
        "location": "Mall",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_boundary_distance_exactly_85(client):
    """distance_to_metro_km == 8.5 must be accepted (edge case)."""
    payload = {
        "ticket_price_rs": 350,
        "days_since_release": 14,
        "star_rating": 3.5,
        "show_hour": 18,
        "screen_seats": 250,
        "distance_to_metro_km": 8.5,  # exactly at boundary
        "language": "hindi",
        "location": "Mall",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_boundary_price_exactly_50(client):
    """ticket_price_rs == 50 must be accepted (edge case)."""
    payload = {
        "ticket_price_rs": 50,  # exactly at boundary
        "days_since_release": 14,
        "star_rating": 3.5,
        "show_hour": 18,
        "screen_seats": 250,
        "distance_to_metro_km": 2.0,
        "language": "hindi",
        "location": "Mall",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
