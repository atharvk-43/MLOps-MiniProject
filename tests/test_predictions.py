"""
tests/test_predictions.py
---------------------------------------------------------------------------
Tests for the predictions output and PSI computation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mp152_cinema.evaluation import compute_psi

# ---------------------------------------------------------------------------
# PSI tests
# ---------------------------------------------------------------------------

def test_psi_identical_distributions():
    """PSI of identical distributions should be near 0."""
    rng = np.random.default_rng(42)
    data = rng.normal(50, 10, 1000)
    psi = compute_psi(data, data.copy(), bins=10)
    assert psi < 0.01, f"Expected PSI ≈ 0 for identical distributions, got {psi:.4f}"


def test_psi_very_different_distributions():
    """PSI between very different distributions should be > 0.2."""
    rng = np.random.default_rng(42)
    ref = rng.normal(50, 5, 1000)
    cmp = rng.normal(80, 5, 1000)
    psi = compute_psi(ref, cmp, bins=10)
    assert psi > 0.2, f"Expected high PSI for distinct distributions, got {psi:.4f}"


def test_psi_non_negative():
    """PSI must always be non-negative."""
    rng = np.random.default_rng(99)
    ref = rng.exponential(scale=5, size=500)
    cmp = rng.exponential(scale=8, size=500)
    psi = compute_psi(ref, cmp, bins=10)
    assert psi >= 0.0, f"PSI must be non-negative, got {psi:.4f}"


# ---------------------------------------------------------------------------
# Predictions output tests (only run if the file exists)
# ---------------------------------------------------------------------------

PREDICTIONS_FILE = ROOT / "predictions" / "holdout_predictions.csv"


@pytest.mark.skipif(
    not PREDICTIONS_FILE.exists(),
    reason="holdout_predictions.csv not yet generated",
)
def test_predictions_file_exists():
    """predictions/holdout_predictions.csv must exist after pipeline run."""
    assert PREDICTIONS_FILE.exists()


@pytest.mark.skipif(
    not PREDICTIONS_FILE.exists(),
    reason="holdout_predictions.csv not yet generated",
)
def test_predictions_columns():
    """Predictions CSV must have exactly: record_id, prediction."""
    df = pd.read_csv(PREDICTIONS_FILE)
    assert list(df.columns) == ["record_id", "prediction"], \
        f"Unexpected columns: {list(df.columns)}"


@pytest.mark.skipif(
    not PREDICTIONS_FILE.exists(),
    reason="holdout_predictions.csv not yet generated",
)
def test_predictions_in_valid_range():
    """All predictions must be in [0, 100]."""
    df = pd.read_csv(PREDICTIONS_FILE)
    assert df["prediction"].between(0, 100).all(), \
        "Some predictions are outside [0, 100]"


@pytest.mark.skipif(
    not PREDICTIONS_FILE.exists(),
    reason="holdout_predictions.csv not yet generated",
)
def test_predictions_row_count():
    """Predictions must have exactly 250 rows (holdout has 250 rows)."""
    df = pd.read_csv(PREDICTIONS_FILE)
    assert len(df) == 250, f"Expected 250 rows, got {len(df)}"
