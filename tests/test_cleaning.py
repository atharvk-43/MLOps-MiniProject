"""
tests/test_cleaning.py
---------------------------------------------------------------------------
Tests for the data cleaning module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mp152_cinema.cleaning import (
    _normalize_language,
    audit_duplicate_rows,
    audit_sentinel_value,
    audit_spelling_variants,
    audit_unit_mixup,
    clean,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal dataframe that exercises all cleaning steps."""
    return pd.DataFrame({
        "record_id": ["R001", "R002", "R003", "R004", "R005", "R005"],
        "date": ["2026-01-01"] * 6,
        "ticket_price_rs": [300.0, 350.0, 25000.0, 25000.0, 400.0, 400.0],
        "days_since_release": [10, 9999, 20, 20, 15, 15],
        "star_rating": [3.5, 2.0, 4.0, 4.0, 1.5, 1.5],
        "show_hour": [14, 18, 20, 20, 10, 10],
        "screen_seats": [200, 150, 300, 300, 250, 250],
        "distance_to_metro_km": [1.5, 2.0, 3.0, 3.0, 4.0, 4.0],
        "language": ["english", "reg.", " Hindi", "hindi", "ENGLISH", "ENGLISH"],
        "location": ["Mall", "City Centre", "Highway", "Highway", "Mall", "Mall"],
        "occupancy_pct": [75.0, 60.0, 80.0, 80.0, 50.0, 50.0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_normalize_language():
    """Language normalization should map variants to 3 canonical values."""
    series = pd.Series(["ENGLISH", "eng.", " hindi", "reg.", "Regional", "english"])
    result = _normalize_language(series)
    assert set(result.unique()) == {"english", "hindi", "regional"}


def test_sentinel_detection(sample_df):
    """Sentinel value 9999 in days_since_release must be detected."""
    result = audit_sentinel_value(sample_df)
    assert result["present"] is True
    assert result["value"] == 9999
    assert result["count"] == 1


def test_duplicate_detection(sample_df):
    """Duplicate rows (excluding record_id) must be detected."""
    result = audit_duplicate_rows(sample_df)
    assert result["present"] is True
    assert result["count"] >= 1


def test_unit_mixup_detection(sample_df):
    """Unit mixup for Highway location should be detected."""
    result = audit_unit_mixup(sample_df)
    assert result["present"] is True
    assert result["column"] == "ticket_price_rs"
    assert result["group_value"] == "Highway"
    assert result["factor"] == 100


def test_spelling_variants_detection(sample_df):
    """Spelling variants in language should be detected with 3 true levels."""
    result = audit_spelling_variants(sample_df)
    assert result["present"] is True
    assert result["true_levels"] == 3


def test_clean_removes_sentinel(sample_df):
    """Cleaned df must not contain sentinel value 9999 in days_since_release."""
    cleaned = clean(sample_df)
    assert (cleaned["days_since_release"] == 9999).sum() == 0


def test_clean_removes_duplicates(sample_df):
    """Cleaned df must not contain duplicate rows."""
    cleaned = clean(sample_df)
    no_id = cleaned.drop(columns=["record_id"])
    assert no_id.duplicated().sum() == 0


def test_clean_fixes_unit_mixup(sample_df):
    """Highway ticket prices should be divided by 100 after cleaning."""
    cleaned = clean(sample_df)
    highway = cleaned[cleaned["location"] == "Highway"]["ticket_price_rs"]
    assert all(highway < 1000), f"Highway prices still too high: {highway.values}"


def test_clean_preserves_valid_columns(sample_df):
    """Valid feature columns including distance_to_metro_km must be preserved."""
    cleaned = clean(sample_df)
    assert "distance_to_metro_km" in cleaned.columns


def test_clean_normalizes_language(sample_df):
    """After cleaning, language column should have only 3 canonical values."""
    cleaned = clean(sample_df)
    valid = {"english", "hindi", "regional"}
    assert set(cleaned["language"].unique()).issubset(valid)
