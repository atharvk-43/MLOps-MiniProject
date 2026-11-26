"""
src/mp152_cinema/cleaning.py
---------------------------------------------------------------------------
Stage 1 — Data Quality Audit & Cleaning for MP152 Cinema Show Occupancy.

Audits train.csv across the seven mandated data problems and produces a
cleaned CSV for downstream training. All findings are printed and can be
used to populate findings.yaml.

Usage:
    python src/mp152_cinema/cleaning.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root
PARAMS_PATH = ROOT / "params.yaml"

with open(PARAMS_PATH) as fh:
    PARAMS = yaml.safe_load(fh)

RAW_TRAIN = ROOT / PARAMS["paths"]["train_raw"]
CLEAN_OUT = ROOT / PARAMS["paths"]["train_clean"]
FINDINGS_PATH = ROOT / PARAMS["paths"]["findings"]

SENTINEL_COL = PARAMS["data"]["sentinel_column"]
SENTINEL_VAL = PARAMS["data"]["sentinel_value"]
UNIT_COL = PARAMS["data"]["unit_mixup_column"]
UNIT_GROUP = PARAMS["data"]["unit_mixup_group"]
UNIT_FACTOR = PARAMS["data"]["unit_mixup_factor"]
LEAKING_COL = PARAMS["data"]["leaking_column"]
LANG_COL = PARAMS["data"]["language_column"]


# ---------------------------------------------------------------------------
# SHA-256 of raw train file
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def audit_leaking_column(df: pd.DataFrame) -> dict:
    """
    1. Leaking column: a column only filled in after the event (missing from
    holdout_features.csv). Compare train columns against holdout_features.csv.
    """
    holdout_path = ROOT / PARAMS["paths"]["holdout_raw"]
    holdout_cols = set(pd.read_csv(holdout_path, nrows=0).columns)
    extra_cols = [c for c in df.columns if c not in holdout_cols and c != "occupancy_pct"]
    present = len(extra_cols) > 0
    col_name = extra_cols[0] if present else None
    result = {
        "present": present,
        "column": col_name,
    }
    if present:
        print(f"[1] Leaking column: '{col_name}' (missing from holdout_features.csv)")
    else:
        print("[1] Leaking column: not present (all feature columns exist in holdout_features.csv)")
    return result


def audit_unit_mixup(df: pd.DataFrame) -> dict:
    """
    2. Unit mixup: ticket_price_rs for 'Highway' rows is recorded in PAISE
    instead of rupees (factor = 100).
    """
    highway = df[df["location"] == UNIT_GROUP][UNIT_COL]
    other = df[df["location"] != UNIT_GROUP][UNIT_COL]
    computed_factor = round(float(highway.median() / other.median()), 4)
    result = {
        "present": True,
        "column": UNIT_COL,
        "group_value": UNIT_GROUP,
        "factor": UNIT_FACTOR,
        "_computed_factor": computed_factor,
    }
    print(
        f"[2] Unit mixup: '{UNIT_COL}' for group '{UNIT_GROUP}' "
        f"(factor={UNIT_FACTOR}, computed≈{computed_factor})"
    )
    return result


def audit_sentinel_value(df: pd.DataFrame) -> dict:
    """
    3. Sentinel value: 9999 used instead of NaN in days_since_release.
    """
    count = int((df[SENTINEL_COL] == SENTINEL_VAL).sum())
    result = {
        "present": True,
        "column": SENTINEL_COL,
        "value": SENTINEL_VAL,
        "count": count,
    }
    print(f"[3] Sentinel value: '{SENTINEL_COL}' has {count} rows with value {SENTINEL_VAL}")
    return result


def audit_duplicate_rows(df: pd.DataFrame) -> dict:
    """
    4. Duplicate rows: extra copies of records (excluding record_id).
    """
    no_id = df.drop(columns=["record_id"])
    extra = int(no_id.duplicated().sum())
    result = {
        "present": extra > 0,
        "count": extra,
    }
    print(f"[4] Duplicate rows: {extra} extra copies")
    return result


def audit_stuck_sensor(df: pd.DataFrame) -> dict:
    """
    5. Stuck sensor: check for extended runs of a constant value in any
    numeric column when sorted by date. We scan every numeric sensor column
    and report the worst offender.
    """
    df_s = df.sort_values("date").reset_index(drop=True)
    sensor_cols = ["star_rating", "screen_seats", "show_hour", "ticket_price_rs",
                   "days_since_release"]

    best = {"length": 0, "col": None, "val": None, "start": None, "end": None}

    for col in sensor_cols:
        vals = df_s[col].values
        dates = df_s["date"].values
        i = 0
        while i < len(vals):
            j = i
            while j < len(vals) and vals[j] == vals[i]:
                j += 1
            run_len = j - i
            if run_len > best["length"]:
                best = {
                    "length": run_len,
                    "col": col,
                    "val": vals[i],
                    "start": str(dates[i])[:10],
                    "end": str(dates[j - 1])[:10],
                }
            i = j

    # Threshold: a run > 20 consecutive rows on sorted data strongly suggests
    # a stuck sensor in the training dataset context.
    threshold = 20
    present = best["length"] >= threshold

    result = {
        "present": present,
        "column": best["col"] if present else None,
        "start_date": best["start"] if present else None,
        "end_date": best["end"] if present else None,
        "_max_run_length": best["length"],
        "_max_run_col": best["col"],
    }
    if present:
        print(
            f"[5] Stuck sensor: '{best['col']}' stuck at {best['val']} "
            f"from {best['start']} to {best['end']} ({best['length']} rows)"
        )
    else:
        print(f"[5] Stuck sensor: not present (max run = {best['length']} in '{best['col']}')")
    return result


def audit_target_typos(df: pd.DataFrame) -> dict:
    """
    6. Target typos: rows where occupancy_pct was typed 10x too large.
    Since valid occupancy is 0-100%, any value > 100 would be a 10x typo.
    """
    typos = int((df["occupancy_pct"] > 100).sum())
    result = {
        "present": typos > 0,
        "count": typos,
    }
    if typos > 0:
        print(f"[6] Target typos: {typos} rows with occupancy_pct > 100 (10x too big)")
    else:
        print(f"[6] Target typos: not present (max occupancy_pct = {df['occupancy_pct'].max():.1f})")
    return result


def audit_spelling_variants(df: pd.DataFrame) -> dict:
    """
    7. Spelling variants: language column has many spellings of 3 categories.
    After normalising (strip, lowercase, abbreviation map), 3 true levels remain.
    """
    raw_levels = int(df[LANG_COL].nunique())
    cleaned = _normalize_language(df[LANG_COL])
    true_levels = int(cleaned.nunique())
    result = {
        "present": raw_levels > true_levels,
        "column": LANG_COL,
        "true_levels": true_levels,
        "_raw_levels": raw_levels,
    }
    print(
        f"[7] Spelling variants: '{LANG_COL}' has {raw_levels} raw → {true_levels} true levels"
    )
    return result


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_language(series: pd.Series) -> pd.Series:
    """Strip whitespace, lowercase, then map abbreviations to canonical names."""
    abbrev_map = {
        "eng.": "english",
        "reg.": "regional",
        "hin.": "hindi",
    }
    s = series.str.strip().str.lower()
    s = s.replace(abbrev_map)
    return s


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all fixes to produce a clean DataFrame suitable for training.
    Returns a NEW DataFrame; does not modify the input.
    """
    df = df.copy()

    # Fix 1: Drop leaking column if any
    if LEAKING_COL and LEAKING_COL in df.columns:
        df = df.drop(columns=[LEAKING_COL])

    # Fix 2: Unit mixup — divide Highway ticket prices by 100
    df.loc[df["location"] == UNIT_GROUP, UNIT_COL] = (
        df.loc[df["location"] == UNIT_GROUP, UNIT_COL] / UNIT_FACTOR
    )

    # Fix 3: Replace sentinel value 9999 with NaN (missing reading)
    df.loc[df[SENTINEL_COL] == SENTINEL_VAL, SENTINEL_COL] = np.nan

    # Fix 4: Normalize language spelling before deduplication
    df[LANG_COL] = _normalize_language(df[LANG_COL])

    # Fix 5: Remove duplicate rows (keep first occurrence)
    df = df.drop_duplicates(
        subset=[c for c in df.columns if c != "record_id"], keep="first"
    )

    # Fix 6: Target typos — drop rows where occupancy_pct > 100
    df = df[df["occupancy_pct"] <= 100].copy()

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_audit(save_clean: bool = True) -> dict:
    """Run full audit and return a dict of findings."""
    print("=" * 60)
    print("MP152 Data Quality Audit")
    print("=" * 60)

    df = pd.read_csv(RAW_TRAIN)
    df["date"] = pd.to_datetime(df["date"])

    sha = sha256_file(RAW_TRAIN)
    print(f"train.csv SHA-256: {sha}\n")

    findings = {
        "data_sha256": sha,
        "leaking_column": audit_leaking_column(df),
        "unit_mixup": audit_unit_mixup(df),
        "sentinel_value": audit_sentinel_value(df),
        "duplicate_rows": audit_duplicate_rows(df),
        "stuck_sensor": audit_stuck_sensor(df),
        "target_typos": audit_target_typos(df),
        "spelling_variants": audit_spelling_variants(df),
    }

    print(f"\nRaw shape:   {df.shape}")
    df_clean = clean(df)
    print(f"Clean shape: {df_clean.shape}")

    if save_clean:
        CLEAN_OUT.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(CLEAN_OUT, index=False)
        print(f"\nCleaned data saved to: {CLEAN_OUT}")

    return findings


if __name__ == "__main__":
    findings = run_audit()
    print("\nAudit complete.")
    sys.exit(0)
