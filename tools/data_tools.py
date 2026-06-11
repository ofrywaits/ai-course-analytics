# -*- coding: utf-8 -*-
"""
Tools for loading, cleaning, and processing the Udemy dataset.
Each function is responsible for exactly one step — easy to test and maintain.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    DATASET_CONTRACT_PATH,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    DROP_COLUMNS,
    TARGET_COLUMN,
    POPULARITY_THRESHOLD,
    OUTLIER_QUANTILE,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw.csv and return a DataFrame."""
    logger.info(f"Loading data from: {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded successfully — {len(df):,} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as exc:
        logger.error(f"Load error: {exc}")
        raise


# ── 2. Clean ─────────────────────────────────────────────────────────────────

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not relevant for analysis."""
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped columns: {cols_to_drop}")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data types — numeric, boolean, datetime."""
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_paid" in df.columns:
        df["is_paid"] = df["is_paid"].astype(bool)

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    logger.info("Data types fixed")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values — fill or drop."""
    before = len(df)

    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: filled with median ({median_val:.2f})")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    if "published_time" in df.columns:
        df = df.dropna(subset=["published_time"])

    after = len(df)
    logger.info(f"After missing value handling: {after:,} rows (removed {before - after:,})")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df):,} duplicates")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers above the 99.5th percentile."""
    before = len(df)
    for col in ["price", "num_subscribers", "num_reviews"]:
        if col in df.columns:
            upper = df[col].quantile(OUTLIER_QUANTILE)
            df = df[df[col] <= upper]
    logger.info(f"Removed {before - len(df):,} extreme outliers")
    return df


# ── 3. Feature Engineering ───────────────────────────────────────────────────

def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target: is_popular (1 = above median subscribers)."""
    threshold = POPULARITY_THRESHOLD or df["num_subscribers"].median()
    df[TARGET_COLUMN] = (df["num_subscribers"] > threshold).astype(int)
    logger.info(f"Target column created — threshold: {threshold:,.0f} subscribers")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the publish date."""
    if "published_time" in df.columns:
        df["publish_year"]        = df["published_time"].dt.year
        df["publish_month"]       = df["published_time"].dt.month
        df["publish_day_of_week"] = df["published_time"].dt.dayofweek
        now = pd.Timestamp.now(tz="UTC")
        df["course_age_years"] = (
            (now - df["published_time"]).dt.days / 365.25
        ).round(2)
    logger.info("Time features added")
    return df


# ── 4. Save ──────────────────────────────────────────────────────────────────

def save_clean_data(df: pd.DataFrame, path: Path = CLEAN_DATA_PATH) -> None:
    """Save the clean dataset."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Clean data saved: {path} ({len(df):,} rows)")


def save_dataset_contract(df: pd.DataFrame, path: Path = DATASET_CONTRACT_PATH) -> None:
    """Save a JSON contract with metadata about the clean dataset."""
    contract = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=2)
    logger.info(f"Dataset contract saved: {path}")


# ── 5. Main Pipeline ─────────────────────────────────────────────────────────

def run_data_pipeline() -> pd.DataFrame:
    """Run all cleaning steps in sequence and return clean DataFrame."""
    logger.info("=== Starting Data Pipeline ===")

    df = load_raw_data()
    df = drop_irrelevant_columns(df)
    df = fix_dtypes(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = add_target_column(df)
    df = add_time_features(df)
    save_clean_data(df)
    save_dataset_contract(df)

    logger.info(f"=== Pipeline complete — {len(df):,} clean rows ===")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = run_data_pipeline()
    print("\nPreview:")
    print(df.head(3).to_string())
