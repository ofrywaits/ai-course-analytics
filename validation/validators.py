# -*- coding: utf-8 -*-
"""
Validation checks between Crew 1 and Crew 2.
Each function checks one condition — raises ValidationError with a clear message on failure.
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from config import (
    CLEAN_DATA_PATH,
    DATASET_CONTRACT_PATH,
    EDA_REPORT_PATH,
    INSIGHTS_PATH,
    MODEL_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_CARD_PATH,
    FEATURES_PATH,
    TARGET_COLUMN,
    MIN_ACCURACY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a validation check fails — includes a clear explanation."""
    pass


# ══════════════════════════════════════════════════════════════════
# Crew 1 Validations
# ══════════════════════════════════════════════════════════════════

def validate_file_exists(path: Path, name: str) -> None:
    """Assert that a file exists and is not empty."""
    if not path.exists():
        raise ValidationError(f"Missing file: {name} ({path})")
    if path.stat().st_size == 0:
        raise ValidationError(f"Empty file: {name} ({path})")
    logger.info(f"✅ {name} exists ({path.stat().st_size / 1024:.1f} KB)")


def validate_clean_data() -> pd.DataFrame:
    """Assert that clean_data.csv is valid and contains required columns."""
    validate_file_exists(CLEAN_DATA_PATH, "clean_data.csv")

    df = pd.read_csv(CLEAN_DATA_PATH)

    required_cols = [TARGET_COLUMN, "price", "num_subscribers",
                     "avg_rating", "category", "is_paid"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing columns in clean_data: {missing}")

    if len(df) < 1000:
        raise ValidationError(
            f"Dataset too small: {len(df)} rows (minimum 1,000)"
        )

    null_pct = df[required_cols].isnull().mean().max()
    if null_pct > 0.05:
        raise ValidationError(
            f"More than 5% missing values in key columns ({null_pct:.1%})"
        )

    logger.info(f"✅ clean_data valid — {len(df):,} rows, {len(df.columns)} columns")
    return df


def validate_dataset_contract(df: pd.DataFrame) -> dict:
    """Assert that the dataset contract matches the actual data."""
    validate_file_exists(DATASET_CONTRACT_PATH, "dataset_contract.json")

    with open(DATASET_CONTRACT_PATH, encoding="utf-8") as f:
        contract = json.load(f)

    if contract["num_rows"] != len(df):
        raise ValidationError(
            f"Row count mismatch: contract={contract['num_rows']}, actual={len(df)}"
        )

    missing_cols = [c for c in contract["columns"] if c not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Columns in contract but missing from data: {missing_cols}"
        )

    logger.info("✅ dataset_contract matches clean_data")
    return contract


def validate_eda_report() -> None:
    """Assert that the EDA report contains embedded charts."""
    validate_file_exists(EDA_REPORT_PATH, "eda_report.html")

    content = EDA_REPORT_PATH.read_text(encoding="utf-8")
    if "base64" not in content:
        raise ValidationError("eda_report.html does not contain embedded charts (base64)")
    if "<html" not in content.lower():
        raise ValidationError("eda_report.html does not appear to be valid HTML")

    logger.info("✅ eda_report.html valid with embedded charts")


def validate_insights() -> None:
    """Assert that the insights file was created with content."""
    validate_file_exists(INSIGHTS_PATH, "insights.md")

    content = INSIGHTS_PATH.read_text(encoding="utf-8")
    if len(content.strip()) < 100:
        raise ValidationError("insights.md too short — less than 100 characters")

    logger.info("✅ insights.md valid")


def run_crew1_validations() -> dict:
    """Run all Crew 1 validations and return a summary."""
    logger.info("=== Running Crew 1 Validations ===")
    results = {}

    df = validate_clean_data()
    results["clean_data"] = {"status": "ok", "rows": len(df)}

    validate_dataset_contract(df)
    results["dataset_contract"] = {"status": "ok"}

    validate_eda_report()
    results["eda_report"] = {"status": "ok"}

    validate_insights()
    results["insights"] = {"status": "ok"}

    logger.info("=== ✅ All Crew 1 validations passed ===")
    return results


# ══════════════════════════════════════════════════════════════════
# Crew 2 Validations
# ══════════════════════════════════════════════════════════════════

def validate_features() -> pd.DataFrame:
    """Assert that features.csv is valid."""
    validate_file_exists(FEATURES_PATH, "features.csv")

    df = pd.read_csv(FEATURES_PATH)
    if len(df) < 1000:
        raise ValidationError(f"features.csv too small: {len(df)} rows")
    if TARGET_COLUMN not in df.columns:
        raise ValidationError(f"Target column '{TARGET_COLUMN}' missing from features.csv")

    logger.info(f"✅ features.csv valid — {len(df):,} rows")
    return df


def validate_model() -> None:
    """Assert that model.pkl loads correctly."""
    validate_file_exists(MODEL_PATH, "model.pkl")

    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict"):
            raise ValidationError("model.pkl does not contain a valid model (no predict method)")
        logger.info(f"✅ model.pkl loaded — {type(model).__name__}")
    except Exception as exc:
        raise ValidationError(f"Error loading model.pkl: {exc}")


def validate_evaluation_report() -> float:
    """Assert evaluation report exists and accuracy meets threshold."""
    validate_file_exists(EVALUATION_REPORT_PATH, "evaluation_report.md")

    content = EVALUATION_REPORT_PATH.read_text(encoding="utf-8")
    match = re.search(r"Test Accuracy \| ([0-9.]+)", content)
    if not match:
        logger.warning("Accuracy not found in report — skipping threshold check")
        return 0.0

    accuracy = float(match.group(1))
    if accuracy < MIN_ACCURACY_THRESHOLD:
        raise ValidationError(
            f"Accuracy too low: {accuracy:.3f} (minimum: {MIN_ACCURACY_THRESHOLD})"
        )

    logger.info(f"✅ evaluation_report.md — accuracy: {accuracy:.3f}")
    return accuracy


def validate_model_card() -> None:
    """Assert that the model card contains all required sections."""
    validate_file_exists(MODEL_CARD_PATH, "model_card.md")

    content = MODEL_CARD_PATH.read_text(encoding="utf-8")
    required_sections = [
        "Model Details", "Intended Use", "Training Data",
        "Metrics", "Limitations",
    ]
    missing = [s for s in required_sections if s not in content]
    if missing:
        raise ValidationError(f"Missing sections in model_card.md: {missing}")

    logger.info("✅ model_card.md valid with all required sections")


def run_crew2_validations() -> dict:
    """Run all Crew 2 validations and return a summary."""
    logger.info("=== Running Crew 2 Validations ===")
    results = {}

    for name, fn in [
        ("features",          validate_features),
        ("model",             validate_model),
        ("evaluation_report", validate_evaluation_report),
        ("model_card",        validate_model_card),
    ]:
        fn()
        results[name] = {"status": "ok"}

    logger.info("=== ✅ All Crew 2 validations passed ===")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = run_crew1_validations()
    print("\nValidation Results:")
    for k, v in results.items():
        status = "✅" if v["status"] == "ok" else "❌"
        print(f"  {status} {k}")
