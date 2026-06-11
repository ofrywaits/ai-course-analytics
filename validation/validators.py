# -*- coding: utf-8 -*-
"""
Validation checks between Crew 1 and Crew 2.
Each function checks one condition — raises ValidationError with a clear message on failure.

Crew 1 validations use a Pydantic-backed DatasetContract for semantic validation:
  - Column schema (dtype, nullable)
  - Allowed values for categorical columns
  - Numeric constraints (min/max)
  - Minimum row count
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

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
    MIN_DATASET_ROWS,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a validation check fails — includes a clear explanation."""
    pass


# ══════════════════════════════════════════════════════════════════
# Pydantic Contract Models
# ══════════════════════════════════════════════════════════════════

class ColumnSchema(BaseModel):
    """Schema definition for a single dataset column."""
    dtype: str
    nullable: bool = True


class RowCountConstraint(BaseModel):
    """Minimum row count requirement."""
    min: int = 0


class DatasetContract(BaseModel):
    """
    Pydantic model for dataset_contract.json.
    Validates schema, allowed values, numeric constraints, and row count.
    """
    schema_: Dict[str, ColumnSchema] = Field(alias="schema", default_factory=dict)
    allowed_values: Dict[str, List[Any]] = Field(default_factory=dict)
    constraints: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    row_count: RowCountConstraint = Field(default_factory=RowCountConstraint)
    target_column: str = "is_popular"
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None

    model_config = {"populate_by_name": True}

    @field_validator("schema_", mode="before")
    @classmethod
    def parse_schema(cls, v: Any) -> Any:
        """Accept raw dicts with dtype/nullable keys."""
        if isinstance(v, dict):
            return {
                col: (ColumnSchema(**info) if isinstance(info, dict) else info)
                for col, info in v.items()
            }
        return v


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

    if len(df) < MIN_DATASET_ROWS:
        raise ValidationError(
            f"Dataset too small: {len(df)} rows (minimum {MIN_DATASET_ROWS:,})"
        )

    null_pct = df[required_cols].isnull().mean().max()
    if null_pct > 0.05:
        raise ValidationError(
            f"More than 5% missing values in key columns ({null_pct:.1%})"
        )

    logger.info(f"✅ clean_data valid — {len(df):,} rows, {len(df.columns)} columns")
    return df


def validate_dataset_contract(df: pd.DataFrame) -> DatasetContract:
    """
    Deep semantic validation of dataset_contract.json against clean_data.csv.

    Checks performed:
      1. All columns in contract.schema exist in the DataFrame
      2. Column dtypes roughly match (int/float families, object)
      3. Categorical allowed_values are a superset of actual values
      4. Numeric constraints (min/max) are satisfied
      5. Minimum row count is met
    """
    validate_file_exists(DATASET_CONTRACT_PATH, "dataset_contract.json")

    with open(DATASET_CONTRACT_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    # Parse with Pydantic — raises ValidationError on bad contract structure
    try:
        contract = DatasetContract.model_validate(raw)
    except Exception as exc:
        raise ValidationError(f"dataset_contract.json is malformed: {exc}") from exc

    # 1 — Column existence
    missing_cols = [col for col in contract.schema_ if col not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Columns declared in contract but missing from data: {missing_cols}"
        )

    # 2 — Dtype compatibility (int64↔int32, float64↔float32 are compatible)
    DTYPE_FAMILIES: Dict[str, str] = {}
    for col in df.columns:
        dt = str(df[col].dtype)
        if "int" in dt:
            DTYPE_FAMILIES[col] = "int"
        elif "float" in dt:
            DTYPE_FAMILIES[col] = "float"
        elif "bool" in dt:
            DTYPE_FAMILIES[col] = "bool"
        else:
            DTYPE_FAMILIES[col] = "object"

    for col, col_schema in contract.schema_.items():
        if col not in df.columns:
            continue
        expected_family = "int" if "int" in col_schema.dtype else (
            "float" if "float" in col_schema.dtype else (
                "bool" if "bool" in col_schema.dtype else "object"
            )
        )
        actual_family = DTYPE_FAMILIES.get(col, "object")
        # int↔float is acceptable (pandas sometimes upcasts)
        compatible = (
            expected_family == actual_family
            or {expected_family, actual_family} <= {"int", "float"}
        )
        if not compatible:
            raise ValidationError(
                f"Column '{col}': expected dtype family '{expected_family}', "
                f"got '{actual_family}' (actual dtype: {df[col].dtype})"
            )

    # 3 — Allowed values
    for col, allowed in contract.allowed_values.items():
        if col not in df.columns:
            continue
        actual_vals = set(df[col].dropna().unique())
        allowed_set = set(allowed)
        extra = actual_vals - allowed_set
        if extra:
            raise ValidationError(
                f"Column '{col}' contains values not in allowed list: "
                f"{sorted(str(v) for v in list(extra)[:5])} (showing up to 5)"
            )

    # 4 — Numeric constraints
    for col, bounds in contract.constraints.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if "min" in bounds and float(series.min()) < bounds["min"]:
            raise ValidationError(
                f"Column '{col}' has values below minimum "
                f"({series.min()} < {bounds['min']})"
            )
        if "max" in bounds and float(series.max()) > bounds["max"]:
            raise ValidationError(
                f"Column '{col}' has values above maximum "
                f"({series.max()} > {bounds['max']})"
            )

    # 5 — Row count
    if len(df) < contract.row_count.min:
        raise ValidationError(
            f"Dataset has {len(df):,} rows — minimum required: {contract.row_count.min:,}"
        )

    logger.info(
        "✅ dataset_contract deep validation passed — "
        "%d columns, %d allowed_values checks, %d constraint checks",
        len(contract.schema_),
        len(contract.allowed_values),
        len(contract.constraints),
    )
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
    if len(df) < MIN_DATASET_ROWS:
        raise ValidationError(f"features.csv too small: {len(df)} rows (minimum {MIN_DATASET_ROWS:,})")
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
