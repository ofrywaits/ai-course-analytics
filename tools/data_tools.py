# -*- coding: utf-8 -*-
"""
כלים לטעינה, ניקוי ועיבוד הדאטאסט של Udemy.
כל פונקציה אחראית על שלב אחד בלבד — קל לבדוק ולתחזק.
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
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. טעינה
# ──────────────────────────────────────────────

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """טוען את raw.csv ומחזיר DataFrame."""
    logger.info(f"טוען נתונים מ: {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"נטען בהצלחה — {len(df):,} שורות, {len(df.columns)} עמודות")
        return df
    except FileNotFoundError:
        logger.error(f"קובץ לא נמצא: {path}")
        raise
    except Exception as exc:
        logger.error(f"שגיאה בטעינה: {exc}")
        raise


# ──────────────────────────────────────────────
# 2. ניקוי
# ──────────────────────────────────────────────

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """מסיר עמודות שאינן רלוונטיות לניתוח."""
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"הוסרו עמודות: {cols_to_drop}")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """מתקן טיפוסי נתונים — מספרים, בוליאן, תאריכים."""
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_paid" in df.columns:
        df["is_paid"] = df["is_paid"].astype(bool)

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    logger.info("טיפוסי נתונים תוקנו")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """מטפל בערכים חסרים — ממלא או מסיר."""
    before = len(df)

    # מספריים — ממלא בחציון
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: ערכים חסרים מולאו בחציון ({median_val:.2f})")

    # קטגוריות — ממלא ב-"Unknown"
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    # תאריכים — מסיר שורות ללא תאריך פרסום
    if "published_time" in df.columns:
        df = df.dropna(subset=["published_time"])

    after = len(df)
    logger.info(f"לאחר טיפול בחסרים: {after:,} שורות (הוסרו {before - after:,})")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """מסיר שורות כפולות."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"הוסרו {before - after:,} כפילויות")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """מסיר outliers קיצוניים בעמודות מספריות (מעל 99.5 אחוזון)."""
    before = len(df)
    for col in ["price", "num_subscribers", "num_reviews"]:
        if col in df.columns:
            upper = df[col].quantile(0.995)
            df = df[df[col] <= upper]
    after = len(df)
    logger.info(f"הוסרו {before - after:,} outliers קיצוניים")
    return df


# ──────────────────────────────────────────────
# 3. הנדסת עמודות בסיסית
# ──────────────────────────────────────────────

def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """מוסיף עמודת מטרה: is_popular (1 = מעל חציון מנויים)."""
    threshold = POPULARITY_THRESHOLD or df["num_subscribers"].median()
    df[TARGET_COLUMN] = (df["num_subscribers"] > threshold).astype(int)
    logger.info(f"עמודת מטרה נוצרה — סף: {threshold:,.0f} מנויים")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """מוסיף עמודות זמן שימושיות מתאריך הפרסום."""
    if "published_time" in df.columns:
        df["publish_year"]  = df["published_time"].dt.year
        df["publish_month"] = df["published_time"].dt.month
        df["publish_day_of_week"] = df["published_time"].dt.dayofweek
        # גיל הקורס בשנים
        now = pd.Timestamp.now(tz="UTC")
        df["course_age_years"] = (
            (now - df["published_time"]).dt.days / 365.25
        ).round(2)
    logger.info("עמודות זמן נוספו")
    return df


# ──────────────────────────────────────────────
# 4. שמירה וחוזה
# ──────────────────────────────────────────────

def save_clean_data(df: pd.DataFrame, path: Path = CLEAN_DATA_PATH) -> None:
    """שומר את הדאטה הנקי."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"נתונים נקיים נשמרו: {path} ({len(df):,} שורות)")


def save_dataset_contract(df: pd.DataFrame, path: Path = DATASET_CONTRACT_PATH) -> None:
    """שומר חוזה JSON עם מטא-דאטה על הדאטאסט הנקי."""
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
    logger.info(f"חוזה דאטאסט נשמר: {path}")


# ──────────────────────────────────────────────
# 5. Pipeline ראשי
# ──────────────────────────────────────────────

def run_data_pipeline() -> pd.DataFrame:
    """מריץ את כל שלבי הניקוי ברצף ומחזיר DataFrame נקי."""
    logger.info("=== מתחיל Data Pipeline ===")

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

    logger.info(f"=== Pipeline הושלם — {len(df):,} שורות נקיות ===")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = run_data_pipeline()
    print("\n📊 תצוגה מקדימה:")
    print(df.head(3).to_string())
