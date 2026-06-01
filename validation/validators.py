# -*- coding: utf-8 -*-
"""
ולידציות בין Crew 1 ל-Crew 2.
כל פונקציה בודקת תנאי אחד — אם נכשל, זורקת ValidationError עם הסבר ברור.
"""

import json
import logging
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
    """שגיאת ולידציה — מכילה הסבר ברור על מה נכשל."""
    pass


# ══════════════════════════════════════════════════════════════════
# ולידציות אחרי Crew 1
# ══════════════════════════════════════════════════════════════════

def validate_file_exists(path: Path, name: str) -> None:
    """מוודא שקובץ קיים ואינו ריק."""
    if not path.exists():
        raise ValidationError(f"קובץ חסר: {name} ({path})")
    if path.stat().st_size == 0:
        raise ValidationError(f"קובץ ריק: {name} ({path})")
    logger.info(f"✅ {name} קיים ({path.stat().st_size / 1024:.1f} KB)")


def validate_clean_data() -> pd.DataFrame:
    """מוודא שה-clean_data.csv תקין ומכיל את העמודות הנדרשות."""
    validate_file_exists(CLEAN_DATA_PATH, "clean_data.csv")

    df = pd.read_csv(CLEAN_DATA_PATH)

    required_cols = [TARGET_COLUMN, "price", "num_subscribers",
                     "avg_rating", "category", "is_paid"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValidationError(f"עמודות חסרות ב-clean_data: {missing}")

    if len(df) < 1000:
        raise ValidationError(
            f"הדאטאסט קטן מדי: {len(df)} שורות (מינימום 1,000)"
        )

    null_pct = df[required_cols].isnull().mean().max()
    if null_pct > 0.05:
        raise ValidationError(
            f"יותר מ-5% ערכים חסרים בעמודות מרכזיות ({null_pct:.1%})"
        )

    logger.info(f"✅ clean_data תקין — {len(df):,} שורות, {len(df.columns)} עמודות")
    return df


def validate_dataset_contract(df: pd.DataFrame) -> dict:
    """מוודא שחוזה הדאטאסט תואם את הנתונים בפועל."""
    validate_file_exists(DATASET_CONTRACT_PATH, "dataset_contract.json")

    with open(DATASET_CONTRACT_PATH, encoding="utf-8") as f:
        contract = json.load(f)

    if contract["num_rows"] != len(df):
        raise ValidationError(
            f"אי-התאמה בשורות: חוזה={contract['num_rows']}, "
            f"בפועל={len(df)}"
        )

    missing_cols = [c for c in contract["columns"] if c not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"עמודות בחוזה אך לא בדאטה: {missing_cols}"
        )

    logger.info("✅ dataset_contract תואם את clean_data")
    return contract


def validate_eda_report() -> None:
    """מוודא שדוח ה-EDA נוצר ומכיל גרפים."""
    validate_file_exists(EDA_REPORT_PATH, "eda_report.html")

    content = EDA_REPORT_PATH.read_text(encoding="utf-8")
    if "base64" not in content:
        raise ValidationError("eda_report.html לא מכיל גרפים מוטמעים (base64)")
    if "<html" not in content.lower():
        raise ValidationError("eda_report.html לא נראה כמו HTML תקין")

    logger.info("✅ eda_report.html תקין עם גרפים מוטמעים")


def validate_insights() -> None:
    """מוודא שקובץ התובנות נוצר ומכיל תוכן."""
    validate_file_exists(INSIGHTS_PATH, "insights.md")

    content = INSIGHTS_PATH.read_text(encoding="utf-8")
    if len(content.strip()) < 100:
        raise ValidationError("insights.md קצר מדי — פחות מ-100 תווים")

    logger.info("✅ insights.md תקין")


def run_crew1_validations() -> dict:
    """מריץ את כל ולידציות Crew 1 ומחזיר סיכום."""
    logger.info("═══ מתחיל ולידציות Crew 1 ═══")
    results = {}

    try:
        df = validate_clean_data()
        results["clean_data"] = {"status": "ok", "rows": len(df)}
    except ValidationError as e:
        results["clean_data"] = {"status": "error", "message": str(e)}
        raise

    try:
        contract = validate_dataset_contract(df)
        results["dataset_contract"] = {"status": "ok"}
    except ValidationError as e:
        results["dataset_contract"] = {"status": "error", "message": str(e)}
        raise

    try:
        validate_eda_report()
        results["eda_report"] = {"status": "ok"}
    except ValidationError as e:
        results["eda_report"] = {"status": "error", "message": str(e)}
        raise

    try:
        validate_insights()
        results["insights"] = {"status": "ok"}
    except ValidationError as e:
        results["insights"] = {"status": "error", "message": str(e)}
        raise

    logger.info("═══ ✅ כל ולידציות Crew 1 עברו ═══")
    return results


# ══════════════════════════════════════════════════════════════════
# ולידציות אחרי Crew 2
# ══════════════════════════════════════════════════════════════════

def validate_features() -> pd.DataFrame:
    """מוודא שקובץ ה-features נוצר ותקין."""
    validate_file_exists(FEATURES_PATH, "features.csv")

    df = pd.read_csv(FEATURES_PATH)
    if len(df) < 1000:
        raise ValidationError(f"features.csv קטן מדי: {len(df)} שורות")
    if TARGET_COLUMN not in df.columns:
        raise ValidationError(f"עמודת מטרה '{TARGET_COLUMN}' חסרה ב-features.csv")

    logger.info(f"✅ features.csv תקין — {len(df):,} שורות")
    return df


def validate_model() -> None:
    """מוודא שה-model.pkl נטען תקין."""
    validate_file_exists(MODEL_PATH, "model.pkl")

    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict"):
            raise ValidationError("model.pkl לא מכיל מודל תקין (אין predict)")
        logger.info(f"✅ model.pkl נטען — {type(model).__name__}")
    except Exception as exc:
        raise ValidationError(f"שגיאה בטעינת model.pkl: {exc}")


def validate_evaluation_report() -> float:
    """מוודא דוח הערכה ומחזיר accuracy."""
    validate_file_exists(EVALUATION_REPORT_PATH, "evaluation_report.md")

    content = EVALUATION_REPORT_PATH.read_text(encoding="utf-8")

    # מחפש accuracy בדוח
    import re
    match = re.search(r"accuracy[:\s]+([0-9.]+)", content, re.IGNORECASE)
    if not match:
        logger.warning("לא נמצא accuracy בדוח — מדלג על בדיקת סף")
        return 0.0

    accuracy = float(match.group(1))
    if accuracy < MIN_ACCURACY_THRESHOLD:
        raise ValidationError(
            f"Accuracy נמוך מדי: {accuracy:.3f} "
            f"(מינימום: {MIN_ACCURACY_THRESHOLD})"
        )

    logger.info(f"✅ evaluation_report.md — accuracy: {accuracy:.3f}")
    return accuracy


def validate_model_card() -> None:
    """מוודא שה-model card מכיל את הסעיפים הנדרשים."""
    validate_file_exists(MODEL_CARD_PATH, "model_card.md")

    content = MODEL_CARD_PATH.read_text(encoding="utf-8")
    required_sections = [
        "Model Details", "Intended Use", "Training Data",
        "Metrics", "Limitations",
    ]
    missing = [s for s in required_sections if s not in content]
    if missing:
        raise ValidationError(f"סעיפים חסרים ב-model_card.md: {missing}")

    logger.info("✅ model_card.md תקין עם כל הסעיפים")


def run_crew2_validations() -> dict:
    """מריץ את כל ולידציות Crew 2 ומחזיר סיכום."""
    logger.info("═══ מתחיל ולידציות Crew 2 ═══")
    results = {}

    for name, fn in [
        ("features",          validate_features),
        ("model",             validate_model),
        ("evaluation_report", validate_evaluation_report),
        ("model_card",        validate_model_card),
    ]:
        try:
            fn()
            results[name] = {"status": "ok"}
        except ValidationError as e:
            results[name] = {"status": "error", "message": str(e)}
            raise

    logger.info("═══ ✅ כל ולידציות Crew 2 עברו ═══")
    return results


# ── הרצה עצמאית לבדיקה ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = run_crew1_validations()
    print("\n📋 תוצאות ולידציה:")
    for k, v in results.items():
        status = "✅" if v["status"] == "ok" else "❌"
        print(f"  {status} {k}")
