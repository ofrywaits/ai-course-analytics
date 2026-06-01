# -*- coding: utf-8 -*-
"""
כלים לאימון, הערכה ושמירת מודלי ML.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    CLEAN_DATA_PATH,
    FEATURES_PATH,
    MODEL_PATH,
    EVALUATION_REPORT_PATH,
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """הנדסת features — encoding, scaling, ניקוי לפני אימון."""
    logger.info("מתחיל feature engineering...")
    df = df.copy()

    # מסיר עמודות לא שימושיות לאימון
    drop = ["title", "instructor_name", "published_time",
            "last_update_date", "num_subscribers"]
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

    # encoding לעמודות קטגוריות
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # is_paid → int
    if "is_paid" in df.columns:
        df["is_paid"] = df["is_paid"].astype(int)

    # מסיר עמודות עם NaN שנשארו
    df = df.dropna()

    logger.info(f"Features מוכנים — {len(df):,} שורות, {len(df.columns)} עמודות")
    return df


def save_features(df: pd.DataFrame) -> None:
    """שומר את ה-features לקובץ."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    logger.info(f"features.csv נשמר: {FEATURES_PATH}")


# ──────────────────────────────────────────────
# אימון מודלים
# ──────────────────────────────────────────────

def train_and_compare(df: pd.DataFrame) -> tuple:
    """מאמן Random Forest ו-Logistic Regression, בוחר את הטוב יותר."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # סקאלינג ל-Logistic Regression
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Random Forest
    logger.info("מאמן Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc  = accuracy_score(y_test, rf.predict(X_test))
    rf_cv   = cross_val_score(rf, X_train, y_train, cv=CV_FOLDS).mean()
    logger.info(f"Random Forest — test acc: {rf_acc:.4f}, CV: {rf_cv:.4f}")

    # Logistic Regression
    logger.info("מאמן Logistic Regression...")
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X_train_sc, y_train)
    lr_acc  = accuracy_score(y_test, lr.predict(X_test_sc))
    lr_cv   = cross_val_score(lr, X_train_sc, y_train, cv=CV_FOLDS).mean()
    logger.info(f"Logistic Regression — test acc: {lr_acc:.4f}, CV: {lr_cv:.4f}")

    # בחירת הטוב ביותר לפי CV
    if rf_cv >= lr_cv:
        best_model, best_name = rf, "Random Forest"
        best_acc, best_cv = rf_acc, rf_cv
        y_pred = rf.predict(X_test)
        feature_imp = dict(zip(X.columns, rf.feature_importances_))
    else:
        best_model, best_name = lr, "Logistic Regression"
        best_acc, best_cv = lr_acc, lr_cv
        y_pred = lr.predict(X_test_sc)
        feature_imp = {}

    logger.info(f"מודל נבחר: {best_name} (CV={best_cv:.4f})")

    metrics = {
        "best_model":    best_name,
        "accuracy":      round(best_acc, 4),
        "cv_score":      round(best_cv, 4),
        "rf_accuracy":   round(rf_acc, 4),
        "lr_accuracy":   round(lr_acc, 4),
        "report":        classification_report(y_test, y_pred),
        "confusion":     confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": dict(
            sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
        "X_columns":     list(X.columns),
    }

    return best_model, metrics


def save_model(model) -> None:
    """שומר את המודל לקובץ pkl."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"model.pkl נשמר: {MODEL_PATH}")


def save_evaluation_report(metrics: dict) -> None:
    """שומר דוח הערכה מפורט ב-Markdown."""
    fi_rows = "\n".join([
        f"| {k} | {v:.4f} |"
        for k, v in metrics["feature_importance"].items()
    ]) if metrics["feature_importance"] else "| N/A | N/A |"

    cm = metrics["confusion"]
    report = f"""# Evaluation Report — {metrics['best_model']}

## Results Summary

| Metric | Value |
|--------|-------|
| Best Model | {metrics['best_model']} |
| Test Accuracy | {metrics['accuracy']} |
| CV Score ({CV_FOLDS}-fold) | {metrics['cv_score']} |
| Random Forest Accuracy | {metrics['rf_accuracy']} |
| Logistic Regression Accuracy | {metrics['lr_accuracy']} |

## Classification Report

```
{metrics['report']}
```

## Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| Actual 0 | {cm[0][0]} | {cm[0][1]} |
| Actual 1 | {cm[1][0]} | {cm[1][1]} |

## Top 10 Feature Importances

| Feature | Importance |
|---------|-----------|
{fi_rows}
"""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    EVALUATION_REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info(f"evaluation_report.md נשמר: {EVALUATION_REPORT_PATH}")


def run_ml_pipeline() -> dict:
    """מריץ את כל pipeline ה-ML ומחזיר metrics."""
    logger.info("=== מתחיל ML Pipeline ===")
    df_clean   = pd.read_csv(CLEAN_DATA_PATH)
    df_features = build_features(df_clean)
    save_features(df_features)

    model, metrics = train_and_compare(df_features)
    save_model(model)
    save_evaluation_report(metrics)

    logger.info(f"=== ML Pipeline הושלם — accuracy: {metrics['accuracy']} ===")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    metrics = run_ml_pipeline()
    print(f"\n✅ accuracy: {metrics['accuracy']}")
    print(f"✅ מודל נבחר: {metrics['best_model']}")
