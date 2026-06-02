# -*- coding: utf-8 -*-
"""
Unit tests for tools/model_tools.py — ML pipeline functions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def features_df() -> pd.DataFrame:
    """Minimal feature dataframe that mirrors the engineered features schema."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "price":               np.random.uniform(0, 200, n),
        "avg_rating":          np.random.uniform(2.0, 5.0, n),
        "num_reviews":         np.random.randint(0, 5000, n),
        "num_lectures":        np.random.randint(5, 300, n),
        "content_length_min":  np.random.randint(60, 3000, n),
        "is_paid":             np.random.randint(0, 2, n),
        "category":            np.random.randint(0, 10, n),
        "subcategory":         np.random.randint(0, 50, n),
        "language":            np.random.randint(0, 15, n),
        "level":               np.random.randint(0, 4, n),
        "publish_year":        np.random.randint(2012, 2024, n),
        "publish_month":       np.random.randint(1, 13, n),
        "is_popular":          np.random.randint(0, 2, n),
    })


# ── Tests: build_features ────────────────────────────────────────────────────

class TestBuildFeatures:
    """Tests for the feature engineering function."""

    def test_target_column_preserved(self, features_df):
        from tools.model_tools import build_features
        result = build_features(features_df.copy())
        assert "is_popular" in result.columns, "Target column must be preserved"

    def test_drop_columns_removed(self):
        from tools.model_tools import build_features
        df = pd.DataFrame({
            "title":            ["Python Course"],
            "instructor_name":  ["Alice"],
            "published_time":   ["2020-01-01"],
            "last_update_date": ["2023-01-01"],
            "num_subscribers":  [10000],
            "price":            [49.99],
            "avg_rating":       [4.5],
            "is_popular":       [1],
        })
        result = build_features(df)
        for col in ["title", "instructor_name", "published_time",
                    "last_update_date", "num_subscribers"]:
            assert col not in result.columns, f"Column '{col}' should be dropped"

    def test_no_nulls_in_output(self, features_df):
        from tools.model_tools import build_features
        result = build_features(features_df.copy())
        assert result.isna().sum().sum() == 0, "Output should have no nulls"

    def test_returns_dataframe(self, features_df):
        from tools.model_tools import build_features
        result = build_features(features_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_original_not_mutated(self, features_df):
        from tools.model_tools import build_features
        original_cols = list(features_df.columns)
        build_features(features_df.copy())
        assert list(features_df.columns) == original_cols, \
            "build_features should not mutate the input dataframe"


# ── Tests: train_and_compare ─────────────────────────────────────────────────

class TestTrainAndCompare:
    """Tests for the model training and comparison function."""

    def test_returns_model_and_metrics(self, features_df):
        from tools.model_tools import build_features, train_and_compare
        df = build_features(features_df.copy())
        model, metrics = train_and_compare(df)
        assert model is not None, "Should return a fitted model"
        assert isinstance(metrics, dict), "Should return a metrics dict"

    def test_metrics_has_required_keys(self, features_df):
        from tools.model_tools import build_features, train_and_compare
        df = build_features(features_df.copy())
        _, metrics = train_and_compare(df)
        required = ["best_model", "accuracy", "cv_score", "report", "confusion",
                    "feature_importance", "rf_accuracy", "lr_accuracy"]
        for key in required:
            assert key in metrics, f"Metrics dict missing key: '{key}'"

    def test_accuracy_in_valid_range(self, features_df):
        from tools.model_tools import build_features, train_and_compare
        df = build_features(features_df.copy())
        _, metrics = train_and_compare(df)
        assert 0.0 <= metrics["accuracy"] <= 1.0, \
            f"Accuracy {metrics['accuracy']} is out of [0, 1] range"

    def test_best_model_is_valid_name(self, features_df):
        from tools.model_tools import build_features, train_and_compare
        df = build_features(features_df.copy())
        _, metrics = train_and_compare(df)
        assert metrics["best_model"] in ("Random Forest", "Logistic Regression"), \
            f"Unexpected model name: {metrics['best_model']}"

    def test_confusion_matrix_shape(self, features_df):
        from tools.model_tools import build_features, train_and_compare
        df = build_features(features_df.copy())
        _, metrics = train_and_compare(df)
        cm = metrics["confusion"]
        assert len(cm) == 2 and len(cm[0]) == 2, \
            "Confusion matrix should be 2×2 for binary classification"


# ── Tests: save_model ────────────────────────────────────────────────────────

class TestSaveModel:
    """Tests for the model persistence function."""

    def test_model_file_created(self, tmp_path, features_df):
        from tools.model_tools import build_features, train_and_compare
        from sklearn.ensemble import RandomForestClassifier
        import joblib

        df = build_features(features_df.copy())
        model, _ = train_and_compare(df)

        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)
        assert model_path.exists(), "model.pkl should be created"

    def test_saved_model_is_loadable(self, tmp_path, features_df):
        from tools.model_tools import build_features, train_and_compare
        import joblib

        df = build_features(features_df.copy())
        model, _ = train_and_compare(df)

        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)
        assert hasattr(loaded, "predict"), "Loaded model should have .predict()"
