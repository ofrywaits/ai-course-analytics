# -*- coding: utf-8 -*-
"""
Integration tests — verify the full data → features → model pipeline
works end-to-end without running the LLM agents.

These tests:
1. Run the Python data pipeline (data_tools) step-by-step
2. Run the Python ML pipeline (model_tools) end-to-end
3. Verify outputs are consistent with each other
4. Verify the saved model can predict on new data

They do NOT call CrewAI agents or the Groq API.
"""

import sys
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib
import unittest.mock as mock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Shared synthetic dataset ──────────────────────────────────────────────────

def _make_raw_df(n: int = 1200) -> pd.DataFrame:
    """
    Build a synthetic DataFrame with real learnable patterns.
    Popular courses: English, Development/IT, high rating, 50-100 lectures.
    Unpopular: non-English, low rating, very few or very many lectures.
    """
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=n, freq="3D", tz="UTC")

    # Half popular (high subscribers), half not
    half = n // 2
    subscribers = np.concatenate([
        np.random.randint(5000, 100000, half),   # popular
        np.random.randint(10,   1000,   n - half), # not popular
    ])
    np.random.shuffle(subscribers)

    # Features that correlate with popularity
    avg_rating = np.where(
        subscribers > np.median(subscribers),
        np.random.uniform(4.0, 5.0, n),
        np.random.uniform(1.5, 3.5, n),
    )
    num_lectures = np.where(
        subscribers > np.median(subscribers),
        np.random.randint(50, 200, n),
        np.random.randint(3,  30,  n),
    )
    category = np.where(
        subscribers > np.median(subscribers),
        np.random.choice(["Development", "IT & Software"], n),
        np.random.choice(["Design", "Lifestyle", "Music"], n),
    )
    price = np.where(
        subscribers > np.median(subscribers),
        np.random.choice([19.99, 29.99, 49.99], n),
        np.random.choice([84.99, 99.99, 199.99], n),
    )

    return pd.DataFrame({
        "id":                 range(n),
        "title":              [f"Course {i}" for i in range(n)],
        "is_paid":            (price > 0).astype(bool),
        "price":              price,
        "num_subscribers":    subscribers,
        "avg_rating":         avg_rating,
        "num_reviews":        (subscribers * np.random.uniform(0.01, 0.05, n)).astype(int),
        "num_comments":       np.random.randint(0, 100, n),
        "num_lectures":       num_lectures,
        "content_length_min": (num_lectures * np.random.uniform(8, 15, n)).astype(int),
        "published_time":     dates.astype(str),
        "last_update_date":   pd.date_range(
            "2022-01-01", periods=n, freq="2D", tz="UTC").astype(str),
        "category":           category,
        "subcategory":        np.random.choice(
            ["Python", "Excel", "Photoshop", "SEO", "AWS"], n),
        "language":           np.random.choice(
            ["English", "Spanish", "Portuguese", "German"], n),
        "instructor_name":    [f"Instructor {i % 20}" for i in range(n)],
    })


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    """Run data pipeline steps on synthetic data and return clean DataFrame."""
    from tools.data_tools import (
        fix_dtypes, handle_missing_values, remove_duplicates,
        remove_outliers, add_target_column, add_time_features,
    )
    df = _make_raw_df()
    df = fix_dtypes(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = add_target_column(df)
    df = add_time_features(df)
    return df


@pytest.fixture(scope="module")
def features_df(clean_df) -> pd.DataFrame:
    """Run build_features on the clean DataFrame."""
    from tools.model_tools import build_features
    return build_features(clean_df.copy())


@pytest.fixture(scope="module")
def trained_outputs(features_df, tmp_path_factory):
    """Train model, save it, return (model, metrics, tmp_dir)."""
    from tools.model_tools import train_and_compare, save_evaluation_report
    tmp = tmp_path_factory.mktemp("ml_out")

    model, metrics = train_and_compare(features_df.copy())

    model_path = tmp / "model.pkl"
    eval_path  = tmp / "evaluation_report.md"
    joblib.dump(model, model_path)

    with mock.patch("tools.model_tools.EVALUATION_REPORT_PATH", eval_path):
        with mock.patch("tools.model_tools.OUTPUTS_DIR", tmp):
            save_evaluation_report(metrics)

    return model, metrics, tmp


# ── Integration Test 1: Data Pipeline ────────────────────────────────────────

class TestDataPipelineIntegration:
    """Full data pipeline produces correct artefacts."""

    def test_clean_df_has_rows(self, clean_df):
        assert len(clean_df) > 0, "Pipeline must return a non-empty DataFrame"

    def test_target_column_is_binary(self, clean_df):
        assert set(clean_df["is_popular"].unique()).issubset({0, 1})

    def test_all_required_columns_present(self, clean_df):
        required = ["is_popular", "publish_year", "publish_month",
                    "price", "avg_rating", "category", "language"]
        for col in required:
            assert col in clean_df.columns, f"Missing column: {col}"

    def test_no_nulls_in_critical_columns(self, clean_df):
        for col in ["price", "avg_rating", "is_popular", "category"]:
            assert clean_df[col].isna().sum() == 0, \
                f"Null values in {col} after pipeline"

    def test_publish_year_in_valid_range(self, clean_df):
        years = clean_df["publish_year"].dropna()
        assert (years >= 2010).all() and (years <= 2030).all()

    def test_contract_file_saved_correctly(self, clean_df, tmp_path):
        """save_dataset_contract must write JSON matching the DataFrame."""
        from tools.data_tools import save_dataset_contract
        contract_path = tmp_path / "contract.json"
        save_dataset_contract(clean_df, path=contract_path)

        assert contract_path.exists()
        with open(contract_path) as f:
            contract = json.load(f)
        assert contract["num_rows"] == len(clean_df), \
            "Contract row count must match DataFrame"
        assert "is_popular" in contract["columns"]


# ── Integration Test 2: ML Pipeline ──────────────────────────────────────────

class TestMLPipelineIntegration:
    """Full ML pipeline: features → train → evaluate → save → load → predict."""

    def test_features_df_has_target(self, features_df):
        assert "is_popular" in features_df.columns

    def test_features_df_no_nulls(self, features_df):
        assert features_df.isna().sum().sum() == 0

    def test_features_drop_unwanted_cols(self, features_df):
        dropped = ["title", "instructor_name", "published_time",
                   "last_update_date", "num_subscribers"]
        for col in dropped:
            assert col not in features_df.columns, f"{col} should be dropped"

    def test_trained_model_accuracy_above_baseline(self, trained_outputs):
        _, metrics, _ = trained_outputs
        assert metrics["accuracy"] > 0.50, \
            f"Accuracy {metrics['accuracy']} must beat random baseline (0.50)"

    def test_metrics_has_all_required_keys(self, trained_outputs):
        _, metrics, _ = trained_outputs
        for key in ["best_model", "accuracy", "cv_score",
                    "report", "confusion", "feature_importance"]:
            assert key in metrics

    def test_evaluation_report_saved(self, trained_outputs):
        _, _, tmp = trained_outputs
        assert (tmp / "evaluation_report.md").exists()

    def test_evaluation_report_contains_accuracy(self, trained_outputs):
        import re
        _, _, tmp = trained_outputs
        content = (tmp / "evaluation_report.md").read_text()
        assert re.search(r"Test Accuracy \| [0-9.]+", content), \
            "Evaluation report must contain 'Test Accuracy'"

    def test_saved_model_is_loadable(self, trained_outputs):
        model, _, tmp = trained_outputs
        loaded = joblib.load(tmp / "model.pkl")
        assert hasattr(loaded, "predict")

    def test_loaded_model_predicts_binary(self, trained_outputs, features_df):
        model, _, _ = trained_outputs
        X = features_df.drop(columns=["is_popular"])
        preds = model.predict(X)
        assert set(preds).issubset({0, 1}), "Predictions must be 0 or 1"
        assert len(preds) == len(X)

    def test_model_returns_probabilities(self, trained_outputs, features_df):
        model, _, _ = trained_outputs
        X = features_df.drop(columns=["is_popular"])
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2), "predict_proba must return shape (n, 2)"
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), \
            "Probabilities must sum to 1.0"


# ── Integration Test 3: Pipeline Consistency ─────────────────────────────────

class TestPipelineConsistency:
    """Cross-artefact consistency checks."""

    def test_features_row_count_close_to_clean(self, clean_df, features_df):
        """features.csv should keep ≥ 90% of clean_data rows (only dropna removes rows)."""
        assert len(features_df) >= len(clean_df) * 0.90, (
            f"features ({len(features_df)}) lost >10% of clean_data rows "
            f"({len(clean_df)})"
        )

    def test_target_distribution_stable(self, clean_df, features_df):
        """Target class ratio should be roughly the same before and after feature engineering."""
        clean_ratio    = clean_df["is_popular"].mean()
        features_ratio = features_df["is_popular"].mean()
        assert abs(clean_ratio - features_ratio) < 0.05, (
            f"Target ratio shifted too much: "
            f"clean={clean_ratio:.3f}, features={features_ratio:.3f}"
        )

    def test_validation_gate_passes_after_real_pipeline(
        self, clean_df, tmp_path
    ):
        """After saving real pipeline outputs, Crew 1 validators must all pass."""
        from tools.data_tools import save_clean_data, save_dataset_contract
        from validation import validators
        from validation.validators import ValidationError

        clean_path    = tmp_path / "clean_data.csv"
        contract_path = tmp_path / "dataset_contract.json"
        eda_path      = tmp_path / "eda_report.html"
        insights_path = tmp_path / "insights.md"

        save_clean_data(clean_df, path=clean_path)
        save_dataset_contract(clean_df, path=contract_path)

        eda_path.write_text(
            '<html><body>'
            '<img src="data:image/png;base64,'
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk'
            '+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" />'
            '</body></html>'
        )
        insights_path.write_text(
            "# Executive Summary\n"
            "Development courses priced around $49.99, published in Q1, "
            "consistently attract above-median subscribers. Free courses "
            "gain reach but paid courses show higher engagement and review rates.\n"
            "# Top Findings\n- Development is the top category by subscribers\n"
        )

        monkeypatches = {
            "CLEAN_DATA_PATH":        clean_path,
            "DATASET_CONTRACT_PATH":  contract_path,
            "EDA_REPORT_PATH":        eda_path,
            "INSIGHTS_PATH":          insights_path,
        }
        with mock.patch.multiple(validators, **monkeypatches):
            result = validators.run_crew1_validations()

        assert result["clean_data"]["status"] == "ok"
        assert result["dataset_contract"]["status"] == "ok"
        assert result["eda_report"]["status"] == "ok"
        assert result["insights"]["status"] == "ok"
