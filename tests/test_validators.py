# -*- coding: utf-8 -*-
"""
Unit tests for validation/validators.py — output validation checks.
"""

import sys
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_clean_csv(path: Path, n: int = 1200) -> pd.DataFrame:
    """Write a valid clean_data.csv with n rows to path and return the DataFrame."""
    np.random.seed(0)
    df = pd.DataFrame({
        "price":           np.random.uniform(0, 200, n),
        "num_subscribers": np.random.randint(100, 50000, n),
        "avg_rating":      np.random.uniform(2.0, 5.0, n),
        "category":        np.random.choice(["Development", "Business"], n),
        "is_paid":         np.random.choice([True, False], n),
        "is_popular":      np.random.randint(0, 2, n),
    })
    df.to_csv(path, index=False)
    return df


def _make_contract(path: Path, df: pd.DataFrame) -> None:
    """Write a dataset_contract.json that matches df."""
    contract = {
        "num_rows":    len(df),
        "num_columns": len(df.columns),
        "columns":     list(df.columns),
    }
    path.write_text(json.dumps(contract))


def _make_eda_html(path: Path) -> None:
    """Write a minimal valid eda_report.html with a base64 image."""
    path.write_text(
        "<html><body>"
        '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" />'
        "</body></html>"
    )


def _make_insights_md(path: Path) -> None:
    """Write a minimal insights.md with more than 100 characters."""
    path.write_text(
        "# Executive Summary\n"
        "Courses in the Development category with a price around $49.99 published "
        "in Q1 tend to attract the most subscribers. Free courses get more reach "
        "but paid courses generate more reviews and engagement.\n\n"
        "# Top Findings\n"
        "- Development is the top category by subscriber count\n"
    )


# ── Tests: ValidationError ───────────────────────────────────────────────────

class TestValidationError:
    """Tests for the custom ValidationError exception."""

    def test_is_exception(self):
        from validation.validators import ValidationError
        assert issubclass(ValidationError, Exception)

    def test_message_preserved(self):
        from validation.validators import ValidationError
        err = ValidationError("test message")
        assert "test message" in str(err)

    def test_can_be_raised_and_caught(self):
        from validation.validators import ValidationError
        with pytest.raises(ValidationError, match="boom"):
            raise ValidationError("boom")


# ── Tests: validate_file_exists ───────────────────────────────────────────────

class TestValidateFileExists:
    """Tests for the file-existence helper."""

    def test_passes_when_file_exists(self, tmp_path):
        from validation.validators import validate_file_exists
        f = tmp_path / "test.txt"
        f.write_text("hello")
        validate_file_exists(f, "test.txt")  # should not raise

    def test_raises_when_file_missing(self, tmp_path):
        from validation.validators import ValidationError, validate_file_exists
        with pytest.raises(ValidationError, match="Missing"):
            validate_file_exists(tmp_path / "ghost.txt", "ghost.txt")

    def test_raises_when_file_empty(self, tmp_path):
        from validation.validators import ValidationError, validate_file_exists
        f = tmp_path / "empty.txt"
        f.write_text("")
        with pytest.raises(ValidationError, match="Empty"):
            validate_file_exists(f, "empty.txt")


# ── Tests: run_crew1_validations ─────────────────────────────────────────────

class TestCrew1Validations:
    """Integration tests for the Crew 1 validation pipeline."""

    def _patch_paths(self, monkeypatch, validators, tmp_path,
                     clean, contract, eda, insights):
        monkeypatch.setattr(validators, "CLEAN_DATA_PATH",        clean)
        monkeypatch.setattr(validators, "DATASET_CONTRACT_PATH",  contract)
        monkeypatch.setattr(validators, "EDA_REPORT_PATH",        eda)
        monkeypatch.setattr(validators, "INSIGHTS_PATH",          insights)

    def test_passes_when_all_files_valid(self, tmp_path, monkeypatch):
        from validation import validators

        clean    = tmp_path / "clean_data.csv"
        contract = tmp_path / "dataset_contract.json"
        eda      = tmp_path / "eda_report.html"
        insights = tmp_path / "insights.md"

        df = _make_clean_csv(clean)
        _make_contract(contract, df)
        _make_eda_html(eda)
        _make_insights_md(insights)
        self._patch_paths(monkeypatch, validators, tmp_path,
                          clean, contract, eda, insights)

        result = validators.run_crew1_validations()
        assert result["clean_data"]["status"] == "ok"

    def test_fails_when_clean_data_missing(self, tmp_path, monkeypatch):
        from validation import validators
        from validation.validators import ValidationError
        self._patch_paths(monkeypatch, validators, tmp_path,
                          tmp_path / "ghost.csv",
                          tmp_path / "ghost.json",
                          tmp_path / "ghost.html",
                          tmp_path / "ghost.md")
        with pytest.raises(ValidationError):
            validators.run_crew1_validations()

    def test_fails_when_dataset_too_small(self, tmp_path, monkeypatch):
        """clean_data.csv with fewer than 1000 rows → ValidationError."""
        from validation import validators
        from validation.validators import ValidationError

        clean    = tmp_path / "clean_data.csv"
        contract = tmp_path / "dataset_contract.json"
        eda      = tmp_path / "eda_report.html"
        insights = tmp_path / "insights.md"

        # Only 5 rows → should fail the row-count check
        df = _make_clean_csv(clean, n=5)
        _make_contract(contract, df)
        _make_eda_html(eda)
        _make_insights_md(insights)
        self._patch_paths(monkeypatch, validators, tmp_path,
                          clean, contract, eda, insights)

        with pytest.raises(ValidationError, match="small|rows|minimum"):
            validators.run_crew1_validations()

    def test_fails_when_eda_has_no_base64(self, tmp_path, monkeypatch):
        """eda_report.html without base64 images → ValidationError."""
        from validation import validators
        from validation.validators import ValidationError

        clean    = tmp_path / "clean_data.csv"
        contract = tmp_path / "dataset_contract.json"
        eda      = tmp_path / "eda_report.html"
        insights = tmp_path / "insights.md"

        df = _make_clean_csv(clean)
        _make_contract(contract, df)
        eda.write_text("<html><body>No charts here</body></html>")  # missing base64
        _make_insights_md(insights)
        self._patch_paths(monkeypatch, validators, tmp_path,
                          clean, contract, eda, insights)

        with pytest.raises(ValidationError, match="base64|charts"):
            validators.run_crew1_validations()


# ── Tests: run_crew2_validations ─────────────────────────────────────────────

class TestCrew2Validations:
    """Integration tests for the Crew 2 validation pipeline."""

    def _patch_paths(self, monkeypatch, validators,
                     features, model_path, eval_md, card):
        monkeypatch.setattr(validators, "FEATURES_PATH",           features)
        monkeypatch.setattr(validators, "MODEL_PATH",              model_path)
        monkeypatch.setattr(validators, "EVALUATION_REPORT_PATH",  eval_md)
        monkeypatch.setattr(validators, "MODEL_CARD_PATH",         card)

    def test_passes_when_all_files_valid(self, tmp_path, monkeypatch):
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from validation import validators

        features   = tmp_path / "features.csv"
        model_path = tmp_path / "model.pkl"
        eval_md    = tmp_path / "evaluation_report.md"
        card       = tmp_path / "model_card.md"

        # 1200-row features file
        np.random.seed(0)
        n = 1200
        pd.DataFrame({
            "price": np.random.uniform(0, 200, n),
            "is_popular": np.random.randint(0, 2, n),
        }).to_csv(features, index=False)

        joblib.dump(RandomForestClassifier(), model_path)

        eval_md.write_text(
            "# Evaluation Report\n"
            "| Test Accuracy | 0.87 |\n"
            "| CV Score (5-fold) | 0.85 |\n"
        )
        card.write_text(
            "# Model Card\n"
            "## Model Details\nRandom Forest.\n"
            "## Intended Use\nCourse analytics.\n"
            "## Training Data\nUdemy courses.\n"
            "## Metrics\nAccuracy: 87%.\n"
            "## Limitations\nOnly Udemy data.\n"
        )

        monkeypatch.setattr(validators, "MIN_ACCURACY_THRESHOLD", 0.0)
        self._patch_paths(monkeypatch, validators,
                          features, model_path, eval_md, card)

        result = validators.run_crew2_validations()
        assert result["model"]["status"] == "ok"

    def test_fails_when_model_missing(self, tmp_path, monkeypatch):
        from validation import validators
        from validation.validators import ValidationError
        self._patch_paths(monkeypatch, validators,
                          tmp_path / "f.csv",
                          tmp_path / "missing.pkl",
                          tmp_path / "e.md",
                          tmp_path / "c.md")
        with pytest.raises(ValidationError):
            validators.run_crew2_validations()

    def test_fails_when_model_card_missing_sections(self, tmp_path, monkeypatch):
        """model_card.md missing required sections → ValidationError."""
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from validation import validators
        from validation.validators import ValidationError

        features   = tmp_path / "features.csv"
        model_path = tmp_path / "model.pkl"
        eval_md    = tmp_path / "evaluation_report.md"
        card       = tmp_path / "model_card.md"

        np.random.seed(0)
        n = 1200
        pd.DataFrame({
            "price": np.random.uniform(0, 200, n),
            "is_popular": np.random.randint(0, 2, n),
        }).to_csv(features, index=False)
        joblib.dump(RandomForestClassifier(), model_path)
        eval_md.write_text("| Test Accuracy | 0.87 |")
        card.write_text("# Model Card\nJust a title, no required sections.")  # incomplete

        monkeypatch.setattr(validators, "MIN_ACCURACY_THRESHOLD", 0.0)
        self._patch_paths(monkeypatch, validators,
                          features, model_path, eval_md, card)

        with pytest.raises(ValidationError, match="[Mm]issing|sections"):
            validators.run_crew2_validations()
