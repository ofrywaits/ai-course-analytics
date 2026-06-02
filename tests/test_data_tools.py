# -*- coding: utf-8 -*-
"""
Unit tests for tools/data_tools.py — data pipeline functions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal dataframe with proper dtypes matching what the pipeline produces."""
    np.random.seed(42)
    n = 20
    df = pd.DataFrame({
        "id":                  range(n),
        "title":               [f"Course {i}" for i in range(n)],
        "is_paid":             [True] * (n - 2) + [False, False],
        "price":               np.random.uniform(0, 200, n),
        "num_subscribers":     np.random.randint(100, 50000, n),
        "avg_rating":          np.random.uniform(2.0, 5.0, n),
        "num_reviews":         np.random.randint(0, 5000, n),
        "num_lectures":        np.random.randint(5, 300, n),
        "content_length_min":  np.random.randint(60, 3000, n),
        # Use proper datetime dtype
        "published_time":      pd.to_datetime(
            ["2020-01-15", "2019-06-20", "2021-03-01", "2018-11-10",
             "2022-07-04", "2017-05-12", "2023-02-28", "2016-09-01",
             "2020-12-15", "2021-08-20", "2019-03-10", "2022-11-05",
             "2018-04-22", "2020-07-30", "2021-01-14", "2019-10-08",
             "2023-05-19", "2017-12-31", "2020-03-03", "2022-06-15"],
            utc=True,
        ),
        "last_update_date":    pd.to_datetime(["2023-01-01"] * n, utc=True),
        "category":            np.random.choice(
            ["Development", "Business", "Design", "Marketing"], n),
        "subcategory":         [f"Sub{i % 10}" for i in range(n)],
        "language":            ["English"] * n,
        "instructor_name":     [f"Instructor {i}" for i in range(n)],
        "level":               np.random.choice(
            ["Beginner", "All Levels", "Intermediate", "Advanced"], n),
    })
    return df


# ── Tests: add_target_column ──────────────────────────────────────────────────

class TestAddTargetColumn:
    """Tests for the is_popular target column creation."""

    def test_target_column_created(self, sample_df):
        from tools.data_tools import add_target_column
        result = add_target_column(sample_df.copy())
        assert "is_popular" in result.columns, "is_popular column must be created"

    def test_target_is_binary(self, sample_df):
        from tools.data_tools import add_target_column
        result = add_target_column(sample_df.copy())
        unique_vals = set(result["is_popular"].unique())
        assert unique_vals.issubset({0, 1}), f"is_popular must be 0 or 1, got {unique_vals}"

    def test_popular_courses_have_higher_subscribers(self, sample_df):
        from tools.data_tools import add_target_column
        result = add_target_column(sample_df.copy())
        popular     = result[result["is_popular"] == 1]["num_subscribers"]
        not_popular = result[result["is_popular"] == 0]["num_subscribers"]
        if len(popular) > 0 and len(not_popular) > 0:
            assert popular.mean() > not_popular.mean(), \
                "Popular courses should have higher avg subscribers"

    def test_target_uses_num_subscribers(self, sample_df):
        """Courses above the median subscribers threshold should be marked popular."""
        from tools.data_tools import add_target_column
        result = add_target_column(sample_df.copy())
        # At least some courses should be popular and some not
        assert result["is_popular"].sum() > 0, "At least one course should be popular"
        assert (result["is_popular"] == 0).sum() > 0, "At least one course should NOT be popular"


# ── Tests: add_time_features ──────────────────────────────────────────────────

class TestAddTimeFeatures:
    """Tests for date-derived feature columns."""

    def test_publish_year_created(self, sample_df):
        from tools.data_tools import add_time_features
        result = add_time_features(sample_df.copy())
        assert "publish_year" in result.columns

    def test_publish_month_created(self, sample_df):
        from tools.data_tools import add_time_features
        result = add_time_features(sample_df.copy())
        assert "publish_month" in result.columns

    def test_publish_year_valid_range(self, sample_df):
        from tools.data_tools import add_time_features
        result = add_time_features(sample_df.copy())
        years = result["publish_year"].dropna()
        assert (years >= 2010).all() and (years <= 2030).all(), \
            "publish_year values should be in a realistic range"

    def test_publish_month_valid_range(self, sample_df):
        from tools.data_tools import add_time_features
        result = add_time_features(sample_df.copy())
        months = result["publish_month"].dropna()
        assert (months >= 1).all() and (months <= 12).all(), \
            "publish_month must be between 1 and 12"

    def test_course_age_years_non_negative(self, sample_df):
        from tools.data_tools import add_time_features
        result = add_time_features(sample_df.copy())
        if "course_age_years" in result.columns:
            ages = result["course_age_years"].dropna()
            assert (ages >= 0).all(), "course_age_years should never be negative"


# ── Tests: remove_outliers ────────────────────────────────────────────────────

class TestRemoveOutliers:
    """Tests for the outlier removal step."""

    def test_extreme_price_removed(self):
        from tools.data_tools import remove_outliers
        # Need enough rows so the 99.5th percentile is below the extreme value
        prices = [19.99] * 195 + [84.99] * 4 + [999999.0]  # 200 rows, last is extreme
        df = pd.DataFrame({
            "price":           prices,
            "num_subscribers": [100] * 200,
            "num_reviews":     [50] * 200,
        })
        result = remove_outliers(df.copy())
        assert result["price"].max() < 999999.0, "Extreme price should be removed"

    def test_output_is_dataframe(self, sample_df):
        from tools.data_tools import remove_outliers
        result = remove_outliers(sample_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_columns_preserved(self, sample_df):
        from tools.data_tools import remove_outliers
        result = remove_outliers(sample_df.copy())
        assert set(sample_df.columns) == set(result.columns), \
            "remove_outliers should not add or remove columns"


# ── Tests: handle_missing_values ─────────────────────────────────────────────

class TestHandleMissingValues:
    """Tests for the missing-value handling step."""

    def test_no_nulls_in_numeric_columns(self, sample_df):
        from tools.data_tools import handle_missing_values
        sample_df.loc[0, "avg_rating"] = np.nan
        sample_df.loc[1, "price"] = np.nan
        result = handle_missing_values(sample_df.copy())
        for col in ["avg_rating", "price"]:
            if col in result.columns:
                assert result[col].isna().sum() == 0, \
                    f"Column {col} should have no nulls after handling"

    def test_returns_dataframe(self, sample_df):
        from tools.data_tools import handle_missing_values
        result = handle_missing_values(sample_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_row_count_stable_without_nulls(self, sample_df):
        from tools.data_tools import handle_missing_values
        result = handle_missing_values(sample_df.copy())
        # Without null published_time, no rows should be dropped
        assert len(result) == len(sample_df), \
            "Rows should not be dropped when there are no null dates"
