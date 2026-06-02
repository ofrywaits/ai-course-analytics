# -*- coding: utf-8 -*-
"""
pytest configuration and shared fixtures for the AI Course Analytics test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path for all test modules
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def outputs_dir(project_root) -> Path:
    """Return the outputs/ directory path."""
    return project_root / "outputs"


@pytest.fixture
def tiny_clean_df() -> pd.DataFrame:
    """
    Small but schema-valid DataFrame that mirrors clean_data.csv.
    Useful for fast unit tests that don't need scale.
    """
    np.random.seed(99)
    n = 50
    return pd.DataFrame({
        "price":               np.random.uniform(0, 200, n),
        "num_subscribers":     np.random.randint(100, 50000, n),
        "avg_rating":          np.random.uniform(2.0, 5.0, n),
        "num_reviews":         np.random.randint(0, 5000, n),
        "num_lectures":        np.random.randint(5, 300, n),
        "content_length_min":  np.random.randint(60, 3000, n),
        "is_paid":             np.random.choice([True, False], n),
        "category":            np.random.choice(
            ["Development", "Business", "Design", "Marketing"], n),
        "subcategory":         [f"Sub{i % 10}" for i in range(n)],
        "language":            np.random.choice(["English", "Spanish", "French"], n),
        "level":               np.random.choice(
            ["Beginner", "All Levels", "Intermediate", "Advanced"], n),
        "publish_year":        np.random.randint(2012, 2024, n),
        "publish_month":       np.random.randint(1, 13, n),
        "is_popular":          np.random.randint(0, 2, n),
    })
