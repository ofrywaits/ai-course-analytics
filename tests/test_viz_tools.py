# -*- coding: utf-8 -*-
"""
Unit tests for tools/viz_tools.py.

All 6 chart functions must:
1. Return a non-empty string
2. Contain a base64-embedded <img> tag (data:image/png;base64,...)
3. Not raise exceptions on valid DataFrame input
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame with all columns required by viz_tools."""
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        "price":              np.random.choice([0, 19.99, 29.99, 49.99], n),
        "is_paid":            np.random.choice([True, False], n),
        "num_subscribers":    np.random.randint(10, 50000, n),
        "avg_rating":         np.random.uniform(1.0, 5.0, n),
        "category":           np.random.choice(["Development", "Design", "IT & Software"], n),
        "subcategory":        np.random.choice(["Python", "Excel", "AWS"], n),
        "published_time":     pd.date_range("2015-01-01", periods=n, freq="30D")
                                .strftime("%Y-%m-%d").tolist(),
        "publish_year":       np.random.randint(2015, 2024, n),
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_html_img(result: str, func_name: str) -> None:
    """Assert result is a non-empty HTML string with an embedded base64 image."""
    assert isinstance(result, str), f"{func_name} must return a str"
    assert len(result) > 100,       f"{func_name} returned suspiciously short string"
    assert "data:image/png;base64," in result, \
        f"{func_name} must embed a base64 PNG image"
    assert "<img" in result, f"{func_name} must contain an <img> tag"


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVizTools:
    """Smoke tests for all 6 chart functions in viz_tools."""

    def test_plot_price_distribution(self, sample_df):
        from tools.viz_tools import plot_price_distribution
        result = plot_price_distribution(sample_df)
        _assert_html_img(result, "plot_price_distribution")

    def test_plot_subscribers_by_category(self, sample_df):
        from tools.viz_tools import plot_subscribers_by_category
        result = plot_subscribers_by_category(sample_df)
        _assert_html_img(result, "plot_subscribers_by_category")

    def test_plot_correlation_heatmap(self, sample_df):
        from tools.viz_tools import plot_correlation_heatmap
        result = plot_correlation_heatmap(sample_df)
        _assert_html_img(result, "plot_correlation_heatmap")

    def test_plot_top_subcategories(self, sample_df):
        from tools.viz_tools import plot_top_subcategories
        result = plot_top_subcategories(sample_df)
        _assert_html_img(result, "plot_top_subcategories")

    def test_plot_courses_over_time(self, sample_df):
        from tools.viz_tools import plot_courses_over_time
        result = plot_courses_over_time(sample_df)
        _assert_html_img(result, "plot_courses_over_time")

    def test_plot_rating_distribution(self, sample_df):
        from tools.viz_tools import plot_rating_distribution
        result = plot_rating_distribution(sample_df)
        _assert_html_img(result, "plot_rating_distribution")

    def test_all_charts_return_div_wrapper(self, sample_df):
        """Each chart should be wrapped in a <div class='chart-block'>."""
        from tools.viz_tools import (
            plot_price_distribution, plot_subscribers_by_category,
            plot_correlation_heatmap, plot_top_subcategories,
            plot_courses_over_time, plot_rating_distribution,
        )
        funcs = [
            plot_price_distribution, plot_subscribers_by_category,
            plot_correlation_heatmap, plot_top_subcategories,
            plot_courses_over_time, plot_rating_distribution,
        ]
        for fn in funcs:
            result = fn(sample_df)
            assert "chart-block" in result, \
                f"{fn.__name__} must wrap output in <div class='chart-block'>"

    def test_no_matplotlib_figure_leak(self, sample_df):
        """All chart functions must close the figure (no memory leak)."""
        import matplotlib.pyplot as plt
        from tools.viz_tools import plot_price_distribution
        before = len(plt.get_fignums())
        plot_price_distribution(sample_df)
        after = len(plt.get_fignums())
        assert after == before, \
            f"Figure not closed after plot_price_distribution (leaked {after - before})"
