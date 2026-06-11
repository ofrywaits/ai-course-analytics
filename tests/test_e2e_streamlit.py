# -*- coding: utf-8 -*-
"""
E2E tests for the Streamlit dashboard using pytest-playwright.

These tests require a running Streamlit server:
    streamlit run app.py

Skip marker is applied so they do NOT run in standard CI/CD
(which has no running server). Run manually:
    pytest tests/test_e2e_streamlit.py -v -m e2e

Prerequisites:
    pip install pytest-playwright
    playwright install chromium
"""

import pytest

# All tests in this module require a running Streamlit server.
pytestmark = pytest.mark.skip(reason="requires running Streamlit server on localhost:8501")

BASE_URL = "http://localhost:8501"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _wait_for_app(page) -> None:
    """Wait until the Streamlit app has finished loading."""
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=15_000)


# ── Test 1: App loads ─────────────────────────────────────────────────────────

class TestAppLoads:
    """Verify the app launches with the correct title and header."""

    def test_page_title_contains_analytics(self, page):
        """Browser tab title should contain 'Analytics'."""
        page.goto(BASE_URL)
        _wait_for_app(page)
        assert "Analytics" in page.title()

    def test_main_heading_visible(self, page):
        """The hero heading '🎓 AI Course Analytics Platform' must appear."""
        page.goto(BASE_URL)
        _wait_for_app(page)
        heading = page.locator("h1, h2").first
        assert "Analytics" in heading.inner_text()

    def test_no_uncaught_error_on_load(self, page):
        """App should not display Streamlit's red error banner on load."""
        page.goto(BASE_URL)
        _wait_for_app(page)
        error_banner = page.locator('[data-testid="stException"]')
        assert error_banner.count() == 0, "Uncaught exception displayed on load"


# ── Test 2: Tab navigation ────────────────────────────────────────────────────

class TestTabNavigation:
    """All 5 tabs must be present and navigable."""

    EXPECTED_TABS = ["Overview", "EDA Report", "Model Results",
                     "Course Predictor", "Downloads"]

    def test_all_tabs_visible(self, page):
        """All 5 tab labels must be visible in the navigation bar."""
        page.goto(BASE_URL)
        _wait_for_app(page)
        tabs = page.locator('[role="tab"]')
        tab_texts = [tabs.nth(i).inner_text() for i in range(tabs.count())]
        for expected in self.EXPECTED_TABS:
            assert any(expected in t for t in tab_texts), \
                f"Tab '{expected}' not found. Visible: {tab_texts}"

    def test_overview_tab_has_metric_cards(self, page):
        """Overview tab should display at least 3 metric cards."""
        page.goto(BASE_URL)
        _wait_for_app(page)
        # Click Overview tab (first tab)
        page.locator('[role="tab"]').first.click()
        page.wait_for_timeout(1000)
        metrics = page.locator('[data-testid="metric-container"]')
        assert metrics.count() >= 3, \
            f"Expected ≥3 metric cards on Overview, got {metrics.count()}"

    def test_eda_tab_loads_without_error(self, page):
        """EDA Report tab must render without exception."""
        page.goto(BASE_URL)
        _wait_for_app(page)
        page.locator('[role="tab"]').nth(1).click()
        page.wait_for_timeout(2000)
        assert page.locator('[data-testid="stException"]').count() == 0

    def test_model_results_tab_shows_accuracy(self, page):
        """Model Results tab must show accuracy somewhere on the page."""
        page.goto(BASE_URL)
        _wait_for_app(page)
        page.locator('[role="tab"]').nth(2).click()
        page.wait_for_timeout(2000)
        content = page.locator('[data-testid="stAppViewContainer"]').inner_text()
        assert "accuracy" in content.lower() or "Accuracy" in content, \
            "Model Results tab does not mention accuracy"

    def test_downloads_tab_has_buttons(self, page):
        """Downloads tab must show at least 4 download buttons."""
        page.goto(BASE_URL)
        _wait_for_app(page)
        downloads_tab = page.locator('[role="tab"]').nth(4)
        downloads_tab.click()
        page.wait_for_timeout(2000)
        download_buttons = page.locator('[data-testid="stDownloadButton"]')
        assert download_buttons.count() >= 4, \
            f"Expected ≥4 download buttons, got {download_buttons.count()}"


# ── Test 3: Course Predictor tab ──────────────────────────────────────────────

class TestCoursePredictor:
    """Course Predictor tab — inputs + prediction flow."""

    def _open_predictor(self, page) -> None:
        page.goto(BASE_URL)
        _wait_for_app(page)
        # 4th tab (index 3)
        page.locator('[role="tab"]').nth(3).click()
        page.wait_for_timeout(1500)

    def test_predictor_tab_loads(self, page):
        """Predictor tab must load without Streamlit exception."""
        self._open_predictor(page)
        assert page.locator('[data-testid="stException"]').count() == 0

    def test_predictor_has_category_selector(self, page):
        """Category selectbox must be present in the Predictor tab."""
        self._open_predictor(page)
        # Selectbox labels
        labels = page.locator('[data-testid="stSelectbox"] label').all_inner_texts()
        assert any("Category" in lbl for lbl in labels), \
            f"Category selectbox not found. Labels: {labels}"

    def test_predictor_has_price_slider(self, page):
        """Price slider must be present in the Predictor tab."""
        self._open_predictor(page)
        sliders = page.locator('[data-testid="stSlider"]')
        assert sliders.count() >= 1, "No sliders found in Predictor tab"

    def test_predict_button_exists(self, page):
        """'Predict Popularity' button must be present."""
        self._open_predictor(page)
        btn = page.locator('button', has_text="Predict")
        assert btn.count() >= 1, "No 'Predict' button found in Predictor tab"

    def test_maya_scenario_button_exists(self, page):
        """Demo scenario button must be present."""
        self._open_predictor(page)
        btn = page.locator('button', has_text="Maya")
        assert btn.count() >= 1, "Maya demo button not found"

    def test_predict_button_shows_result(self, page):
        """Clicking Predict must show a probability gauge or result text."""
        self._open_predictor(page)
        predict_btn = page.locator('button', has_text="Predict").first
        predict_btn.click()
        page.wait_for_timeout(3000)
        # Check for plotly gauge or text containing probability
        gauge_or_text = (
            page.locator('[class*="js-plotly-plot"]').count() > 0
            or "popular" in page.locator('[data-testid="stAppViewContainer"]')
                               .inner_text().lower()
        )
        assert gauge_or_text, "No prediction result visible after clicking Predict"
