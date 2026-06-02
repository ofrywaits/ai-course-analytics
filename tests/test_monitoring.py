# -*- coding: utf-8 -*-
"""
Unit tests for monitoring.py — health checks and run metrics.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunHealthCheck:
    """Tests for the output-file health check."""

    def test_all_ok_when_all_files_present(self, tmp_path):
        from monitoring import run_health_check, EXPECTED_FILES
        for fname in EXPECTED_FILES:
            (tmp_path / fname).write_text("content")
        report = run_health_check(outputs_dir=tmp_path)
        assert report.all_ok is True
        assert report.ok_count == len(EXPECTED_FILES)
        assert report.missing == []

    def test_degraded_when_files_missing(self, tmp_path):
        from monitoring import run_health_check
        # Write only 4 of 8 files
        for fname in ["clean_data.csv", "eda_report.html",
                      "model.pkl", "model_card.md"]:
            (tmp_path / fname).write_text("content")
        report = run_health_check(outputs_dir=tmp_path)
        assert report.all_ok is False
        assert report.ok_count == 4
        assert len(report.missing) == 4

    def test_empty_file_counts_as_missing(self, tmp_path):
        from monitoring import run_health_check, EXPECTED_FILES
        # Write all files but one is empty
        for fname in EXPECTED_FILES:
            (tmp_path / fname).write_text("content")
        (tmp_path / "model.pkl").write_text("")  # empty
        report = run_health_check(outputs_dir=tmp_path)
        assert report.all_ok is False
        assert "model.pkl" in report.missing

    def test_summary_string_contains_status(self, tmp_path):
        from monitoring import run_health_check
        report = run_health_check(outputs_dir=tmp_path)
        summary = report.summary()
        assert "DEGRADED" in summary or "HEALTHY" in summary


class TestLoadRunMetrics:
    """Tests for parsing run_summary.txt."""

    def test_returns_none_when_missing(self, tmp_path):
        from monitoring import load_run_metrics
        result = load_run_metrics(outputs_dir=tmp_path)
        assert result is None

    def test_parses_runtime(self, tmp_path):
        from monitoring import load_run_metrics
        summary = (
            "========================================\n"
            "  AI Course Analytics Platform\n"
            "========================================\n"
            "  Runtime:        47.3 seconds\n"
            "  Rows processed: 206,885\n"
            "  Crew 1:         ✅ Passed\n"
            "  Crew 2:         ✅ Passed\n"
            "  Accuracy:       0.8736\n"
            "========================================\n"
        )
        (tmp_path / "run_summary.txt").write_text(summary)
        metrics = load_run_metrics(outputs_dir=tmp_path)
        assert metrics is not None
        assert metrics.runtime_seconds == 47.3
        assert metrics.rows_processed == 206885
        assert metrics.accuracy == pytest.approx(0.8736)

    def test_accuracy_pct_property(self, tmp_path):
        from monitoring import load_run_metrics
        (tmp_path / "run_summary.txt").write_text("Accuracy:       0.8736\n")
        metrics = load_run_metrics(outputs_dir=tmp_path)
        assert metrics is not None
        assert "87.4%" in metrics.accuracy_pct
