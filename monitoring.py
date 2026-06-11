# -*- coding: utf-8 -*-
"""
Monitoring utilities for the AI Course Analytics Platform.

Provides:
- HealthCheck: verifies all 8 expected output files exist and are non-empty
- RunMetrics: reads the last run summary and exposes key numbers
- log_system_info: logs Python version, memory, and disk usage at startup

Used by app.py (Streamlit dashboard) and main.py to surface errors early
and give operators visibility into platform health without external tooling.
"""

import logging
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Expected output files ─────────────────────────────────────────────────────

_OUTPUTS_DIR = Path(__file__).parent / "outputs"

EXPECTED_FILES: dict[str, str] = {
    "clean_data.csv":        "Cleaned dataset (Crew 1)",
    "dataset_contract.json": "Dataset metadata contract (Crew 1)",
    "eda_report.html":       "EDA report with 6 charts (Crew 1)",
    "insights.md":           "Business insights report (Crew 1)",
    "features.csv":          "Engineered ML features (Crew 2)",
    "model.pkl":             "Trained ML model (Crew 2)",
    "evaluation_report.md":  "Model evaluation report (Crew 2)",
    "model_card.md":         "Model Card — Google standard (Crew 2)",
}


# ── Health Check ──────────────────────────────────────────────────────────────

@dataclass
class FileStatus:
    """Status of a single expected output file."""
    name:    str
    present: bool
    size_kb: float = 0.0
    note:    str   = ""


@dataclass
class HealthReport:
    """Aggregated health-check result for all output files."""
    files:      list[FileStatus] = field(default_factory=list)
    all_ok:     bool             = False
    ok_count:   int              = 0
    total:      int              = 0
    missing:    list[str]        = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ HEALTHY" if self.all_ok else "⚠️  DEGRADED"
        return (
            f"{status} — {self.ok_count}/{self.total} output files present"
            + (f" | Missing: {', '.join(self.missing)}" if self.missing else "")
        )


def run_health_check(outputs_dir: Optional[Path] = None) -> HealthReport:
    """
    Check that all 8 expected output files exist and are non-empty.

    Parameters
    ----------
    outputs_dir : Path, optional
        Directory to scan. Defaults to outputs/.

    Returns
    -------
    HealthReport
        Dataclass with per-file status and an overall health flag.
    """
    directory = outputs_dir or _OUTPUTS_DIR
    report    = HealthReport(total=len(EXPECTED_FILES))

    for fname, desc in EXPECTED_FILES.items():
        path = directory / fname
        if path.exists() and path.stat().st_size > 0:
            size_kb = path.stat().st_size / 1024
            report.files.append(FileStatus(
                name=fname, present=True, size_kb=round(size_kb, 1), note=desc
            ))
            report.ok_count += 1
        else:
            report.files.append(FileStatus(
                name=fname, present=False,
                note=f"MISSING — {desc}"
            ))
            report.missing.append(fname)

    report.all_ok = report.ok_count == report.total
    severity = logging.INFO if report.all_ok else logging.WARNING
    logger.log(severity, report.summary())
    return report


# ── Run Metrics ───────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    """Key metrics extracted from the last pipeline run summary."""
    runtime_seconds: float = 0.0
    rows_processed:  int   = 0
    crew1_passed:    bool  = False
    crew2_passed:    bool  = False
    accuracy:        float = 0.0
    raw_summary:     str   = ""

    @property
    def accuracy_pct(self) -> str:
        return f"{self.accuracy * 100:.1f}%" if self.accuracy else "N/A"


def load_run_metrics(outputs_dir: Optional[Path] = None) -> Optional[RunMetrics]:
    """
    Parse run_summary.txt and return a RunMetrics dataclass.

    Returns None if the file doesn't exist yet (pipeline not run).
    """
    path = (outputs_dir or _OUTPUTS_DIR) / "run_summary.txt"
    if not path.exists():
        logger.info("run_summary.txt not found — pipeline has not run yet")
        return None

    text    = path.read_text(encoding="utf-8")
    metrics = RunMetrics(raw_summary=text)

    m = re.search(r"Runtime:\s+([\d.]+)", text)
    if m:
        metrics.runtime_seconds = float(m.group(1))

    m = re.search(r"Rows processed:\s+([\d,]+)", text)
    if m:
        metrics.rows_processed = int(m.group(1).replace(",", ""))

    metrics.crew1_passed = bool(re.search(r"Crew 1:.*✅", text))
    metrics.crew2_passed = bool(re.search(r"Crew 2:.*✅", text))

    m = re.search(r"Accuracy:\s+([\d.]+)", text)
    if m:
        metrics.accuracy = float(m.group(1))

    logger.info(
        f"Run metrics loaded — runtime: {metrics.runtime_seconds}s, "
        f"rows: {metrics.rows_processed:,}, accuracy: {metrics.accuracy_pct}"
    )
    return metrics


# ── System Info ───────────────────────────────────────────────────────────────

def log_system_info() -> None:
    """
    Log Python version, OS, and available disk space at startup.
    Helps diagnose environment issues in production.
    """
    import shutil
    disk       = shutil.disk_usage("/")
    disk_free  = disk.free / (1024 ** 3)

    logger.info("─── System Info ───────────────────────────────────")
    logger.info(f"  Python  : {sys.version.split()[0]}")
    logger.info(f"  Platform: {platform.system()} {platform.release()}")
    logger.info(f"  Disk    : {disk_free:.1f} GB free")
    logger.info("───────────────────────────────────────────────────")


# ── Convenience: print health to stdout ──────────────────────────────────────

def print_health() -> HealthReport:
    """Run health check and print a formatted table to stdout."""
    report = run_health_check()
    print(f"\n{report.summary()}")
    print(f"{'File':<30} {'Status':<8} {'Size':>8}")
    print("─" * 50)
    for f in report.files:
        status = "✅ OK" if f.present else "❌ MISSING"
        size   = f"{f.size_kb:.1f} KB" if f.present else "—"
        print(f"{f.name:<30} {status:<8} {size:>8}")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log_system_info()
    print_health()
    metrics = load_run_metrics()
    if metrics:
        print(f"\nLast run: {metrics.runtime_seconds}s | "
              f"{metrics.rows_processed:,} rows | "
              f"Accuracy: {metrics.accuracy_pct}")
