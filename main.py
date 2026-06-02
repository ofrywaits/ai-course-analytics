# -*- coding: utf-8 -*-
"""
Entry point — runs the full CrewAI Flow.
"""

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from config import LOG_PATH, LOG_FORMAT, LOG_DATE_FORMAT, OUTPUTS_DIR, LOGS_DIR


def setup_logging() -> logging.Logger:
    """Configure logging to both console and file."""
    LOGS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )
    return logging.getLogger("main")


def main() -> None:
    """Run the full analytics pipeline."""
    logger = setup_logging()
    logger.info("=== AI Course Analytics Platform — Starting ===")

    from monitoring import log_system_info
    log_system_info()

    try:
        from flow import RetailAnalyticsFlow
        flow = RetailAnalyticsFlow()
        flow.kickoff()
        logger.info("=== Run completed successfully ===")
    except Exception as exc:
        logger.error(f"Critical error: {exc}", exc_info=True)
        sys.exit(1)

    # Post-run health check — warns if any output file is missing
    from monitoring import run_health_check
    report = run_health_check()
    if not report.all_ok:
        logger.warning(f"Post-run health check: {report.summary()}")


if __name__ == "__main__":
    main()
