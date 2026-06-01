# -*- coding: utf-8 -*-
"""
נקודת הכניסה לפרויקט — מריץ את ה-Flow המלא
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# טוען משתני סביבה לפני כל דבר אחר
load_dotenv()

from config import LOG_PATH, LOG_FORMAT, LOG_DATE_FORMAT, OUTPUTS_DIR, LOGS_DIR


def setup_logging() -> logging.Logger:
    """מגדיר לוגים לקונסול ולקובץ במקביל."""
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
    """מריץ את הפרויקט המלא."""
    logger = setup_logging()
    logger.info("=== AI Retail Analytics Platform — מתחיל ===")

    try:
        from flow import RetailAnalyticsFlow
        flow = RetailAnalyticsFlow()
        flow.kickoff()
        logger.info("=== הרצה הושלמה בהצלחה ===")
    except Exception as exc:
        logger.error(f"שגיאה קריטית: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
