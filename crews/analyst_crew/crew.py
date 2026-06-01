# -*- coding: utf-8 -*-
"""
Crew 1 — Data Analyst Crew.
מריץ 3 סוכנים ברצף: Data Engineer → Data Analyst → BI Analyst.
"""

import logging
import sys
from pathlib import Path

from crewai import Crew, Process

# מוסיף את שורש הפרויקט ל-Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crews.analyst_crew.agents import (
    build_data_engineer,
    build_data_analyst,
    build_bi_analyst,
)
from crews.analyst_crew.tasks import (
    build_data_engineering_task,
    build_eda_task,
    build_insights_task,
)
from config import CREW_VERBOSE

logger = logging.getLogger(__name__)


def build_analyst_crew() -> Crew:
    """בונה ומחזיר את ה-Analyst Crew מוכן להרצה."""

    # סוכנים
    data_engineer = build_data_engineer()
    data_analyst  = build_data_analyst()
    bi_analyst    = build_bi_analyst()

    # Tasks ברצף — כל task מקבל context מהקודם
    task_engineering = build_data_engineering_task(data_engineer)
    task_eda         = build_eda_task(data_analyst, [task_engineering])
    task_insights    = build_insights_task(bi_analyst, [task_engineering, task_eda])

    crew = Crew(
        agents=[data_engineer, data_analyst, bi_analyst],
        tasks=[task_engineering, task_eda, task_insights],
        process=Process.sequential,
        verbose=CREW_VERBOSE,
    )

    logger.info("Analyst Crew נבנה בהצלחה")
    return crew


def run_analyst_crew() -> dict:
    """מריץ את Crew 1 ומחזיר תוצאות."""
    from config import INSIGHTS_PATH, OUTPUTS_DIR
    logger.info("=== מתחיל Crew 1: Data Analyst Crew ===")
    crew   = build_analyst_crew()
    result = crew.kickoff()

    # שומר את תוצאת הסוכן האחרון כ-insights.md
    OUTPUTS_DIR.mkdir(exist_ok=True)
    INSIGHTS_PATH.write_text(str(result), encoding="utf-8")
    logger.info(f"insights.md נשמר: {INSIGHTS_PATH}")
    logger.info("=== Crew 1 הושלם ===")
    return {"crew1_result": str(result)}


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    # חובה: Groq צריך את המפתח לפני import של crewai
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_analyst_crew()
