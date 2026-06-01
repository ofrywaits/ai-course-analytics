# -*- coding: utf-8 -*-
"""
Crew 1 — Data Analyst Crew.
Runs 3 agents in sequence: Data Engineer → Data Analyst → BI Analyst.
"""

import logging
import sys
from pathlib import Path

from crewai import Crew, Process

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
from config import CREW_VERBOSE, INSIGHTS_PATH, OUTPUTS_DIR

logger = logging.getLogger(__name__)


def build_analyst_crew() -> Crew:
    """Build and return the Analyst Crew ready to run."""
    data_engineer = build_data_engineer()
    data_analyst  = build_data_analyst()
    bi_analyst    = build_bi_analyst()

    task_engineering = build_data_engineering_task(data_engineer)
    task_eda         = build_eda_task(data_analyst, [task_engineering])
    task_insights    = build_insights_task(bi_analyst, [task_engineering, task_eda])

    crew = Crew(
        agents=[data_engineer, data_analyst, bi_analyst],
        tasks=[task_engineering, task_eda, task_insights],
        process=Process.sequential,
        verbose=CREW_VERBOSE,
    )

    logger.info("Analyst Crew built successfully")
    return crew


def run_analyst_crew() -> dict:
    """Run Crew 1 and return results."""
    logger.info("=== Starting Crew 1: Data Analyst Crew ===")
    crew   = build_analyst_crew()
    result = crew.kickoff()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    INSIGHTS_PATH.write_text(str(result), encoding="utf-8")
    logger.info(f"insights.md saved: {INSIGHTS_PATH}")
    logger.info("=== Crew 1 complete ===")
    return {"crew1_result": str(result)}


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_analyst_crew()
