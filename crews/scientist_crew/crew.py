# -*- coding: utf-8 -*-
"""
Crew 2 — Data Scientist Crew.
Feature Engineer → ML Engineer → Ethics Specialist.
"""

import logging
import sys
from pathlib import Path

from crewai import Crew, Process

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crews.scientist_crew.agents import (
    build_feature_engineer,
    build_ml_engineer,
    build_ethics_specialist,
)
from crews.scientist_crew.tasks import (
    build_feature_engineering_task,
    build_ml_training_task,
    build_model_card_task,
)
from config import CREW_VERBOSE, MODEL_CARD_PATH, OUTPUTS_DIR

logger = logging.getLogger(__name__)


def build_scientist_crew() -> Crew:
    """בונה ומחזיר את ה-Scientist Crew."""
    feature_engineer   = build_feature_engineer()
    ml_engineer        = build_ml_engineer()
    ethics_specialist  = build_ethics_specialist()

    task_features  = build_feature_engineering_task(feature_engineer)
    task_ml        = build_ml_training_task(ml_engineer, [task_features])
    task_card      = build_model_card_task(ethics_specialist, [task_features, task_ml])

    crew = Crew(
        agents=[feature_engineer, ml_engineer, ethics_specialist],
        tasks=[task_features, task_ml, task_card],
        process=Process.sequential,
        verbose=CREW_VERBOSE,
    )

    logger.info("Scientist Crew נבנה בהצלחה")
    return crew


def run_scientist_crew() -> dict:
    """מריץ את Crew 2 ומחזיר תוצאות."""
    logger.info("=== מתחיל Crew 2: Data Scientist Crew ===")
    crew   = build_scientist_crew()
    result = crew.kickoff()

    # שומר את ה-model card מתוצאת הסוכן האחרון
    OUTPUTS_DIR.mkdir(exist_ok=True)
    MODEL_CARD_PATH.write_text(str(result), encoding="utf-8")
    logger.info(f"model_card.md נשמר: {MODEL_CARD_PATH}")
    logger.info("=== Crew 2 הושלם ===")
    return {"crew2_result": str(result)}


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_scientist_crew()
