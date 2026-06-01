# -*- coding: utf-8 -*-
"""
RetailAnalyticsFlow — מחבר את Crew 1 ו-Crew 2 עם ולידציות ביניהם.
"""

import logging
import time
from typing import Optional

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

from config import OUTPUTS_DIR, LOGS_DIR
from validation.validators import (
    ValidationError,
    run_crew1_validations,
    run_crew2_validations,
)

logger = logging.getLogger(__name__)


class RetailFlowState(BaseModel):
    crew1_result:   str = ""
    crew2_result:   str = ""
    crew1_valid:    bool = False
    crew2_valid:    bool = False
    start_time:     float = 0.0
    accuracy:       float = 0.0
    error_message:  Optional[str] = None


class RetailAnalyticsFlow(Flow[RetailFlowState]):
    """Flow ראשי — מריץ Crew 1 → ולידציה → Crew 2 → ולידציה → סיכום."""

    @start()
    def run_analyst_crew(self):
        """שלב 1: מריץ את Crew 1 — Data Analyst."""
        OUTPUTS_DIR.mkdir(exist_ok=True)
        LOGS_DIR.mkdir(exist_ok=True)
        self.state.start_time = time.time()
        logger.info("━━━ Flow התחיל ━━━")

        from crews.analyst_crew.crew import run_analyst_crew
        result = run_analyst_crew()
        self.state.crew1_result = result.get("crew1_result", "")
        logger.info("Crew 1 הושלם")

    @listen(run_analyst_crew)
    def validate_analyst_outputs(self):
        """שלב 2: מוודא שכל פלטי Crew 1 תקינים."""
        logger.info("מוודא פלטי Crew 1...")
        try:
            run_crew1_validations()
            self.state.crew1_valid = True
            logger.info("✅ ולידציות Crew 1 עברו")
        except ValidationError as e:
            self.state.error_message = str(e)
            logger.error(f"❌ ולידציה נכשלה: {e}")
            raise

    @listen(validate_analyst_outputs)
    def run_scientist_crew(self):
        """שלב 3: מריץ את Crew 2 — Data Scientist."""
        from crews.scientist_crew.crew import run_scientist_crew
        result = run_scientist_crew()
        self.state.crew2_result = result.get("crew2_result", "")
        logger.info("Crew 2 הושלם")

    @listen(run_scientist_crew)
    def validate_scientist_outputs(self):
        """שלב 4: מוודא שכל פלטי Crew 2 תקינים."""
        logger.info("מוודא פלטי Crew 2...")
        try:
            run_crew2_validations()
            self.state.crew2_valid = True
            logger.info("✅ ולידציות Crew 2 עברו")
        except ValidationError as e:
            self.state.error_message = str(e)
            logger.error(f"❌ ולידציה נכשלה: {e}")
            raise

    @listen(validate_scientist_outputs)
    def generate_summary(self):
        """שלב 5: מייצר סיכום הרצה מלא."""
        elapsed = round(time.time() - self.state.start_time, 1)

        import re
        eval_path = OUTPUTS_DIR / "evaluation_report.md"
        if eval_path.exists():
            content = eval_path.read_text(encoding="utf-8")
            m = re.search(r"Test Accuracy \| ([0-9.]+)", content)
            self.state.accuracy = float(m.group(1)) if m else 0.0

        import pandas as pd
        clean_path = OUTPUTS_DIR / "clean_data.csv"
        rows = len(pd.read_csv(clean_path)) if clean_path.exists() else 0

        summary = f"""
╔══════════════════════════════════════════════════════╗
║       🎓 AI Course Analytics Platform — סיכום       ║
╠══════════════════════════════════════════════════════╣
║  זמן הרצה:        {elapsed:.1f} שניות
║  שורות עובדו:     {rows:,}
║  Crew 1:          {'✅ הצליח' if self.state.crew1_valid else '❌ נכשל'}
║  Crew 2:          {'✅ הצליח' if self.state.crew2_valid else '❌ נכשל'}
║  Accuracy:        {self.state.accuracy:.4f}
║  קבצי פלט:       8/8 ✅
╚══════════════════════════════════════════════════════╝
"""
        print(summary)
        logger.info(summary)

        summary_path = OUTPUTS_DIR / "run_summary.txt"
        summary_path.write_text(summary, encoding="utf-8")
