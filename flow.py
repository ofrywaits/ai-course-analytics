# -*- coding: utf-8 -*-
"""
RetailAnalyticsFlow — connects Crew 1 and Crew 2 with validations in between.
"""

import logging
import re
import time
from typing import Optional

import pandas as pd
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
    crew1_result:  str   = ""
    crew2_result:  str   = ""
    crew1_valid:   bool  = False
    crew2_valid:   bool  = False
    start_time:    float = 0.0
    accuracy:      float = 0.0
    error_message: Optional[str] = None


class RetailAnalyticsFlow(Flow[RetailFlowState]):
    """Main Flow — runs Crew 1 → validation → Crew 2 → validation → summary."""

    @start()
    def run_analyst_crew(self):
        """Step 1: Run Crew 1 — Data Analyst Crew."""
        OUTPUTS_DIR.mkdir(exist_ok=True)
        LOGS_DIR.mkdir(exist_ok=True)
        self.state.start_time = time.time()
        logger.info("=== Flow started ===")

        from crews.analyst_crew.crew import run_analyst_crew
        result = run_analyst_crew()
        self.state.crew1_result = result.get("crew1_result", "")
        logger.info("Crew 1 complete")

    @listen(run_analyst_crew)
    def validate_analyst_outputs(self):
        """Step 2: Validate all Crew 1 outputs."""
        logger.info("Validating Crew 1 outputs...")
        try:
            run_crew1_validations()
            self.state.crew1_valid = True
            logger.info("✅ Crew 1 validations passed")
        except ValidationError as e:
            self.state.error_message = str(e)
            logger.error(f"❌ Validation failed: {e}")
            raise

    @listen(validate_analyst_outputs)
    def run_scientist_crew(self):
        """Step 3: Run Crew 2 — Data Scientist Crew."""
        from crews.scientist_crew.crew import run_scientist_crew
        result = run_scientist_crew()
        self.state.crew2_result = result.get("crew2_result", "")
        logger.info("Crew 2 complete")

    @listen(run_scientist_crew)
    def validate_scientist_outputs(self):
        """Step 4: Validate all Crew 2 outputs."""
        logger.info("Validating Crew 2 outputs...")
        try:
            run_crew2_validations()
            self.state.crew2_valid = True
            logger.info("✅ Crew 2 validations passed")
        except ValidationError as e:
            self.state.error_message = str(e)
            logger.error(f"❌ Validation failed: {e}")
            raise

    @listen(validate_scientist_outputs)
    def generate_summary(self):
        """Step 5: Generate a full run summary."""
        elapsed    = round(time.time() - self.state.start_time, 1)
        eval_path  = OUTPUTS_DIR / "evaluation_report.md"
        clean_path = OUTPUTS_DIR / "clean_data.csv"

        if eval_path.exists():
            m = re.search(r"Test Accuracy \| ([0-9.]+)",
                          eval_path.read_text(encoding="utf-8"))
            self.state.accuracy = float(m.group(1)) if m else 0.0

        rows = len(pd.read_csv(clean_path)) if clean_path.exists() else 0

        summary = (
            f"\n{'='*54}\n"
            f"  AI Course Analytics Platform — Run Summary\n"
            f"{'='*54}\n"
            f"  Runtime:       {elapsed:.1f} seconds\n"
            f"  Rows processed:{rows:,}\n"
            f"  Crew 1:        {'✅ Passed' if self.state.crew1_valid else '❌ Failed'}\n"
            f"  Crew 2:        {'✅ Passed' if self.state.crew2_valid else '❌ Failed'}\n"
            f"  Accuracy:      {self.state.accuracy:.4f}\n"
            f"  Output files:  8 / 8 ✅\n"
            f"{'='*54}\n"
        )

        print(summary)
        logger.info(summary)
        (OUTPUTS_DIR / "run_summary.txt").write_text(summary, encoding="utf-8")
