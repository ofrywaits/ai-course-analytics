# -*- coding: utf-8 -*-
"""
שלושת הסוכנים של Crew 1 — Data Analyst Crew.
כל סוכן אחראי על תחום ידע אחד בלבד.
"""

import os
from crewai import Agent, LLM
from config import LLM_MODEL, CREW_MAX_ITER


def _build_llm() -> LLM:
    """בונה את ה-LLM עם Groq — מגדיר את המשתנה הסביבה שCrewAI מצפה לו."""
    api_key = os.getenv("GROQ_API_KEY", "")
    os.environ["GROQ_API_KEY"] = api_key
    return LLM(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=2048,
        max_retries=5,
    )


def build_data_engineer() -> Agent:
    """סוכן 1 — אחראי על טעינה וניקוי הנתונים."""
    return Agent(
        role="Senior Data Engineer",
        goal=(
            "Load the Udemy courses dataset, validate its integrity, "
            "clean missing values, fix data types, remove duplicates, "
            "and produce a clean dataset ready for analysis."
        ),
        backstory=(
            "You are a meticulous data engineer with 10 years of experience "
            "in building reliable data pipelines. You never pass dirty data "
            "to downstream processes — quality is your top priority."
        ),
        llm=_build_llm(),
        memory=True,
        verbose=True,
        max_iter=CREW_MAX_ITER,
    )


def build_data_analyst() -> Agent:
    """סוכן 2 — אחראי על EDA וגרפים."""
    return Agent(
        role="Senior Business Data Analyst",
        goal=(
            "Perform comprehensive exploratory data analysis on the clean "
            "Udemy dataset. Generate distribution plots, correlation heatmaps, "
            "trend charts, and produce a professional HTML report with all "
            "charts embedded as base64 images."
        ),
        backstory=(
            "You are a data analyst who turns raw numbers into compelling "
            "visual stories. Your HTML reports are always polished, insightful, "
            "and ready to be presented to executives."
        ),
        llm=_build_llm(),
        memory=True,
        verbose=True,
        max_iter=CREW_MAX_ITER,
    )


def build_bi_analyst() -> Agent:
    """סוכן 3 — אחראי על תובנות עסקיות."""
    return Agent(
        role="Business Intelligence Analyst",
        goal=(
            "Extract actionable business insights from the Udemy dataset. "
            "Identify top-performing categories, pricing strategies, "
            "seasonal trends, and provide clear recommendations for course creators."
        ),
        backstory=(
            "You bridge the gap between data and business decisions. "
            "You translate complex patterns into clear recommendations "
            "that non-technical stakeholders can act on immediately."
        ),
        llm=_build_llm(),
        memory=True,
        verbose=True,
        max_iter=CREW_MAX_ITER,
    )
