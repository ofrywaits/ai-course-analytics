# -*- coding: utf-8 -*-
"""
Three agents for Crew 2 — Data Scientist Crew.
"""

import logging
import os
from crewai import Agent, LLM
from config import LLM_MODEL, CREW_MAX_ITER
from tools.mcp_tools import get_mcp_file_tools

logger = logging.getLogger(__name__)


def _build_llm() -> LLM:
    """Build the LLM with Groq credentials."""
    api_key = os.getenv("GROQ_API_KEY", "")
    os.environ["GROQ_API_KEY"] = api_key
    return LLM(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=2048,
        max_retries=5,
    )


def build_feature_engineer() -> Agent:
    """Agent 4 — feature engineering."""
    return Agent(
        role="Senior Feature Engineer",
        goal=(
            "Read the clean Udemy dataset, apply encoding and scaling, "
            "create meaningful features, and produce a features dataset "
            "ready for machine learning model training."
        ),
        backstory=(
            "You specialize in extracting signal from raw data. "
            "You know exactly which transformations make models perform better, "
            "and you never let data leakage slip through."
        ),
        llm=_build_llm(),
        memory=True,
        verbose=True,
        max_iter=CREW_MAX_ITER,
    )


def build_ml_engineer() -> Agent:
    """Agent 5 — model training and selection."""
    return Agent(
        role="Senior Machine Learning Engineer",
        goal=(
            "Train Random Forest and Logistic Regression models on the Udemy "
            "features dataset, compare their performance, select the best one, "
            "and produce a detailed evaluation report."
        ),
        backstory=(
            "You have shipped ML models to production at scale. "
            "You always validate with cross-validation and never trust "
            "a single train/test split result alone."
        ),
        llm=_build_llm(),
        memory=True,
        verbose=True,
        max_iter=CREW_MAX_ITER,
    )


def build_ethics_specialist() -> Agent:
    """Agent 6 — Model Card and AI ethics.

    Equipped with MCP filesystem tools so it can read the evaluation report
    directly from outputs/ when writing the Model Card.
    """
    mcp_tools = get_mcp_file_tools()
    if mcp_tools:
        logger.info(f"Ethics Specialist: {len(mcp_tools)} MCP tool(s) attached")
    else:
        logger.info("Ethics Specialist: running without MCP tools (fallback mode)")

    return Agent(
        role="AI Ethics and Documentation Specialist",
        goal=(
            "Write a professional Model Card following Google's standard "
            "for the trained Udemy popularity prediction model, covering "
            "intended use, limitations, ethical considerations, and metrics."
        ),
        backstory=(
            "You believe AI systems must be transparent and accountable. "
            "Your model cards are the gold standard — clear, honest, "
            "and actionable for both technical and non-technical readers. "
            "You use MCP filesystem access to read evaluation reports "
            "directly from disk before writing documentation."
        ),
        llm=_build_llm(),
        tools=mcp_tools,
        memory=True,
        verbose=True,
        max_iter=CREW_MAX_ITER,
    )
