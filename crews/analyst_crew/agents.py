# -*- coding: utf-8 -*-
"""
Three agents for Crew 1 — Data Analyst Crew.
Each agent is responsible for exactly one domain.
"""

import logging
from crewai import Agent
from config import CREW_MAX_ITER
from crews.shared import build_llm
from tools.mcp_tools import get_mcp_file_tools

logger = logging.getLogger(__name__)


def build_data_engineer() -> Agent:
    """Agent 1 — responsible for loading and cleaning the data."""
    return Agent(
        role="Senior Data Engineer",
        goal=(
            "Load the Udemy courses dataset, validate its integrity, "
            "clean missing values, fix data types, remove duplicates, "
            "and produce a clean dataset ready for analysis."
        ),
        backstory=(
            "You are a meticulous data engineer with 10 years of experience "
            "building reliable data pipelines. You never pass dirty data "
            "to downstream processes — quality is your top priority."
        ),
        llm=build_llm(),
        memory=True,
        verbose=True,
        max_iter=CREW_MAX_ITER,
    )


def build_data_analyst() -> Agent:
    """Agent 2 — responsible for EDA and visualizations."""
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
        llm=build_llm(),
        memory=True,
        verbose=True,
        max_iter=CREW_MAX_ITER,
    )


def build_bi_analyst() -> Agent:
    """Agent 3 — responsible for business insights.

    Equipped with MCP filesystem tools so it can directly read
    the generated EDA report and clean data from the outputs/ directory.
    """
    mcp_tools = get_mcp_file_tools()
    if mcp_tools:
        logger.info(f"BI Analyst: {len(mcp_tools)} MCP tool(s) attached")
    else:
        logger.info("BI Analyst: running without MCP tools (fallback mode)")

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
            "that non-technical stakeholders can act on immediately. "
            "You use MCP filesystem access to read the latest generated "
            "reports directly from disk."
        ),
        llm=build_llm(),
        tools=mcp_tools,
        memory=True,
        verbose=True,
        max_iter=CREW_MAX_ITER,
    )
