# -*- coding: utf-8 -*-
"""
Shared utilities for all crews — single source of truth for LLM setup.
"""

import os
import logging
from crewai import LLM
from config import LLM_MODEL

logger = logging.getLogger(__name__)


def build_llm() -> LLM:
    """
    Build the shared LLM instance with Groq credentials.

    Raises:
        EnvironmentError: if GROQ_API_KEY is not set in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )
    logger.debug("Building LLM — model: %s", LLM_MODEL)
    return LLM(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=2048,
        max_retries=5,
    )
