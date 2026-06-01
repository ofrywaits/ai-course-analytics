# -*- coding: utf-8 -*-
"""
Tasks של Crew 2 — Python עושה את העבודה הכבדה, LLM כותב את הניתוח.
"""

import pandas as pd
from crewai import Task
from crewai.agent import Agent

from config import (
    FEATURES_PATH,
    MODEL_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_CARD_PATH,
    TARGET_COLUMN,
)
from tools.model_tools import run_ml_pipeline


def build_feature_engineering_task(agent: Agent) -> Task:
    """Task 4 — הנדסת features + סיכום מהסוכן."""
    metrics = run_ml_pipeline()   # מריץ הכל מראש — Python עושה את העבודה

    feat_summary = (
        f"Features dataset saved to {FEATURES_PATH}.\n"
        f"Model trained: {metrics['best_model']}\n"
        f"Accuracy: {metrics['accuracy']}\n"
        f"CV Score: {metrics['cv_score']}\n"
        f"Top features: {list(metrics['feature_importance'].keys())[:5]}\n"
    )

    return Task(
        description=(
            f"The ML pipeline has completed. Here is the summary:\n\n"
            f"{feat_summary}\n\n"
            f"Write a professional feature engineering report (under 250 words):\n"
            f"1. Which features were created and why\n"
            f"2. How categorical variables were encoded\n"
            f"3. Which features appear most predictive\n"
            f"4. Any data quality observations during feature creation"
        ),
        expected_output=(
            "A feature engineering report under 250 words covering "
            "transformations applied and feature importance observations."
        ),
        agent=agent,
    )


def build_ml_training_task(agent: Agent, context_tasks: list) -> Task:
    """Task 5 — ניתוח תוצאות המודל."""
    eval_content = EVALUATION_REPORT_PATH.read_text(encoding="utf-8") \
        if EVALUATION_REPORT_PATH.exists() else "No report yet."

    return Task(
        description=(
            f"The model evaluation report has been generated:\n\n"
            f"{eval_content[:800]}\n\n"
            f"Write a professional model analysis (under 300 words):\n"
            f"1. Why the winning model outperformed the other\n"
            f"2. What the accuracy score means in business terms\n"
            f"3. How to interpret the confusion matrix results\n"
            f"4. Confidence level in deploying this model to production"
        ),
        expected_output=(
            "A model analysis report under 300 words with business interpretation "
            "of the results and deployment recommendation."
        ),
        agent=agent,
        context=context_tasks,
    )


def build_model_card_task(agent: Agent, context_tasks: list) -> Task:
    """Task 6 — כתיבת Model Card מלא ושמירתו."""
    eval_content = EVALUATION_REPORT_PATH.read_text(encoding="utf-8") \
        if EVALUATION_REPORT_PATH.exists() else ""

    import re
    acc_match = re.search(r"Test Accuracy \| ([0-9.]+)", eval_content)
    accuracy  = acc_match.group(1) if acc_match else "N/A"
    model_match = re.search(r"Best Model \| (.+)", eval_content)
    model_name  = model_match.group(1).strip() if model_match else "Unknown"

    return Task(
        description=(
            f"Write a complete Model Card for the Udemy course popularity "
            f"prediction model. Model: {model_name}, Accuracy: {accuracy}.\n\n"
            f"Structure EXACTLY as follows:\n\n"
            f"# Model Card — Udemy Course Popularity Predictor\n\n"
            f"## Model Details\n"
            f"[model type, version, training date]\n\n"
            f"## Intended Use\n"
            f"[primary use, out-of-scope uses]\n\n"
            f"## Training Data\n"
            f"[dataset description, size, features used]\n\n"
            f"## Metrics\n"
            f"[accuracy, CV score, confusion matrix summary]\n\n"
            f"## Limitations\n"
            f"[known limitations, bias risks]\n\n"
            f"## Ethical Considerations\n"
            f"[fairness, transparency, recommendations]\n\n"
            f"Save this model card to {MODEL_CARD_PATH}"
        ),
        expected_output=(
            "A complete model card saved to model_card.md with all 6 sections: "
            "Model Details, Intended Use, Training Data, Metrics, Limitations, "
            "Ethical Considerations."
        ),
        agent=agent,
        context=context_tasks,
    )
