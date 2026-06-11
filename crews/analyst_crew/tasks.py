# -*- coding: utf-8 -*-
"""
Tasks for Crew 1 — agents receive pre-computed data and write text analysis only.
This keeps LLM token usage low and results fast and reliable.
"""

import json
import logging
import pandas as pd
from crewai import Task
from crewai.agent import Agent

logger = logging.getLogger(__name__)
from config import (
    CLEAN_DATA_PATH,
    EDA_REPORT_PATH,
    INSIGHTS_PATH,
    OUTPUTS_DIR,
)
from tools.data_tools import run_data_pipeline
from tools.viz_tools import (
    plot_price_distribution,
    plot_subscribers_by_category,
    plot_correlation_heatmap,
    plot_top_subcategories,
    plot_courses_over_time,
    plot_rating_distribution,
)


def _prepare_data_summary() -> str:
    """Run the data pipeline and return a JSON summary string for the agent."""
    logger.info("Running data pipeline for Task 1 summary...")
    try:
        df = run_data_pipeline()
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        raise
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "top_categories": df["category"].value_counts().head(5).to_dict(),
        "avg_price": round(df["price"].mean(), 2),
        "avg_subscribers": round(df["num_subscribers"].mean(), 2),
        "avg_rating": round(df["avg_rating"].mean(), 2),
        "popular_ratio": round(df["is_popular"].mean() * 100, 1),
    }
    logger.info(f"Data summary ready — {len(df):,} rows")
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _build_eda_html(df: pd.DataFrame) -> None:
    """Build a full HTML report with embedded charts and save to file."""
    charts = (
        plot_price_distribution(df)
        + plot_subscribers_by_category(df)
        + plot_correlation_heatmap(df)
        + plot_top_subcategories(df)
        + plot_courses_over_time(df)
        + plot_rating_distribution(df)
    )

    stats      = df[["price", "num_subscribers", "avg_rating",
                      "num_reviews", "num_lectures"]].describe().round(2)
    stats_html = stats.to_html(classes="stats-table", border=0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Udemy EDA Report</title>
  <style>
    body        {{ font-family:'Segoe UI',sans-serif; background:#f8f9fa; color:#333; margin:0; padding:20px; }}
    h1          {{ background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; padding:30px; border-radius:12px; text-align:center; }}
    h2          {{ color:#667eea; border-bottom:2px solid #667eea; padding-bottom:8px; margin-top:40px; }}
    h3          {{ color:#555; }}
    .chart-block{{ background:#fff; border-radius:12px; padding:20px; margin:20px 0; box-shadow:0 2px 12px rgba(0,0,0,.08); }}
    .chart-block img{{ width:100%; border-radius:8px; }}
    .stats-table{{ width:100%; border-collapse:collapse; font-size:14px; }}
    .stats-table th{{ background:#667eea; color:#fff; padding:10px; }}
    .stats-table td{{ padding:8px; border:1px solid #e0e0e0; text-align:center; }}
    .stats-table tr:nth-child(even){{ background:#f0f0f0; }}
    .kpi-grid   {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:20px 0; }}
    .kpi        {{ background:#fff; border-radius:12px; padding:20px; text-align:center; box-shadow:0 2px 12px rgba(0,0,0,.08); }}
    .kpi .value {{ font-size:2em; font-weight:700; color:#667eea; }}
    .kpi .label {{ color:#888; font-size:.9em; margin-top:4px; }}
  </style>
</head>
<body>
  <h1>🎓 Udemy Courses — EDA Report</h1>
  <div class="kpi-grid">
    <div class="kpi"><div class="value">{len(df):,}</div><div class="label">Total Courses</div></div>
    <div class="kpi"><div class="value">{df['category'].nunique()}</div><div class="label">Categories</div></div>
    <div class="kpi"><div class="value">{df['avg_rating'].mean():.2f} ★</div><div class="label">Average Rating</div></div>
    <div class="kpi"><div class="value">${df['price'].mean():.0f}</div><div class="label">Average Price</div></div>
    <div class="kpi"><div class="value">{df['num_subscribers'].mean():,.0f}</div><div class="label">Avg Subscribers</div></div>
    <div class="kpi"><div class="value">{df['is_popular'].mean()*100:.0f}%</div><div class="label">Popular Courses</div></div>
  </div>
  <h2>Basic Statistics</h2>
  <div class="chart-block">{stats_html}</div>
  <h2>Charts</h2>
  {charts}
</body>
</html>"""

    OUTPUTS_DIR.mkdir(exist_ok=True)
    try:
        EDA_REPORT_PATH.write_text(html, encoding="utf-8")
        logger.info(f"EDA report saved: {EDA_REPORT_PATH}")
    except OSError as e:
        logger.error(f"Failed to save EDA report: {e}")
        raise


def build_data_engineering_task(agent: Agent) -> Task:
    """Task 1 — agent receives data summary and writes a quality report."""
    data_summary = _prepare_data_summary()
    return Task(
        description=(
            f"The data pipeline has already run. Here is the summary:\n\n"
            f"{data_summary}\n\n"
            f"Write a professional data quality report confirming:\n"
            f"1. The dataset was loaded and cleaned successfully\n"
            f"2. Key statistics (rows, columns, missing values handled)\n"
            f"3. The target variable 'is_popular' distribution\n"
            f"4. Any data quality observations\n"
            f"Keep your response under 300 words."
        ),
        expected_output=(
            "A concise data quality report (under 300 words) confirming "
            "the pipeline completed successfully with key statistics."
        ),
        agent=agent,
    )


def build_eda_task(agent: Agent, context_tasks: list) -> Task:
    """Task 2 — builds HTML EDA report from code, asks agent for written summary."""
    logger.info("Building EDA HTML report...")
    try:
        df = pd.read_csv(CLEAN_DATA_PATH)
    except FileNotFoundError as e:
        logger.error(f"clean_data.csv not found: {e}")
        raise
    _build_eda_html(df)

    top_cats = (
        df.groupby("category")["num_subscribers"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    top_cats_str = "\n".join(
        [f"- {k}: {v:,.0f} avg subscribers" for k, v in top_cats.items()]
    )

    return Task(
        description=(
            f"The EDA report HTML has been generated and saved to {EDA_REPORT_PATH}.\n\n"
            f"Top categories by subscribers:\n{top_cats_str}\n\n"
            f"Write a professional 3-paragraph EDA summary:\n"
            f"1. Overview of the dataset distribution\n"
            f"2. Key patterns found in pricing and ratings\n"
            f"3. Most important correlations discovered\n"
            f"Keep your response under 250 words."
        ),
        expected_output=(
            "A 3-paragraph EDA summary (under 250 words) highlighting "
            "key findings from the Udemy dataset analysis."
        ),
        agent=agent,
        context=context_tasks,
    )


def build_insights_task(agent: Agent, context_tasks: list) -> Task:
    """Task 3 — agent writes structured business insights from pre-computed stats."""
    logger.info("Computing business insight statistics for Task 3...")
    try:
        df = pd.read_csv(CLEAN_DATA_PATH)
    except FileNotFoundError as e:
        logger.error(f"clean_data.csv not found: {e}")
        raise

    best_price = df[df["is_popular"] == 1]["price"].mode().iloc[0] if len(df) > 0 else "N/A"
    best_cat   = df.groupby("category")["num_subscribers"].mean().idxmax()
    best_month = df.groupby("publish_month")["num_subscribers"].mean().idxmax()
    free_avg   = df[~df["is_paid"]]["num_subscribers"].mean()
    paid_avg   = df[df["is_paid"]]["num_subscribers"].mean()

    context_str = (
        f"Key statistics:\n"
        f"- Best performing category: {best_cat}\n"
        f"- Most common price for popular courses: ${best_price}\n"
        f"- Best month to publish: Month {best_month}\n"
        f"- Free courses avg subscribers: {free_avg:,.0f}\n"
        f"- Paid courses avg subscribers: {paid_avg:,.0f}\n"
    )

    return Task(
        description=(
            f"{context_str}\n\n"
            f"Write actionable business insights for course creators.\n"
            f"Structure your response EXACTLY as:\n\n"
            f"# Executive Summary\n[2-3 sentences]\n\n"
            f"# Top Findings\n[5 bullet points with data]\n\n"
            f"# Recommendations\n[5 actionable tips]\n\n"
            f"# Key Metrics Table\n[markdown table]\n\n"
            f"Save this content to {INSIGHTS_PATH}"
        ),
        expected_output=(
            "A markdown insights report with: executive summary, "
            "5 findings, 5 recommendations, and a metrics table."
        ),
        agent=agent,
        context=context_tasks,
    )
