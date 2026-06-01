# -*- coding: utf-8 -*-
"""
Tasks של Crew 1 — הסוכנים מקבלים נתונים מוכנים ועושים רק ניתוח טקסטואלי.
כך חוסכים tokens ומגיעים לתוצאות מהירות ויציבות.
"""

import json
import pandas as pd
from crewai import Task
from crewai.agent import Agent
from config import (
    CLEAN_DATA_PATH,
    DATASET_CONTRACT_PATH,
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


# ── עוזר: מריץ pipeline ומכין סיכום קצר לסוכן ──────────────────────────────

def _prepare_data_summary() -> str:
    """מריץ את ה-pipeline ומחזיר סיכום קצר לסוכן."""
    df = run_data_pipeline()
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "top_categories": df["category"].value_counts().head(5).to_dict(),
        "avg_price": round(df["price"].mean(), 2),
        "avg_subscribers": round(df["num_subscribers"].mean(), 2),
        "avg_rating": round(df["avg_rating"].mean(), 2),
        "popular_ratio": round(df["is_popular"].mean() * 100, 1),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _build_eda_html(df: pd.DataFrame) -> None:
    """בונה דוח HTML מלא עם גרפים ושומר לקובץ."""
    charts = (
        plot_price_distribution(df)
        + plot_subscribers_by_category(df)
        + plot_correlation_heatmap(df)
        + plot_top_subcategories(df)
        + plot_courses_over_time(df)
        + plot_rating_distribution(df)
    )

    stats = df[["price", "num_subscribers", "avg_rating",
                "num_reviews", "num_lectures"]].describe().round(2)
    stats_html = stats.to_html(classes="stats-table", border=0)

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8"/>
  <title>Udemy EDA Report</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background:#f8f9fa;
            color:#333; margin:0; padding:20px; }}
    h1   {{ background: linear-gradient(135deg,#667eea,#764ba2);
            color:#fff; padding:30px; border-radius:12px; text-align:center; }}
    h2   {{ color:#667eea; border-bottom:2px solid #667eea;
            padding-bottom:8px; margin-top:40px; }}
    h3   {{ color:#555; }}
    .chart-block {{ background:#fff; border-radius:12px;
                    padding:20px; margin:20px 0;
                    box-shadow:0 2px 12px rgba(0,0,0,.08); }}
    .chart-block img {{ width:100%; border-radius:8px; }}
    .stats-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .stats-table th {{ background:#667eea; color:#fff; padding:10px; }}
    .stats-table td {{ padding:8px; border:1px solid #e0e0e0; text-align:center; }}
    .stats-table tr:nth-child(even) {{ background:#f0f0f0; }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px;
                 margin:20px 0; }}
    .kpi {{ background:#fff; border-radius:12px; padding:20px; text-align:center;
            box-shadow:0 2px 12px rgba(0,0,0,.08); }}
    .kpi .value {{ font-size:2em; font-weight:700; color:#667eea; }}
    .kpi .label {{ color:#888; font-size:.9em; margin-top:4px; }}
  </style>
</head>
<body>
  <h1>🎓 Udemy Courses — EDA Report</h1>

  <div class="kpi-grid">
    <div class="kpi">
      <div class="value">{len(df):,}</div>
      <div class="label">סה"כ קורסים</div>
    </div>
    <div class="kpi">
      <div class="value">{df['category'].nunique()}</div>
      <div class="label">קטגוריות</div>
    </div>
    <div class="kpi">
      <div class="value">{df['avg_rating'].mean():.2f} ★</div>
      <div class="label">דירוג ממוצע</div>
    </div>
    <div class="kpi">
      <div class="value">${df['price'].mean():.0f}</div>
      <div class="label">מחיר ממוצע</div>
    </div>
    <div class="kpi">
      <div class="value">{df['num_subscribers'].mean():,.0f}</div>
      <div class="label">מנויים ממוצע</div>
    </div>
    <div class="kpi">
      <div class="value">{df['is_popular'].mean()*100:.0f}%</div>
      <div class="label">קורסים פופולריים</div>
    </div>
  </div>

  <h2>📊 סטטיסטיקות בסיסיות</h2>
  <div class="chart-block">{stats_html}</div>

  <h2>📈 גרפים</h2>
  {charts}

</body>
</html>"""

    OUTPUTS_DIR.mkdir(exist_ok=True)
    EDA_REPORT_PATH.write_text(html, encoding="utf-8")


# ── Task builders ────────────────────────────────────────────────────────────

def build_data_engineering_task(agent: Agent) -> Task:
    """Task 1 — הסוכן מקבל סיכום נתונים ומאשר את הניקוי."""
    data_summary = _prepare_data_summary()

    return Task(
        description=(
            f"The data pipeline has already run. Here is the summary:\n\n"
            f"{data_summary}\n\n"
            f"Your job: Write a professional data quality report confirming:\n"
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
    """Task 2 — בונה HTML מהקוד ומבקש מהסוכן לכתוב תקציר."""
    df = pd.read_csv(CLEAN_DATA_PATH)
    _build_eda_html(df)

    top_cats = df.groupby("category")["num_subscribers"].mean().sort_values(ascending=False).head(5)
    top_cats_str = "\n".join([f"- {k}: {v:,.0f} avg subscribers" for k, v in top_cats.items()])

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
    """Task 3 — הסוכן כותב תובנות עסקיות ושומר לקובץ."""
    df = pd.read_csv(CLEAN_DATA_PATH)

    # מחשבים סטטיסטיקות מהקוד — לא מה-LLM
    best_price  = df[df["is_popular"] == 1]["price"].mode().iloc[0] if len(df) > 0 else "N/A"
    best_cat    = df.groupby("category")["num_subscribers"].mean().idxmax()
    best_month  = df.groupby("publish_month")["num_subscribers"].mean().idxmax()
    free_avg    = df[df["is_paid"] == False]["num_subscribers"].mean()
    paid_avg    = df[df["is_paid"] == True]["num_subscribers"].mean()

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
            "A markdown insights report saved to insights.md with: "
            "executive summary, 5 findings, 5 recommendations, metrics table."
        ),
        agent=agent,
        context=context_tasks,
    )
