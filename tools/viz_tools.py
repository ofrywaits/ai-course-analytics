# -*- coding: utf-8 -*-
"""
כלים ליצירת גרפים מוטמעים כ-base64 בתוך HTML.
כל גרף נשמר בזיכרון — אין קבצי תמונה חיצוניים.
"""

import base64
import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE_WIDE  = (14, 5)
FIGSIZE_SQ    = (8, 6)
FIGSIZE_TALL  = (10, 8)


def _fig_to_base64(fig: plt.Figure) -> str:
    """ממיר Figure ל-string base64 לשימוש ב-HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _img_tag(b64: str, title: str = "") -> str:
    """מחזיר תג HTML של תמונה עם כותרת."""
    return (
        f'<div class="chart-block">'
        f'<h3>{title}</h3>'
        f'<img src="data:image/png;base64,{b64}" alt="{title}"/>'
        f'</div>'
    )


def plot_price_distribution(df: pd.DataFrame) -> str:
    """היסטוגרמה של התפלגות מחירים."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    paid = df[df["is_paid"] == True]["price"]
    axes[0].hist(paid[paid > 0], bins=50, color="#4C72B0", edgecolor="white")
    axes[0].set_title("התפלגות מחירי קורסים בתשלום")
    axes[0].set_xlabel("מחיר ($)")
    axes[0].set_ylabel("מספר קורסים")

    free_vs_paid = df["is_paid"].value_counts()
    axes[1].pie(
        free_vs_paid.values,
        labels=["בתשלום", "חינמי"],
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"],
        startangle=90,
    )
    axes[1].set_title("חינמי vs. בתשלום")
    fig.tight_layout()
    return _img_tag(_fig_to_base64(fig), "התפלגות מחירים")


def plot_subscribers_by_category(df: pd.DataFrame) -> str:
    """עמודות — ממוצע מנויים לפי קטגוריה."""
    avg = (
        df.groupby("category")["num_subscribers"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.barplot(x=avg.values, y=avg.index, palette="Blues_r", ax=ax)
    ax.set_title("ממוצע מנויים לפי קטגוריה (Top 15)")
    ax.set_xlabel("ממוצע מנויים")
    ax.set_ylabel("")
    fig.tight_layout()
    return _img_tag(_fig_to_base64(fig), "מנויים לפי קטגוריה")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """מפת חום של מתאמים בין עמודות מספריות."""
    num_cols = ["price", "num_subscribers", "avg_rating",
                "num_reviews", "num_lectures", "content_length_min"]
    existing = [c for c in num_cols if c in df.columns]
    corr = df[existing].corr()
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title("מטריצת מתאמים")
    fig.tight_layout()
    return _img_tag(_fig_to_base64(fig), "מטריצת מתאמים")


def plot_top_subcategories(df: pd.DataFrame) -> str:
    """Top 10 תת-קטגוריות לפי מספר קורסים."""
    top = df["subcategory"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.barplot(x=top.values, y=top.index, palette="viridis", ax=ax)
    ax.set_title("Top 10 תת-קטגוריות")
    ax.set_xlabel("מספר קורסים")
    fig.tight_layout()
    return _img_tag(_fig_to_base64(fig), "Top 10 תת-קטגוריות")


def plot_courses_over_time(df: pd.DataFrame) -> str:
    """גרף קווי — קורסים חדשים לפי שנה."""
    if "publish_year" not in df.columns:
        return ""
    yearly = df.groupby("publish_year").size()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(yearly.index, yearly.values, marker="o", color="#4C72B0", linewidth=2)
    ax.fill_between(yearly.index, yearly.values, alpha=0.15, color="#4C72B0")
    ax.set_title("קורסים חדשים לפי שנה")
    ax.set_xlabel("שנה")
    ax.set_ylabel("מספר קורסים")
    fig.tight_layout()
    return _img_tag(_fig_to_base64(fig), "צמיחת קורסים לאורך זמן")


def plot_rating_distribution(df: pd.DataFrame) -> str:
    """היסטוגרמה של דירוגים."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    df["avg_rating"].hist(bins=40, color="#55A868", edgecolor="white", ax=ax)
    ax.axvline(df["avg_rating"].mean(), color="red", linestyle="--",
               label=f'ממוצע: {df["avg_rating"].mean():.2f}')
    ax.set_title("התפלגות דירוגי קורסים")
    ax.set_xlabel("דירוג")
    ax.set_ylabel("מספר קורסים")
    ax.legend()
    fig.tight_layout()
    return _img_tag(_fig_to_base64(fig), "התפלגות דירוגים")
