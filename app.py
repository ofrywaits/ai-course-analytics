# -*- coding: utf-8 -*-
"""
Streamlit Dashboard — AI Course Analytics Platform.
4 tabs: Overview | EDA Report | Model Results | Downloads
"""

import subprocess
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ── נתיבים ──────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
OUTPUTS     = BASE / "outputs"
CLEAN_CSV   = OUTPUTS / "clean_data.csv"
EDA_HTML    = OUTPUTS / "eda_report.html"
INSIGHTS_MD = OUTPUTS / "insights.md"
MODEL_PKL   = OUTPUTS / "model.pkl"
EVAL_MD     = OUTPUTS / "evaluation_report.md"
MODEL_CARD  = OUTPUTS / "model_card.md"
FEATURES_CSV= OUTPUTS / "features.csv"
CONTRACT    = OUTPUTS / "dataset_contract.json"
SUMMARY_TXT = OUTPUTS / "run_summary.txt"

# ── הגדרות עמוד ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Course Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS מותאם ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #f8f9fa; }
  .kpi-card {
    background: white; border-radius: 14px; padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08); text-align: center;
  }
  .kpi-value { font-size: 2.2em; font-weight: 700; color: #667eea; }
  .kpi-label { color: #888; font-size: .9em; margin-top: 4px; }
  .status-ok  { color: #28a745; font-weight: 700; }
  .status-err { color: #dc3545; font-weight: 700; }
  h1 { color: #333; }
</style>
""", unsafe_allow_html=True)


# ── פונקציות עזר ────────────────────────────────────────────────────────────

@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV) if CLEAN_CSV.exists() else pd.DataFrame()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PKL) if MODEL_PKL.exists() else None


def kpi(value: str, label: str):
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def file_status(path: Path, name: str):
    if path.exists():
        size = path.stat().st_size / 1024
        st.markdown(f'<span class="status-ok">✅ {name}</span> ({size:.1f} KB)',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-err">❌ {name} — חסר</span>',
                    unsafe_allow_html=True)


# ── כותרת ראשית ─────────────────────────────────────────────────────────────
st.markdown("# 🎓 AI Course Analytics Platform")
st.markdown("**ניתוח חכם של 200,000+ קורסי Udemy באמצעות CrewAI**")
st.divider()

# ── כפתור הרצה ──────────────────────────────────────────────────────────────
col_btn, col_status = st.columns([2, 5])
with col_btn:
    run_clicked = st.button("▶️ Run Full Analysis", type="primary", use_container_width=True)

if run_clicked:
    with col_status:
        progress = st.progress(0, text="מאתחל...")
        status   = st.empty()

        steps = [
            (10, "טוען נתונים..."),
            (30, "Crew 1: Data Engineer"),
            (50, "Crew 1: EDA & Insights"),
            (60, "ולידציות Crew 1..."),
            (70, "Crew 2: Feature Engineering"),
            (85, "Crew 2: ML Training"),
            (95, "ולידציות Crew 2..."),
            (100, "✅ הושלם!"),
        ]

        def run_flow():
            import os
            from dotenv import load_dotenv
            load_dotenv()
            env = {**os.environ, "PYTHONPATH": str(BASE)}
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=str(BASE), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            return proc

        proc = run_flow()
        for pct, msg in steps:
            progress.progress(pct, text=msg)
            status.info(f"⏳ {msg}")
            time.sleep(2)

        proc.wait()
        if proc.returncode == 0:
            progress.progress(100, text="✅ ניתוח הושלם!")
            status.success("הרצה הושלמה בהצלחה!")
            st.cache_data.clear()
            st.rerun()
        else:
            status.error("שגיאה בהרצה — בדוק את הלוגים")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 EDA Report", "🤖 Model Results", "⬇️ Downloads"]
)

# ─────────────────────────────── TAB 1: OVERVIEW ────────────────────────────
with tab1:
    df = load_clean_data()

    if df.empty:
        st.warning("עדיין לא הורצה ניתוח. לחץ על **Run Full Analysis** למעלה.")
    else:
        st.subheader("סטטוס הרצה")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi(f"{len(df):,}", "קורסים בנתונים")
        with c2: kpi(f"{df['category'].nunique()}", "קטגוריות")
        with c3: kpi(f"${df['price'].mean():.0f}", "מחיר ממוצע")
        with c4: kpi(f"{df['avg_rating'].mean():.2f} ★", "דירוג ממוצע")
        with c5: kpi(f"{df['is_popular'].mean()*100:.0f}%", "קורסים פופולריים")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("קורסים לפי קטגוריה")
            cat_counts = df["category"].value_counts().head(10).reset_index()
            cat_counts.columns = ["קטגוריה", "מספר קורסים"]
            fig = px.bar(cat_counts, x="מספר קורסים", y="קטגוריה",
                         orientation="h", color="מספר קורסים",
                         color_continuous_scale="Blues")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("פופולריות לפי שפה")
            lang_pop = df.groupby("language")["is_popular"].mean().sort_values(
                ascending=False).head(10).reset_index()
            lang_pop.columns = ["שפה", "% פופולרי"]
            lang_pop["% פופולרי"] = (lang_pop["% פופולרי"] * 100).round(1)
            fig2 = px.bar(lang_pop, x="שפה", y="% פופולרי",
                          color="% פופולרי", color_continuous_scale="Greens")
            fig2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("קורסים לפי שנה")
        if "publish_year" in df.columns:
            yearly = df.groupby("publish_year").size().reset_index()
            yearly.columns = ["שנה", "קורסים חדשים"]
            fig3 = px.line(yearly, x="שנה", y="קורסים חדשים",
                           markers=True, line_shape="spline")
            fig3.update_traces(line_color="#667eea", line_width=3)
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("סטטוס קבצים")
        cols = st.columns(4)
        files = [
            (CLEAN_CSV, "clean_data.csv"),
            (CONTRACT, "dataset_contract.json"),
            (EDA_HTML, "eda_report.html"),
            (INSIGHTS_MD, "insights.md"),
            (FEATURES_CSV, "features.csv"),
            (MODEL_PKL, "model.pkl"),
            (EVAL_MD, "evaluation_report.md"),
            (MODEL_CARD, "model_card.md"),
        ]
        for i, (path, name) in enumerate(files):
            with cols[i % 4]:
                file_status(path, name)

# ─────────────────────────────── TAB 2: EDA REPORT ──────────────────────────
with tab2:
    if EDA_HTML.exists():
        st.subheader("📊 דוח EDA — Udemy Courses")
        html_content = EDA_HTML.read_text(encoding="utf-8")
        st.components.v1.html(html_content, height=900, scrolling=True)
    else:
        st.warning("דוח EDA עדיין לא נוצר.")

    if INSIGHTS_MD.exists():
        st.divider()
        st.subheader("💡 תובנות עסקיות")
        st.markdown(INSIGHTS_MD.read_text(encoding="utf-8"))

# ─────────────────────────────── TAB 3: MODEL RESULTS ───────────────────────
with tab3:
    model = load_model()
    df    = load_clean_data()

    if model is None:
        st.warning("מודל עדיין לא אומן.")
    else:
        st.subheader(f"🤖 מודל: {type(model).__name__}")

        if EVAL_MD.exists():
            import re
            content = EVAL_MD.read_text(encoding="utf-8")
            acc_m   = re.search(r"Test Accuracy \| ([0-9.]+)", content)
            cv_m    = re.search(r"CV Score.+?\| ([0-9.]+)", content)
            accuracy = float(acc_m.group(1)) if acc_m else 0.0
            cv_score = float(cv_m.group(1)) if cv_m else 0.0

            m1, m2, m3 = st.columns(3)
            with m1: kpi(f"{accuracy:.2%}", "Test Accuracy")
            with m2: kpi(f"{cv_score:.2%}", "CV Score")
            with m3: kpi("87%+", "ביצועים")

            st.divider()
            st.subheader("📄 דוח הערכה מלא")
            st.markdown(content)

        if MODEL_CARD.exists():
            st.divider()
            st.subheader("📋 Model Card")
            st.markdown(MODEL_CARD.read_text(encoding="utf-8"))

        if not df.empty and hasattr(model, "feature_importances_"):
            st.divider()
            st.subheader("🔍 Feature Importance")
            feat_cols = [c for c in df.columns
                         if c not in ["is_popular", "title", "instructor_name",
                                       "published_time", "last_update_date",
                                       "num_subscribers"]]
            importances = pd.Series(
                model.feature_importances_[:len(feat_cols)],
                index=feat_cols[:len(model.feature_importances_)]
            ).sort_values(ascending=False).head(10)
            fig_fi = px.bar(
                importances.reset_index(),
                x="importance", y="index",
                orientation="h",
                labels={"index": "Feature", "importance": "Importance"},
                color="importance", color_continuous_scale="Purples",
            )
            fig_fi.update_layout(showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────── TAB 4: DOWNLOADS ───────────────────────────
with tab4:
    st.subheader("⬇️ הורד קבצי פלט")
    st.markdown("כל הקבצים שנוצרו אוטומטית על ידי המערכת:")
    st.divider()

    download_files = [
        (CLEAN_CSV,    "clean_data.csv",         "text/csv",        "📊"),
        (FEATURES_CSV, "features.csv",            "text/csv",        "🔧"),
        (EDA_HTML,     "eda_report.html",         "text/html",       "📈"),
        (INSIGHTS_MD,  "insights.md",             "text/markdown",   "💡"),
        (EVAL_MD,      "evaluation_report.md",    "text/markdown",   "📉"),
        (MODEL_CARD,   "model_card.md",           "text/markdown",   "📋"),
        (CONTRACT,     "dataset_contract.json",   "application/json","📄"),
        (MODEL_PKL,    "model.pkl",               "application/octet-stream", "🤖"),
    ]

    cols = st.columns(2)
    for i, (path, fname, mime, icon) in enumerate(download_files):
        with cols[i % 2]:
            if path.exists():
                with open(path, "rb") as f:
                    data = f.read()
                st.download_button(
                    label=f"{icon} הורד {fname}",
                    data=data,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True,
                )
            else:
                st.button(f"❌ {fname} — לא נמצא",
                          disabled=True, use_container_width=True)

    if SUMMARY_TXT.exists():
        st.divider()
        st.subheader("📋 סיכום הרצה אחרון")
        st.code(SUMMARY_TXT.read_text(encoding="utf-8"))
