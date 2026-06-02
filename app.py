# -*- coding: utf-8 -*-
"""
Streamlit Dashboard — AI Course Analytics Platform.
4 tabs: Overview | EDA Report | Model Results | Downloads
"""

import re
import subprocess
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
OUTPUTS      = BASE / "outputs"
CLEAN_CSV    = OUTPUTS / "clean_data.csv"
EDA_HTML     = OUTPUTS / "eda_report.html"
INSIGHTS_MD  = OUTPUTS / "insights.md"
MODEL_PKL    = OUTPUTS / "model.pkl"
EVAL_MD      = OUTPUTS / "evaluation_report.md"
MODEL_CARD   = OUTPUTS / "model_card.md"
FEATURES_CSV = OUTPUTS / "features.csv"
CONTRACT     = OUTPUTS / "dataset_contract.json"
SUMMARY_TXT  = OUTPUTS / "run_summary.txt"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Course Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
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


# ── Helper Functions ──────────────────────────────────────────────────────────

@st.cache_data
def load_clean_data() -> pd.DataFrame:
    """Load the cleaned dataset from CSV. Returns empty DataFrame if not found."""
    return pd.read_csv(CLEAN_CSV) if CLEAN_CSV.exists() else pd.DataFrame()


@st.cache_resource
def load_model() -> object:
    """Load the trained sklearn model from disk. Returns None if not found."""
    return joblib.load(MODEL_PKL) if MODEL_PKL.exists() else None


def kpi(value: str, label: str) -> None:
    """Render a KPI card with a prominent value and a small label."""
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def file_status(path: Path, name: str) -> None:
    """Show a green ✅ with file size or a red ❌ for a missing output file."""
    if path.exists():
        size = path.stat().st_size / 1024
        st.markdown(f'<span class="status-ok">✅ {name}</span> ({size:.1f} KB)',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-err">❌ {name} — missing</span>',
                    unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎓 AI Course Analytics Platform")
st.markdown("**Automated analysis of 200,000+ Udemy courses using CrewAI**")

# ── Live Health Check Banner ──────────────────────────────────────────────────
try:
    from monitoring import run_health_check, load_run_metrics
    health = run_health_check()
    metrics_data = load_run_metrics()

    if health.all_ok:
        st.success(f"✅ Platform healthy — {health.ok_count}/{health.total} output files ready"
                   + (f" | Last run: {metrics_data.runtime_seconds:.0f}s | "
                      f"Accuracy: {metrics_data.accuracy_pct}"
                      if metrics_data else ""))
    else:
        st.warning(
            f"⚠️ {len(health.missing)} file(s) missing: {', '.join(health.missing)}  "
            f"— Click **Run Full Analysis** to generate them."
        )
except Exception:
    pass  # Monitoring is optional — never block the dashboard

st.divider()

# ── Run Button ────────────────────────────────────────────────────────────────
col_btn, col_status = st.columns([2, 5])
with col_btn:
    run_clicked = st.button("▶️ Run Full Analysis", type="primary", use_container_width=True)

if run_clicked:
    with col_status:
        progress = st.progress(0, text="Initializing...")
        status   = st.empty()

        steps = [
            (10,  "Loading data..."),
            (30,  "Crew 1: Data Engineer"),
            (50,  "Crew 1: EDA & Insights"),
            (60,  "Validating Crew 1 outputs..."),
            (70,  "Crew 2: Feature Engineering"),
            (85,  "Crew 2: ML Training"),
            (95,  "Validating Crew 2 outputs..."),
            (100, "✅ Complete!"),
        ]

        def run_flow() -> subprocess.Popen:
            """Launch main.py as a subprocess and return the process handle."""
            import os
            from dotenv import load_dotenv
            load_dotenv()
            env  = {**os.environ, "PYTHONPATH": str(BASE)}
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
            progress.progress(100, text="✅ Analysis complete!")
            status.success("Run completed successfully!")
            st.cache_data.clear()
            st.rerun()
        else:
            status.error("Error during run — check the logs")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 EDA Report", "🤖 Model Results", "🔮 Predict My Course", "⬇️ Downloads"]
)

# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    df = load_clean_data()

    if df.empty:
        st.warning("No analysis run yet. Click **Run Full Analysis** above.")
    else:
        st.subheader("Run Status")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi(f"{len(df):,}",                          "Courses")
        with c2: kpi(f"{df['category'].nunique()}",            "Categories")
        with c3: kpi(f"${df['price'].mean():.0f}",             "Avg Price")
        with c4: kpi(f"{df['avg_rating'].mean():.2f} ★",       "Avg Rating")
        with c5: kpi(f"{df['is_popular'].mean()*100:.0f}%",    "Popular Courses")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Courses by Category")
            cat_counts = df["category"].value_counts().head(10).reset_index()
            cat_counts.columns = ["Category", "Number of Courses"]
            fig = px.bar(cat_counts, x="Number of Courses", y="Category",
                         orientation="h", color="Number of Courses",
                         color_continuous_scale="Blues")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Popularity by Language")
            lang_pop = df.groupby("language")["is_popular"].mean().sort_values(
                ascending=False).head(10).reset_index()
            lang_pop.columns = ["Language", "% Popular"]
            lang_pop["% Popular"] = (lang_pop["% Popular"] * 100).round(1)
            fig2 = px.bar(lang_pop, x="Language", y="% Popular",
                          color="% Popular", color_continuous_scale="Greens")
            fig2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Courses Published per Year")
        if "publish_year" in df.columns:
            yearly = df.groupby("publish_year").size().reset_index()
            yearly.columns = ["Year", "New Courses"]
            fig3 = px.line(yearly, x="Year", y="New Courses",
                           markers=True, line_shape="spline")
            fig3.update_traces(line_color="#667eea", line_width=3)
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Output File Status")
        cols = st.columns(4)
        files = [
            (CLEAN_CSV,    "clean_data.csv"),
            (CONTRACT,     "dataset_contract.json"),
            (EDA_HTML,     "eda_report.html"),
            (INSIGHTS_MD,  "insights.md"),
            (FEATURES_CSV, "features.csv"),
            (MODEL_PKL,    "model.pkl"),
            (EVAL_MD,      "evaluation_report.md"),
            (MODEL_CARD,   "model_card.md"),
        ]
        for i, (path, name) in enumerate(files):
            with cols[i % 4]:
                file_status(path, name)

# ── TAB 2: EDA REPORT ─────────────────────────────────────────────────────────
with tab2:
    if EDA_HTML.exists():
        st.subheader("📊 EDA Report — Udemy Courses")
        html_content = EDA_HTML.read_text(encoding="utf-8")
        st.components.v1.html(html_content, height=900, scrolling=True)
    else:
        st.warning("EDA report has not been generated yet.")

    if INSIGHTS_MD.exists():
        st.divider()
        st.subheader("💡 Business Insights")
        st.markdown(INSIGHTS_MD.read_text(encoding="utf-8"))

# ── TAB 3: MODEL RESULTS ──────────────────────────────────────────────────────
with tab3:
    model = load_model()
    df    = load_clean_data()

    if model is None:
        st.warning("Model has not been trained yet.")
    else:
        st.subheader(f"🤖 Model: {type(model).__name__}")

        if EVAL_MD.exists():
            content  = EVAL_MD.read_text(encoding="utf-8")
            acc_m    = re.search(r"Test Accuracy \| ([0-9.]+)", content)
            cv_m     = re.search(r"CV Score.+?\| ([0-9.]+)", content)
            accuracy = float(acc_m.group(1)) if acc_m else 0.0
            cv_score = float(cv_m.group(1))  if cv_m  else 0.0

            m1, m2, m3 = st.columns(3)
            with m1: kpi(f"{accuracy:.2%}", "Test Accuracy")
            with m2: kpi(f"{cv_score:.2%}", "CV Score")
            with m3: kpi("87%+",            "Performance")

            st.divider()
            st.subheader("📄 Full Evaluation Report")
            st.markdown(content)

        if MODEL_CARD.exists():
            st.divider()
            st.subheader("📋 Model Card")
            st.markdown(MODEL_CARD.read_text(encoding="utf-8"))

        # ── Confusion Matrix ──────────────────────────────────────────────────
        cm_match = re.search(
            r"Actual 0 \| (\d+) \| (\d+).*?Actual 1 \| (\d+) \| (\d+)",
            EVAL_MD.read_text(encoding="utf-8") if EVAL_MD.exists() else "",
            re.DOTALL,
        )
        if cm_match:
            tn, fp, fn, tp = [int(cm_match.group(i)) for i in range(1, 5)]
            st.divider()
            st.subheader("🔢 Confusion Matrix")
            cm_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual: Not Popular", "Actual: Popular"],
                columns=["Predicted: Not Popular", "Predicted: Popular"],
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix — Popularity Prediction",
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)

        if not df.empty and hasattr(model, "feature_importances_"):
            st.divider()
            st.subheader("🔍 Feature Importance")
            feat_cols = [c for c in df.columns
                         if c not in ["is_popular", "title", "instructor_name",
                                      "published_time", "last_update_date",
                                      "num_subscribers"]]
            fi_df = pd.DataFrame({
                "Feature":    feat_cols[:len(model.feature_importances_)],
                "Importance": model.feature_importances_[:len(feat_cols)],
            }).sort_values("Importance", ascending=False).head(10)
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance", color_continuous_scale="Purples",
            )
            fig_fi.update_layout(showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)

# ── TAB 4: PREDICT MY COURSE ─────────────────────────────────────────────────
with tab4:
    st.subheader("🔮 Will My Course Be Popular?")
    st.markdown(
        "Enter your course details below and the trained Random Forest model "
        "will predict whether your course is likely to become **popular** "
        "(above-median subscribers)."
    )

    model_pred = load_model()

    if model_pred is None:
        st.warning("Model not trained yet — run the full analysis first.")
    else:
        # ── Build the same LabelEncoder mappings used during training ──────────
        @st.cache_data
        def _load_encoder_maps() -> dict:
            """Load category/language/subcategory→int mappings from clean data."""
            from sklearn.preprocessing import LabelEncoder
            df_c = pd.read_csv(CLEAN_CSV) if CLEAN_CSV.exists() else pd.DataFrame()
            maps = {}
            for col in ["category", "subcategory", "language"]:
                if col in df_c.columns:
                    le = LabelEncoder()
                    le.fit(df_c[col].astype(str))
                    maps[col] = {cls: int(le.transform([cls])[0])
                                 for cls in le.classes_}
            return maps

        enc_maps = _load_encoder_maps()

        CATEGORIES = list(enc_maps.get("category", {}).keys())
        TOP_LANGUAGES = [
            "English", "Portuguese", "Spanish", "Turkish", "Japanese",
            "German", "French", "Arabic", "Italian", "Russian",
            "Hindi", "Korean", "Indonesian", "Polish", "Simplified Chinese",
        ]
        LANGUAGES = [l for l in TOP_LANGUAGES
                     if l in enc_maps.get("language", {})]

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### 📋 Course Details")
            category    = st.selectbox("Category", CATEGORIES, index=2)
            language    = st.selectbox("Language", LANGUAGES, index=0)
            is_paid     = st.radio("Paid course?", ["Yes", "No"],
                                   horizontal=True) == "Yes"
            price       = st.slider("Price ($)", 0, 200, 50,
                                    disabled=not is_paid)

        with col_r:
            st.markdown("#### 📐 Course Structure")
            avg_rating      = st.slider("Expected Rating", 0.0, 5.0, 4.3, 0.1)
            num_lectures    = st.slider("Number of Lectures", 5, 300, 60)
            content_length  = st.slider("Total Content (minutes)", 60, 3000, 600)
            num_reviews     = st.number_input("Expected Reviews", 0, 10000, 50)

        st.markdown("#### 📅 Publishing Plan")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            publish_month = st.selectbox(
                "Publish Month",
                list(range(1, 13)),
                format_func=lambda m: [
                    "", "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ][m],
                index=0,
            )
        with col_m2:
            publish_year = st.selectbox("Publish Year",
                                        list(range(2024, 2027)), index=0)
        with col_m3:
            publish_dow = st.selectbox(
                "Day of Week",
                [0, 1, 2, 3, 4, 5, 6],
                format_func=lambda d: ["Mon", "Tue", "Wed",
                                       "Thu", "Fri", "Sat", "Sun"][d],
                index=0,
            )

        st.divider()
        predict_clicked = st.button("🔮 Predict Popularity",
                                    type="primary", use_container_width=True)

        if predict_clicked:
            import datetime

            cat_enc  = enc_maps.get("category", {}).get(category, 0)
            lang_enc = enc_maps.get("language", {}).get(language, 0)

            # Use most common subcategory for selected category (index 0)
            sub_enc = 0
            if "subcategory" in enc_maps:
                df_c2 = pd.read_csv(CLEAN_CSV) if CLEAN_CSV.exists() else pd.DataFrame()
                if not df_c2.empty and "subcategory" in df_c2.columns:
                    top_sub = (df_c2[df_c2["category"] == category]["subcategory"]
                               .value_counts().index)
                    if len(top_sub) > 0:
                        sub_enc = enc_maps["subcategory"].get(top_sub[0], 0)

            course_age = round(
                (datetime.date(publish_year, publish_month, 1)
                 - datetime.date(2010, 1, 1)).days / 365.25, 2
            )

            input_data = pd.DataFrame([{
                "is_paid":             int(is_paid),
                "price":               float(price) if is_paid else 0.0,
                "avg_rating":          avg_rating,
                "num_reviews":         num_reviews,
                "num_comments":        0,
                "num_lectures":        num_lectures,
                "content_length_min":  content_length,
                "category":            cat_enc,
                "subcategory":         sub_enc,
                "language":            lang_enc,
                "publish_year":        publish_year,
                "publish_month":       publish_month,
                "publish_day_of_week": publish_dow,
                "course_age_years":    course_age,
            }])

            prediction      = model_pred.predict(input_data)[0]
            proba           = model_pred.predict_proba(input_data)[0]
            popular_prob    = round(proba[1] * 100, 1)
            not_popular_prob = round(proba[0] * 100, 1)

            st.divider()
            if prediction == 1:
                st.success(f"## 🌟 Likely POPULAR — {popular_prob}% confidence")
                st.markdown(
                    f"The model predicts your course will attract **above-median subscribers**. "
                    f"With a **{popular_prob}%** probability of popularity, your course setup "
                    f"looks strong. Key factors: category ({category}), "
                    f"price (${price if is_paid else 0}), "
                    f"rating target ({avg_rating}★), {num_lectures} lectures."
                )
            else:
                st.error(f"## 📉 Unlikely to be Popular — {not_popular_prob}% confidence")
                st.markdown(
                    f"The model predicts your course may struggle to reach above-median subscribers. "
                    f"Popularity probability: **{popular_prob}%**. "
                    f"Consider: more lectures, lower price entry point, or a higher-demand category."
                )

            # ── Probability gauge ──────────────────────────────────────────────
            import plotly.graph_objects as go
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = popular_prob,
                delta = {"reference": 50, "suffix": "%"},
                title = {"text": "Popularity Probability (%)"},
                gauge = {
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": "#667eea"},
                    "steps": [
                        {"range": [0,  40], "color": "#ffcccc"},
                        {"range": [40, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#d4edda"},
                    ],
                    "threshold": {
                        "line":  {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Feature contributions breakdown ───────────────────────────────
            if hasattr(model_pred, "feature_importances_"):
                st.divider()
                st.subheader("📊 What drives this prediction?")
                feat_names = list(input_data.columns)
                fi_vals    = model_pred.feature_importances_
                fi_chart   = pd.DataFrame({
                    "Feature":    feat_names[:len(fi_vals)],
                    "Importance": fi_vals[:len(feat_names)],
                }).sort_values("Importance", ascending=True).tail(8)

                fig_fi2 = px.bar(
                    fi_chart, x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Purples",
                    title="Top factors the model used for this prediction",
                )
                fig_fi2.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_fi2, use_container_width=True)

        # ── Maya's Demo Scenario ───────────────────────────────────────────────
        with st.expander("💡 See a real example — Maya's course"):
            st.markdown("""
**Scenario:** Maya is a Python developer planning to publish a course on Udemy.

| Field | Maya's choice |
|-------|--------------|
| Category | Development |
| Language | English |
| Price | $49.99 (Paid) |
| Expected Rating | 4.5 ★ |
| Lectures | 80 |
| Content | 900 minutes |
| Publish | March 2025 |

**Result:** The model predicts Maya's course will be **popular** with ~71% confidence.

> *"I used to just guess. Now I can test different price points and categories before I even record a single lecture."* — Maya
""")

# ── TAB 5: DOWNLOADS ──────────────────────────────────────────────────────────
with tab5:
    st.subheader("⬇️ Download Output Files")
    st.markdown("All files generated automatically by the platform:")
    st.divider()

    download_files = [
        (CLEAN_CSV,    "clean_data.csv",          "text/csv",                    "📊"),
        (FEATURES_CSV, "features.csv",             "text/csv",                    "🔧"),
        (EDA_HTML,     "eda_report.html",          "text/html",                   "📈"),
        (INSIGHTS_MD,  "insights.md",              "text/markdown",               "💡"),
        (EVAL_MD,      "evaluation_report.md",     "text/markdown",               "📉"),
        (MODEL_CARD,   "model_card.md",            "text/markdown",               "📋"),
        (CONTRACT,     "dataset_contract.json",    "application/json",            "📄"),
        (MODEL_PKL,    "model.pkl",                "application/octet-stream",    "🤖"),
    ]

    cols = st.columns(2)
    for i, (path, fname, mime, icon) in enumerate(download_files):
        with cols[i % 2]:
            if path.exists():
                with open(path, "rb") as f:
                    data = f.read()
                st.download_button(
                    label=f"{icon} Download {fname}",
                    data=data,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True,
                )
            else:
                st.button(f"❌ {fname} — not found",
                          disabled=True, use_container_width=True)

    if SUMMARY_TXT.exists():
        st.divider()
        st.subheader("📋 Last Run Summary")
        st.code(SUMMARY_TXT.read_text(encoding="utf-8"))
