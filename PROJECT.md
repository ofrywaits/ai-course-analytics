# AI Course Analytics Platform — Project Definition

## Step 1: The Problem

**What is the problem?**
Course creators on Udemy publish courses without data-driven guidance.
They don't know which category, price point, or publishing time maximizes reach.
The result: most courses attract fewer than 500 subscribers despite significant effort.

**Who experiences this pain?**
Independent instructors and small content teams publishing on online learning platforms.
They typically have domain expertise but no data science background.

**Why is it worth solving?**
- 200,000+ courses on Udemy compete for the same learners
- A course that earns 10,000 subscribers vs. 500 represents a 20× revenue difference
- No free, automated tool exists that analyses the full Udemy catalogue and gives actionable advice

**What does a good solution look like?**
An automated platform that ingests raw course data, runs full EDA, trains a popularity
prediction model, and surfaces clear business recommendations — all without requiring
the instructor to write a single line of code.

---

## Step 2: The User

**User Persona — "Maya, the Independent Instructor"**

| Attribute | Detail |
|-----------|--------|
| Age | 32 |
| Role | Full-time instructor, former software developer |
| Tech Level | Intermediate — comfortable with spreadsheets, not data science |
| Daily Challenge | Spends weeks building a course, then wonders why it gets no traction |
| Goal | Understand what makes a Udemy course popular before investing time |

**What blocks Maya today?**
- Raw data is publicly available but too large and messy to analyse manually
- Existing tools either cost money or require coding skills
- No clear answer to: "Should I price at $19.99 or $84.99? Does the month I publish matter?"

**Desired Outcome**
- *"I want to see, in under 5 minutes, exactly what pricing, category, and timing gives the
  best chance of my course becoming popular."*
- Maya feels: **empowered and confident** before launch
- Workflow change: before publishing → checks the platform → adjusts strategy

**User Story**
> As an independent course creator, I want to analyse 200,000+ Udemy courses
> so that I can make data-driven decisions about pricing, category, and launch timing
> before I publish my next course.

---

## Step 3: The MVP

### Must Have ✅
- Automated data pipeline: load raw CSV → clean → save `clean_data.csv`
- EDA report with 6 embedded charts saved as `eda_report.html`
- Business insights report (`insights.md`) with actionable recommendations
- ML model trained on engineered features — predicts course popularity (is_popular)
- Model evaluation report (`evaluation_report.md`) with accuracy, CV score, confusion matrix
- Model Card (`model_card.md`) documenting fairness, limitations, intended use
- Streamlit dashboard with 4 tabs: Overview, EDA Report, Model Results, Downloads
- One-click "Run Full Analysis" button in the dashboard

### Nice to Have 🔵
- Course popularity predictor — input your own course details, get a prediction
- Email/export of insights report
- Trend analysis over time (which categories are growing?)

### Out of Scope ❌
- Real-time Udemy API integration
- Multi-platform support (Coursera, edX)
- Payment processing or user accounts
- Mobile app

---

## Step 4: UX — User Flow

```
[User opens dashboard]
        │
        ▼
[Overview Tab — sees KPIs: total courses, avg price, avg rating, % popular]
        │
        ├── First time? → Click "Run Full Analysis" (progress bar shows 8 steps)
        │                        │
        │                        ▼
        │              [Pipeline runs: clean → EDA → insights → ML → model card]
        │                        │
        │                        ▼
        │              [Page refreshes with all data loaded]
        │
        ▼
[EDA Report Tab — scrollable HTML with 6 charts + business insights text]
        │
        ▼
[Model Results Tab — accuracy KPIs, confusion matrix, feature importance chart, model card]
        │
        ▼
[Downloads Tab — one-click download for all 8 output files]
```

**Where might the user get stuck?**
- If the pipeline hasn't run yet → clear warning message with call-to-action button
- If a file is missing → red ❌ badge with file name (not a silent crash)

---

## Step 5: UI Design Decisions

| Element | Choice | Reasoning |
|---------|--------|-----------|
| Framework | Streamlit | Zero front-end code, fast iteration |
| Color scheme | Purple gradient (#667eea → #764ba2) | Professional, distinct from default Streamlit blue |
| KPI cards | Custom HTML cards with large bold numbers | Immediate visual impact, scannable |
| Charts | Plotly (interactive) in dashboard, Matplotlib (embedded) in HTML report | Best tool per context |
| Layout | Wide layout, tabbed navigation | Maximises data density without scrolling |
| Run button | Primary type, full width | Clear primary call-to-action |
| File status | Green ✅ / Red ❌ badges | Instant system health check |

---

## Architecture Summary

```
data/raw.csv (209,735 rows)
        │
        ▼
  tools/data_tools.py  ──►  outputs/clean_data.csv
        │                    outputs/dataset_contract.json
        ▼
  tools/viz_tools.py   ──►  (charts embedded in EDA report)
        │
        ▼
  Crew 1 (analyst_crew)
  ├── Agent 1: Senior Data Engineer     → data quality report
  ├── Agent 2: Senior Data Analyst      → EDA summary (3 paragraphs)
  └── Agent 3: BI Analyst               → outputs/insights.md
        │
        ▼  [Validation: clean_data, eda_report, insights all exist]
        │
        ▼
  tools/model_tools.py ──►  outputs/features.csv
                             outputs/model.pkl  (Random Forest, 87.4% accuracy)
                             outputs/evaluation_report.md
        │
        ▼
  Crew 2 (scientist_crew)
  ├── Agent 4: Senior Feature Engineer  → feature engineering report
  ├── Agent 5: Senior ML Engineer       → model evaluation analysis
  └── Agent 6: AI Ethics Specialist     → outputs/model_card.md
        │
        ▼  [Validation: features, model, eval report, model card all exist]
        │
        ▼
  app.py (Streamlit)  ──►  Interactive dashboard on localhost:8501
```
