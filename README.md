# 🎓 AI Course Analytics Platform

An automated CrewAI Flow system for analyzing 200,000+ Udemy courses.

## Architecture

```
Raw Data → Crew 1 (Analyst) → Validation → Crew 2 (Scientist) → Dashboard
```

**Crew 1** — 3 agents: Data Engineer → Data Analyst → BI Analyst  
**Crew 2** — 3 agents: Feature Engineer → ML Engineer → Ethics Specialist

## Quick Start

```bash
cd "retail-analytics-crewai"
source venv/bin/activate

# Run the full pipeline
python main.py

# Launch the dashboard only
streamlit run app.py
```

## Output Files

| File | Description |
|------|-------------|
| `clean_data.csv` | 206,885 cleaned courses |
| `eda_report.html` | EDA report with 6 embedded charts |
| `insights.md` | Business insights and recommendations |
| `dataset_contract.json` | Dataset metadata contract |
| `features.csv` | Engineered ML features |
| `model.pkl` | Random Forest model (87% accuracy) |
| `evaluation_report.md` | Model evaluation report |
| `model_card.md` | Model Card following Google standard |

## Tech Stack

- **CrewAI** — multi-agent orchestration
- **Groq LLM** — llama-3.3-70b-versatile (free tier)
- **scikit-learn** — Random Forest & Logistic Regression
- **Streamlit** — interactive dashboard
- **pandas / matplotlib / seaborn / plotly** — data processing & visualization

## Project Structure

```
retail-analytics-crewai/
├── main.py                  # Entry point
├── flow.py                  # CrewAI Flow orchestration
├── app.py                   # Streamlit dashboard
├── config.py                # All constants and paths
├── data/raw.csv             # Udemy dataset (209K courses)
├── crews/
│   ├── analyst_crew/        # Crew 1: Data analysis
│   └── scientist_crew/      # Crew 2: ML pipeline
├── tools/
│   ├── data_tools.py        # Data cleaning pipeline
│   ├── viz_tools.py         # Chart generation
│   └── model_tools.py       # ML training & evaluation
├── validation/
│   └── validators.py        # Output validation checks
└── outputs/                 # All generated files
```
