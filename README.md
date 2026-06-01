# 🎓 AI Course Analytics Platform

מערכת CrewAI Flow אוטומטית לניתוח 200,000+ קורסי Udemy.

## ארכיטקטורה

```
Raw Data → Crew 1 (Analyst) → Validation → Crew 2 (Scientist) → Dashboard
```

**Crew 1** — 3 סוכנים: Data Engineer → Data Analyst → BI Analyst
**Crew 2** — 3 סוכנים: Feature Engineer → ML Engineer → Ethics Specialist

## הרצה מהירה

```bash
cd "retail-analytics-crewai"
source venv/bin/activate

# Flow מלא
python main.py

# Dashboard בלבד
streamlit run app.py
```

## פלטים

| קובץ | תיאור |
|------|-------|
| `clean_data.csv` | 206,885 קורסים נקיים |
| `eda_report.html` | דוח EDA עם 6 גרפים |
| `insights.md` | תובנות עסקיות |
| `features.csv` | features מעובדים |
| `model.pkl` | Random Forest (87% accuracy) |
| `evaluation_report.md` | דוח הערכת מודל |
| `model_card.md` | Model Card לפי תקן Google |

## טכנולוגיות

- **CrewAI** — orchestration של סוכנים
- **Groq LLM** — llama-3.3-70b-versatile
- **scikit-learn** — Random Forest
- **Streamlit** — dashboard
- **pandas / matplotlib / seaborn** — עיבוד וויזואליזציה
