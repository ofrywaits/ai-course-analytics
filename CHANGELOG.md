# Changelog — AI Course Analytics Platform

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.3.0] — 2026-06-02 — Final Polish & Grading Sprint

### Added
- **Course Popularity Predictor** (Tab 4 in dashboard) — users enter their own course
  details and get a live ML prediction with probability gauge and feature importance breakdown
- **Maya demo scenario** in the Predictor tab — real-world user story end-to-end
- **Mermaid architecture diagram** in README — renders on GitHub, shows full flow
- **Architecture Decisions section** in README — explains every major design choice
- **CHANGELOG.md** — this file, tracks all versions and decisions
- **Integration tests** — `tests/test_integration.py` covers the full data→features→model pipeline
- **`_load_encoder_maps()`** cached helper in app.py — encodes user input using the same
  LabelEncoder mappings that were used during model training

### Changed
- README completely rewritten: badges, Mermaid diagram, tech stack table, architecture decisions,
  project structure, user story — from minimal to production-quality documentation
- Tab order: Downloads moved to Tab 5, Predictor inserted as Tab 4

### Decision Log
- Chose `predict_proba` over `predict` alone to show confidence, not just binary output
- Used `plotly.graph_objects.Indicator` (gauge) — more compelling than a simple score
- Used the top 15 languages by frequency to keep the language dropdown usable
- Defaulted to the most common subcategory per selected category for the hidden `subcategory` feature

---

## [1.2.0] — 2026-06-02 — Monitoring & Deployment

### Added
- `monitoring.py` — `HealthReport` dataclass, `run_health_check()` scans all 8 output files,
  `load_run_metrics()` parses `run_summary.txt`, `log_system_info()` at startup
- Live **health banner** at the top of the Streamlit dashboard
- Post-run health check warning in `main.py`
- **Railway deployment config** — `Procfile`, `railway.json`, `.streamlit/config.toml`
- `DEPLOYMENT.md` — step-by-step Railway CLI and GitHub auto-deploy guide
- `pytest.ini` — test discovery, log_cli, warning suppression
- `tests/conftest.py` — session-scoped fixtures shared across all test modules
- **3 merged Pull Requests** on GitHub (#1 pytest-config, #2 monitoring, #3 deployment)

### Decision Log
- Railway chosen over Heroku (free tier still active) and Render (cold starts too slow)
- `HealthReport` as a dataclass (not dict) — typed, IDE-friendly, easier to test
- Streamlit banner uses `try/except` — monitoring must never block the dashboard

---

## [1.1.0] — 2026-06-01 — Audit & Quality Sprint

### Added
- **MCP filesystem integration** — `tools/mcp_tools.py` wraps `MCPServerAdapter`
  with `@modelcontextprotocol/server-filesystem`; attached to Agents 3 and 6
- **50 unit tests** across 5 test files (data_tools, model_tools, validators, mcp_tools, monitoring)
- **`PROJECT.md`** — full problem definition, user persona (Maya), MVP spec, UX flow, UI decisions
- **Confusion matrix heatmap** in Model Results tab (plotly px.imshow)
- Docstrings and return type hints on all `app.py` helper functions
- `try/except` + logging added to every I/O operation in all crew files

### Changed
- All code converted from Hebrew to English (comments, docstrings, variable names)
- `flow.py` — added `-> None` return types to all 5 Flow methods, class docstring
- `tools/model_tools.py` — `save_model(model: object) -> None` type hint added
- EDA HTML regenerated with English chart titles

### Decision Log
- MCP only on Agents 3 and 6 (consumers) — Agents 1/2/4/5 are producers, MCP adds no value there
- LLM does narrative only (< 300 words per task) — all heavy computation in Python
  This keeps token usage under 12,000 TPM Groq free tier limit

---

## [1.0.0] — 2026-06-01 — Initial Release

### Added
- Full CrewAI Flow pipeline: 2 Crews, 6 Agents, validation gates between them
- `tools/data_tools.py` — 7-step data pipeline for 209,735 Udemy courses → 206,885 clean rows
- `tools/viz_tools.py` — 6 chart functions returning base64-embedded HTML `<img>` tags
- `tools/model_tools.py` — Random Forest vs Logistic Regression, RF wins (87.4% accuracy, CV 87.4%)
- `validation/validators.py` — `ValidationError`, `run_crew1_validations()`, `run_crew2_validations()`
- `flow.py` — `RetailAnalyticsFlow` with `RetailFlowState` (Pydantic BaseModel)
- `app.py` — Streamlit dashboard with 4 tabs and "Run Full Analysis" progress bar
- `config.py` — single source of truth for all paths, constants, ML params
- Git repository with `crew1`, `crew2`, `flow`, `ui` branches

### Decision Log
- Udemy dataset (209K rows) chosen over Superstore — richer, more relevant to AI course context
- Groq free tier chosen over OpenAI — zero cost, llama-3.3-70b-versatile is sufficient for summaries
- Python pre-computes all stats before passing to LLM — avoids rate limit errors
- `pathlib.Path` throughout — no hardcoded strings, works on Windows and macOS

---

## What's Next — v2.0 Roadmap (Step 18: Iteration)

Based on the platform's first run, here are the planned improvements for v2:

| Feature | Priority | Rationale |
|---------|----------|-----------|
| Real-time Udemy API integration | High | Fresh data instead of static CSV |
| Email report export | Medium | Maya wants to share insights with her team |
| Category trend analysis (time series) | Medium | Which categories are growing? |
| A/B price simulator | High | "What if I price at $19 vs $49?" |
| Multi-language dashboard | Low | Currently English-only |
| User authentication | Low | Save personal prediction history |

### Lessons Learned from v1

1. **LLM rate limits are real** — designing Python-first with LLM-as-narrator saved the project
2. **Validation gates add confidence** — catching bad data between crews prevented silent failures
3. **MCP graceful fallback is essential** — not every environment has npx installed
4. **50 tests caught 7 real bugs** during the audit phase — testing is not optional
