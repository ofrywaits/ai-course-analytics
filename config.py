# -*- coding: utf-8 -*-
"""
Central configuration file — all constants and paths are defined here.
No hardcoded values anywhere else in the project.
"""

from pathlib import Path

# ===== Base Paths =====
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR    = BASE_DIR / "logs"
CREWS_DIR   = BASE_DIR / "crews"

# ===== Input Files =====
RAW_DATA_PATH = DATA_DIR / "raw.csv"

# ===== Output Files =====
CLEAN_DATA_PATH        = OUTPUTS_DIR / "clean_data.csv"
EDA_REPORT_PATH        = OUTPUTS_DIR / "eda_report.html"
INSIGHTS_PATH          = OUTPUTS_DIR / "insights.md"
DATASET_CONTRACT_PATH  = OUTPUTS_DIR / "dataset_contract.json"
FEATURES_PATH          = OUTPUTS_DIR / "features.csv"
MODEL_PATH             = OUTPUTS_DIR / "model.pkl"
EVALUATION_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.md"
MODEL_CARD_PATH        = OUTPUTS_DIR / "model_card.md"

# ===== Logging =====
LOG_PATH        = LOGS_DIR / "run.log"
LOG_FORMAT      = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ===== Dataset Columns — Udemy Courses =====
TARGET_COLUMN       = "is_popular"           # Binary target: num_subscribers > median
NUMERIC_COLUMNS     = [
    "price",
    "num_subscribers",
    "avg_rating",
    "num_reviews",
    "num_comments",
    "num_lectures",
    "content_length_min",
]
CATEGORICAL_COLUMNS = [
    "category",
    "subcategory",
    "language",
    "is_paid",
]
DATE_COLUMNS  = ["published_time", "last_update_date"]
DROP_COLUMNS  = ["id", "course_url", "instructor_url", "headline", "topic"]
TEXT_COLUMNS  = ["title", "instructor_name"]

# ===== ML Model =====
TEST_SIZE              = 0.2
RANDOM_STATE           = 42
CV_FOLDS               = 5
MIN_ACCURACY_THRESHOLD = 0.70
POPULARITY_THRESHOLD   = None    # None = auto median

# ===== CrewAI =====
CREW_VERBOSE  = True
CREW_MAX_ITER = 10
LLM_MODEL     = "groq/llama-3.3-70b-versatile"

# ===== Streamlit =====
APP_TITLE  = "AI Course Analytics Platform"
APP_ICON   = "🎓"
APP_LAYOUT = "wide"
