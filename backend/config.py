import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(PROJECT_ROOT / ".env")

SCALEDOWN_URL = os.getenv("SCALEDOWN_URL", "https://api.scaledown.xyz/compress/raw/")
SCALEDOWN_API_KEY = os.getenv("SCALEDOWN_API_KEY", "")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
APP_API_KEY = os.getenv("APP_API_KEY", "")

ENABLE_SEMANTIC_RAG = os.getenv("ENABLE_SEMANTIC_RAG", "0").strip() == "1"
SEMANTIC_MODEL_NAME = os.getenv("SEMANTIC_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K_SMART = int(os.getenv("RAG_TOP_K_SMART", "60"))
RAG_TOP_K_NAIVE = int(os.getenv("RAG_TOP_K_NAIVE", "120"))

SOCIAL_X_HANDLES = [h.strip() for h in os.getenv("SOCIAL_X_HANDLES", "PIBFactCheck,RBI").split(",") if h.strip()]
SOCIAL_INSTAGRAM_HANDLES = [
	h.strip() for h in os.getenv("SOCIAL_INSTAGRAM_HANDLES", "pib_india").split(",") if h.strip()
]

# Approximate per-1K-token pricing used for workshop benchmarking.
INPUT_COST_PER_1K = float(os.getenv("INPUT_COST_PER_1K", "0.0025"))
