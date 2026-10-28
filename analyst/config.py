"""Central configuration. Values can be overridden with environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

if not os.getenv("ANALYST_SANDBOX"):  # the sandbox worker must never load the API keys from .env
    load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_CSV = BASE_DIR / "data" / "sales_synthetic.csv"
DEMO_DICTIONARY = BASE_DIR / "data" / "sales_dictionary.json"

LLM_MODEL = os.getenv("ANALYST_LLM_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("ANALYST_EMBED_MODEL", "text-embedding-3-small")

# --- graph limits -----------------------------------------------------------
MAX_ATTEMPTS = 3          # C2: total executions allowed per code version (2 repairs)
MAX_REGEN = 1             # C3: answer/code regenerations allowed after a failed validation
RECURSION_LIMIT = 40      # hard stop for any graph loop

# --- sandbox ----------------------------------------------------------------
EXEC_TIMEOUT_S = 15
MEMORY_LIMIT_MB = 2048
RESULT_MAX_ROWS = 200
RESULT_MAX_CHARS = 8000
MAX_CODE_CHARS = 6000

# --- data limits ------------------------------------------------------------
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 50_000
MAX_COLS = 60

# --- web search -------------------------------------------------------------
WEB_MAX_RESULTS = 3
WEB_SNIPPET_CHARS = 800

# --- public demo protection -------------------------------------------------
MAX_INPUT_CHARS = 500
MAX_QUERIES_PER_SESSION = 20


def get_secret(name: str):
    """Read a secret from the environment first, then from Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None
