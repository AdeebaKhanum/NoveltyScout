"""Central configuration for NoveltyScout.

Everything tunable lives here so that the viva answer to
"where do you set X?" is always "config.py".
"""
import os
from pathlib import Path

try:  # optional: read GOOGLE_API_KEY etc. from a .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv not installed -> use real environment only
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"          # persistent vector store on disk
SAMPLE_CORPUS = DATA_DIR / "sample_prior_art.json"

# ---- LLM -------------------------------------------------------------
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")   # "gemini" | "ollama"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# Tried automatically if the main Gemini model is overloaded (503) or unavailable. Comma separated.
GEMINI_FALLBACKS = [m.strip() for m in os.getenv("GEMINI_FALLBACKS", "gemini-3.7-flash,gemini-3.6-flash").split(",") if m.strip()]
LLM_TEMPERATURE = 0.2          # used for local (Ollama) models; Gemini 3.x manages sampling itself

# ---- Retrieval / vector DB ------------------------------------------
COLLECTION_NAME = "prior_art"
TOP_K = 4                      # evidence chunks retrieved per feature
LOW_SIMILARITY = 0.30          # below this the agent thinks "evidence is weak"
MAX_SEARCH_ROUNDS = 2          # ReAct-style loop limit (search -> reflect -> search)
RESULTS_PER_SOURCE = 5         # live results pulled per query per source
CHUNK_WORDS = 180              # chunk size for user-uploaded long text
CHUNK_OVERLAP = 30

# ---- Network ---------------------------------------------------------
HTTP_TIMEOUT = 15
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")   # optional "polite pool" e-mail

DISCLAIMER = (
    "This is an automated screening aid, NOT a legal patentability opinion. "
    "It searches a limited set of sources and an AI model judges the overlap, so it can miss "
    "relevant prior art. Consult a registered patent agent before filing."
)