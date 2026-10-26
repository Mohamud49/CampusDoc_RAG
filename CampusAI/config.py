"""
config.py — CampusAI global settings
------------------------------------
Waxaan halkan ku haynaa dhammaan settings-ka app-ka (API keys, paths, chunk sizes).
English: Central place for environment variables and defaults. Beginners: change .env, not this file often.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env file in project root (if it exists).
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- Paths ---
DATA_DIR = _PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_PERSIST_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db"))
)

# --- Text splitting (RAG chunk size) ---
# Chunk size: how many characters per piece of document.
# Overlap: small repeat so sentences at boundaries are not lost.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- LLM provider ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower().strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))

# --- Login (demo accounts seeded into data/users.json on first run) ---
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
USER_USERNAME = os.getenv("USER_USERNAME", "student").strip().lower()
USER_PASSWORD = os.getenv("USER_PASSWORD", "student123")


def ensure_directories() -> None:
    """
    Create data folders if missing.
    Somali: Samee faylasha data haddii aysan jirin.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
