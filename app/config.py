
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GROQ_MODEL)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "data/chroma"))

if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = BASE_DIR / CHROMA_DIR


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MMR_FETCH_K = int(os.getenv("MMR_FETCH_K", "20"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))

MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.25"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".html",
    ".htm",
    ".md",
    ".markdown"
}


if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than zero")

if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

if DEFAULT_TOP_K <= 0:
    raise ValueError("DEFAULT_TOP_K must be greater than zero")

if MMR_FETCH_K < DEFAULT_TOP_K:
    raise ValueError("MMR_FETCH_K must be >= DEFAULT_TOP_K")

if not 0 <= MMR_LAMBDA <= 1:
    raise ValueError("MMR_LAMBDA must be between 0 and 1")

if not -1 <= MIN_RELEVANCE <= 1:
    raise ValueError("MIN_RELEVANCE must be between -1 and 1")

if MAX_UPLOAD_MB <= 0:
    raise ValueError("MAX_UPLOAD_MB must be greater than zero")
