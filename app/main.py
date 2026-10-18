
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import (
    BASE_DIR,
    MAX_UPLOAD_MB,
    SUPPORTED_EXTENSIONS,
)
from app.ingestion import process_document
from app.store import add_documents, get_stats, list_documents
from app.rag import answer_question


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cost-Efficient RAG Application",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    metadata_filter: dict | None = None


@app.get("/")
def home():
    return {
        "message": "Cost-Efficient RAG Application is running"
    }


@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    filename = (file.filename or "").replace("\\", "/")
    filename = Path(filename).name

    if not filename:
        raise HTTPException(400, "Filename is required")

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            "Only PDF, HTML and Markdown files are supported"
        )

    content = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)

    if not content:
        raise HTTPException(400, "Uploaded file is empty")

    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            413,
            f"File size exceeds {MAX_UPLOAD_MB} MB"
        )

    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(content)

    try:
        documents = process_document(file_path)
    except Exception as e:
        raise HTTPException(
            400,
            f"Could not read the document: {e}"
        ) from e

    result = add_documents(documents)

    return {
        "filename": filename,
        "chunks_found": len(documents),
        **result
    }


@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter
        )

        logger.info(
            "query_complete latency_ms=%s chunks=%s tokens=%s",
            result["total_latency_ms"],
            result["chunks_retrieved"],
            result["total_tokens"]
        )

        return result

    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(
            502,
            "Query failed. Check the application logs."
        ) from e


@app.get("/stats")
def stats():
    return get_stats()


@app.get("/documents")
def documents():
    return list_documents()
