
import hashlib
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

def get_file_hash(file_path: str | Path) -> str:
    """Create a stable hash to identify the document."""
    file_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            file_hash.update(block)
    return file_hash.hexdigest()


def extract_text(file_path: str | Path) -> list[dict]:
    """Extract text from PDF, HTML or Markdown files."""
    path = Path(file_path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")
    
    pages = []

    if extension == ".pdf":
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "text": text.strip(),
                    "page": page_number,
                })
    elif extension in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

        if text:
            pages.append({
                "text": text,
                "page": 1,
            })
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": 1,
            })
    if not pages:
        raise ValueError(
            f"No readable text found in {path.name}. " "Scanned PDFs may require OCR."
        )
    return pages

def split_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    text = text.strip()

    if not text: return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a word boundary when possible.
        if end < len(text):
            boundary = max(
                text.rfind(" ", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - CHUNK_OVERLAP

        # Ensure the loop always moves forward.
        start = max(next_start, start + 1)

    return chunks


def process_document(file_path: str | Path) -> list[dict]:
    """Extract, chunk and attach metadata to a document."""
    path = Path(file_path)
    document_id = get_file_hash(path)
    pages = extract_text(path)

    result = []
    chunk_index = 0

    for page_data in pages:
        chunks = split_text(page_data["text"])

        for chunk in chunks:
            metadata = {
                "document_id": document_id,
                "source": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "page": page_data["page"],
                "chunk_index": chunk_index,
            }

            result.append({
                "id": f"{document_id}_{chunk_index}",
                "text": chunk,
                "metadata": metadata,
            })

            chunk_index += 1

    return result
