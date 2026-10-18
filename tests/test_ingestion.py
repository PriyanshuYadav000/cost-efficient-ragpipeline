
from pathlib import Path

import pytest

from app.ingestion import (
    extract_text,
    get_file_hash,
    process_document,
    split_text,
)


def test_empty_text():
    assert split_text("") == []


def test_short_text():
    assert split_text("Hello world") == ["Hello world"]


def test_long_text_is_split():
    text = "word " * 600
    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_size():
    text = "a" * 2500
    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_chunk_overlap():
    text = "a" * 2000
    chunks = split_text(text)

    assert chunks[0][-150:] == chunks[1][:150]


def test_markdown_extraction(tmp_path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Leave Policy\nEmployees receive 18 days of leave.",
        encoding="utf-8",
    )

    pages = extract_text(file_path)

    assert len(pages) == 1
    assert "18 days of leave" in pages[0]["text"]


def test_html_extraction(tmp_path):
    file_path = tmp_path / "policy.html"
    file_path.write_text(
        "<html><body><h1>Policy</h1>"
        "<p>Urgent tickets take one hour.</p>"
        "<script>hidden text</script></body></html>",
        encoding="utf-8",
    )

    pages = extract_text(file_path)
    text = pages[0]["text"]

    assert "Urgent tickets take one hour" in text
    assert "hidden text" not in text


def test_unsupported_file(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Some text", encoding="utf-8")

    with pytest.raises(ValueError):
        extract_text(file_path)


def test_file_hash_is_stable(tmp_path):
    file_path = tmp_path / "policy.md"
    file_path.write_text("Same content", encoding="utf-8")

    first_hash = get_file_hash(file_path)
    second_hash = get_file_hash(file_path)

    assert first_hash == second_hash


def test_processed_document_has_metadata(tmp_path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Policy\nEmployees receive 18 days of annual leave.",
        encoding="utf-8",
    )

    documents = process_document(file_path)

    assert documents
    assert documents[0]["id"]
    assert documents[0]["text"]
    assert documents[0]["metadata"]["source"] == "policy.md"
    assert documents[0]["metadata"]["page"] == 1
    assert documents[0]["metadata"]["document_id"]
