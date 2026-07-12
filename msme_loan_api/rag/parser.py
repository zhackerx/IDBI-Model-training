from pathlib import Path

import fitz

from rag.security import sanitize_text_for_rag


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def extract_sanitized_text_from_pdf(pdf_path: Path) -> tuple[str, dict]:
    """Extract PDF text and remove sensitive or hazardous content before retrieval."""
    raw_text = extract_text_from_pdf(pdf_path)
    return sanitize_text_for_rag(raw_text)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Chunk text with overlap to preserve context across boundaries."""
    if not text:
        return []

    normalized = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break
        start = max(0, end - overlap)

    return chunks
