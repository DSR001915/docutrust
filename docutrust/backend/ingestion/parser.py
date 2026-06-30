"""
PDF -> plain text parsing.

Uses pypdf for text extraction. Returns per-page text so downstream
chunking can keep page-number provenance, which is what lets DocuTrust
cite "Source: policy.pdf, page 4" instead of just a filename.
"""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int  # 1-indexed, matches what a human would see in a PDF viewer
    text: str


def parse_pdf(file_path: str | Path) -> list[PageText]:
    """Extract text from every page of a PDF, preserving page numbers."""
    reader = PdfReader(str(file_path))
    pages: list[PageText] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = _clean_text(raw_text)
        if cleaned.strip():
            pages.append(PageText(page_number=idx, text=cleaned))

    if not pages:
        raise ValueError(
            f"No extractable text found in {file_path}. "
            "This may be a scanned/image-only PDF requiring OCR, which "
            "DocuTrust does not currently support."
        )

    return pages


def _clean_text(text: str) -> str:
    """Light normalization: collapse excessive whitespace, fix hyphenation
    artifacts from line-wrapped PDFs."""
    # Collapse runs of whitespace (but not newlines that separate paragraphs)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)
