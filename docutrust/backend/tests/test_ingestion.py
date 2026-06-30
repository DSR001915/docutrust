"""Tests for ingestion: parsing, chunking, embedding shape, idempotent re-ingest."""
import shutil
import tempfile
from pathlib import Path

import pytest

from backend.ingestion.chunker import chunk_pages
from backend.ingestion.embedder import EMBEDDING_DIM, MockEmbedder
from backend.ingestion.parser import PageText
from backend.ingestion.service import ingest_pdf, _content_hash_id
from backend.ingestion.vector_store import VectorStore


@pytest.fixture
def tmp_chroma_dir():
    d = tempfile.mkdtemp(prefix="docutrust_test_chroma_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_chunk_pages_no_duplication_at_overlap_boundary():
    """Regression test for the classic recursive-overlap bug: overlap text
    must appear exactly once at each chunk boundary, not stack up across
    multiple split passes.

    Uses non-repetitive sentences deliberately: a naive test with
    repetitive input (e.g. the same sentence repeated many times) would
    show high substring-occurrence counts even from a CORRECT chunker,
    since the repetition is in the source text itself -- that would be a
    false positive, not a real signal of the overlap bug.
    """
    sentences = [
        "The quarterly revenue report shows growth in the northeast region.",
        "Employee onboarding now requires a signed confidentiality agreement.",
        "The data center migration is scheduled for the third week of March.",
        "Customer support tickets increased by twelve percent last month.",
        "The new firmware update addresses three critical security vulnerabilities.",
        "Marketing spend was reallocated toward the enterprise segment this quarter.",
        "The compliance audit identified two minor process gaps to remediate.",
        "Warehouse inventory counts will be reconciled at the end of the fiscal year.",
    ]
    text = " ".join(sentences)
    pages = [PageText(page_number=1, text=text)]

    chunks = chunk_pages(pages, document_id="doc-1", document_name="test.pdf", chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1, "Expected multiple chunks given chunk_size << total text length"

    # Each sentence appears exactly once in the source text, so it should
    # appear at most twice in any single chunk: once normally, and at most
    # once more if it happens to fall within an overlap region shared with
    # the adjacent chunk. Three or more occurrences of the SAME sentence in
    # one chunk would indicate compounding overlap duplication.
    for chunk in chunks:
        for sentence in sentences:
            occurrences = chunk.text.count(sentence)
            assert occurrences <= 1, (
                f"Possible overlap duplication bug: sentence appears {occurrences} "
                f"times within a single chunk: {sentence!r}"
            )


def test_chunk_pages_preserves_page_numbers():
    pages = [
        PageText(page_number=1, text="Page one content. " * 20),
        PageText(page_number=2, text="Page two content. " * 20),
    ]
    chunks = chunk_pages(pages, document_id="doc-1", document_name="test.pdf", chunk_size=100, chunk_overlap=20)

    page_1_chunks = [c for c in chunks if c.page_number == 1]
    page_2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(page_1_chunks) > 0
    assert len(page_2_chunks) > 0
    assert all("Page one" in c.text for c in page_1_chunks)
    assert all("Page two" in c.text for c in page_2_chunks)


def test_mock_embedder_dimension():
    embedder = MockEmbedder()
    vec = embedder.embed_query("test query")
    assert len(vec) == EMBEDDING_DIM


def test_mock_embedder_relevant_scores_higher_than_irrelevant():
    """The core property the whole CRAG grading mechanism depends on:
    relevant text must score higher than irrelevant text against a query."""
    import numpy as np

    embedder = MockEmbedder()
    query = "data retention policy for customer records"
    relevant = "customer records must be retained under the data retention policy"
    irrelevant = "the cafeteria serves lunch on Tuesdays and Thursdays"

    qv = np.array(embedder.embed_query(query))
    rv = np.array(embedder.embed_documents([relevant])[0])
    iv = np.array(embedder.embed_documents([irrelevant])[0])

    cos_relevant = np.dot(qv, rv) / (np.linalg.norm(qv) * np.linalg.norm(rv))
    cos_irrelevant = np.dot(qv, iv) / (np.linalg.norm(qv) * np.linalg.norm(iv))

    assert cos_relevant > cos_irrelevant


def test_vector_store_upsert_and_query(tmp_chroma_dir):
    pages = [PageText(page_number=1, text="The retention policy requires seven years. " * 5)]
    chunks = chunk_pages(pages, document_id="doc-1", document_name="test.pdf", chunk_size=200, chunk_overlap=20)

    embedder = MockEmbedder()
    vectors = embedder.embed_documents([c.text for c in chunks])

    store = VectorStore(persist_dir=tmp_chroma_dir, collection_name="test")
    store.upsert_chunks(chunks, vectors)

    assert store.document_count() == len(chunks)

    query_vector = embedder.embed_query("retention policy years")
    results = store.query(query_vector, top_k=5)
    assert len(results) > 0
    assert results[0].document_name == "test.pdf"


def test_vector_store_delete_document(tmp_chroma_dir):
    pages = [PageText(page_number=1, text="Content for deletion test. " * 5)]
    chunks = chunk_pages(pages, document_id="doc-to-delete", document_name="test.pdf", chunk_size=200, chunk_overlap=20)

    embedder = MockEmbedder()
    vectors = embedder.embed_documents([c.text for c in chunks])

    store = VectorStore(persist_dir=tmp_chroma_dir, collection_name="test")
    store.upsert_chunks(chunks, vectors)
    assert store.document_count() == len(chunks)

    store.delete_document("doc-to-delete")
    assert store.document_count() == 0


def test_ingest_pdf_is_idempotent_for_identical_content(tmp_path, monkeypatch):
    """Regression test: re-uploading the exact same file content must not
    duplicate chunks in the vector store. See backend/ingestion/service.py
    docstring for the bug this prevents."""
    from backend.ingestion import vector_store as vs_module

    # Force a fresh vector store instance pointed at a temp dir for this test,
    # bypassing the process-wide singleton so tests don't interfere with each other.
    test_store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test_idempotent")
    monkeypatch.setattr(vs_module, "_store_singleton", test_store)

    # Build a minimal real PDF on disk so parse_pdf has something to read.
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Test document content for idempotency check.")
    c.save()

    result_1 = ingest_pdf(pdf_path, original_filename="sample.pdf")
    result_2 = ingest_pdf(pdf_path, original_filename="sample.pdf")

    assert result_1["document_id"] == result_2["document_id"]
    assert test_store.document_count() == result_1["chunk_count"]


def test_content_hash_id_differs_for_different_content(tmp_path):
    from reportlab.pdfgen import canvas

    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"

    c = canvas.Canvas(str(pdf_a))
    c.drawString(100, 750, "Content A")
    c.save()

    c = canvas.Canvas(str(pdf_b))
    c.drawString(100, 750, "Content B, totally different")
    c.save()

    assert _content_hash_id(pdf_a) != _content_hash_id(pdf_b)
