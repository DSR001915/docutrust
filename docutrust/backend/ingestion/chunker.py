"""
Chunking strategy: recursive character splitting with overlap.

Why this over fixed-size splitting: splitting on paragraph/sentence
boundaries first (and only falling back to hard character cuts when a
single paragraph is too long) keeps semantic units intact, which
matters a lot for retrieval quality and for keeping citations readable
to a human checking them against the source.

Why this over full semantic/embedding-based chunking: semantic chunking
buys you better topic-boundary detection but costs an embedding call per
candidate split and adds another tunable to justify. For policy/contract
style documents with reasonably clean paragraph structure, recursive
splitting gets ~90% of the benefit for near-zero extra cost. This is a
documented tradeoff, not an oversight -- see docs/ARCHITECTURE.md.
"""
import uuid
from dataclasses import dataclass, field

from backend.ingestion.parser import PageText

# Separators tried in order, largest semantic unit first.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    chunk_index: int  # position within the document, for debugging/ordering


def chunk_pages(
    pages: list[PageText],
    document_id: str,
    document_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split parsed pages into overlapping chunks, tagging each with the
    page number it came from so citations can point to an exact page."""
    chunks: list[Chunk] = []
    global_index = 0

    for page in pages:
        page_splits = _recursive_split(page.text, chunk_size, chunk_overlap)
        for split_text in page_splits:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page.page_number,
                    text=split_text,
                    chunk_index=global_index,
                )
            )
            global_index += 1

    return chunks


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into raw (non-overlapped) pieces on the largest separator
    that produces pieces under chunk_size, falling back to smaller
    separators, then apply overlap exactly once at the end.

    Overlap is intentionally applied only at the top level: recursing with
    overlap already baked in causes overlap text to get re-split and
    re-merged at each recursion level, stacking duplicated text. Splitting
    "clean" first and overlapping last avoids that entirely.
    """
    raw_pieces = _split_clean(text, chunk_size)
    return _apply_overlap(raw_pieces, chunk_overlap)


def _split_clean(text: str, chunk_size: int) -> list[str]:
    """Split text into pieces under chunk_size with NO overlap applied.
    Safe to call recursively."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in _SEPARATORS:
        if sep == "":
            return _hard_split_clean(text, chunk_size)

        pieces = text.split(sep)
        if len(pieces) == 1:
            continue  # this separator doesn't appear; try the next one

        return _merge_pieces_clean(pieces, sep, chunk_size)

    return _hard_split_clean(text, chunk_size)


def _merge_pieces_clean(pieces: list[str], sep: str, chunk_size: int) -> list[str]:
    """Greedily merge small pieces back together up to chunk_size. No
    overlap is applied here -- that happens once, at the top level."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = current + sep + piece if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(piece) > chunk_size:
                chunks.extend(_split_clean(piece, chunk_size))
                current = ""
            else:
                current = piece

    if current.strip():
        chunks.append(current)

    return chunks


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    """Prepend the tail of the previous chunk to each chunk so context
    isn't lost at chunk boundaries."""
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for prev, curr in zip(chunks, chunks[1:]):
        tail = prev[-chunk_overlap:]
        overlapped.append(tail + curr)

    return overlapped


def _hard_split_clean(text: str, chunk_size: int) -> list[str]:
    """Fallback: split on raw character count, no overlap, when no natural
    separator produces small enough pieces (e.g. a giant unbroken table row)."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size) if text[i : i + chunk_size].strip()]
