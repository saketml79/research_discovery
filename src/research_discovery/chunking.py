"""Chunking.

Rules, in priority order:

1. A table or figure caption is never split and never merged with a neighbour -
   it is the evidence a result claim points at.
2. A chunk never spans two pages, so every chunk keeps one page reference.
3. Within those bounds, adjacent text blocks in the same section are packed up
   to ``max_chunk_chars``; a block longer than that is split on sentence
   boundaries.
4. A chunk below ``min_chunk_chars`` is merged forward when a compatible
   neighbour exists, and kept otherwise - dropping it would lose evidence.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from .config import RESTRICTED_EXCERPT_CHARS, Config
from .models import BlockType, Chunk
from .parsers.base import ParsedBlock, ParsedDocument

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

#: Block types that must stand alone as their own chunk.
_ATOMIC = frozenset({BlockType.TABLE, BlockType.FIGURE_CAPTION})

#: Block types excluded from the retrieval corpus: a bibliography produces
#: nothing but false-positive matches on other people's titles.
_EXCLUDED = frozenset({BlockType.REFERENCES})


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into pieces of at most ``max_chars`` on sentence bounds.

    Falls back to a hard character split for a single sentence longer than the
    limit, so the function always terminates and never drops content.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_END.split(text):
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer)
            buffer = ""
        if len(sentence) <= max_chars:
            buffer = sentence
        else:
            for start in range(0, len(sentence), max_chars):
                fragment = sentence[start : start + max_chars]
                if len(fragment) == max_chars:
                    pieces.append(fragment)
                else:
                    buffer = fragment
    if buffer:
        pieces.append(buffer)
    return pieces


def _packable(current: ParsedBlock, nxt: ParsedBlock) -> bool:
    """True when two blocks may share a chunk."""
    return (
        current.block_type not in _ATOMIC
        and nxt.block_type not in _ATOMIC
        and current.page_number == nxt.page_number
        and current.section_title == nxt.section_title
        and current.warning == nxt.warning
    )


def _group(blocks: Sequence[ParsedBlock], max_chars: int) -> list[list[ParsedBlock]]:
    """Group blocks into packable runs bounded by ``max_chars``."""
    groups: list[list[ParsedBlock]] = []
    for block in blocks:
        if block.block_type in _EXCLUDED:
            continue
        if not groups:
            groups.append([block])
            continue
        current = groups[-1]
        projected = sum(len(b.text) + 1 for b in current) + len(block.text)
        if _packable(current[-1], block) and projected <= max_chars:
            current.append(block)
        else:
            groups.append([block])
    return groups


def chunk_document(
    document: ParsedDocument,
    *,
    source_version_id: str,
    source_id: str,
    config: Config,
    storage_permitted: bool = True,
    parser_warning: str | None = None,
) -> list[Chunk]:
    """Turn a parsed document into persisted chunk records.

    Args:
        document: Backend-independent parse result.
        source_version_id: Version the chunks are derived from.
        source_id: Denormalized source id, for index metadata filters.
        config: Supplies the chunk size bounds.
        storage_permitted: When false, chunk text is truncated to the permitted
            excerpt length. The licence decision is enforced here, once, rather
            than trusted to every downstream caller.
        parser_warning: Adapter-level warning (e.g. a parser fallback) stamped
            onto every chunk produced from this document.

    Returns:
        Chunks in reading order with contiguous ``chunk_index`` values.
    """
    groups = _group(list(document.blocks), config.max_chunk_chars)

    staged: list[tuple[str, ParsedBlock]] = []
    for group in groups:
        head = group[0]
        merged = " ".join(b.text.strip() for b in group).strip()
        if not merged:
            continue
        for piece in split_long_text(merged, config.max_chunk_chars):
            staged.append((piece, head))

    merged_staged = _merge_short(staged, config.min_chunk_chars, config.max_chunk_chars)

    chunks: list[Chunk] = []
    for index, (text, head) in enumerate(merged_staged):
        body = text if storage_permitted else text[:RESTRICTED_EXCERPT_CHARS]
        warning = head.warning or parser_warning
        if not storage_permitted:
            warning = ",".join(filter(None, (warning, "TRUNCATED_LICENCE")))
        chunks.append(
            Chunk(
                source_version_id=source_version_id,
                source_id=source_id,
                chunk_index=index,
                text=body,
                block_type=head.block_type,
                page_number=head.page_number,
                section_title=head.section_title,
                parser_name=document.parser_name,
                parser_version=document.parser_version,
                extraction_warning=warning or None,
            )
        )
    return chunks


def _merge_short(
    staged: Sequence[tuple[str, ParsedBlock]], min_chars: int, max_chars: int
) -> list[tuple[str, ParsedBlock]]:
    """Merge undersized text chunks forward where the neighbour is compatible."""
    result: list[tuple[str, ParsedBlock]] = []
    for text, head in staged:
        if (
            result
            and len(result[-1][0]) < min_chars
            and head.block_type not in _ATOMIC
            and result[-1][1].block_type not in _ATOMIC
            and _packable(result[-1][1], head)
            and len(result[-1][0]) + len(text) + 1 <= max_chars
        ):
            previous_text, previous_head = result.pop()
            result.append((f"{previous_text} {text}".strip(), previous_head))
        else:
            result.append((text, head))
    return result


def iter_chunk_texts(chunks: Iterable[Chunk]) -> Iterable[str]:
    """Yield chunk texts. Small helper used by extractors and index builds."""
    for chunk in chunks:
        yield chunk.text
