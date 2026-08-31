"""Parser adapter interface.

Every parser returns the same ``ParsedDocument`` regardless of backend, so the
rest of the pipeline never depends on pypdf, Docling, or a Databricks AI
function. Swapping a parser is a config change plus a re-parse, never a code
change downstream.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Sequence

from ..models import BlockType


class ParserError(RuntimeError):
    """Raised when a document cannot be parsed at all.

    A parser that succeeds with degraded output must NOT raise; it records a
    warning on the affected blocks so reviewers can see the degradation.
    """


class ParserUnavailableError(ParserError):
    """Raised when a parser's backend is not installed or not enabled."""


@dataclass(slots=True)
class ParsedBlock:
    """One structural block of a parsed document.

    Attributes:
        text: Block text as extracted.
        page_number: One-based page. Required for paginated formats.
        block_type: Structural role, used to keep tables and captions intact.
        section_title: Nearest preceding heading, when recovered.
        warning: Degradation marker, e.g. ``OCR_LOW_CONFIDENCE``.
        confidence: Backend confidence in [0, 1] when the backend reports one.
    """

    text: str
    page_number: int | None = None
    block_type: BlockType = BlockType.TEXT
    section_title: str | None = None
    warning: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        self.block_type = BlockType(self.block_type)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(slots=True)
class ParsedDocument:
    """Backend-independent parse result.

    Attributes:
        blocks: Ordered blocks in reading order.
        parser_name: Adapter key, persisted on every derived chunk.
        parser_version: Adapter version. Bumping it forces a re-parse.
        page_count: Pages seen by the backend, when known.
        metadata: Backend document metadata (title, author) as strings.
        warnings: Document-level degradation markers.
    """

    blocks: Sequence[ParsedBlock]
    parser_name: str
    parser_version: str
    page_count: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when no block carried usable text."""
        return not any(b.text.strip() for b in self.blocks)


class DocumentParser(abc.ABC):
    """Base class for parser adapters."""

    name: str = "base"
    version: str = "0"

    @abc.abstractmethod
    def parse(self, content: bytes, *, source_uri: str, content_type: str) -> ParsedDocument:
        """Parse raw bytes into a ``ParsedDocument``.

        Args:
            content: Raw document bytes exactly as fetched.
            source_uri: Where the bytes came from, for error messages only.
            content_type: MIME type as fetched.

        Returns:
            The parsed document, possibly carrying warnings.

        Raises:
            ParserError: The document could not be parsed at all.
            ParserUnavailableError: The backend is unavailable in this runtime.
        """

    def supports(self, content_type: str) -> bool:
        """Whether this adapter handles ``content_type``. Override as needed."""
        return True

    def is_available(self) -> bool:
        """Whether this adapter's backend can actually run in this runtime.

        Adapters with an optional dependency or a workspace-gated feature
        override this. The registry consults it before selecting an adapter, so
        an unavailable backend is detected at selection time rather than
        discovered halfway through a batch.
        """
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}@{self.version}>"
