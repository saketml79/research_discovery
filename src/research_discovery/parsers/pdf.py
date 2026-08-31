"""PDF parser adapters.

Three backends behind one interface:

* ``PypdfParser`` - text extraction for digital PDFs. Cheap, dependency-light,
  and honest about what it cannot do: it records ``LAYOUT_UNRECOVERED`` on pages
  whose text density suggests a scan or a complex layout.
* ``DoclingParser`` - layout-aware parsing (tables, headings, reading order)
  when the ``docling`` package is installed on the cluster.
* ``AiParseDocumentParser`` - the Databricks ``ai_parse_document`` SQL function,
  used only when explicitly enabled and validated in the target workspace.

None of them is imported at module load; each backend is resolved lazily so an
absent dependency degrades to a clear ``ParserUnavailableError`` rather than an
import failure at job start.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..models import BlockType
from .base import DocumentParser, ParsedBlock, ParsedDocument, ParserError, ParserUnavailableError

#: Below this many characters per page, a digital-text extraction is suspect and
#: the page is almost certainly a scan needing OCR.
_LOW_TEXT_PAGE_CHARS = 120

_CAPTION = re.compile(r"^\s*(figure|fig\.|table)\s+\d+", re.IGNORECASE)
_REFERENCES_HEADING = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
_ABSTRACT_HEADING = re.compile(r"^\s*abstract\s*$", re.IGNORECASE)
_HEADING = re.compile(r"^\s*(\d+(\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,:\-]{2,60}\s*$")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _classify(text: str, in_references: bool) -> BlockType:
    """Assign a structural role to a paragraph of PDF text."""
    if in_references:
        return BlockType.REFERENCES
    if _CAPTION.match(text):
        return BlockType.FIGURE_CAPTION
    return BlockType.TEXT


class PypdfParser(DocumentParser):
    """Digital-PDF text extraction via ``pypdf``.

    Produces page-scoped blocks so that every downstream chunk keeps a page
    number. Pages with implausibly little text are emitted with an
    ``OCR_REQUIRED`` warning instead of being silently dropped.
    """

    name = "pypdf"
    version = "1.1.0"

    def __init__(self, reader_factory: Callable[[Any], Any] | None = None) -> None:
        """Args:
        reader_factory: Injection point for tests; defaults to ``pypdf.PdfReader``.
        """
        self._reader_factory = reader_factory

    def supports(self, content_type: str) -> bool:
        return "pdf" in content_type.lower()

    def is_available(self) -> bool:
        """True when pypdf is importable, or a reader factory was injected."""
        if self._reader_factory is not None:
            return True
        try:
            import pypdf  # noqa: PLC0415, F401
        except ImportError:
            return False
        return True

    def _reader(self, stream: Any) -> Any:
        if self._reader_factory is not None:
            return self._reader_factory(stream)
        try:
            from pypdf import PdfReader  # noqa: PLC0415 - lazy backend resolution
        except ImportError as exc:  # pragma: no cover - exercised on clusters
            raise ParserUnavailableError(
                "pypdf is not installed; add it to the job environment or select another parser"
            ) from exc
        return PdfReader(stream)

    def parse(self, content: bytes, *, source_uri: str, content_type: str) -> ParsedDocument:
        import io  # noqa: PLC0415 - only needed on this path

        try:
            reader = self._reader(io.BytesIO(content))
            pages = list(reader.pages)
        except ParserUnavailableError:
            raise
        except Exception as exc:
            raise ParserError(f"pypdf could not open {source_uri}: {exc}") from exc

        blocks: list[ParsedBlock] = []
        warnings: list[str] = []
        section: str | None = None
        in_references = False

        for page_index, page in enumerate(pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # a single bad page must not fail the document
                warnings.append(f"PAGE_{page_index}_EXTRACT_FAILED")
                blocks.append(
                    ParsedBlock(
                        text=f"[page {page_index} could not be extracted: {exc}]",
                        page_number=page_index,
                        warning="PAGE_EXTRACT_FAILED",
                        confidence=0.0,
                    )
                )
                continue

            page_warning = None
            if len(page_text.strip()) < _LOW_TEXT_PAGE_CHARS:
                page_warning = "OCR_REQUIRED"
                warnings.append(f"PAGE_{page_index}_LOW_TEXT")

            for paragraph in _PARAGRAPH_BREAK.split(page_text):
                cleaned = " ".join(paragraph.split())
                if not cleaned:
                    continue
                if _REFERENCES_HEADING.match(cleaned):
                    in_references = True
                if _ABSTRACT_HEADING.match(cleaned):
                    section = "Abstract"
                    continue
                if len(cleaned) < 80 and _HEADING.match(cleaned):
                    section = cleaned
                    continue
                block_type = _classify(cleaned, in_references)
                if section == "Abstract" and block_type is BlockType.TEXT:
                    block_type = BlockType.ABSTRACT
                blocks.append(
                    ParsedBlock(
                        text=cleaned,
                        page_number=page_index,
                        block_type=block_type,
                        section_title=section,
                        warning=page_warning,
                        confidence=0.0 if page_warning else None,
                    )
                )

        if not blocks:
            raise ParserError(f"no extractable text in {source_uri}; OCR is required")

        return ParsedDocument(
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            page_count=len(pages),
            warnings=warnings,
        )


class DoclingParser(DocumentParser):
    """Layout-aware parsing via Docling, when installed.

    Docling recovers reading order, headings and table structure, which matters
    for research PDFs where a result table is the actual evidence. It is heavier
    than pypdf, so it is opt-in through configuration.
    """

    name = "docling"
    version = "1.0.0"

    def __init__(self, converter: Any | None = None) -> None:
        """Args:
        converter: Pre-built Docling ``DocumentConverter``; injected in tests.
        """
        self._converter = converter

    def supports(self, content_type: str) -> bool:
        lowered = content_type.lower()
        return any(k in lowered for k in ("pdf", "word", "presentation", "html"))

    def is_available(self) -> bool:
        """True when Docling is importable, or a converter was injected."""
        if self._converter is not None:
            return True
        try:
            import docling  # noqa: PLC0415, F401
        except ImportError:
            return False
        return True

    def _resolve(self) -> Any:
        if self._converter is not None:
            return self._converter
        try:
            from docling.document_converter import DocumentConverter  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised on clusters
            raise ParserUnavailableError(
                "docling is not installed; set parser=pypdf or add docling to the job environment"
            ) from exc
        self._converter = DocumentConverter()
        return self._converter

    def parse(self, content: bytes, *, source_uri: str, content_type: str) -> ParsedDocument:
        converter = self._resolve()
        try:
            result = converter.convert(source_uri)
            document = getattr(result, "document", result)
            items = list(getattr(document, "texts", []) or [])
        except ParserUnavailableError:
            raise
        except Exception as exc:
            raise ParserError(f"docling failed on {source_uri}: {exc}") from exc

        blocks: list[ParsedBlock] = []
        section: str | None = None
        for item in items:
            text = " ".join(str(getattr(item, "text", "")).split())
            if not text:
                continue
            label = str(getattr(item, "label", "text")).lower()
            if "title" in label or "heading" in label or "section" in label:
                section = text
                continue
            page = getattr(item, "page_no", None)
            blocks.append(
                ParsedBlock(
                    text=text,
                    page_number=int(page) if isinstance(page, int) else None,
                    block_type=(
                        BlockType.TABLE
                        if "table" in label
                        else BlockType.FIGURE_CAPTION
                        if "caption" in label
                        else BlockType.TEXT
                    ),
                    section_title=section,
                )
            )

        if not blocks:
            raise ParserError(f"docling returned no text blocks for {source_uri}")

        return ParsedDocument(
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            page_count=getattr(document, "num_pages", None),
        )


class AiParseDocumentParser(DocumentParser):
    """Adapter for the Databricks ``ai_parse_document`` SQL function.

    Disabled by default. ``ai_parse_document`` varies by workspace in syntax,
    supported file types, model access, cost and region, so this adapter refuses
    to run unless a caller passes an explicit ``sql_runner`` and confirms
    availability. Validate it in the target workspace before making it a
    production dependency.
    """

    name = "ai_parse_document"
    version = "0.1.0"

    def __init__(self, sql_runner: Callable[[str, dict[str, Any]], list[dict[str, Any]]] | None = None):
        """Args:
        sql_runner: Callable executing SQL and returning rows as dicts. Wired to
            a SQL warehouse client by the job; ``None`` disables the adapter.
        """
        self._sql_runner = sql_runner

    def supports(self, content_type: str) -> bool:
        return "pdf" in content_type.lower() or "image" in content_type.lower()

    def is_available(self) -> bool:
        """Only available when a caller supplied a validated sql_runner."""
        return self._sql_runner is not None

    def parse(self, content: bytes, *, source_uri: str, content_type: str) -> ParsedDocument:
        if self._sql_runner is None:
            raise ParserUnavailableError(
                "ai_parse_document is not enabled for this deployment; validate workspace "
                "support, then construct AiParseDocumentParser with a sql_runner"
            )
        query = "SELECT ai_parse_document(:uri) AS parsed"
        try:
            rows = self._sql_runner(query, {"uri": source_uri})
        except Exception as exc:
            raise ParserError(f"ai_parse_document failed on {source_uri}: {exc}") from exc
        if not rows:
            raise ParserError(f"ai_parse_document returned no rows for {source_uri}")

        parsed = rows[0].get("parsed") or {}
        elements = parsed.get("elements") or parsed.get("pages") or []
        blocks = [
            ParsedBlock(
                text=" ".join(str(element.get("content", "")).split()),
                page_number=element.get("page_number"),
                block_type=BlockType.TABLE if element.get("type") == "table" else BlockType.TEXT,
                section_title=element.get("section"),
                confidence=element.get("confidence"),
            )
            for element in elements
            if str(element.get("content", "")).strip()
        ]
        if not blocks:
            raise ParserError(f"ai_parse_document produced no usable content for {source_uri}")
        return ParsedDocument(
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            warnings=["AI_PARSE_WORKSPACE_DEPENDENT"],
        )
