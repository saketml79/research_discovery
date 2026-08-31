"""Plain-text and HTML parser adapters.

The HTML adapter uses only the standard library, so the pipeline can be
exercised end to end without any third-party dependency. It is a deliberate
baseline, not a claim that it handles arbitrary modern web pages well: it
records a warning whenever it has to fall back to naive tag stripping.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser as _StdHTMLParser

from ..models import BlockType
from .base import DocumentParser, ParsedBlock, ParsedDocument, ParserError

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_REFERENCES_HEADING = re.compile(r"^\s*(references|bibliography|works cited)\s*$", re.IGNORECASE)
_BLOCK_TAGS = {"p", "li", "pre", "blockquote", "td", "th", "figcaption", *_HEADING_TAGS}
_SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class _Collector(_StdHTMLParser):
    """Collects block-level text and the heading in scope for each block."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str, str | None]] = []  # (tag, text, section)
        self._stack: list[str] = []
        self._buffer: list[str] = []
        self._current_section: str | None = None
        self._title: str | None = None
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001 - stdlib signature
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self._flush()
        self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self._flush(tag)
        while self._stack and self._stack.pop() != tag:
            continue

    def handle_data(self, data: str) -> None:
        # The title check precedes the skip check: <title> lives inside <head>,
        # which is itself skipped, so checking skip first would drop the title.
        if self._in_title and self._title is None:
            self._title = data.strip() or None
            return
        if any(t in _SKIP_TAGS for t in self._stack):
            return
        if data.strip():
            self._buffer.append(data)

    def _flush(self, tag: str | None = None) -> None:
        text = " ".join(" ".join(self._buffer).split())
        self._buffer.clear()
        if not text:
            return
        effective = tag or (self._stack[-1] if self._stack else "p")
        if effective in _HEADING_TAGS:
            self._current_section = text
            return
        self.blocks.append((effective, text, self._current_section))

    def close(self) -> None:
        super().close()
        self._flush()

    @property
    def title(self) -> str | None:
        return self._title


class HtmlParser(DocumentParser):
    """Standard-library HTML parser producing block-level chunks."""

    name = "html"
    version = "1.0.0"

    def supports(self, content_type: str) -> bool:
        return "html" in content_type.lower() or "xml" in content_type.lower()

    def parse(self, content: bytes, *, source_uri: str, content_type: str) -> ParsedDocument:
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - decode with replace rarely fails
            raise ParserError(f"could not decode {source_uri}: {exc}") from exc

        collector = _Collector()
        warnings: list[str] = []
        try:
            collector.feed(text)
            collector.close()
        except Exception as exc:
            raise ParserError(f"malformed HTML at {source_uri}: {exc}") from exc

        blocks = [
            ParsedBlock(
                text=body,
                page_number=None,
                block_type=(
                    # A bibliography entry names other people's results; left in
                    # the corpus it is a reliable source of false matches.
                    BlockType.REFERENCES
                    if section and _REFERENCES_HEADING.match(section)
                    else BlockType.TABLE
                    if tag in {"td", "th"}
                    else BlockType.FIGURE_CAPTION
                    if tag == "figcaption"
                    else BlockType.TEXT
                ),
                section_title=section,
            )
            for tag, body, section in collector.blocks
        ]

        if not blocks:
            stripped = " ".join(re.sub(r"<[^>]+>", " ", text).split())
            if stripped:
                warnings.append("HTML_STRUCTURE_UNRECOVERED")
                blocks = [ParsedBlock(text=stripped, warning="HTML_STRUCTURE_UNRECOVERED")]

        metadata = {"title": collector.title} if collector.title else {}
        return ParsedDocument(
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata=metadata,
            warnings=warnings,
        )


class PlainTextParser(DocumentParser):
    """Splits UTF-8 text on blank lines. Used for transcripts and fixtures."""

    name = "plaintext"
    version = "1.0.0"

    def supports(self, content_type: str) -> bool:
        return content_type.lower().startswith("text/plain")

    def parse(self, content: bytes, *, source_uri: str, content_type: str) -> ParsedDocument:
        text = content.decode("utf-8", errors="replace")
        paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
        if not paragraphs:
            raise ParserError(f"no text content in {source_uri}")
        return ParsedDocument(
            blocks=[ParsedBlock(text=p) for p in paragraphs],
            parser_name=self.name,
            parser_version=self.version,
        )
