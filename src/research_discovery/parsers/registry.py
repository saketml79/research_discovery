"""Parser selection.

Resolution order: an explicitly configured parser wins; otherwise the first
adapter whose ``supports()`` accepts the content type is used. If the configured
parser is unavailable at runtime, ``resolve_with_fallback`` degrades to a
working adapter and reports the substitution rather than failing the batch -
with the substitution recorded on every chunk it produces.
"""

from __future__ import annotations

from typing import Callable, Iterable

from .base import DocumentParser, ParserError
from .ocr import OcrParser
from .pdf import AiParseDocumentParser, DoclingParser, PypdfParser
from .text import HtmlParser, PlainTextParser

#: Adapter constructors keyed by config value.
PARSERS: dict[str, Callable[[], DocumentParser]] = {
    "pypdf": PypdfParser,
    "docling": DoclingParser,
    "ai_parse_document": AiParseDocumentParser,
    "ocr": OcrParser,
    "html": HtmlParser,
    "plaintext": PlainTextParser,
}

#: Order tried when no parser is configured, most specific first.
_AUTO_ORDER = ("pypdf", "ocr", "html", "plaintext")

#: Fallbacks tried when the configured parser is unavailable.
_FALLBACK_ORDER = ("pypdf", "ocr", "html", "plaintext")


def get_parser(name: str) -> DocumentParser:
    """Instantiate the adapter registered under ``name``.

    Raises:
        KeyError: ``name`` is not a registered adapter.
    """
    try:
        factory = PARSERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown parser {name!r}; known: {sorted(PARSERS)}") from exc
    return factory()


def select_parser(content_type: str, candidates: Iterable[str] = _AUTO_ORDER) -> DocumentParser:
    """Return the first registered adapter that supports ``content_type``.

    Raises:
        ParserError: No candidate supports the content type.
    """
    for name in candidates:
        parser = get_parser(name)
        if parser.supports(content_type):
            return parser
    raise ParserError(f"no parser supports content type {content_type!r}")


def resolve_with_fallback(
    preferred: str, content_type: str
) -> tuple[DocumentParser, str | None]:
    """Resolve a usable parser, degrading gracefully.

    Args:
        preferred: Configured adapter key.
        content_type: MIME type of the document to parse.

    Returns:
        A ``(parser, warning)`` pair. ``warning`` is ``None`` when the preferred
        adapter was used, otherwise a marker such as
        ``PARSER_FALLBACK_docling_TO_pypdf`` that is persisted on every chunk.

    Raises:
        ParserError: Neither the preferred parser nor any fallback is usable.
    """
    try:
        parser = get_parser(preferred)
    except KeyError as exc:
        raise ParserError(str(exc)) from exc

    if parser.supports(content_type) and _is_available(parser):
        return parser, None

    for name in _FALLBACK_ORDER:
        if name == preferred:
            continue
        candidate = get_parser(name)
        if candidate.supports(content_type) and _is_available(candidate):
            return candidate, f"PARSER_FALLBACK_{preferred}_TO_{name}"

    raise ParserError(
        f"no usable parser for content type {content_type!r}; preferred {preferred!r} unavailable"
    )


def _is_available(parser: DocumentParser) -> bool:
    """Ask the adapter whether its backend can run here.

    Adapters declare this directly rather than being probed with fake input, so
    an unavailable backend never reaches a real document.
    """
    try:
        return parser.is_available()
    except Exception:  # noqa: BLE001 - a probe must never fail selection
        return False
