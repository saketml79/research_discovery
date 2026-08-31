"""OCR adapter for scanned and image-heavy documents.

OCR output is the least reliable text in the system, so this adapter makes that
visible rather than smoothing it over: every block carries the engine's own
per-block confidence, blocks below the floor are marked ``OCR_LOW_CONFIDENCE``,
and a page whose mean confidence is poor is flagged at document level. Reviewers
see the degradation; the extractor's validation gate then refuses to build a
numeric claim on text it cannot trust.

Backends are resolved lazily (pytesseract, then easyocr), so a deployment
without either degrades to a clear ``ParserUnavailableError`` instead of an
import failure at job start.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, Sequence

from ..models import BlockType
from .base import DocumentParser, ParsedBlock, ParsedDocument, ParserError, ParserUnavailableError

logger = logging.getLogger(__name__)

#: Per-block confidence below which text is marked as unreliable.
LOW_CONFIDENCE_FLOOR = 0.60

#: Page mean confidence below which the whole page is flagged for review.
POOR_PAGE_FLOOR = 0.75


class OcrEngine(Protocol):
    """An OCR backend returning text blocks with confidences."""

    def recognize(self, image_bytes: bytes) -> Sequence[tuple[str, float]]:
        """Return ``(text, confidence)`` pairs for one page image."""


class PytesseractEngine:
    """``OcrEngine`` over pytesseract's per-word confidence output."""

    def __init__(self, image_opener: Callable[[bytes], Any] | None = None) -> None:
        self._image_opener = image_opener

    def recognize(self, image_bytes: bytes) -> Sequence[tuple[str, float]]:
        try:
            import pytesseract  # noqa: PLC0415 - lazy backend
        except ImportError as exc:  # pragma: no cover - exercised on clusters
            raise ParserUnavailableError(
                "pytesseract is not installed; install research-discovery[ocr] or choose "
                "another parser"
            ) from exc

        image = (self._image_opener or _open_image)(image_bytes)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        blocks: list[tuple[str, float]] = []
        words: list[str] = []
        confidences: list[float] = []
        last_block = None

        for index, word in enumerate(data.get("text", [])):
            block_number = data.get("block_num", [0] * len(data["text"]))[index]
            confidence = float(data.get("conf", ["-1"])[index] or -1)
            if last_block is not None and block_number != last_block and words:
                blocks.append((" ".join(words), _mean(confidences)))
                words, confidences = [], []
            last_block = block_number
            if word.strip() and confidence >= 0:
                words.append(word)
                confidences.append(confidence / 100.0)

        if words:
            blocks.append((" ".join(words), _mean(confidences)))
        return blocks


class OcrParser(DocumentParser):
    """Parses scanned documents page by page through an OCR engine."""

    name = "ocr"
    version = "1.0.0"

    def __init__(
        self,
        engine: OcrEngine | None = None,
        *,
        page_renderer: Callable[[bytes], dict[int, bytes]] | None = None,
    ) -> None:
        """Args:
        engine: OCR backend. Defaults to pytesseract.
        page_renderer: Renders a document's pages to images. Defaults to the
            PyMuPDF renderer shared with vision extraction.
        """
        self._engine = engine or PytesseractEngine()
        self._page_renderer = page_renderer

    def supports(self, content_type: str) -> bool:
        lowered = content_type.lower()
        return "pdf" in lowered or "image" in lowered or "tiff" in lowered

    def is_available(self) -> bool:
        """True when both a page renderer and an OCR engine can run."""
        if self._page_renderer is None:
            try:
                import fitz  # noqa: PLC0415, F401 - PyMuPDF
            except ImportError:
                return False
        if isinstance(self._engine, PytesseractEngine):
            try:
                import pytesseract  # noqa: PLC0415, F401
            except ImportError:
                return False
        return True

    def _render(self, content: bytes) -> dict[int, bytes]:
        if self._page_renderer is not None:
            return self._page_renderer(content)
        from .. extract.vision import render_page_images  # noqa: PLC0415 - shared renderer

        # Page count is unknown before opening; the renderer clamps out-of-range
        # pages, and 200 pages is far past any realistic research paper.
        return render_page_images(content, range(1, 201))

    def parse(self, content: bytes, *, source_uri: str, content_type: str) -> ParsedDocument:
        if not content:
            raise ParserError(f"no content to OCR for {source_uri}")

        try:
            pages = self._render(content)
        except ParserUnavailableError:
            raise
        except Exception as exc:
            raise ParserError(f"could not render pages of {source_uri}: {exc}") from exc

        blocks: list[ParsedBlock] = []
        warnings: list[str] = []

        for page_number in sorted(pages):
            try:
                recognized = self._engine.recognize(pages[page_number])
            except ParserUnavailableError:
                raise
            except Exception as exc:
                warnings.append(f"PAGE_{page_number}_OCR_FAILED")
                logger.warning("OCR failed on page %d of %s: %s", page_number, source_uri, exc)
                continue

            page_confidences = [c for _, c in recognized if c >= 0]
            if page_confidences and _mean(page_confidences) < POOR_PAGE_FLOOR:
                warnings.append(f"PAGE_{page_number}_POOR_OCR")

            for text, confidence in recognized:
                cleaned = " ".join(text.split())
                if not cleaned:
                    continue
                blocks.append(
                    ParsedBlock(
                        text=cleaned,
                        page_number=page_number,
                        block_type=BlockType.TEXT,
                        warning=(
                            "OCR_LOW_CONFIDENCE" if confidence < LOW_CONFIDENCE_FLOOR else "OCR"
                        ),
                        confidence=confidence,
                    )
                )

        if not blocks:
            raise ParserError(f"OCR recovered no text from {source_uri}")

        return ParsedDocument(
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            page_count=len(pages),
            warnings=warnings,
        )


def _open_image(image_bytes: bytes) -> Any:  # pragma: no cover - backend dependent
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise ParserUnavailableError(
            "Pillow is not installed; install research-discovery[ocr]"
        ) from exc
    import io  # noqa: PLC0415

    return Image.open(io.BytesIO(image_bytes))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
