"""Vision extraction for figures, charts and slides.

A number read off a chart is not a stated number. This module keeps that
distinction structural rather than advisory:

* figure readings land in ``research_figure``, never directly in
  ``research_claim``;
* every reading carries the model, the prompt version, a confidence and a stored
  image reference, so a reviewer can check it against the original;
* a claim derived from a figure gets ``figure_id`` set, which the agent surfaces
  as a visual interpretation rather than a quotation.

The vision client is injected, so the whole path is testable with a stub.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from ..ids import stable_id
from ..models import Claim, ClaimType, ReviewStatus, utcnow
from .base import CandidateClaim, ExtractionError, ExtractorUnavailableError

logger = logging.getLogger(__name__)

VISION_PROMPT_VERSION = "2026.08.1"

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

VISION_SYSTEM_PROMPT = """\
You read a single figure from a research document and report what is printed in it.

Rules:
1. Report only what is visibly printed: axis labels, tick values, series names,
   legend entries, cell values, and the caption. Do not infer a trend the figure
   does not label, and do not use outside knowledge of the paper.
2. Every value you report must be readable in the image. If a bar's value is not
   printed and cannot be read against a labelled axis, omit it. Never estimate a
   value from pixel height and never round to a "likely" number.
3. confidence is how sure you are that you READ THE IMAGE correctly - legibility,
   resolution, occlusion - not whether the underlying result is true.
4. If the image is unreadable, return entities as an empty array with a
   confidence of 0 and say why in extracted_text. That is a correct answer.
5. Return ONLY JSON matching the schema. No prose, no code fence.

Schema:
{
  "figure_type": "CHART|DIAGRAM|TABLE_IMAGE|SCREENSHOT|SLIDE",
  "extracted_text": "axis labels, legend entries and any printed values, verbatim",
  "entities": [{"label": "series or cell name", "value": number|null,
                "unit": "string|null", "readable": true|false}],
  "confidence": 0.0
}
"""


class VisionClient(Protocol):
    """Minimal multimodal interface: an image plus a prompt in, text out."""

    def describe(self, image_bytes: bytes, prompt: str, *, system: str) -> str:
        """Return the model's textual response for ``image_bytes``."""


class ServingEndpointVisionClient:
    """``VisionClient`` over a vision-capable Databricks serving endpoint."""

    def __init__(self, endpoint: str, workspace_client: Any | None = None) -> None:
        self._endpoint = endpoint
        self._client = workspace_client

    def describe(self, image_bytes: bytes, prompt: str, *, system: str) -> str:
        import base64  # noqa: PLC0415

        try:
            from databricks.sdk import WorkspaceClient  # noqa: PLC0415
            from databricks.sdk.service.serving import ChatMessage, ChatMessageRole  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised on clusters
            raise ExtractorUnavailableError(
                "databricks-sdk is not installed; vision extraction is unavailable"
            ) from exc

        client = self._client or WorkspaceClient()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = client.serving_endpoints.query(
            name=self._endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=system),
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=json.dumps(
                        [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded}"},
                            },
                        ]
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=1200,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ExtractionError("vision endpoint returned no choices")
        return choices[0].message.content or ""


@dataclass(slots=True)
class FigureReading:
    """A vision model's interpretation of one figure, before persistence."""

    source_version_id: str
    source_id: str
    page_number: int
    figure_index: int
    figure_type: str
    extracted_text: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    caption: str | None = None
    image_uri: str | None = None
    bounding_box: str | None = None
    vision_model: str = "unknown"
    prompt_version: str = VISION_PROMPT_VERSION
    review_status: str = ReviewStatus.CANDIDATE.value
    extracted_at: Any = field(default_factory=utcnow)
    figure_id: str = ""

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("a figure reading requires a one-based page number")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.review_status != ReviewStatus.CANDIDATE.value:
            raise ValueError("a fresh figure reading is always CANDIDATE")
        self.figure_id = self.figure_id or stable_id(
            "fig", self.source_version_id, str(self.page_number), str(self.figure_index)
        )

    @property
    def readable_entities(self) -> list[dict[str, Any]]:
        """Entities the model reported as legible with a numeric value."""
        return [
            entity
            for entity in self.entities
            if entity.get("readable") and entity.get("value") is not None
        ]


class FigureExtractor:
    """Turns page images into reviewable figure readings."""

    name = "vision"

    def __init__(
        self,
        client: VisionClient,
        *,
        model_name: str,
        min_confidence: float = 0.4,
    ) -> None:
        """Args:
        client: Vision backend.
        model_name: Recorded on every reading for provenance.
        min_confidence: Readings below this are kept but never proposed as
            claims — an illegible chart is evidence of nothing.
        """
        self._client = client
        self._model_name = model_name
        self._min_confidence = min_confidence

    def read_figure(
        self,
        image_bytes: bytes,
        *,
        source_version_id: str,
        source_id: str,
        page_number: int,
        figure_index: int,
        caption: str | None = None,
        image_uri: str | None = None,
        bounding_box: str | None = None,
    ) -> FigureReading:
        """Read one figure image into a candidate reading.

        Raises:
            ExtractionError: The backend failed or returned unusable output.
        """
        prompt = (
            f"Read this figure from page {page_number} of a research document."
            + (f' Its printed caption is: "{caption}".' if caption else "")
        )
        raw = self._client.describe(image_bytes, prompt, system=VISION_SYSTEM_PROMPT)
        payload = _parse_json(raw)

        entities = payload.get("entities") or []
        if not isinstance(entities, list):
            raise ExtractionError("vision response 'entities' must be an array")

        return FigureReading(
            source_version_id=source_version_id,
            source_id=source_id,
            page_number=page_number,
            figure_index=figure_index,
            figure_type=str(payload.get("figure_type") or "CHART").upper(),
            extracted_text=str(payload.get("extracted_text") or "").strip(),
            entities=[e for e in entities if isinstance(e, dict)],
            confidence=_as_float(payload.get("confidence")) or 0.0,
            caption=caption,
            image_uri=image_uri,
            bounding_box=bounding_box,
            vision_model=self._model_name,
        )

    def propose_claims(self, reading: FigureReading) -> list[CandidateClaim]:
        """Propose candidate claims from a figure reading.

        A reading below ``min_confidence``, or with no legibly-valued entity,
        proposes nothing: the figure is persisted for a reviewer, but it does not
        become a claim. Every proposed claim inherits the reading's confidence
        and names the figure in its missing-field reason, so the visual origin
        travels with it.
        """
        if reading.confidence < self._min_confidence:
            logger.info(
                "figure %s below confidence floor (%.2f); no claims proposed",
                reading.figure_id,
                reading.confidence,
            )
            return []

        candidates: list[CandidateClaim] = []
        for entity in reading.readable_entities:
            label = str(entity.get("label") or "").strip()
            if not label:
                continue
            value = _as_float(entity.get("value"))
            if value is None:
                continue
            candidates.append(
                CandidateClaim(
                    claim_text=(
                        f"Figure {reading.figure_index + 1} on page {reading.page_number} "
                        f"reports {label} = {value}"
                        + (f" {entity['unit']}" if entity.get("unit") else "")
                        + "."
                    ),
                    claim_type=ClaimType.PERFORMANCE.value,
                    metric=label.lower() or None,
                    metric_value=value,
                    metric_unit=(str(entity["unit"]) if entity.get("unit") else None),
                    evidence_excerpt=reading.extracted_text[:600] or None,
                    confidence=reading.confidence,
                    missing_field_reason=(
                        f"FIGURE_DERIVED:{reading.figure_id} - read from a chart by "
                        f"{reading.vision_model}; task, method, benchmark and conditions are "
                        "not readable from the figure alone and must be supplied by a reviewer"
                    ),
                    warnings=["VISION_EXTRACTION", f"CONFIDENCE:{reading.confidence:.2f}"],
                )
            )
        return candidates


def figure_claim(
    candidate: CandidateClaim,
    reading: FigureReading,
    *,
    source_url: str,
) -> Claim:
    """Build a persistable claim from a figure-derived candidate.

    Bypasses the text-excerpt checks in ``extract.base`` — which require a value
    to appear in chunk text — because the value's provenance is an image, not a
    passage. In exchange the claim is pinned to its ``figure_id``, so it can
    never be presented without its visual origin and confidence.
    """
    if candidate.metric_value is None:
        raise ExtractionError("a figure-derived claim must carry the value it read")
    return Claim(
        source_version_id=reading.source_version_id,
        source_id=reading.source_id,
        chunk_id=None,
        figure_id=reading.figure_id,
        claim_text=candidate.claim_text,
        claim_type=ClaimType(candidate.claim_type),
        metric=candidate.metric,
        metric_value=candidate.metric_value,
        metric_unit=candidate.metric_unit,
        evidence_excerpt=candidate.evidence_excerpt,
        page_number=reading.page_number,
        source_url=source_url,
        extractor_name="vision",
        extractor_version=f"{reading.vision_model}/{reading.prompt_version}",
        extraction_confidence=candidate.confidence,
        missing_field_reason=candidate.missing_field_reason,
        review_status=ReviewStatus.CANDIDATE,
    )


def render_page_images(
    content: bytes, pages: Sequence[int]
) -> dict[int, bytes]:  # pragma: no cover - backend dependent
    """Render selected PDF pages to PNG bytes for vision extraction.

    Uses PyMuPDF when available. Raises ``ExtractorUnavailableError`` otherwise,
    so a deployment without it degrades to text-only extraction rather than
    failing the pipeline.
    """
    try:
        import fitz  # noqa: PLC0415 - PyMuPDF
    except ImportError as exc:
        raise ExtractorUnavailableError(
            "PyMuPDF is not installed; install research-discovery[vision] for figure extraction"
        ) from exc

    images: dict[int, bytes] = {}
    with fitz.open(stream=content, filetype="pdf") as document:
        for page_number in pages:
            if not 1 <= page_number <= document.page_count:
                continue
            page = document.load_page(page_number - 1)
            images[page_number] = page.get_pixmap(dpi=200).tobytes("png")
    return images


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise ExtractionError(f"vision response was not JSON: {text[:200]!r}") from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"vision response was not valid JSON: {exc}") from exc


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
