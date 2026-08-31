"""Adapter for the Databricks ``ai_extract`` SQL function.

Schema-guided extraction executed in the warehouse rather than through a serving
endpoint. Availability, cost and supported argument shape vary by workspace and
runtime, so this adapter is inert unless a caller supplies a ``sql_runner`` -
the same rule the parser layer applies to ``ai_parse_document``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from ..models import Chunk
from .base import CandidateClaim, ClaimExtractor, ExtractionError, ExtractorUnavailableError

#: Labels handed to ai_extract. They intentionally mirror the claim columns so
#: the mapping back to CandidateClaim is one-to-one and reviewable.
EXTRACT_LABELS: tuple[str, ...] = (
    "claim_text",
    "claim_type",
    "task",
    "method",
    "metric",
    "metric_value",
    "metric_unit",
    "benchmark",
    "condition_text",
    "evidence_excerpt",
)


class AiExtractClaimExtractor(ClaimExtractor):
    """Extract claim fields with ``ai_extract`` over chunk text."""

    name = "ai_extract"
    version = "0.1.0"

    def __init__(
        self,
        sql_runner: Callable[[str, dict[str, Any]], list[dict[str, Any]]] | None = None,
        *,
        labels: Sequence[str] = EXTRACT_LABELS,
        default_confidence: float = 0.55,
    ) -> None:
        """Args:
        sql_runner: Executes SQL and returns rows as dicts. ``None`` disables
            the adapter, which then raises ``ExtractorUnavailableError``.
        labels: Field labels requested from ``ai_extract``.
        default_confidence: Confidence recorded when the function reports none.
            Deliberately mid-range: an unscored extraction is not a confident one.
        """
        self._sql_runner = sql_runner
        self._labels = tuple(labels)
        self._default_confidence = default_confidence

    def extract(self, chunk: Chunk) -> Sequence[CandidateClaim]:
        if self._sql_runner is None:
            raise ExtractorUnavailableError(
                "ai_extract is not enabled for this deployment; validate workspace support, "
                "then construct AiExtractClaimExtractor with a sql_runner"
            )
        query = "SELECT ai_extract(:text, :labels) AS extracted"
        try:
            rows = self._sql_runner(
                query, {"text": chunk.text, "labels": list(self._labels)}
            )
        except Exception as exc:
            raise ExtractionError(f"ai_extract failed on chunk {chunk.chunk_id}: {exc}") from exc
        if not rows:
            return []

        payload = rows[0].get("extracted")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ExtractionError(f"ai_extract returned non-JSON text: {exc}") from exc
        if not isinstance(payload, dict):
            return []

        claim_text = _clean(payload.get("claim_text"))
        if not claim_text:
            return []

        missing = [
            label
            for label in ("task", "method", "metric", "benchmark", "condition_text")
            if not _clean(payload.get(label))
        ]
        return [
            CandidateClaim(
                claim_text=claim_text,
                claim_type=_clean(payload.get("claim_type")) or "METHOD_DESCRIPTION",
                task=_clean(payload.get("task")),
                method=_clean(payload.get("method")),
                metric=_clean(payload.get("metric")),
                metric_value=_as_float(payload.get("metric_value")),
                metric_unit=_clean(payload.get("metric_unit")),
                benchmark=_clean(payload.get("benchmark")),
                condition_text=_clean(payload.get("condition_text")),
                evidence_excerpt=_clean(payload.get("evidence_excerpt")),
                confidence=_as_float(payload.get("confidence")) or self._default_confidence,
                missing_field_reason=(
                    f"MISSING:{','.join(missing)} - ai_extract did not locate these labels "
                    "in this passage"
                    if missing
                    else None
                ),
                warnings=["AI_EXTRACT_WORKSPACE_DEPENDENT"],
            )
        ]


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
