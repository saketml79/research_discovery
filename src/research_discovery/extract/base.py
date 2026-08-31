"""Claim extractor interface and the shared validation gate.

Every extractor - LLM, ``ai_extract``, or rule-based - returns
``CandidateClaim`` objects that pass through one validation function. That is
where the project's central rule is enforced: an extractor may not invent a
metric value, and a missing scope field must be an explicit statement of
ignorance rather than a silent gap.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Sequence

from ..models import Chunk, Claim, ClaimType, ReviewStatus, utcnow


class ExtractionError(RuntimeError):
    """Raised when extraction fails for reasons the caller should see."""


class ExtractorUnavailableError(ExtractionError):
    """Raised when the extractor's backend is not available in this runtime."""


@dataclass(slots=True)
class CandidateClaim:
    """An extractor's proposal, before validation and persistence."""

    claim_text: str
    claim_type: str
    task: str | None = None
    method: str | None = None
    metric: str | None = None
    metric_value: float | None = None
    metric_unit: str | None = None
    benchmark: str | None = None
    condition_text: str | None = None
    evidence_excerpt: str | None = None
    confidence: float | None = None
    missing_field_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


class ClaimExtractor(abc.ABC):
    """Base class for claim extractors."""

    name: str = "base"
    version: str = "0"

    @abc.abstractmethod
    def extract(self, chunk: Chunk) -> Sequence[CandidateClaim]:
        """Propose claims found in ``chunk``.

        Implementations must return an empty sequence rather than raise when a
        chunk simply contains no claim.

        Raises:
            ExtractionError: The backend failed in a way the caller must see.
        """

    def extract_many(self, chunks: Sequence[Chunk]) -> list[tuple[Chunk, CandidateClaim]]:
        """Extract over several chunks, keeping each candidate's origin chunk.

        A failure on one chunk is recorded as a warning on a zero-confidence
        candidate rather than aborting the batch: one unparseable page must not
        cost the corpus the other forty.
        """
        results: list[tuple[Chunk, CandidateClaim]] = []
        for chunk in chunks:
            try:
                for candidate in self.extract(chunk):
                    results.append((chunk, candidate))
            except ExtractorUnavailableError:
                raise
            except ExtractionError as exc:
                results.append(
                    (
                        chunk,
                        CandidateClaim(
                            claim_text=f"[extraction failed for chunk {chunk.chunk_id}]",
                            claim_type=ClaimType.LIMITATION.value,
                            confidence=0.0,
                            missing_field_reason=f"EXTRACTION_FAILED: {exc}",
                            warnings=["EXTRACTION_FAILED"],
                        ),
                    )
                )
        return results


def validate_candidate(candidate: CandidateClaim, chunk: Chunk) -> list[str]:
    """Return the reasons ``candidate`` may not be persisted as-is.

    An empty list means the candidate is well-formed. It does NOT mean the claim
    is true - only a human reviewer decides that.
    """
    problems: list[str] = []

    if not candidate.claim_text.strip():
        problems.append("EMPTY_CLAIM_TEXT")
    try:
        ClaimType(candidate.claim_type)
    except ValueError:
        problems.append(f"INVALID_CLAIM_TYPE:{candidate.claim_type}")

    if candidate.confidence is not None and not 0.0 <= candidate.confidence <= 1.0:
        problems.append("CONFIDENCE_OUT_OF_RANGE")

    # A number must be traceable to the chunk it was read from. This is the
    # guard against a fluent model inventing a plausible benchmark score.
    if candidate.metric_value is not None and not _value_appears_in(candidate.metric_value, chunk.text):
        problems.append("METRIC_VALUE_NOT_IN_SOURCE_TEXT")

    if candidate.evidence_excerpt:
        if _normalized(candidate.evidence_excerpt) not in _normalized(chunk.text):
            problems.append("EXCERPT_NOT_VERBATIM")

    missing = [
        name
        for name in ("task", "method", "metric", "benchmark", "condition_text")
        if getattr(candidate, name) in (None, "")
    ]
    if missing and not candidate.missing_field_reason:
        problems.append("MISSING_FIELD_REASON_REQUIRED:" + ",".join(missing))

    return problems


def to_claim(
    candidate: CandidateClaim,
    chunk: Chunk,
    *,
    source_url: str,
    extractor_name: str,
    extractor_version: str,
) -> Claim:
    """Build a persistable ``Claim`` from a validated candidate.

    The claim is always created as ``CANDIDATE``: nothing an extractor produces
    is runtime-visible until a reviewer accepts it.
    """
    problems = validate_candidate(candidate, chunk)
    if problems:
        raise ExtractionError("candidate failed validation: " + "; ".join(problems))

    missing = [
        name
        for name in ("task", "method", "metric", "benchmark", "condition_text")
        if getattr(candidate, name) in (None, "")
    ]
    reason = candidate.missing_field_reason
    if missing and reason and not reason.startswith("MISSING:"):
        reason = f"MISSING:{','.join(missing)} - {reason}"

    return Claim(
        source_version_id=chunk.source_version_id,
        source_id=chunk.source_id,
        chunk_id=chunk.chunk_id,
        claim_text=candidate.claim_text.strip(),
        claim_type=ClaimType(candidate.claim_type),
        task=candidate.task,
        method=candidate.method,
        metric=candidate.metric,
        metric_value=candidate.metric_value,
        metric_unit=candidate.metric_unit,
        benchmark=candidate.benchmark,
        condition_text=candidate.condition_text,
        evidence_excerpt=candidate.evidence_excerpt,
        page_number=chunk.page_number,
        source_url=source_url,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        extraction_confidence=candidate.confidence,
        missing_field_reason=reason,
        extracted_at=utcnow(),
        review_status=ReviewStatus.CANDIDATE,
    )


def _normalized(text: str) -> str:
    return " ".join(text.split()).lower()


def _value_appears_in(value: float, text: str) -> bool:
    """True when ``value`` is recoverable from ``text``.

    Accepts the raw number, a trailing-zero-trimmed form, and a percentage form
    (0.62 matched by "62%"), because sources state the same figure both ways.
    """
    haystack = text.replace(",", "")
    candidates = {
        f"{value:g}",
        f"{value}",
        f"{value:.1f}",
        f"{value:.2f}",
    }
    if 0.0 < abs(value) <= 1.0:
        candidates.update({f"{value * 100:g}", f"{value * 100:.1f}"})
    if abs(value) > 1.0:
        candidates.add(f"{value / 100:g}")
    return any(c in haystack for c in candidates)
