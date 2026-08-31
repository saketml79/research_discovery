"""Review queue construction and decision application.

The queue is the boundary between extracted candidate knowledge and the
agent's runtime surface. Nothing crosses it without a human decision, and every
crossing records who decided and when.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..config import Config
from ..models import (
    Claim,
    ClaimRelationship,
    ComparabilityStatus,
    RelationshipType,
    ReviewItem,
    ReviewStatus,
    utcnow,
)


class ReviewError(RuntimeError):
    """Raised when a review decision is not applicable to its target."""


def queue_reasons(claim: Claim, config: Config) -> list[tuple[str, str]]:
    """Return ``(priority, reason)`` pairs explaining why a claim needs review."""
    reasons: list[tuple[str, str]] = []
    confidence = claim.extraction_confidence
    if confidence is not None and confidence < config.review_confidence_threshold:
        reasons.append(("HIGH", f"LOW_CONFIDENCE:{confidence:.2f}"))
    missing = claim.missing_scope_fields()
    if missing:
        reasons.append(("HIGH" if len(missing) >= 3 else "NORMAL", "MISSING_SCOPE:" + ",".join(missing)))
    if claim.metric_value is not None and not claim.evidence_excerpt:
        reasons.append(("HIGH", "NUMERIC_CLAIM_WITHOUT_EXCERPT"))
    if not reasons:
        reasons.append(("NORMAL", "ROUTINE_EXTRACTION"))
    return reasons


def build_claim_queue(claims: Sequence[Claim], config: Config) -> list[ReviewItem]:
    """Create one queue item per unreviewed claim, at its highest priority.

    A claim with several problems is queued once - reviewers work on claims, not
    on findings about claims - carrying every reason so the worst one drives the
    priority and none of the detail is lost.
    """
    items: list[ReviewItem] = []
    for claim in claims:
        if claim.review_status is not ReviewStatus.CANDIDATE:
            continue
        reasons = queue_reasons(claim, config)
        priority = _highest({p for p, _ in reasons})
        items.append(
            ReviewItem(
                target_type="CLAIM",
                target_id=claim.claim_id,
                reason=";".join(r for _, r in reasons),
                priority=priority,
            )
        )
    return items


def build_relationship_queue(relationships: Sequence[ClaimRelationship]) -> list[ReviewItem]:
    """Queue relationships that a human must adjudicate.

    A contradiction candidate is always HIGH: it is the assertion most likely to
    be repeated outside the system, so it gets the most scrutiny.
    """
    items: list[ReviewItem] = []
    for relationship in relationships:
        if relationship.review_status is not ReviewStatus.CANDIDATE:
            continue
        if relationship.relationship_type is RelationshipType.CONTRADICTS:
            priority, reason = "HIGH", "CONTRADICTION_CANDIDATE"
        elif relationship.comparability_status is ComparabilityStatus.PARTIALLY_COMPARABLE:
            priority, reason = "NORMAL", "PARTIAL_COMPARABILITY"
        elif relationship.relationship_type is RelationshipType.NOT_COMPARABLE_YET:
            # Not worth a reviewer's time individually: it is already reported
            # honestly as an open question by v_research_open_questions.
            continue
        else:
            priority, reason = "LOW", relationship.relationship_type.value
        items.append(
            ReviewItem(
                target_type="RELATIONSHIP",
                target_id=relationship.relationship_id,
                reason=reason,
                priority=priority,
            )
        )
    return items


#: Fields a reviewer may correct. A reviewer fixes the record's scope and
#: citation; they never restate what the source found.
AMENDABLE_FIELDS: frozenset[str] = frozenset(
    {"task", "method", "metric", "benchmark", "condition_text", "metric_unit", "page_number"}
)


def validate_decision(
    decision: str,
    *,
    reviewer: str,
    note: str | None,
    amendments: Mapping[str, object] | None,
) -> None:
    """Check a review decision against the rules, independent of storage.

    Shared by the in-process path (:func:`apply_claim_decision`) and the review
    app's SQL path, so the rules cannot drift between them.

    Raises:
        ReviewError: The decision is invalid, unattributed or unexplained.
    """
    if not reviewer:
        raise ReviewError("a review decision must record the reviewer")
    if decision not in {"ACCEPTED", "AMENDED", "REJECTED"}:
        raise ReviewError(f"unknown decision {decision!r}")
    if decision == "AMENDED":
        if not amendments:
            raise ReviewError("AMENDED requires at least one field amendment")
        if not note:
            raise ReviewError("AMENDED requires a note explaining the amendment")
        illegal = set(amendments) - AMENDABLE_FIELDS
        if illegal:
            raise ReviewError(f"fields may not be amended by review: {sorted(illegal)}")
    if decision == "REJECTED" and not note:
        raise ReviewError("REJECTED requires a note explaining the rejection")


def apply_claim_decision(
    claim: Claim,
    *,
    decision: str,
    reviewer: str,
    note: str | None = None,
    amendments: dict[str, object] | None = None,
) -> Claim:
    """Apply a reviewer's decision to a claim.

    Args:
        claim: The claim under review.
        decision: ``ACCEPTED``, ``AMENDED`` or ``REJECTED``.
        reviewer: Principal recording the decision. Required - an anonymous
            review is not a review.
        note: Reviewer rationale, mandatory for rejection and amendment.
        amendments: Field updates applied with ``AMENDED``. Only scope and
            citation fields may be amended; a reviewer corrects the record, they
            do not restate the source's finding.

    Returns:
        The updated claim.

    Raises:
        ReviewError: The decision is invalid, unattributed or unexplained.
    """
    validate_decision(decision, reviewer=reviewer, note=note, amendments=amendments)
    if claim.review_status in (ReviewStatus.REVIEWED, ReviewStatus.REJECTED):
        raise ReviewError(f"claim {claim.claim_id} is already {claim.review_status.value}")

    if decision == "REJECTED":
        claim.review_status = ReviewStatus.REJECTED
    else:
        for field_name, value in (amendments or {}).items():
            setattr(claim, field_name, value)
        claim.review_status = ReviewStatus.REVIEWED

    claim.reviewed_by = reviewer
    claim.reviewed_at = utcnow()
    claim.review_note = note
    return claim


# Supersession lives in review.supersession: retiring a reviewed claim is a
# decision about a source *change*, and it needs the old/new pairing that module
# builds. There is deliberately no simpler one-line version here to reach for.


def backlog(items: Iterable[ReviewItem]) -> dict[str, int]:
    """Count open queue items by priority."""
    counts = {"HIGH": 0, "NORMAL": 0, "LOW": 0}
    for item in items:
        if item.status == "OPEN":
            counts[item.priority] += 1
    return counts


def _highest(priorities: set[str]) -> str:
    for level in ("HIGH", "NORMAL", "LOW"):
        if level in priorities:
            return level
    return "NORMAL"
