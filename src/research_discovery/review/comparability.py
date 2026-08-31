"""Comparability and relationship detection.

This module is where the project's central claim is enforced in code: *a
contradiction is not opposite wording, it is a disagreement between claims whose
scope actually overlaps.* Two claims are compared only after task, metric and
benchmark match; conditions govern whether the comparison is qualified. Anything
short of that is ``NOT_COMPARABLE_YET`` with the missing dimensions named.

The scoring is deliberately transparent and deterministic: weighted dimension
overlap, no model in the loop, so a reviewer can replay any verdict by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

from ..ids import normalize_text
from ..models import (
    Claim,
    ClaimRelationship,
    ComparabilityStatus,
    RelationshipType,
    ReviewStatus,
)

DETECTOR_NAME = "rule_comparability_v1"

#: Dimension weights. Task, metric and benchmark are gating; condition presence
#: only distinguishes a full comparison from a qualified one.
WEIGHTS: dict[str, float] = {
    "task": 0.30,
    "metric": 0.30,
    "benchmark": 0.25,
    "condition": 0.15,
}

#: Relative difference below which two values of the same metric are treated as
#: agreement rather than disagreement. Sources round; 0.5 vs 0.51 is not a fight.
AGREEMENT_TOLERANCE = 0.05

#: Relative difference above which a same-scope value gap is a real conflict.
CONTRADICTION_THRESHOLD = 0.15


@dataclass(frozen=True, slots=True)
class ComparabilityVerdict:
    """Result of comparing the scope of two claims.

    Attributes:
        status: Whether the pair may be compared, and how strictly.
        score: Weighted dimension overlap in [0, 1].
        missing_dimensions: Scope dimensions absent or mismatched.
        rationale: Plain-language explanation a reviewer can check.
    """

    status: ComparabilityStatus
    score: float
    missing_dimensions: tuple[str, ...]
    rationale: str

    @property
    def is_comparable(self) -> bool:
        """True when a difference between the claims may be reported at all."""
        return self.status is not ComparabilityStatus.INSUFFICIENT_EVIDENCE


def _same(left: str | None, right: str | None) -> bool:
    """Case- and whitespace-insensitive equality, with ``None`` never equal."""
    if left is None or right is None:
        return False
    return normalize_text(left) == normalize_text(right)


def assess_comparability(a: Claim, b: Claim) -> ComparabilityVerdict:
    """Decide whether two claims may be compared.

    Args:
        a: First claim.
        b: Second claim.

    Returns:
        A verdict carrying the status, the weighted overlap score, the missing
        dimensions and a reviewer-checkable rationale.
    """
    task_match = _same(a.task, b.task)
    metric_match = _same(a.metric, b.metric)
    benchmark_match = _same(a.benchmark, b.benchmark)
    condition_present = bool(a.condition_text) and bool(b.condition_text)

    score = (
        WEIGHTS["task"] * task_match
        + WEIGHTS["metric"] * metric_match
        + WEIGHTS["benchmark"] * benchmark_match
        + WEIGHTS["condition"] * condition_present
    )
    missing = tuple(
        name
        for name, matched in (
            ("task", task_match),
            ("metric", metric_match),
            ("benchmark", benchmark_match),
            ("condition", condition_present),
        )
        if not matched
    )

    if task_match and metric_match and benchmark_match and condition_present:
        return ComparabilityVerdict(
            ComparabilityStatus.COMPARABLE,
            round(score, 4),
            missing,
            "Same task, metric and benchmark, with stated conditions on both claims. "
            "A difference in reported values is a genuine disagreement.",
        )

    if task_match and metric_match and benchmark_match:
        return ComparabilityVerdict(
            ComparabilityStatus.PARTIALLY_COMPARABLE,
            round(score, 4),
            missing,
            "Same task, metric and benchmark, but at least one claim does not state its "
            "conditions (corpus size, model, retrieval budget). Any difference must be "
            "reported as conditional, naming the missing conditions.",
        )

    return ComparabilityVerdict(
        ComparabilityStatus.INSUFFICIENT_EVIDENCE,
        round(score, 4),
        missing,
        "Scope does not overlap on " + ", ".join(missing) + ". These claims cannot be "
        "called contradictory or agreeing; report insufficient evidence to compare.",
    )


def _relative_difference(x: float, y: float) -> float:
    """Relative difference between two values, scaled by the larger magnitude."""
    scale = max(abs(x), abs(y))
    return 0.0 if scale == 0 else abs(x - y) / scale


def classify_relationship(a: Claim, b: Claim, verdict: ComparabilityVerdict) -> RelationshipType:
    """Choose the edge type for a claim pair given its comparability verdict.

    ``CONTRADICTS`` is reachable only for a comparable pair with numeric values
    that differ beyond ``CONTRADICTION_THRESHOLD``. Without numbers on both
    sides, a comparable pair is at most ``REFINES``: two prose claims about the
    same benchmark restrict each other's scope, they do not refute each other.
    """
    if not verdict.is_comparable:
        return RelationshipType.NOT_COMPARABLE_YET

    if a.metric_value is None or b.metric_value is None:
        if normalize_text(a.claim_text) == normalize_text(b.claim_text):
            return RelationshipType.DUPLICATES
        return RelationshipType.REFINES

    difference = _relative_difference(a.metric_value, b.metric_value)
    if difference <= AGREEMENT_TOLERANCE:
        return RelationshipType.SUPPORTS
    if difference >= CONTRADICTION_THRESHOLD:
        # A partially comparable pair still differs materially, but the missing
        # conditions could explain it - that is a refinement, not a refutation.
        return (
            RelationshipType.CONTRADICTS
            if verdict.status is ComparabilityStatus.COMPARABLE
            else RelationshipType.REFINES
        )
    return RelationshipType.REFINES


def build_relationship(a: Claim, b: Claim) -> ClaimRelationship:
    """Build the reviewable edge between two claims."""
    verdict = assess_comparability(a, b)
    relationship_type = classify_relationship(a, b, verdict)
    detail = ""
    if a.metric_value is not None and b.metric_value is not None:
        detail = (
            f" Reported values {a.metric_value} vs {b.metric_value}"
            f" (relative difference {_relative_difference(a.metric_value, b.metric_value):.0%})."
        )
    return ClaimRelationship(
        from_claim_id=a.claim_id,
        to_claim_id=b.claim_id,
        relationship_type=relationship_type,
        comparability_status=verdict.status,
        comparability_score=verdict.score,
        missing_dimensions=",".join(verdict.missing_dimensions) or None,
        rationale=verdict.rationale + detail,
        detector_name=DETECTOR_NAME,
        review_status=ReviewStatus.CANDIDATE,
    )


def detect_relationships(
    claims: Sequence[Claim], *, reviewed_only: bool = True
) -> list[ClaimRelationship]:
    """Build edges for every claim pair drawn from different sources.

    Args:
        claims: Claims to relate.
        reviewed_only: When true (the default), only reviewed claims are
            related. Relating candidate claims would let unverified extractions
            manufacture a disagreement.

    Returns:
        One relationship per eligible pair, deduplicated by relationship id.
    """
    pool = [c for c in claims if c.is_runtime_visible] if reviewed_only else list(claims)
    seen: set[str] = set()
    relationships: list[ClaimRelationship] = []
    for a, b in combinations(pool, 2):
        # Two claims from one source version are the same author's voice; an
        # apparent conflict there is a reading error, not a research dispute.
        if a.source_id == b.source_id:
            continue
        relationship = build_relationship(a, b)
        if relationship.relationship_id in seen:
            continue
        seen.add(relationship.relationship_id)
        relationships.append(relationship)
    return relationships


def summarize(relationships: Iterable[ClaimRelationship]) -> dict[str, int]:
    """Count relationships by type. Used by pipeline telemetry and tests."""
    counts: dict[str, int] = {}
    for relationship in relationships:
        key = relationship.relationship_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts
