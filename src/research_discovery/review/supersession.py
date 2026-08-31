"""Source-change handling: what happens to reviewed knowledge when a source moves.

The rule from the brief is precise and easy to get wrong: a changed source
creates a new version, and previously reviewed knowledge is superseded **only
after review** — not automatically when new bytes arrive. A silent auto-supersede
would let an edited preprint quietly retract a finding a human had accepted, with
no one seeing the change.

So a new source version does three things:

1. it never touches the old claims' review status by itself;
2. it queues every affected reviewed claim for re-review with reason
   ``SOURCE_UPDATED``, carrying the matched replacement when one exists;
3. only when a reviewer accepts the replacement does :func:`apply_supersession`
   mark the old claim ``SUPERSEDED``.

Matching old to new is deliberately conservative: claims are paired on identical
normalized text first, then on identical scope, and anything unmatched is
surfaced to the reviewer rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..ids import normalize_text
from ..models import Chunk, Claim, ReviewItem, ReviewStatus, utcnow
from .queue import ReviewError


@dataclass(frozen=True, slots=True)
class SupersessionCandidate:
    """A proposed old→new claim replacement awaiting a reviewer's decision.

    Attributes:
        previous_claim_id: The reviewed claim that may be replaced.
        replacement_claim_id: The candidate claim from the new source version,
            or ``None`` when the finding vanished from the new revision.
        match_basis: How the pair was matched — ``IDENTICAL_TEXT``,
            ``IDENTICAL_SCOPE`` or ``NO_MATCH``.
        note: Reviewer-facing explanation of what changed.
    """

    previous_claim_id: str
    replacement_claim_id: str | None
    match_basis: str
    note: str


def _scope_key(claim: Claim) -> tuple[str, ...]:
    """Scope identity used for the second matching pass."""
    return tuple(
        normalize_text(getattr(claim, field) or "")
        for field in ("task", "method", "metric", "benchmark")
    )


def plan_supersession(
    previous_claims: Sequence[Claim], replacement_claims: Sequence[Claim]
) -> list[SupersessionCandidate]:
    """Pair reviewed claims from an old version against a new version's claims.

    Args:
        previous_claims: Claims from the superseded source version. Only
            reviewed, non-superseded claims are considered; a candidate claim
            that was never accepted has nothing to lose.
        replacement_claims: Freshly extracted claims from the new version.

    Returns:
        One candidate per affected reviewed claim, including the ones with no
        match — a finding that disappeared from a revision is the case a
        reviewer most needs to see.
    """
    affected = [c for c in previous_claims if c.is_runtime_visible]
    if not affected:
        return []

    by_text = {normalize_text(c.claim_text): c for c in replacement_claims}
    by_scope: dict[tuple[str, ...], list[Claim]] = {}
    for claim in replacement_claims:
        by_scope.setdefault(_scope_key(claim), []).append(claim)

    used: set[str] = set()
    candidates: list[SupersessionCandidate] = []

    for previous in affected:
        text_match = by_text.get(normalize_text(previous.claim_text))
        if text_match is not None and text_match.claim_id not in used:
            used.add(text_match.claim_id)
            candidates.append(
                SupersessionCandidate(
                    previous.claim_id,
                    text_match.claim_id,
                    "IDENTICAL_TEXT",
                    "The new revision states this claim unchanged. Accepting carries the "
                    "review forward to the new source version.",
                )
            )
            continue

        scope_matches = [c for c in by_scope.get(_scope_key(previous), []) if c.claim_id not in used]
        if scope_matches:
            replacement = scope_matches[0]
            used.add(replacement.claim_id)
            change = _describe_change(previous, replacement)
            candidates.append(
                SupersessionCandidate(
                    previous.claim_id,
                    replacement.claim_id,
                    "IDENTICAL_SCOPE",
                    f"The new revision reports the same task, method, metric and benchmark "
                    f"with different wording or values. {change}",
                )
            )
            continue

        candidates.append(
            SupersessionCandidate(
                previous.claim_id,
                None,
                "NO_MATCH",
                "This reviewed claim has no counterpart in the new revision. It may have "
                "been retracted, moved, or missed by extraction. Do not supersede it "
                "without checking the new source.",
            )
        )

    return candidates


def _describe_change(previous: Claim, replacement: Claim) -> str:
    """One sentence describing what moved between two matched claims."""
    if previous.metric_value is None or replacement.metric_value is None:
        return "Reported wording changed; no numeric comparison is possible."
    if previous.metric_value == replacement.metric_value:
        return f"The reported value is unchanged at {previous.metric_value}."
    return (
        f"The reported value changed from {previous.metric_value} to "
        f"{replacement.metric_value}."
    )


def build_supersession_queue(candidates: Sequence[SupersessionCandidate]) -> list[ReviewItem]:
    """Queue every supersession candidate for a human decision.

    A vanished claim is queued HIGH: an unexplained disappearance is the change
    most likely to matter and least likely to be noticed.
    """
    items: list[ReviewItem] = []
    for candidate in candidates:
        vanished = candidate.replacement_claim_id is None
        items.append(
            ReviewItem(
                target_type="CLAIM",
                target_id=candidate.previous_claim_id,
                reason=(
                    f"SOURCE_UPDATED:{candidate.match_basis}"
                    + (f";REPLACEMENT:{candidate.replacement_claim_id}" if not vanished else "")
                ),
                priority="HIGH" if vanished else "NORMAL",
            )
        )
    return items


def apply_supersession(
    previous: Claim, replacement: Claim | None, *, reviewer: str, note: str | None = None
) -> Claim:
    """Mark ``previous`` superseded after a reviewer accepted the change.

    Args:
        previous: The reviewed claim being retired.
        replacement: The accepted replacement, or ``None`` when the reviewer
            confirmed the finding is gone from the new revision.
        reviewer: Principal recording the decision.
        note: Reviewer rationale. Required when there is no replacement, because
            retiring a finding with nothing in its place needs an explanation.

    Returns:
        The updated previous claim.

    Raises:
        ReviewError: The decision is unattributed, unexplained, or applied to a
            claim that was never runtime-visible.
    """
    if not reviewer:
        raise ReviewError("a supersession decision must record the reviewer")
    if not previous.is_runtime_visible:
        raise ReviewError(
            f"claim {previous.claim_id} is not currently runtime-visible; nothing to supersede"
        )
    if replacement is None and not note:
        raise ReviewError("retiring a claim with no replacement requires a note")
    if replacement is not None and replacement.claim_id == previous.claim_id:
        raise ReviewError("a claim cannot supersede itself")

    previous.superseded_by_claim_id = replacement.claim_id if replacement else None
    previous.review_status = ReviewStatus.SUPERSEDED
    previous.reviewed_by = reviewer
    previous.reviewed_at = utcnow()
    previous.review_note = note or "Superseded by a newer reviewed source version."
    return previous


def supersede_chunks(chunks: Iterable[Chunk], superseded_version_id: str) -> list[Chunk]:
    """Mark chunks of a retired source version so retrieval stops serving them."""
    updated: list[Chunk] = []
    for chunk in chunks:
        if chunk.source_version_id == superseded_version_id:
            chunk.lifecycle_state = "SUPERSEDED"
            updated.append(chunk)
    return updated
