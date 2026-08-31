"""Proposal writing: the only write path the agent has.

Two invariants are enforced here rather than trusted to the model:

1. A proposal is created as ``PENDING_APPROVAL`` and by no other status. There
   is no code path in this project that approves or executes one.
2. A proposal may only reference claim ids and source URLs that the agent
   actually saw in tool output. That closes the gap where a fluent model
   proposes work on a record it invented.
"""

from __future__ import annotations

import json
from typing import Any, Collection, Mapping

from ..models import Proposal

#: Required payload keys per proposal type.
REQUIRED_PAYLOAD_FIELDS: Mapping[str, tuple[str, ...]] = {
    "REVIEW_CLAIM": ("claim_id", "requested_action"),
    "INGEST_SOURCE": ("canonical_url", "source_type", "why_relevant"),
    "RESOLVE_CONTRADICTION": ("from_claim_id", "to_claim_id", "missing_dimensions"),
    "OPEN_QUESTION": ("question_text", "supporting_claim_ids"),
}

MAX_PAYLOAD_BYTES = 16_384


class ProposalValidationError(ValueError):
    """Raised when a proposal is malformed or cites unseen evidence."""


def validate_payload(
    proposal_type: str,
    payload: Mapping[str, Any],
    *,
    known_claim_ids: Collection[str] = (),
) -> None:
    """Validate a proposal payload before it is written.

    Args:
        proposal_type: One of ``REQUIRED_PAYLOAD_FIELDS``.
        payload: Proposal body.
        known_claim_ids: Claim ids the agent retrieved during this turn. When
            provided, every claim id in the payload must appear here.

    Raises:
        ProposalValidationError: The payload is invalid or cites unseen claims.
    """
    if proposal_type not in REQUIRED_PAYLOAD_FIELDS:
        raise ProposalValidationError(
            f"unknown proposal_type {proposal_type!r}; expected one of "
            f"{sorted(REQUIRED_PAYLOAD_FIELDS)}"
        )

    missing = [f for f in REQUIRED_PAYLOAD_FIELDS[proposal_type] if not payload.get(f)]
    if missing:
        raise ProposalValidationError(
            f"{proposal_type} payload is missing required fields: {missing}"
        )

    encoded = json.dumps(payload, sort_keys=True, default=str)
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ProposalValidationError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")

    if known_claim_ids:
        cited = _cited_claim_ids(payload)
        unknown = sorted(cited - set(known_claim_ids))
        if unknown:
            raise ProposalValidationError(
                "proposal cites claim ids that did not appear in tool output: " + ", ".join(unknown)
            )

    if proposal_type == "INGEST_SOURCE":
        url = str(payload["canonical_url"])
        if not url.startswith(("http://", "https://")):
            raise ProposalValidationError("canonical_url must be an absolute http(s) URL")


def build_proposal(
    proposal_type: str,
    payload: Mapping[str, Any],
    *,
    created_by: str,
    rationale: str,
    investigation_id: str | None = None,
    known_claim_ids: Collection[str] = (),
) -> Proposal:
    """Validate and build a ``PENDING_APPROVAL`` proposal.

    Raises:
        ProposalValidationError: Validation failed; nothing is written.
    """
    if not rationale or not rationale.strip():
        raise ProposalValidationError("a proposal must state its rationale")
    validate_payload(proposal_type, payload, known_claim_ids=known_claim_ids)
    return Proposal(
        proposal_type=proposal_type,
        payload_json=json.dumps(payload, sort_keys=True, default=str),
        created_by=created_by,
        rationale=rationale.strip(),
        investigation_id=investigation_id,
    )


def _cited_claim_ids(payload: Mapping[str, Any]) -> set[str]:
    """Collect every claim id referenced anywhere in a payload."""
    found: set[str] = set()

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                walk(item, key)
        elif isinstance(node, str) and key and "claim_id" in key:
            found.add(node)

    walk(payload)
    return found
