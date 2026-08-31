"""The agent's answer contract.

The Research Discovery agent answers in a fixed shape so that an answer can be
checked mechanically, not just read. ``validate_answer`` is used by the
benchmark harness and can be used at serving time to reject a malformed answer
before a user sees it.
"""

from __future__ import annotations

from typing import Any, Collection, Mapping

ANSWER_SCHEMA: dict = {
    "type": "object",
    "required": ["answer", "supporting_claims", "limitations", "recommended_next_step"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "supporting_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim_id", "claim_text", "source_url", "review_status"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "claim_text": {"type": "string"},
                    "source_url": {"type": "string"},
                    "page_number": {"type": ["integer", "null"]},
                    "review_status": {"type": "string"},
                },
            },
        },
        "contrary_claims": {"type": "array"},
        "comparability": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from_claim_id", "to_claim_id", "comparability_status"],
                "properties": {
                    "from_claim_id": {"type": "string"},
                    "to_claim_id": {"type": "string"},
                    "comparability_status": {"type": "string"},
                    "missing_dimensions": {"type": ["string", "null"]},
                },
            },
        },
        "scope_conditions": {"type": "array", "items": {"type": "string"}},
        "unresolved_questions": {"type": "array", "items": {"type": "string"}},
        "recommended_next_step": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "unreviewed_leads": {"type": "array"},
    },
}

#: Words that assert a disagreement. Using one obliges the answer to carry a
#: COMPARABLE verdict for the pair it is talking about.
CONFLICT_WORDS = ("contradict", "disagree", "conflict", "refute", "inconsistent with")

#: Words that assert agreement across sources, which obliges >1 distinct source.
CONSENSUS_WORDS = ("consensus", "sources agree", "widely reported", "consistently show")


def validate_answer(
    answer: Mapping[str, Any], *, retrieved_claim_ids: Collection[str] = ()
) -> list[str]:
    """Return contract violations in ``answer``. Empty means well-formed.

    Checks performed, beyond structure:

    * every cited claim is REVIEWED and was actually retrieved;
    * the words "contradict"/"disagree" appear only alongside a COMPARABLE
      comparability verdict;
    * a consensus statement is backed by at least two distinct sources;
    * unreviewed leads are labelled as such and never cited as support.
    """
    problems: list[str] = []

    for required in ANSWER_SCHEMA["required"]:
        if required not in answer:
            problems.append(f"MISSING_FIELD:{required}")
    if problems:
        return problems

    supporting = answer.get("supporting_claims") or []
    contrary = answer.get("contrary_claims") or []
    comparability = answer.get("comparability") or []
    cited = list(supporting) + list(contrary)

    for entry in cited:
        claim_id = entry.get("claim_id")
        if not claim_id:
            problems.append("CITED_CLAIM_WITHOUT_ID")
            continue
        if entry.get("review_status") != "REVIEWED":
            problems.append(f"UNREVIEWED_CLAIM_CITED:{claim_id}")
        if not entry.get("source_url"):
            problems.append(f"CLAIM_WITHOUT_SOURCE_URL:{claim_id}")
        if retrieved_claim_ids and claim_id not in retrieved_claim_ids:
            problems.append(f"CLAIM_NOT_RETRIEVED:{claim_id}")

    prose = str(answer.get("answer", "")).lower()

    if any(word in prose for word in CONFLICT_WORDS):
        comparable = [
            c for c in comparability if c.get("comparability_status") == "COMPARABLE"
        ]
        if not comparable:
            problems.append("CONFLICT_ASSERTED_WITHOUT_COMPARABLE_VERDICT")

    if any(word in prose for word in CONSENSUS_WORDS):
        sources = {e.get("source_url") for e in supporting if e.get("source_url")}
        if len(sources) < 2:
            problems.append("CONSENSUS_ASSERTED_FROM_FEWER_THAN_TWO_SOURCES")

    for lead in answer.get("unreviewed_leads") or []:
        if lead.get("review_status") == "REVIEWED":
            problems.append("REVIEWED_CLAIM_LISTED_AS_UNREVIEWED_LEAD")

    if not answer.get("limitations"):
        problems.append("EMPTY_LIMITATIONS")

    return problems


EXAMPLE_ANSWER: dict = {
    "answer": (
        "Two reviewed sources report GraphRAG comprehensiveness win rates on different "
        "corpora, so their numbers cannot be compared directly."
    ),
    "supporting_claims": [
        {
            "claim_id": "clm-example00001",
            "claim_text": "Graph-based global search won 72-83% of comprehensiveness comparisons "
            "against naive RAG on the podcast transcript corpus.",
            "source_url": "https://arxiv.org/abs/2404.16130",
            "page_number": 1,
            "review_status": "REVIEWED",
        }
    ],
    "contrary_claims": [],
    "comparability": [
        {
            "from_claim_id": "clm-example00001",
            "to_claim_id": "clm-example00002",
            "comparability_status": "INSUFFICIENT_EVIDENCE",
            "missing_dimensions": "benchmark,condition",
        }
    ],
    "scope_conditions": ["One corpus is podcast transcripts; the other is multi-hop QA."],
    "unresolved_questions": [
        "No reviewed source reports comprehensiveness win rate and multi-hop F1 on the same corpus."
    ],
    "recommended_next_step": (
        "Proposal recorded as PENDING_APPROVAL: ingest a source evaluating both metrics "
        "on one corpus."
    ),
    "limitations": [
        "The corpus contains 12 reviewed sources on this topic; absence of a finding here "
        "is not evidence that none exists."
    ],
    "unreviewed_leads": [],
}
