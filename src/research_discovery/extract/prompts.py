"""Extraction prompt and response schema.

Kept in one module so the prompt is versioned, diffable and testable. Bumping
``EXTRACTION_PROMPT_VERSION`` changes ``extractor_version`` on every claim
produced afterwards, which is what makes a prompt change auditable.
"""

from __future__ import annotations

EXTRACTION_PROMPT_VERSION = "2026.08.1"

#: JSON Schema the model must satisfy. Enforced again in Python after the call -
#: a schema in a prompt is a request, not a guarantee.
CLAIM_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "required": ["claims"],
    "additionalProperties": False,
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim_text", "claim_type", "confidence"],
                "additionalProperties": False,
                "properties": {
                    "claim_text": {"type": "string", "minLength": 10},
                    "claim_type": {
                        "type": "string",
                        "enum": [
                            "PERFORMANCE",
                            "LIMITATION",
                            "METHOD_DESCRIPTION",
                            "RESOURCE_COST",
                            "NEGATIVE_RESULT",
                            "RECOMMENDATION",
                        ],
                    },
                    "task": {"type": ["string", "null"]},
                    "method": {"type": ["string", "null"]},
                    "metric": {"type": ["string", "null"]},
                    "metric_value": {"type": ["number", "null"]},
                    "metric_unit": {"type": ["string", "null"]},
                    "benchmark": {"type": ["string", "null"]},
                    "condition_text": {"type": ["string", "null"]},
                    "evidence_excerpt": {"type": ["string", "null"], "maxLength": 600},
                    "missing_field_reason": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        }
    },
}

SYSTEM_PROMPT = """\
You extract structured research claims from one passage of one source document.

A claim is what the source ASSERTS, not what it is about. "This paper studies
GraphRAG on multi-hop QA" is a topic, not a claim. "GraphRAG local search beat
vector RAG on comprehensiveness win rate on the podcast corpus" is a claim.

Rules you must follow exactly:
1. Extract only claims stated in the passage you were given. Do not use outside
   knowledge, and do not carry over context from other passages.
2. Never invent a number. metric_value may only be a figure that appears
   literally in the passage. If a result is described in words without a number,
   set metric_value to null.
3. evidence_excerpt must be a verbatim substring of the passage, at most 600
   characters. Do not paraphrase it, do not fix its typos.
4. Scope fields (task, method, metric, benchmark, condition_text) may be null.
   When any is null you MUST set missing_field_reason naming which fields are
   absent and why, for example "benchmark and condition not stated in this
   passage; results section may state them".
5. condition_text records what would make this result comparable to another:
   corpus, corpus size, model used, retrieval budget, hardware, hyperparameters.
6. confidence is your confidence that you read the passage correctly, not your
   belief that the claim is true. A clearly-worded claim you are unsure is
   correct still gets high confidence.
7. If the passage contains no assertion - it is a heading, an acknowledgement,
   a bibliography entry, boilerplate - return an empty claims array. Returning
   nothing is a correct and common answer.
8. Return ONLY JSON matching the provided schema. No prose, no code fence.
"""

USER_PROMPT_TEMPLATE = """\
Source: {source_title}
Source type: {source_type}
URL: {source_url}
Section: {section_title}
Page: {page_number}

Passage:
\"\"\"
{chunk_text}
\"\"\"

Return ONLY a single JSON object with exactly this shape (no other keys, no
prose, no markdown code fence):

{{"claims": [{{"claim_text": "...", "claim_type": "PERFORMANCE", "task": null,
"method": null, "metric": null, "metric_value": null, "metric_unit": null,
"benchmark": null, "condition_text": null, "evidence_excerpt": "...",
"missing_field_reason": null, "confidence": 0.9}}]}}

If the passage contains no claim, return exactly {{"claims": []}}.
claim_type must be one of: PERFORMANCE, LIMITATION, METHOD_DESCRIPTION,
RESOURCE_COST, NEGATIVE_RESULT, RECOMMENDATION.
Use exactly these field names - do not rename "claim_text" to "claim", and do
not nest task/method/metric/benchmark/condition_text under a "scope" key.
Extract at most {max_claims} claims.
"""


def build_messages(
    *,
    chunk_text: str,
    source_title: str | None,
    source_type: str,
    source_url: str,
    section_title: str | None,
    page_number: int | None,
    max_claims: int = 3,
) -> list[dict[str, str]]:
    """Build the chat messages for one chunk."""
    user = USER_PROMPT_TEMPLATE.format(
        source_title=source_title or "(untitled)",
        source_type=source_type,
        source_url=source_url,
        section_title=section_title or "(none)",
        page_number=page_number if page_number is not None else "(none)",
        chunk_text=chunk_text,
        max_claims=max_claims,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
