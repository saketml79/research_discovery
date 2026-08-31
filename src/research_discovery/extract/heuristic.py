"""Rule-based claim extractor.

Not a replacement for the LLM extractor. It exists so that the pipeline, the
review queue and the Genie surface can be exercised deterministically in CI and
in a workspace without model-serving access, and so there is always a baseline
to measure the LLM extractor against.

It is conservative by design: it fires only on sentences carrying an explicit
comparative or result verb near a number, and it never guesses a benchmark.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..models import Chunk, ClaimType
from .base import CandidateClaim, ClaimExtractor

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

_RESULT_VERB = re.compile(
    r"\b(outperform\w*|improv\w+|reduc\w+|increas\w+|decreas\w+|achiev\w+|"
    r"beat\w*|exceed\w+|underperform\w*|degrad\w+|fail\w*|cost\w*)\b",
    re.IGNORECASE,
)
_LIMITATION = re.compile(
    r"\b(limitation|does not generalize|we did not|cannot|fails to|"
    r"only when|is limited to|caveat)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|percent|points?|x|ms|s|usd|\$)?")
_METRIC_TERM = re.compile(
    r"\b(win rate|comprehensiveness|diversity|f1|em|exact match|accuracy|recall@?\d*|"
    r"precision|ndcg@?\d*|mrr|latency|throughput|token cost|index(?:ing)? cost)\b",
    re.IGNORECASE,
)
_METHOD_TERM = re.compile(
    r"\b(graphrag|graph rag|global search|local search|drift search|vector rag|"
    r"baseline rag|naive rag|hybrid retrieval|bm25|hipporag|lightrag|raptor)\b",
    re.IGNORECASE,
)
_BENCHMARK_TERM = re.compile(
    r"\b(hotpotqa|musique|2wikimultihopqa|multihop-rag|narrativeqa|triviaqa|"
    r"nq|natural questions|quality|podcast transcripts|news articles|ultradomain)\b",
    re.IGNORECASE,
)
_TASK_TERM = re.compile(
    r"\b(multi-?hop (?:qa|question answering)|question answering|summarization|"
    r"query-?focused summarization|sense-?making|retrieval)\b",
    re.IGNORECASE,
)


class HeuristicClaimExtractor(ClaimExtractor):
    """Deterministic baseline extractor over sentence-level patterns."""

    name = "heuristic"
    version = "1.1.0"

    def __init__(self, max_claims_per_chunk: int = 3) -> None:
        self._max_claims = max_claims_per_chunk

    def extract(self, chunk: Chunk) -> Sequence[CandidateClaim]:
        candidates: list[CandidateClaim] = []
        for sentence in _SENTENCE.split(chunk.text):
            sentence = sentence.strip()
            if len(sentence) < 40:
                continue
            is_result = bool(_RESULT_VERB.search(sentence))
            is_limitation = bool(_LIMITATION.search(sentence))
            if not (is_result or is_limitation):
                continue

            value, unit = _first_number(sentence)
            claim_type = (
                ClaimType.LIMITATION.value
                if is_limitation and value is None
                else ClaimType.PERFORMANCE.value
            )
            fields = {
                "task": _match(_TASK_TERM, sentence),
                "method": _match(_METHOD_TERM, sentence),
                "metric": _match(_METRIC_TERM, sentence),
                "benchmark": _match(_BENCHMARK_TERM, sentence),
                "condition_text": None,
            }
            missing = [k for k, v in fields.items() if not v]
            candidates.append(
                CandidateClaim(
                    claim_text=sentence,
                    claim_type=claim_type,
                    metric_value=value,
                    metric_unit=unit,
                    evidence_excerpt=sentence[:600],
                    confidence=0.45 if value is not None else 0.35,
                    missing_field_reason=(
                        f"MISSING:{','.join(missing)} - rule-based extractor recognises only "
                        "controlled-vocabulary surface forms; a reviewer must supply the rest"
                        if missing
                        else None
                    ),
                    warnings=["HEURISTIC_EXTRACTION"],
                    **fields,
                )
            )
            if len(candidates) >= self._max_claims:
                break
        return candidates


def _match(pattern: re.Pattern[str], text: str) -> str | None:
    found = pattern.search(text)
    return found.group(0).strip().lower() if found else None


def _first_number(text: str) -> tuple[float | None, str | None]:
    """Return the first number in ``text`` with its unit, if any."""
    match = _NUMBER.search(text)
    if not match:
        return None, None
    try:
        value = float(match.group(1))
    except ValueError:  # pragma: no cover - regex guarantees a numeric group
        return None, None
    raw_unit = (match.group(2) or "").lower()
    unit = {"%": "percent", "percent": "percent", "$": "usd", "usd": "usd"}.get(raw_unit)
    return value, unit or (raw_unit or None)
