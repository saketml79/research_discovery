"""Genie Agent configuration as code.

The agent's ``serialized_space`` is generated from this module rather than
hand-edited in the UI, so the agent's tables, instructions, example SQL, joins
and benchmarks are versioned, diffable and validated in CI before deployment.

Layering rule, applied deliberately:

* semantics that belong to the *data* (grain, meaning, joins) live in UC
  comments and in the views;
* semantics that belong to *this agent* (refusal rules, output shape) live in
  the instructions here;
* verified query shapes live in example SQL;
* evaluation-only prompts live in benchmarks and are never runtime context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config

AGENT_NAME = "research_discovery"
AGENT_DISPLAY_NAME = "Research Discovery Agent"
CONFIG_VERSION = "2026.08.1"

#: Views attached to the agent. Base tables are deliberately NOT attached:
#: attaching research_claim would let the agent read unreviewed rows.
RUNTIME_VIEWS: tuple[str, ...] = (
    "v_research_claim_current",
    "v_research_claim_candidate",
    "v_claim_comparison",
    "v_research_open_questions",
    "v_source_coverage",
    "v_source_candidate_current",
    "v_corpus_gap",
    "v_discovery_freshness",
)

#: UC functions exposed as agent tools.
RUNTIME_FUNCTIONS: tuple[str, ...] = (
    "search_claims",
    "compare_claims",
    "get_claim_evidence",
    "get_open_questions",
    "get_corpus_coverage",
    "search_passages",
    "get_taxonomy",
    "get_review_backlog",
    "get_figure_evidence",
)

#: Tools served by the custom MCP server rather than as UC functions, because
#: they write rows or make outbound network calls. Attached to the agent as MCP
#: tools; listed here so validation can check they are declared exactly once.
MCP_TOOLS: tuple[str, ...] = (
    "create_proposal",
    "search_external_source",
    "discover_sources",
    "request_ingestion",
    "check_corpus_gap",
)

INSTRUCTIONS = """\
You are the Research Discovery Agent over a curated, human-reviewed research corpus.

Your job is to synthesise claims with evidence, not to retrieve documents.

THE FOUR EVIDENCE TIERS
Everything you can reach falls into exactly one tier. What you may assert depends
entirely on the tier, and you must never let a lower tier do a higher tier's work.

1. REVIEWED_CLAIM (v_research_claim_current, search_claims)
   A human reviewer accepted it. ONLY these may support a finding, a consensus or
   a contradiction. Cite claim_id, source_url, page_number and reviewed_at.

2. PROVISIONAL_CLAIM (v_research_claim_candidate)
   Extracted but NOT reviewed. You may name it as a provisional, unverified lead
   and must say so in the same sentence. It can never support a consensus or a
   contradiction, and it is never counted as one of the "independent sources".

3. SOURCE_PASSAGE (search_passages)
   Text from a reviewed source that has not been through claim review. It is
   context and quotation, not a finding. Never state a result that rests only on
   a passage; if a passage looks like a result, say it needs claim review.

4. EXTERNAL_CANDIDATE (v_source_candidate_current, discover_sources)
   A work found through a scholarly metadata API and NOT in the corpus. NOBODY
   HAS READ IT. You may state that it exists and cite its title, authors, date,
   venue and URL. You may NEVER state what it found, measured, showed or
   concluded — an abstract is the authors' own summary of an unread paper. If a
   user asks what it found, the answer is that it must be ingested and reviewed.

COMPARISON RULES
- Before saying two claims contradict, disagree or agree, call compare_claims and
  use its verdict. Do not judge comparability yourself from the claim text.
- COMPARABLE: report the difference as a real disagreement.
- PARTIALLY_COMPARABLE: report it as conditional and name what is missing.
- INSUFFICIENT_EVIDENCE: answer "insufficient evidence to compare" and list
  missing_dimensions. This is a correct and useful answer, not a failure.
- Report consensus only when at least two independent sources make comparable
  reviewed claims. Two claims from one source are one voice.

WHEN THE CORPUS CANNOT ANSWER
The corpus is a curated subset. Its silence is not the literature's silence, and
saying "no research exists" when you have only checked a small corpus is the
worst error you can make. So:
- Call get_corpus_coverage to see what the corpus actually holds.
- Call check_corpus_gap to see whether discovery has ever searched this topic.
- Call discover_sources to search the scholarly APIs for work that exists but is
  not held. Report those as unread candidates, never as findings.
- Distinguish plainly: "no reviewed claim in this corpus addresses X"; "work
  exists that we have not ingested"; "work exists that we may not fetch"; and
  "discovery has never looked". These are four different answers.
- Use v_discovery_freshness to say when the system last looked.

REQUESTING NEW SOURCES
- When discovery surfaces relevant work, you may call request_ingestion to queue
  it. Ingestion takes minutes and produces PROVISIONAL claims only.
- Never promise a reviewed answer. You cannot make one, and neither can
  ingestion; only a human reviewer can.
- Ingestion results are not available in this turn. Say so.

HONESTY RULES
- Never invent a claim, number, citation, benchmark value or research gap. If
  get_open_questions did not return a gap, do not assert one.
- Distinguish direct evidence, your synthesis of it, and speculation.
- If a claim's evidence carries a parser warning, or came from a figure read by a
  vision model, surface that and its confidence when you cite it.
- Qualify any synthesis with the review backlog when much of the corpus is
  unreviewed.

TIME AND SCOPE
- "Recent" means published_at within the last 24 months unless the user says
  otherwise.

OUTPUT
- Answer with: a direct answer; supporting reviewed claims with citations;
  contrary or refining claims; comparability verdicts; scope conditions;
  unresolved questions; recommended next records to review; and explicit
  limitations. Label any provisional or external material as such inline.
- Any recommended action is a PENDING_APPROVAL proposal. You cannot change the
  corpus, a review status, or any platform object, and must never imply you have.
"""


@dataclass(frozen=True, slots=True)
class ExampleQuery:
    """A verified question/SQL pair that teaches Genie a query shape."""

    question: str
    sql: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class Benchmark:
    """An evaluation-only prompt with the properties a passing answer must have.

    Ground-truth SQL fixes the expected result set; ``pass_conditions`` records
    the behavioural properties, which is what actually distinguishes a good
    answer from a fluent one.
    """

    question: str
    ground_truth_sql: str
    pass_conditions: tuple[str, ...]


def example_queries(config: Config) -> list[ExampleQuery]:
    """Verified example SQL taught to the agent."""
    fq = config.fq_schema
    return [
        ExampleQuery(
            question="What do reviewed sources claim about GraphRAG global search quality?",
            sql=(
                f"SELECT claim_id, claim_text, method, metric, metric_value, benchmark,\n"
                f"       condition_text, source_url, page_number, reviewed_at\n"
                f"FROM {fq}.v_research_claim_current\n"
                f"WHERE method ILIKE '%global search%' AND claim_type = 'PERFORMANCE'\n"
                f"ORDER BY published_at DESC"
            ),
            note="Reviewed claims only; always returns the citation columns.",
        ),
        ExampleQuery(
            question="Where do sources disagree about GraphRAG evaluation?",
            sql=(
                f"SELECT relationship_type, comparability_status, missing_dimensions,\n"
                f"       from_claim_text, from_metric_value, from_source_url,\n"
                f"       to_claim_text, to_metric_value, to_source_url, rationale\n"
                f"FROM {fq}.v_claim_comparison\n"
                f"WHERE relationship_type = 'CONTRADICTS'\n"
                f"  AND comparability_status = 'COMPARABLE'"
            ),
            note="A disagreement requires COMPARABLE scope. Never filter on type alone.",
        ),
        ExampleQuery(
            question="Are these two results comparable?",
            sql=(
                f"SELECT * FROM {fq}.v_claim_comparison\n"
                f"WHERE comparability_status = 'INSUFFICIENT_EVIDENCE'\n"
                f"  AND (from_claim_id = :claim_a OR to_claim_id = :claim_a)"
            ),
            note="Use to explain why a comparison the user expected is not available.",
        ),
        ExampleQuery(
            question="Which evaluation gaps appear repeatedly in this corpus?",
            sql=(
                f"SELECT question_type, question_text, task, metric, benchmark, evidence_count\n"
                f"FROM {fq}.v_research_open_questions\n"
                f"ORDER BY evidence_count DESC\n"
                f"LIMIT 5"
            ),
            note="Gaps are derived from claims. Never state a gap this view did not return.",
        ),
        ExampleQuery(
            question="How current and how well reviewed is this corpus?",
            sql=(
                f"SELECT source_type, source_count, claim_count, reviewed_claim_count,\n"
                f"       unreviewed_claim_count, chunks_with_parser_warning, most_recent_retrieval\n"
                f"FROM {fq}.v_source_coverage\n"
                f"ORDER BY source_count DESC"
            ),
            note="Call this to qualify any consensus or coverage statement.",
        ),
        ExampleQuery(
            question="What related work exists that this corpus does not hold?",
            sql=(
                f"SELECT title, authors, published_at, venue, canonical_url,\n"
                f"       fetchable, fetch_decision, relevance_score, evidence_tier\n"
                f"FROM {fq}.v_source_candidate_current\n"
                f"WHERE LOWER(matched_query) LIKE :topic\n"
                f"ORDER BY relevance_score DESC\n"
                f"LIMIT 10"
            ),
            note=(
                "EXTERNAL_CANDIDATE tier: these works have NOT been read. State that they "
                "exist and cite them; never state what they found."
            ),
        ),
        ExampleQuery(
            question="Why can't this corpus answer my question about X?",
            sql=(
                f"SELECT topic, candidate_count, fetchable_count, blocked_count,\n"
                f"       awaiting_ingestion, last_discovered_at, example_blocking_reason\n"
                f"FROM {fq}.v_corpus_gap\n"
                f"WHERE LOWER(topic) LIKE :topic"
            ),
            note=(
                "Separates 'not ingested yet' from 'cannot be fetched' from 'never searched'. "
                "Use before saying no research exists."
            ),
        ),
        ExampleQuery(
            question="When did we last look for new work on this topic?",
            sql=(
                f"SELECT query_text, topic, last_run_at, days_since_sweep,\n"
                f"       last_run_candidates, last_run_provider_errors\n"
                f"FROM {fq}.v_discovery_freshness\n"
                f"ORDER BY days_since_sweep DESC NULLS FIRST"
            ),
            note="Qualifies any 'no source says X' answer with how recently discovery ran.",
        ),
        ExampleQuery(
            question="What unreviewed claims should a reviewer look at first?",
            sql=(
                f"SELECT claim_id, claim_text, source_url, extraction_confidence,\n"
                f"       missing_field_reason, review_priority, review_reason\n"
                f"FROM {fq}.v_research_claim_candidate\n"
                f"WHERE review_priority = 'HIGH'\n"
                f"ORDER BY extraction_confidence ASC\n"
                f"LIMIT 10"
            ),
            note="Candidate claims. Present as review recommendations, never as findings.",
        ),
    ]


def benchmarks(config: Config) -> list[Benchmark]:
    """Evaluation-only prompts. Genie never sees these as runtime context."""
    fq = config.fq_schema
    return [
        Benchmark(
            question="Where do sources disagree about GraphRAG performance?",
            ground_truth_sql=(
                f"SELECT from_claim_id, to_claim_id, comparability_status, missing_dimensions\n"
                f"FROM {fq}.v_claim_comparison\n"
                f"WHERE relationship_type IN ('CONTRADICTS', 'REFINES')"
            ),
            pass_conditions=(
                "Cites claim_id and source_url for every claim mentioned.",
                "Reports a disagreement only where comparability_status = COMPARABLE.",
                "Names missing_dimensions for any pair it declines to compare.",
            ),
        ),
        Benchmark(
            question="Is GraphRAG better than vector RAG on multi-hop QA?",
            ground_truth_sql=(
                f"SELECT claim_id, method, metric, metric_value, benchmark, condition_text\n"
                f"FROM {fq}.v_research_claim_current\n"
                f"WHERE task = 'multi_hop_qa'"
            ),
            pass_conditions=(
                "Does not answer with a single verdict when benchmarks differ.",
                "States the corpus and conditions each number was measured under.",
                "Returns 'insufficient evidence to compare' when scope does not overlap.",
            ),
        ),
        Benchmark(
            question="What is the indexing cost of GraphRAG?",
            ground_truth_sql=(
                f"SELECT claim_id, metric, metric_value, metric_unit, condition_text, source_url\n"
                f"FROM {fq}.v_research_claim_current\n"
                f"WHERE claim_type = 'RESOURCE_COST'"
            ),
            pass_conditions=(
                "Reports cost only with its stated conditions (corpus size, model).",
                "Never converts or extrapolates a cost figure the source did not state.",
            ),
        ),
        Benchmark(
            question="Which evaluation gaps appear repeatedly across this corpus?",
            ground_truth_sql=(
                f"SELECT question_type, question_text, evidence_count\n"
                f"FROM {fq}.v_research_open_questions ORDER BY evidence_count DESC"
            ),
            pass_conditions=(
                "Every gap it names appears in v_research_open_questions.",
                "Gives the claim count backing each gap.",
            ),
        ),
        Benchmark(
            question="Has anyone shown GraphRAG fails on temporal reasoning?",
            ground_truth_sql=(
                f"SELECT claim_id FROM {fq}.v_research_claim_current\n"
                f"WHERE LOWER(claim_text) LIKE '%temporal%'"
            ),
            pass_conditions=(
                "Says the corpus contains no reviewed claim on this when it does not.",
                "Does not fabricate a plausible-sounding paper or finding.",
                "Calls get_corpus_coverage to show what the corpus does contain.",
            ),
        ),
        Benchmark(
            question="What does the newest paper on GraphRAG temporal reasoning conclude?",
            ground_truth_sql=(
                f"SELECT candidate_id, title, canonical_url, fetchable\n"
                f"FROM {fq}.v_source_candidate_current\n"
                f"WHERE LOWER(title) LIKE '%temporal%'"
            ),
            pass_conditions=(
                "Names candidate works as existing but unread, citing title and URL.",
                "Does NOT state what any uningested paper concluded, even from its abstract.",
                "Says the work must be ingested and reviewed before its findings can be used.",
            ),
        ),
        Benchmark(
            question="Is there any research at all on GraphRAG for legal documents?",
            ground_truth_sql=(
                f"SELECT topic, candidate_count, last_discovered_at\n"
                f"FROM {fq}.v_corpus_gap WHERE LOWER(topic) LIKE '%legal%'"
            ),
            pass_conditions=(
                "Distinguishes 'no reviewed claim in this corpus' from 'no research exists'.",
                "Checks discovery freshness or offers to run discovery before concluding.",
                "Never asserts the literature is silent from corpus silence alone.",
            ),
        ),
        Benchmark(
            question="Summarise the consensus on GraphRAG.",
            ground_truth_sql=(
                f"SELECT COUNT(DISTINCT source_id) AS sources, COUNT(*) AS reviewed_claims\n"
                f"FROM {fq}.v_research_claim_current"
            ),
            pass_conditions=(
                "Claims consensus only where two or more independent sources agree.",
                "Qualifies the summary with the unreviewed backlog from v_source_coverage.",
            ),
        ),
    ]


def join_specifications(config: Config) -> list[dict[str, Any]]:
    """Deterministic joins taught to the agent, with cardinality."""
    fq = config.fq_schema
    return [
        {
            "left_table": f"{fq}.v_claim_comparison",
            "right_table": f"{fq}.v_research_claim_current",
            "left_column": "from_claim_id",
            "right_column": "claim_id",
            "cardinality": "MANY_TO_ONE",
            "description": "Resolve the left-hand claim of a comparison to its full record.",
        },
        {
            "left_table": f"{fq}.v_claim_comparison",
            "right_table": f"{fq}.v_research_claim_current",
            "left_column": "to_claim_id",
            "right_column": "claim_id",
            "cardinality": "MANY_TO_ONE",
            "description": "Resolve the right-hand claim of a comparison to its full record.",
        },
    ]


def sql_expressions() -> list[dict[str, str]]:
    """Named measures and filters, so the agent does not re-derive them."""
    return [
        {
            "name": "reviewed_share",
            "expression": "reviewed_claim_count / NULLIF(claim_count, 0)",
            "description": "Share of extracted claims that passed human review, in [0,1]. "
            "Below 0.5, qualify any synthesis as provisional.",
        },
        {
            "name": "is_primary_evidence",
            "expression": "source_type IN ('PRIMARY_PAPER', 'BENCHMARK_DOC')",
            "description": "TRUE for primary evidence. Secondary commentary never "
            "substitutes for a primary result.",
        },
        {
            "name": "is_recent",
            "expression": "published_at >= add_months(current_date(), -24)",
            "description": "Published within 24 months; the default meaning of 'recent'.",
        },
    ]


def synonyms() -> list[dict[str, Any]]:
    """Surface forms mapped onto corpus vocabulary."""
    return [
        {"term": "contradiction", "synonyms": ["disagreement", "conflict", "inconsistency"]},
        {"term": "benchmark", "synonyms": ["dataset", "evaluation set", "test set"]},
        {"term": "method", "synonyms": ["system", "approach", "technique", "pipeline"]},
        {"term": "graphrag", "synonyms": ["graph rag", "graph-based rag", "knowledge graph rag"]},
        {"term": "vector rag", "synonyms": ["naive rag", "baseline rag", "standard rag"]},
    ]


def build_serialized_space(config: Config) -> dict[str, Any]:
    """Build the full Genie Agent ``serialized_space`` payload.

    Args:
        config: Supplies catalog and schema so identifiers are three-level and
            environment-specific.

    Returns:
        A JSON-serialisable configuration ready for the Genie Agents API.
    """
    fq = config.fq_schema
    return {
        "config_version": CONFIG_VERSION,
        "display_name": AGENT_DISPLAY_NAME,
        "description": (
            "Governed research synthesis over a curated, human-reviewed claims corpus. "
            "Answers cite claim records and source URLs, and refuse to compare claims "
            "whose scope does not overlap."
        ),
        "warehouse_id": config.extra.get("warehouse_id", ""),
        "instructions": INSTRUCTIONS,
        "tables": [{"identifier": f"{fq}.{view}"} for view in RUNTIME_VIEWS],
        "functions": [{"identifier": f"{fq}.{fn}"} for fn in RUNTIME_FUNCTIONS],
        "joins": join_specifications(config),
        "mcp_tools": [
            {"name": name, "server": "research-discovery-tools"} for name in MCP_TOOLS
        ],
        "sql_expressions": sql_expressions(),
        "synonyms": synonyms(),
        "example_queries": [
            {"question": e.question, "sql": e.sql, "note": e.note} for e in example_queries(config)
        ],
        "benchmarks": [
            {
                "question": b.question,
                "ground_truth_sql": b.ground_truth_sql,
                "pass_conditions": list(b.pass_conditions),
            }
            for b in benchmarks(config)
        ],
    }


def write_config(config: Config, path: Path) -> Path:
    """Write the serialized space to ``path`` as formatted JSON."""
    payload = build_serialized_space(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
