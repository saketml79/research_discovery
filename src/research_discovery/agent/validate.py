"""Pre-deployment validation of the Genie Agent configuration.

Run in CI and again immediately before deployment. It catches the failures that
are cheap to find here and expensive to find in a demo: two-level identifiers, a
base table attached by mistake, an example query that reads unreviewed claims, a
benchmark without pass conditions, an instruction block that lost its refusal
rules.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: Views that must never be attached because they expose unreviewed rows as if
#: they were findings. v_research_claim_candidate is allowed - it is explicitly
#: labelled as candidate data and the instructions govern its use.
FORBIDDEN_TABLES = (
    "research_claim",
    "research_claim_relationship",
    "research_chunk",
    "research_review_queue",
)

#: Instruction phrases that encode the agent's non-negotiable behaviour.
REQUIRED_INSTRUCTION_PHRASES = (
    "REVIEWED_CLAIM",
    "PROVISIONAL_CLAIM",
    "SOURCE_PASSAGE",
    "EXTERNAL_CANDIDATE",
    "compare_claims",
    "insufficient evidence to compare",
    "two independent sources",
    "PENDING_APPROVAL",
    "NOBODY HAS READ IT",
)

#: Tools that must NEVER be declared as Unity Catalog functions. Each either
#: writes rows or makes an outbound call, which a UC SQL function cannot do; a
#: SQL definition could therefore only return a string claiming it worked.
MUST_NOT_BE_UC_FUNCTIONS = (
    "create_proposal",
    "discover_sources",
    "request_ingestion",
    "search_external_source",
)

_THREE_LEVEL = re.compile(r"^[A-Za-z_][\w]*\.[A-Za-z_][\w]*\.[A-Za-z_][\w]*$")


class ConfigValidationError(ValueError):
    """Raised by ``assert_valid`` when the configuration is not deployable."""


def validate_space(space: Mapping[str, Any]) -> list[str]:
    """Return the problems in a ``serialized_space``. Empty means deployable."""
    problems: list[str] = []

    for required in ("display_name", "instructions", "tables", "functions", "benchmarks"):
        if not space.get(required):
            problems.append(f"MISSING_OR_EMPTY:{required}")
    if problems:
        return problems

    for entry in space["tables"]:
        identifier = entry.get("identifier", "")
        if not _THREE_LEVEL.match(identifier):
            problems.append(f"NOT_THREE_LEVEL_IDENTIFIER:{identifier}")
        leaf = identifier.rsplit(".", 1)[-1]
        if leaf in FORBIDDEN_TABLES:
            problems.append(f"BASE_TABLE_ATTACHED:{identifier}")

    for entry in space["functions"]:
        identifier = entry.get("identifier", "")
        if not _THREE_LEVEL.match(identifier):
            problems.append(f"NOT_THREE_LEVEL_IDENTIFIER:{identifier}")
        leaf = identifier.rsplit(".", 1)[-1]
        if leaf in MUST_NOT_BE_UC_FUNCTIONS:
            problems.append(f"WRITE_TOOL_DECLARED_AS_UC_FUNCTION:{leaf}")

    declared_mcp = {t["name"] for t in space.get("mcp_tools", [])}
    for required in ("create_proposal", "discover_sources", "request_ingestion"):
        if required not in declared_mcp:
            problems.append(f"MCP_TOOL_NOT_DECLARED:{required}")

    # Whitespace is normalized on both sides so a required phrase still matches
    # when the instruction text wraps it across two lines.
    instructions = " ".join(space["instructions"].lower().split())
    for phrase in REQUIRED_INSTRUCTION_PHRASES:
        if " ".join(phrase.lower().split()) not in instructions:
            problems.append(f"INSTRUCTIONS_MISSING_RULE:{phrase}")

    examples = space.get("example_queries") or []
    if len(examples) < 5:
        problems.append(f"TOO_FEW_EXAMPLE_QUERIES:{len(examples)}")
    for example in examples:
        sql = example.get("sql", "")
        if not sql.strip():
            problems.append(f"EMPTY_EXAMPLE_SQL:{example.get('question')}")
        # An example that reads a base claim table teaches the agent to bypass
        # the review boundary, whatever the instructions say.
        for forbidden in FORBIDDEN_TABLES:
            if re.search(rf"\b{forbidden}\b(?!_)", sql):
                problems.append(f"EXAMPLE_READS_BASE_TABLE:{forbidden}")

    bench = space.get("benchmarks") or []
    if len(bench) < 5:
        problems.append(f"TOO_FEW_BENCHMARKS:{len(bench)}")
    for item in bench:
        if not item.get("pass_conditions"):
            problems.append(f"BENCHMARK_WITHOUT_PASS_CONDITIONS:{item.get('question')}")
        if not item.get("ground_truth_sql", "").strip():
            problems.append(f"BENCHMARK_WITHOUT_GROUND_TRUTH:{item.get('question')}")

    for join in space.get("joins") or []:
        if join.get("cardinality") not in {"ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY"}:
            problems.append(f"JOIN_WITHOUT_VALID_CARDINALITY:{join.get('left_column')}")

    return problems


def assert_valid(space: Mapping[str, Any]) -> None:
    """Raise ``ConfigValidationError`` when ``space`` is not deployable."""
    problems = validate_space(space)
    if problems:
        raise ConfigValidationError(
            "Genie configuration is not deployable:\n  - " + "\n  - ".join(problems)
        )
