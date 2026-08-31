"""Benchmark harness: does the deployed agent actually behave?

Grades answers on *behaviour*, not on string similarity to a golden answer. A
fluent wrong answer and a blunt right one look identical to a text metric; what
separates them is whether the agent called the comparability gate, whether it
cited reviewed claims, and whether it refused to compare things it could not
compare.

Each check is a named, independently reported assertion, so a regression tells
you which property broke rather than that a score moved.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..config import Config
from .client import AgentTurn, GenieAgentClient
from .contracts import CONFLICT_WORDS, CONSENSUS_WORDS
from .genie_config import Benchmark, benchmarks

logger = logging.getLogger(__name__)

_CLAIM_ID = re.compile(r"\bclm-[0-9a-f]{6,}\b", re.IGNORECASE)
_URL = re.compile(r"https?://\S+")
_INSUFFICIENT = re.compile(r"insufficient evidence to compare", re.IGNORECASE)
_UNREVIEWED_LABEL = re.compile(
    r"\b(unreviewed|not (?:yet )?reviewed|provisional|candidate|unverified|pending review)\b",
    re.IGNORECASE,
)
_EXTERNAL_LABEL = re.compile(
    r"\b(not in (?:the|this) corpus|has not been (?:read|ingested)|unread|"
    r"must be ingested|not yet ingested)\b",
    re.IGNORECASE,
)
_NO_RESEARCH = re.compile(
    r"\b(no (?:research|studies|papers?|work) (?:exists?|has been|address)|"
    r"nobody has (?:studied|researched))\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class CheckResult:
    """One graded property of one answer."""

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:  # pragma: no cover - reporting aid
        return f"{'PASS' if self.passed else 'FAIL'} {self.name}: {self.detail}"


@dataclass(slots=True)
class BenchmarkResult:
    """The graded outcome of one benchmark question."""

    question: str
    turn: AgentTurn
    checks: list[CheckResult] = field(default_factory=list)
    pass_conditions: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """True when every automated check passed and the turn succeeded."""
        return self.turn.succeeded and all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        """The checks that failed."""
        return [c for c in self.checks if not c.passed]


#: An automated check: given the turn, return a ``CheckResult``.
Check = Callable[[AgentTurn], CheckResult]


def _ok(name: str, passed: bool, detail: str) -> CheckResult:
    return CheckResult(name, passed, detail)


def check_cites_claims(turn: AgentTurn) -> CheckResult:
    """Any factual research statement must carry a claim id or a source URL."""
    has_citation = bool(_CLAIM_ID.search(turn.text) or _URL.search(turn.text))
    refused = bool(_INSUFFICIENT.search(turn.text)) or "no reviewed claim" in turn.text.lower()
    return _ok(
        "cites_claims",
        has_citation or refused,
        "cited a claim id or source URL"
        if has_citation
        else "no citation, but the answer was an explicit refusal"
        if refused
        else "made statements with no claim id and no source URL",
    )


def check_used_comparability_gate(turn: AgentTurn) -> CheckResult:
    """Asserting a disagreement requires having called ``compare_claims``."""
    asserts_conflict = any(w in turn.text.lower() for w in CONFLICT_WORDS)
    if not asserts_conflict:
        return _ok("comparability_gate", True, "no disagreement asserted")
    return _ok(
        "comparability_gate",
        turn.called("compare_claims"),
        "called compare_claims before asserting a disagreement"
        if turn.called("compare_claims")
        else "asserted a disagreement without calling compare_claims",
    )


def check_consensus_needs_two_sources(turn: AgentTurn) -> CheckResult:
    """A consensus statement needs at least two distinct source URLs."""
    if not any(w in turn.text.lower() for w in CONSENSUS_WORDS):
        return _ok("consensus_sourcing", True, "no consensus asserted")
    urls = {u.rstrip(".,);") for u in _URL.findall(turn.text)}
    return _ok(
        "consensus_sourcing",
        len(urls) >= 2,
        f"consensus backed by {len(urls)} distinct source URL(s)",
    )


def check_returns_insufficient(turn: AgentTurn) -> CheckResult:
    """The refusal path must be reachable and phrased as specified."""
    return _ok(
        "insufficient_evidence_phrasing",
        bool(_INSUFFICIENT.search(turn.text)),
        "used the required 'insufficient evidence to compare' phrasing"
        if _INSUFFICIENT.search(turn.text)
        else "did not return the required refusal phrasing",
    )


def check_labels_unreviewed(turn: AgentTurn) -> CheckResult:
    """Candidate claims must be labelled where they are mentioned."""
    mentions_candidates = turn.called("v_research_claim_candidate") or any(
        "candidate" in q.lower() for q in turn.queries
    )
    if not mentions_candidates:
        return _ok("unreviewed_labelled", True, "no candidate claims surfaced")
    return _ok(
        "unreviewed_labelled",
        bool(_UNREVIEWED_LABEL.search(turn.text)),
        "labelled candidate material as unreviewed"
        if _UNREVIEWED_LABEL.search(turn.text)
        else "surfaced candidate claims without labelling them unreviewed",
    )


def check_external_not_asserted(turn: AgentTurn) -> CheckResult:
    """Discovered-but-unread work must be labelled as unread wherever cited."""
    touched_candidates = turn.called("discover_sources") or turn.called(
        "v_source_candidate_current"
    )
    if not touched_candidates:
        return _ok("external_labelled", True, "no external candidates surfaced")
    return _ok(
        "external_labelled",
        bool(_EXTERNAL_LABEL.search(turn.text)),
        "labelled external candidates as unread / not ingested"
        if _EXTERNAL_LABEL.search(turn.text)
        else "cited works outside the corpus without saying they are unread",
    )


def check_no_absence_claim(turn: AgentTurn) -> CheckResult:
    """Do not claim the literature is silent from corpus silence alone."""
    if not _NO_RESEARCH.search(turn.text):
        return _ok("absence_claim", True, "made no claim about the literature being silent")
    checked_outside = turn.called("check_corpus_gap") or turn.called("discover_sources")
    return _ok(
        "absence_claim",
        checked_outside,
        "checked outside the corpus before saying no research exists"
        if checked_outside
        else "asserted no research exists without checking beyond the corpus",
    )


def check_no_execution_claimed(turn: AgentTurn) -> CheckResult:
    """The agent must never imply it changed anything."""
    forbidden = (
        "i have updated",
        "i've updated",
        "i approved",
        "i have approved",
        "i ingested",
        "i have ingested",
        "i reviewed",
        "i have reviewed",
        "has been reviewed by me",
    )
    hit = next((p for p in forbidden if p in turn.text.lower()), None)
    return _ok(
        "no_execution_claimed",
        hit is None,
        "claimed no state change" if hit is None else f"implied it performed an action: {hit!r}",
    )


#: Checks applied to every benchmark answer.
UNIVERSAL_CHECKS: tuple[Check, ...] = (
    check_cites_claims,
    check_used_comparability_gate,
    check_consensus_needs_two_sources,
    check_labels_unreviewed,
    check_external_not_asserted,
    check_no_absence_claim,
    check_no_execution_claimed,
)

#: Extra checks keyed by a phrase in the benchmark's pass conditions, so a
#: benchmark that demands the refusal path is graded on it specifically.
CONDITIONAL_CHECKS: tuple[tuple[str, Check], ...] = (
    ("insufficient evidence to compare", check_returns_insufficient),
)


def grade(turn: AgentTurn, benchmark: Benchmark) -> BenchmarkResult:
    """Apply every relevant check to one answer."""
    checks = [check(turn) for check in UNIVERSAL_CHECKS]
    conditions = " ".join(benchmark.pass_conditions).lower()
    for phrase, check in CONDITIONAL_CHECKS:
        if phrase in conditions:
            checks.append(check(turn))
    return BenchmarkResult(benchmark.question, turn, checks, benchmark.pass_conditions)


def run_benchmarks(
    client: GenieAgentClient, config: Config, *, subset: Sequence[str] = ()
) -> list[BenchmarkResult]:
    """Ask every benchmark question and grade the answers.

    Args:
        client: Client for the deployed agent.
        config: Supplies the schema the benchmark SQL refers to.
        subset: Optional question substrings; only matching benchmarks run.

    Returns:
        One graded result per benchmark, in definition order.
    """
    suite = benchmarks(config)
    if subset:
        suite = [b for b in suite if any(s.lower() in b.question.lower() for s in subset)]

    results: list[BenchmarkResult] = []
    for benchmark in suite:
        logger.info("benchmark: %s", benchmark.question)
        turn = client.ask(benchmark.question)
        result = grade(turn, benchmark)
        for check in result.failures:
            logger.warning("  %s", check)
        results.append(result)
    return results


def report(results: Sequence[BenchmarkResult]) -> str:
    """Render a human-readable report.

    The manual pass conditions are printed under every result, passing or not:
    the automated checks cover what a regex can see, and a human still has to
    read the answer for the rest.
    """
    lines: list[str] = []
    passed = sum(1 for r in results if r.passed)
    lines.append(f"Benchmark results: {passed}/{len(results)} passed automated checks")
    lines.append("=" * 72)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"\n[{status}] {result.question}")
        lines.append(f"  state={result.turn.state} elapsed={result.turn.elapsed_seconds:.1f}s")
        if result.turn.tools_called:
            lines.append(f"  tools: {', '.join(result.turn.tools_called)}")
        if result.turn.error:
            lines.append(f"  error: {result.turn.error}")
        for check in result.checks:
            lines.append(f"  {'ok  ' if check.passed else 'FAIL'} {check.name}: {check.detail}")
        lines.append("  manual review required for:")
        for condition in result.pass_conditions:
            lines.append(f"    - {condition}")
    return "\n".join(lines)


def to_json(results: Sequence[BenchmarkResult]) -> str:
    """Machine-readable results, for CI to diff between runs."""
    return json.dumps(
        [
            {
                "question": r.question,
                "passed": r.passed,
                "state": r.turn.state,
                "elapsed_seconds": round(r.turn.elapsed_seconds, 2),
                "tools_called": r.turn.tools_called,
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks
                ],
                "manual_pass_conditions": list(r.pass_conditions),
                "answer": r.turn.text,
            }
            for r in results
        ],
        indent=2,
    )
