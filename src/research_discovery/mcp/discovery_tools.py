"""MCP tools that let the agent reach outside the corpus during a question.

These are the tools that answer "the corpus does not cover this — now what?"
They are MCP tools rather than UC functions because they make outbound network
calls and write rows, neither of which a Genie SQL function can do.

The design rule they encode: **discovery is not evidence.** ``discover_sources``
returns the existence of work, never its findings, and it says so in every
response so the agent cannot quietly promote a title into a result. Getting from
"a paper exists" to "a paper found X" requires ingestion, extraction and human
review, and there is no tool here that shortcuts any of those.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping, Sequence

from ..config import Config
from ..discovery.service import (
    DiscoveryDecision,
    DiscoveryService,
    EvidenceTier,
    IngestionSpeed,
)
from ..ids import stable_id
from ..models import utcnow
from .server import SqlExecutor, ToolResult

logger = logging.getLogger(__name__)

#: How many candidates a live, in-question search returns. Small on purpose: a
#: user asking a research question wants three leads, not a literature dump.
LIVE_SEARCH_LIMIT = 8


class DiscoveryTools:
    """Live discovery and ingestion-request tools."""

    def __init__(
        self,
        config: Config,
        service: DiscoveryService,
        executor: SqlExecutor,
        *,
        principal: str = "research_discovery_agent",
        job_trigger: Callable[[Sequence[str]], str] | None = None,
    ) -> None:
        """Args:
        config: Supplies target table names.
        service: The discovery service wrapping the metadata-API providers.
        executor: SQL executor for candidate persistence.
        principal: Recorded as the requester on discovery runs.
        job_trigger: Starts the provisional-ingestion job for a set of candidate
            ids and returns a run id. When ``None``, ``request_ingestion`` still
            records an approved candidate but tells the agent that ingestion
            runs on the next scheduled pass rather than immediately.
        """
        self._config = config
        self._service = service
        self._executor = executor
        self._principal = principal
        self._job_trigger = job_trigger

    # -- tool: discover_sources --------------------------------------------

    def discover_sources(
        self, query: str, max_results: int = LIVE_SEARCH_LIMIT, include_paywalled: bool = True
    ) -> ToolResult:
        """Search scholarly metadata APIs for work not in the corpus.

        Runs synchronously inside a user's turn and touches no document: it
        queries OpenAlex, arXiv and Semantic Scholar for metadata, de-duplicates
        against the corpus and against itself, and ranks what is left.

        Returns:
            A ``ToolResult`` whose every candidate is tagged
            ``EXTERNAL_CANDIDATE``, with an explicit instruction that these are
            unread works.
        """
        if not query.strip():
            return ToolResult(False, {"error": "EMPTY_QUERY"}, "A search query is required.")

        known = self._known_urls()
        try:
            result = self._service.discover(
                query, known_urls=known, speed=IngestionSpeed.METADATA_ONLY
            )
        except Exception as exc:  # noqa: BLE001 - reported to the agent, not raised
            logger.exception("discovery failed")
            return ToolResult(False, {"error": "DISCOVERY_FAILED"}, f"Discovery failed: {exc}")

        decisions = result.decisions[:max_results]
        if not include_paywalled:
            decisions = [d for d in decisions if d.fetchable]

        self._persist_candidates(decisions, mode="LIVE_QUESTION")
        self._record_run(result, mode="LIVE_QUESTION")

        return ToolResult(
            True,
            {
                "evidence_tier": EvidenceTier.EXTERNAL_CANDIDATE.value,
                "query": query,
                "candidates": [_candidate_payload(d) for d in decisions],
                "already_in_corpus": len(result.already_known),
                "provider_errors": result.provider_errors,
                "fetchable_count": sum(1 for d in decisions if d.fetchable),
            },
            "These are search-API results for works NOT in the corpus. Nobody has read them: "
            "you may state that each work EXISTS and cite its title, authors, date and URL, "
            "and you must NOT state what any of them found, measured or concluded — an "
            "abstract is the authors' summary, not a reviewed claim. To get findings from "
            "one, call request_ingestion and tell the user it must be ingested and reviewed "
            + (
                "first. Some providers errored, so this list is incomplete."
                if result.provider_errors
                else "first."
            ),
        )

    # -- tool: request_ingestion -------------------------------------------

    def request_ingestion(
        self,
        candidate_ids: Sequence[str],
        rationale: str,
        speed: str = IngestionSpeed.PROVISIONAL.value,
    ) -> ToolResult:
        """Approve discovered candidates for ingestion.

        Marks each fetchable candidate ``APPROVED`` and, when a job trigger is
        configured, starts the provisional ingestion run. Ingestion produces
        ``CANDIDATE`` claims only: it never produces reviewed knowledge, and the
        response says so explicitly so the agent cannot promise otherwise.
        """
        if not candidate_ids:
            return ToolResult(False, {"error": "NO_CANDIDATES"}, "No candidate ids were given.")
        if not rationale.strip():
            return ToolResult(
                False, {"error": "NO_RATIONALE"}, "An ingestion request must state why."
            )
        try:
            requested_speed = IngestionSpeed(speed)
        except ValueError:
            return ToolResult(
                False,
                {"error": "INVALID_SPEED"},
                f"speed must be METADATA_ONLY or PROVISIONAL, got {speed!r}. "
                "REVIEWED is not requestable: only a human reviewer produces reviewed knowledge.",
            )
        if requested_speed is IngestionSpeed.REVIEWED:
            return ToolResult(
                False,
                {"error": "REVIEW_NOT_AUTOMATABLE"},
                "Ingestion cannot produce REVIEWED knowledge. Request PROVISIONAL ingestion; a "
                "human reviewer decides what becomes established.",
            )

        rows = self._candidates_by_id(candidate_ids)
        found = {r["candidate_id"]: r for r in rows}
        missing = [c for c in candidate_ids if c not in found]
        blocked = [
            {"candidate_id": r["candidate_id"], "reason": r["fetch_decision"]}
            for r in rows
            if not _truthy(r.get("fetchable"))
        ]
        approved = [r["candidate_id"] for r in rows if _truthy(r.get("fetchable"))]

        if approved:
            self._approve(approved, requested_speed)

        run_id = ""
        if approved and self._job_trigger is not None:
            try:
                run_id = self._job_trigger(approved)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not trigger ingestion job: %s", exc)

        message = (
            f"{len(approved)} candidate(s) approved for ingestion. "
            + (
                f"Ingestion run {run_id} started; results will be CANDIDATE claims, "
                if run_id
                else "Ingestion will run on the next scheduled pass; results will be CANDIDATE claims, "
            )
            + "which are provisional and unreviewed. Tell the user that no finding from these "
            "sources can be stated as established until a reviewer accepts it."
        )
        if blocked:
            message += (
                f" {len(blocked)} candidate(s) could not be approved because their content "
                "cannot be fetched under this deployment's access and licence rules."
            )
        if missing:
            message += f" {len(missing)} candidate id(s) were not found; call discover_sources first."

        return ToolResult(
            bool(approved),
            {
                "approved": approved,
                "blocked": blocked,
                "unknown": missing,
                "ingestion_run_id": run_id,
                "resulting_tier": EvidenceTier.PROVISIONAL_CLAIM.value,
                "rationale": rationale,
            },
            message,
        )

    # -- tool: check_corpus_gap --------------------------------------------

    def check_corpus_gap(self, topic: str) -> ToolResult:
        """Explain why the corpus cannot answer something.

        Distinguishes the three cases a user actually cares about: nobody has
        published on it; work exists but has not been ingested; work exists but
        this deployment may not fetch it.
        """
        rows = self._executor.execute(
            f"""
            SELECT topic, candidate_count, fetchable_count, blocked_count,
                   awaiting_ingestion, last_discovered_at, example_blocking_reason
            FROM {self._config.table('v_corpus_gap')}
            WHERE LOWER(topic) LIKE :needle
            """,
            {"needle": f"%{topic.lower()}%"},
        )
        if not rows:
            return ToolResult(
                True,
                {"topic": topic, "candidates": 0, "searched": False},
                "Discovery has never searched for this topic, so the corpus holding nothing "
                "about it is not evidence that no research exists. Offer to run "
                "discover_sources before concluding anything.",
            )
        row = dict(rows[0])
        return ToolResult(
            True,
            {"topic": topic, **row},
            "Work on this topic exists outside the corpus. Distinguish clearly for the user "
            "between what has not been ingested yet and what cannot be fetched at all, and do "
            "not describe either as a research gap — an unread paper is not an absent finding.",
        )

    # -- persistence -------------------------------------------------------

    def _known_urls(self) -> list[str]:
        rows = self._executor.execute(
            f"SELECT canonical_url FROM {self._config.table('research_source')}", {}
        )
        return [str(r["canonical_url"]) for r in rows]

    def _candidates_by_id(self, candidate_ids: Sequence[str]) -> list[Mapping[str, Any]]:
        placeholders = ", ".join(f":id{i}" for i in range(len(candidate_ids)))
        return list(
            self._executor.execute(
                f"""
                SELECT candidate_id, canonical_url, fetchable, fetch_decision, status
                FROM {self._config.table('research_source_candidate')}
                WHERE candidate_id IN ({placeholders})
                """,
                {f"id{i}": cid for i, cid in enumerate(candidate_ids)},
            )
        )

    def _persist_candidates(self, decisions: Sequence[DiscoveryDecision], *, mode: str) -> None:
        """MERGE candidates so a repeated search does not duplicate rows."""
        for decision in decisions:
            hit = decision.candidate
            self._executor.execute(
                f"""
                MERGE INTO {self._config.table('research_source_candidate')} AS t
                USING (SELECT :candidate_id AS candidate_id) AS s
                  ON t.candidate_id = s.candidate_id
                WHEN NOT MATCHED THEN INSERT (
                  candidate_id, canonical_url, title, provider, external_id, doi, source_type,
                  authors, venue, published_at, abstract, citation_count, is_open_access,
                  pdf_url, license, fetchable, fetch_decision, relevance_score, matched_query,
                  discovery_mode, ingestion_speed, status, discovered_at
                ) VALUES (
                  :candidate_id, :canonical_url, :title, :provider, :external_id, :doi,
                  :source_type, :authors, :venue, :published_at, :abstract, :citation_count,
                  :is_open_access, :pdf_url, :license, :fetchable, :fetch_decision,
                  :relevance_score, :matched_query, :discovery_mode, 'METADATA_ONLY',
                  'DISCOVERED', current_timestamp()
                )
                """,
                {
                    "candidate_id": hit.candidate_id,
                    "canonical_url": hit.canonical_url,
                    "title": hit.title,
                    "provider": hit.provider,
                    "external_id": hit.external_id,
                    "doi": hit.doi,
                    "source_type": hit.source_type.value,
                    "authors": hit.authors,
                    "venue": hit.venue,
                    "published_at": hit.published_at.isoformat() if hit.published_at else None,
                    "abstract": (hit.abstract or "")[:4000] or None,
                    "citation_count": hit.citation_count,
                    "is_open_access": hit.is_open_access,
                    "pdf_url": hit.pdf_url,
                    "license": hit.license,
                    "fetchable": decision.fetchable,
                    "fetch_decision": decision.reason,
                    "relevance_score": hit.relevance_score,
                    "matched_query": hit.matched_query,
                    "discovery_mode": mode,
                },
            )

    def _approve(self, candidate_ids: Sequence[str], speed: IngestionSpeed) -> None:
        placeholders = ", ".join(f":id{i}" for i in range(len(candidate_ids)))
        self._executor.execute(
            f"""
            UPDATE {self._config.table('research_source_candidate')}
            SET status = 'APPROVED', ingestion_speed = :speed,
                decided_by = :principal, decided_at = current_timestamp()
            WHERE candidate_id IN ({placeholders}) AND fetchable
            """,
            {
                "speed": speed.value,
                "principal": self._principal,
                **{f"id{i}": cid for i, cid in enumerate(candidate_ids)},
            },
        )

    def _record_run(self, result: Any, *, mode: str, query_id: str | None = None) -> None:
        self._executor.execute(
            f"""
            INSERT INTO {self._config.table('research_discovery_run')}
              (discovery_run_id, query_text, query_id, discovery_mode, providers_searched,
               provider_errors, candidates_found, candidates_fetchable, already_known,
               requested_by, started_at, finished_at)
            VALUES
              (:run_id, :query_text, :query_id, :mode, :providers, :errors, :found,
               :fetchable, :known, :principal, :started, current_timestamp())
            """,
            {
                "run_id": stable_id("drun", result.query, utcnow().isoformat()),
                "query_text": result.query,
                "query_id": query_id,
                "mode": mode,
                "providers": ",".join(sorted({d.candidate.provider for d in result.decisions}))
                or "none",
                "errors": json.dumps(result.provider_errors) if result.provider_errors else None,
                "found": len(result.decisions),
                "fetchable": len(result.fetchable),
                "known": len(result.already_known),
                "principal": self._principal,
                "started": result.searched_at.isoformat(),
            },
        )


def _candidate_payload(decision: DiscoveryDecision) -> dict[str, Any]:
    """Shape a candidate for the agent, abstract included but clearly labelled."""
    hit = decision.candidate
    return {
        "candidate_id": hit.candidate_id,
        "title": hit.title,
        "authors": hit.authors,
        "published_at": hit.published_at.date().isoformat() if hit.published_at else None,
        "venue": hit.venue,
        "url": hit.canonical_url,
        "doi": hit.doi,
        "citation_count": hit.citation_count,
        "relevance_score": hit.relevance_score,
        "open_access": hit.is_open_access,
        "fetchable": decision.fetchable,
        "fetch_decision": decision.reason,
        "abstract_unreviewed": (hit.abstract or "")[:1000] or None,
        "evidence_tier": EvidenceTier.EXTERNAL_CANDIDATE.value,
        "caution": "Author-written abstract of an unread work. Not a reviewed claim.",
    }


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "t", "yes"}


#: MCP schemas for the discovery tools.
DISCOVERY_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "discover_sources",
        "description": (
            "Search scholarly metadata APIs (OpenAlex, arXiv, Semantic Scholar) for published "
            "work that is NOT in the corpus. Returns titles, authors, dates, venues, DOIs and "
            "author-written abstracts — metadata only. Nothing is downloaded or read, so you "
            "may state that a work EXISTS and cite it, but you must NEVER state what it found, "
            "measured or concluded. Call this when the corpus cannot answer a question, so you "
            "can distinguish 'no research exists' from 'we have not ingested it'. Results are "
            "EXTERNAL_CANDIDATE tier and can never support a consensus or contradiction."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 25},
                "include_paywalled": {"type": "boolean"},
            },
        },
    },
    {
        "name": "request_ingestion",
        "description": (
            "Approve discovered candidates for ingestion so their content can be parsed and "
            "claims extracted. Produces CANDIDATE (provisional, unreviewed) claims only — it "
            "cannot produce reviewed knowledge, and requesting REVIEWED is refused. Ingestion "
            "takes minutes, so the results are not available in this turn. Tell the user the "
            "sources are queued and that nothing from them can be stated as established until "
            "a human reviewer accepts it. Candidates whose content cannot be fetched under the "
            "deployment's access and licence rules are reported as blocked, with the reason."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["candidate_ids", "rationale"],
            "properties": {
                "candidate_ids": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
                "speed": {"type": "string", "enum": ["METADATA_ONLY", "PROVISIONAL"]},
            },
        },
    },
    {
        "name": "check_corpus_gap",
        "description": (
            "Explain why the corpus cannot answer something. Distinguishes three cases the user "
            "cares about: no discovery has ever searched this topic; work exists but has not "
            "been ingested; work exists but cannot be fetched under this deployment's rules. "
            "Call it before telling a user that no research addresses their question — the "
            "corpus being silent is not the literature being silent."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["topic"],
            "properties": {"topic": {"type": "string"}},
        },
    },
]


def dispatch_discovery(
    tools: DiscoveryTools, name: str, arguments: Mapping[str, Any]
) -> ToolResult | None:
    """Route a discovery tool call, or return ``None`` when the name is not ours."""
    if name == "discover_sources":
        return tools.discover_sources(
            query=str(arguments.get("query", "")),
            max_results=int(arguments.get("max_results") or LIVE_SEARCH_LIMIT),
            include_paywalled=bool(arguments.get("include_paywalled", True)),
        )
    if name == "request_ingestion":
        return tools.request_ingestion(
            candidate_ids=tuple(arguments.get("candidate_ids") or ()),
            rationale=str(arguments.get("rationale", "")),
            speed=str(arguments.get("speed") or IngestionSpeed.PROVISIONAL.value),
        )
    if name == "check_corpus_gap":
        return tools.check_corpus_gap(topic=str(arguments.get("topic", "")))
    return None
