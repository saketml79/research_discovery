"""Discovery, MCP tools and supersession tests."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from research_discovery.config import Config
from research_discovery.discovery.providers import (
    ArxivProvider,
    DiscoveredSource,
    DiscoveryError,
    OpenAlexProvider,
    RssProvider,
)
from research_discovery.discovery.service import (
    DiscoveryService,
    EvidenceTier,
    IngestionSpeed,
    StandingQuery,
    sweep,
)
from research_discovery.mcp.discovery_tools import DiscoveryTools, dispatch_discovery
from research_discovery.mcp.server import ToolRegistry, dispatch
from research_discovery.models import ClaimType, ReviewStatus, SourceType
from research_discovery.review.queue import ReviewError
from research_discovery.review.supersession import (
    apply_supersession,
    build_supersession_queue,
    plan_supersession,
)

CONFIG = Config(catalog="main", schema="rd")


def utc(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2404.16130v1</id>
    <title>From Local to Global: A Graph RAG Approach</title>
    <summary>We present a graph-based approach to query-focused summarization.</summary>
    <published>2024-04-24T00:00:00Z</published>
    <author><name>Darren Edge</name></author>
    <author><name>Ha Trinh</name></author>
    <link title="pdf" href="https://arxiv.org/pdf/2404.16130" type="application/pdf"/>
  </entry>
</feed>"""

OPENALEX_JSON = json.dumps(
    {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "title": "Evaluating GraphRAG on multi-hop QA",
                "doi": "https://doi.org/10.1234/abcd",
                "publication_date": "2025-02-01",
                "cited_by_count": 42,
                "open_access": {"is_oa": True},
                "best_oa_location": {
                    "pdf_url": "https://arxiv.org/pdf/2502.00001",
                    "license": "cc-by",
                },
                "primary_location": {"source": {"display_name": "ACL"}},
                "authorships": [{"author": {"display_name": "A. Researcher"}}],
                "abstract_inverted_index": {"GraphRAG": [0], "evaluation": [1], "study": [2]},
            }
        ]
    }
)

RSS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Practical notes on GraphRAG evaluation</title>
    <description>What we learned running graphrag in production.</description>
    <link>https://blog.example.com/graphrag</link>
    <pubDate>Mon, 03 Feb 2025 10:00:00 +0000</pubDate>
  </item>
</channel></rss>"""


class StubTransport:
    def __init__(self, body: str, fail: bool = False):
        self.body, self.fail, self.calls = body, fail, []

    def get_text(self, url: str, *, headers=None) -> str:
        self.calls.append(url)
        if self.fail:
            raise ConnectionError("network down")
        return self.body


class TestProviders(unittest.TestCase):
    def test_arxiv_parses_entries(self):
        provider = ArxivProvider(StubTransport(ARXIV_XML), sleep=lambda _: None)
        [hit] = provider.search("graphrag", limit=5, since=None)
        self.assertEqual(hit.provider, "arxiv")
        self.assertEqual(hit.pdf_url, "https://arxiv.org/pdf/2404.16130")
        self.assertTrue(hit.is_open_access)
        self.assertIn("Darren Edge", hit.authors)
        self.assertEqual(hit.published_at.year, 2024)

    def test_arxiv_rate_limit_is_enforced(self):
        slept: list[float] = []
        clock = iter([0.0, 0.0, 0.1, 0.1, 0.2, 0.2])
        provider = ArxivProvider(
            StubTransport(ARXIV_XML), clock=lambda: next(clock), sleep=slept.append
        )
        provider.search("a", limit=1, since=None)
        provider.search("b", limit=1, since=None)
        self.assertTrue(slept and slept[0] > 2.5, "arXiv terms require ~3s between requests")

    def test_arxiv_respects_since(self):
        provider = ArxivProvider(StubTransport(ARXIV_XML), sleep=lambda _: None)
        self.assertEqual(provider.search("graphrag", limit=5, since=utc(2026)), [])

    def test_openalex_rebuilds_inverted_abstract(self):
        provider = OpenAlexProvider(StubTransport(OPENALEX_JSON), sleep=lambda _: None)
        [hit] = provider.search("graphrag", limit=5, since=None)
        self.assertEqual(hit.abstract, "GraphRAG evaluation study")
        self.assertEqual(hit.doi, "10.1234/abcd")
        self.assertEqual(hit.citation_count, 42)

    def test_openalex_identifies_itself(self):
        transport = StubTransport(OPENALEX_JSON)
        OpenAlexProvider(transport, contact_email="me@example.org", sleep=lambda _: None).search(
            "x", limit=1, since=None
        )
        self.assertIn("mailto=me%40example.org", transport.calls[0])

    def test_provider_failure_raises_discovery_error(self):
        provider = OpenAlexProvider(StubTransport("", fail=True), sleep=lambda _: None)
        with self.assertRaises(DiscoveryError):
            provider.search("x", limit=1, since=None)

    def test_rss_is_always_secondary_evidence(self):
        provider = RssProvider(StubTransport(RSS_XML), ["https://blog.example.com/feed"], sleep=lambda _: None)
        [hit] = provider.search("graphrag", limit=5, since=None)
        self.assertIs(hit.source_type, SourceType.SECONDARY_BLOG)

    def test_rss_filters_on_query_terms(self):
        provider = RssProvider(StubTransport(RSS_XML), ["f"], sleep=lambda _: None)
        self.assertEqual(provider.search("quantum chromodynamics", limit=5, since=None), [])


def make_hit(**overrides) -> DiscoveredSource:
    base = dict(
        canonical_url="https://arxiv.org/abs/2404.16130",
        title="GraphRAG evaluation study",
        provider="arxiv",
        external_id="2404.16130",
        abstract="graphrag evaluation multi-hop qa",
        is_open_access=True,
        pdf_url="https://arxiv.org/pdf/2404.16130",
        license="ARXIV-NONEXCLUSIVE",
    )
    base.update(overrides)
    return DiscoveredSource(**base)


class StubProvider:
    def __init__(self, hits, name="arxiv", error=None):
        self.name, self._hits, self._error = name, hits, error

    def search(self, query, *, limit, since):
        if self._error:
            raise DiscoveryError(self._error)
        return list(self._hits)


class TestDiscoveryService(unittest.TestCase):
    def test_open_access_on_allowlisted_host_is_fetchable(self):
        result = DiscoveryService([StubProvider([make_hit()])]).discover("graphrag evaluation")
        self.assertTrue(result.decisions[0].fetchable)

    def test_paywalled_work_is_metadata_only(self):
        hit = make_hit(is_open_access=False, pdf_url=None)
        [decision] = DiscoveryService([StubProvider([hit])]).discover("graphrag").decisions
        self.assertFalse(decision.fetchable)
        self.assertIn("No open-access full text", decision.reason)

    def test_offsite_pdf_is_refused_even_when_open_access(self):
        hit = make_hit(pdf_url="https://randomsite.example.com/paper.pdf")
        [decision] = DiscoveryService([StubProvider([hit])]).discover("graphrag").decisions
        self.assertFalse(decision.fetchable)
        self.assertIn("allowlist", decision.reason)

    def test_restrictive_licence_permits_fetch_but_flags_storage(self):
        hit = make_hit(license="all-rights-reserved")
        [decision] = DiscoveryService([StubProvider([hit])]).discover("graphrag").decisions
        self.assertTrue(decision.fetchable)
        self.assertIn("short excerpts", decision.reason)

    def test_known_urls_are_not_reproposed(self):
        service = DiscoveryService([StubProvider([make_hit()])])
        result = service.discover("graphrag", known_urls=["https://arxiv.org/abs/2404.16130"])
        self.assertEqual(result.decisions, [])
        self.assertEqual(len(result.already_known), 1)

    def test_same_doi_from_two_providers_is_one_candidate(self):
        a = make_hit(provider="openalex", doi="10.1/x", canonical_url="https://openalex.org/W1")
        b = make_hit(provider="arxiv", doi="10.1/x", canonical_url="https://arxiv.org/abs/1")
        result = DiscoveryService([StubProvider([a, b])]).discover("graphrag")
        self.assertEqual(len(result.decisions), 1)
        self.assertEqual(result.decisions[0].candidate.provider, "arxiv", "arXiv preferred")

    def test_provider_failure_is_recorded_not_raised(self):
        service = DiscoveryService(
            [StubProvider([make_hit()]), StubProvider([], name="openalex", error="429 rate limited")]
        )
        result = service.discover("graphrag")
        self.assertEqual(len(result.decisions), 1)
        self.assertIn("openalex", result.provider_errors)
        self.assertIn("provider errors", result.summary())

    def test_relevance_prefers_term_overlap(self):
        on_topic = make_hit(title="GraphRAG evaluation on multi-hop QA")
        off_topic = make_hit(
            title="Unrelated protein folding work",
            abstract="proteins",
            canonical_url="https://arxiv.org/abs/9999.1",
        )
        result = DiscoveryService([StubProvider([off_topic, on_topic])]).discover(
            "graphrag evaluation multi-hop"
        )
        self.assertIn("GraphRAG", result.decisions[0].candidate.title)

    def test_candidates_are_never_above_external_tier(self):
        result = DiscoveryService([StubProvider([make_hit()])]).discover(
            "graphrag", speed=IngestionSpeed.METADATA_ONLY
        )
        self.assertIs(result.decisions[0].tier, EvidenceTier.EXTERNAL_CANDIDATE)

    def test_provisional_speed_still_never_reaches_reviewed(self):
        result = DiscoveryService([StubProvider([make_hit()])]).discover(
            "graphrag", speed=IngestionSpeed.PROVISIONAL
        )
        self.assertIs(result.decisions[0].tier, EvidenceTier.PROVISIONAL_CLAIM)
        self.assertNotEqual(result.decisions[0].tier, EvidenceTier.REVIEWED_CLAIM)

    def test_to_source_marks_unstorable_licence(self):
        service = DiscoveryService([StubProvider([make_hit(license="all-rights-reserved")])])
        [decision] = service.discover("graphrag").decisions
        self.assertFalse(service.to_source(decision).storage_permitted)

    def test_recency_window_is_applied(self):
        captured: dict = {}

        class Recorder(StubProvider):
            def search(self, query, *, limit, since):
                captured["since"] = since
                return []

        DiscoveryService([Recorder([])]).discover("x", recency_months=12)
        self.assertIsNotNone(captured["since"])
        self.assertLess(
            abs((datetime.now(timezone.utc) - captured["since"]) - timedelta(days=360)),
            timedelta(days=15),
        )

    def test_sweep_does_not_repropose_across_queries(self):
        service = DiscoveryService([StubProvider([make_hit()])])
        results = sweep(
            service,
            [
                StandingQuery(query_text="graphrag evaluation", topic="g"),
                StandingQuery(query_text="graph retrieval", topic="g"),
            ],
        )
        self.assertEqual(len(results[0].decisions), 1)
        self.assertEqual(len(results[1].decisions), 0)

    def test_sweep_skips_disabled_queries(self):
        service = DiscoveryService([StubProvider([make_hit()])])
        results = sweep(service, [StandingQuery(query_text="x", topic="g", enabled=False)])
        self.assertEqual(results, [])


class StubExecutor:
    def __init__(self, rows=None):
        self.statements: list[tuple[str, dict]] = []
        self._rows = rows or {}

    def execute(self, statement, parameters):
        self.statements.append((statement, dict(parameters)))
        for needle, rows in self._rows.items():
            if needle in statement:
                return rows
        return []

    def ran(self, needle: str) -> bool:
        return any(needle in s for s, _ in self.statements)


class TestProposalTool(unittest.TestCase):
    """The defect that mattered most: create_proposal must actually write."""

    def test_valid_proposal_is_inserted(self):
        executor = StubExecutor()
        registry = ToolRegistry(CONFIG, executor)
        result = registry.create_proposal(
            "REVIEW_CLAIM",
            json.dumps({"claim_id": "clm-1", "requested_action": "verify benchmark"}),
            "scope incomplete",
            retrieved_claim_ids=["clm-1"],
        )
        self.assertTrue(result.ok)
        self.assertTrue(executor.ran("INSERT INTO main.rd.agent_proposal"))
        self.assertIn("was written", result.message)

    def test_status_is_a_literal_not_a_parameter(self):
        executor = StubExecutor()
        ToolRegistry(CONFIG, executor).create_proposal(
            "OPEN_QUESTION",
            json.dumps({"question_text": "q", "supporting_claim_ids": ["clm-1"]}),
            "why",
            retrieved_claim_ids=["clm-1"],
        )
        statement, parameters = executor.statements[0]
        self.assertIn("'PENDING_APPROVAL'", statement)
        self.assertNotIn("status", parameters)

    def test_invalid_proposal_writes_nothing(self):
        executor = StubExecutor()
        result = ToolRegistry(CONFIG, executor).create_proposal(
            "REVIEW_CLAIM",
            json.dumps({"claim_id": "clm-ghost", "requested_action": "x"}),
            "why",
            retrieved_claim_ids=["clm-1"],
        )
        self.assertFalse(result.ok)
        self.assertEqual(executor.statements, [])
        self.assertIn("No proposal was created", result.message)

    def test_malformed_json_writes_nothing(self):
        executor = StubExecutor()
        result = ToolRegistry(CONFIG, executor).create_proposal("REVIEW_CLAIM", "{oops", "why")
        self.assertFalse(result.ok)
        self.assertEqual(executor.statements, [])

    def test_external_search_refuses_when_unconfigured(self):
        result = ToolRegistry(CONFIG, StubExecutor()).search_external_source("anything")
        self.assertFalse(result.ok)
        self.assertIn("No approved external source", result.message)

    def test_unknown_tool_is_a_refusal(self):
        result = dispatch(ToolRegistry(CONFIG, StubExecutor()), "delete_everything", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.data["error"], "UNKNOWN_TOOL")


class TestDiscoveryTools(unittest.TestCase):
    def _tools(self, executor=None, **kwargs):
        service = DiscoveryService([StubProvider([make_hit()])])
        return DiscoveryTools(CONFIG, service, executor or StubExecutor(), **kwargs)

    def test_discover_labels_everything_external(self):
        result = self._tools().discover_sources("graphrag evaluation")
        self.assertTrue(result.ok)
        self.assertEqual(result.data["evidence_tier"], "EXTERNAL_CANDIDATE")
        for candidate in result.data["candidates"]:
            self.assertEqual(candidate["evidence_tier"], "EXTERNAL_CANDIDATE")
            self.assertIn("Not a reviewed claim", candidate["caution"])

    def test_discover_message_forbids_stating_findings(self):
        message = self._tools().discover_sources("graphrag").message
        self.assertIn("NOT state what any of them found", message)
        self.assertIn("must be ingested and reviewed", message)

    def test_abstract_is_flagged_as_unreviewed(self):
        [candidate] = self._tools().discover_sources("graphrag").data["candidates"]
        self.assertIn("abstract_unreviewed", candidate)
        self.assertNotIn("abstract", candidate)

    def test_discovery_persists_candidates_and_the_run(self):
        executor = StubExecutor()
        self._tools(executor).discover_sources("graphrag")
        self.assertTrue(executor.ran("MERGE INTO main.rd.research_source_candidate"))
        self.assertTrue(executor.ran("INSERT INTO main.rd.research_discovery_run"))

    def test_empty_query_is_refused(self):
        self.assertFalse(self._tools().discover_sources("  ").ok)

    def test_requesting_reviewed_ingestion_is_refused(self):
        result = self._tools().request_ingestion(["cand-1"], "because", speed="REVIEWED")
        self.assertFalse(result.ok)
        self.assertIn("a human reviewer decides", result.message.lower())

    def test_ingestion_approves_only_fetchable_candidates(self):
        executor = StubExecutor(
            {
                "FROM main.rd.research_source_candidate": [
                    {"candidate_id": "c1", "fetchable": "true", "fetch_decision": "ok", "status": "DISCOVERED"},
                    {"candidate_id": "c2", "fetchable": "false", "fetch_decision": "paywalled", "status": "DISCOVERED"},
                ]
            }
        )
        result = self._tools(executor).request_ingestion(["c1", "c2"], "needed for the question")
        self.assertEqual(result.data["approved"], ["c1"])
        self.assertEqual(result.data["blocked"][0]["candidate_id"], "c2")

    def test_ingestion_promises_only_provisional_results(self):
        executor = StubExecutor(
            {"FROM main.rd.research_source_candidate": [
                {"candidate_id": "c1", "fetchable": "true", "fetch_decision": "ok", "status": "DISCOVERED"}]}
        )
        result = self._tools(executor).request_ingestion(["c1"], "why")
        self.assertEqual(result.data["resulting_tier"], "PROVISIONAL_CLAIM")
        self.assertIn("CANDIDATE claims", result.message)
        self.assertIn("until a reviewer accepts it", result.message)

    def test_ingestion_requires_a_rationale(self):
        self.assertFalse(self._tools().request_ingestion(["c1"], "  ").ok)

    def test_job_trigger_is_reported_when_configured(self):
        executor = StubExecutor(
            {"FROM main.rd.research_source_candidate": [
                {"candidate_id": "c1", "fetchable": "true", "fetch_decision": "ok", "status": "DISCOVERED"}]}
        )
        tools = self._tools(executor, job_trigger=lambda ids: "run-77")
        result = tools.request_ingestion(["c1"], "why")
        self.assertEqual(result.data["ingestion_run_id"], "run-77")
        self.assertIn("run-77", result.message)

    def test_gap_check_distinguishes_never_searched(self):
        result = self._tools().check_corpus_gap("legal documents")
        self.assertTrue(result.ok)
        self.assertFalse(result.data["searched"])
        self.assertIn("not evidence that no research exists", result.message)

    def test_dispatch_routes_discovery_tools(self):
        tools = self._tools()
        self.assertIsNotNone(dispatch_discovery(tools, "discover_sources", {"query": "x"}))
        self.assertIsNone(dispatch_discovery(tools, "create_proposal", {}))

    def test_server_dispatch_includes_discovery_when_supplied(self):
        result = dispatch(
            ToolRegistry(CONFIG, StubExecutor()),
            "discover_sources",
            {"query": "graphrag"},
            discovery=self._tools(),
        )
        self.assertTrue(result.ok)


def claim(source: str, text: str, **overrides):
    from research_discovery.models import Claim

    base = dict(
        source_version_id=f"srcv-{source}",
        source_id=source,
        claim_text=text,
        claim_type=ClaimType.PERFORMANCE,
        source_url=f"https://arxiv.org/abs/{source}",
        extractor_name="manual",
        extractor_version="1.0",
        task="multi_hop_qa",
        method="graphrag",
        metric="f1",
        benchmark="hotpotqa",
        condition_text="top-20",
        review_status=ReviewStatus.REVIEWED,
        reviewed_by="alice",
    )
    base.update(overrides)
    return Claim(**base)


class TestSupersession(unittest.TestCase):
    def test_identical_text_matches_across_versions(self):
        old = claim("v1", "Graph retrieval reaches 0.62 F1.", metric_value=0.62)
        new = claim(
            "v2", "Graph retrieval reaches 0.62 F1.", metric_value=0.62,
            review_status=ReviewStatus.CANDIDATE, reviewed_by=None,
        )
        [candidate] = plan_supersession([old], [new])
        self.assertEqual(candidate.match_basis, "IDENTICAL_TEXT")

    def test_same_scope_different_value_is_flagged_with_the_change(self):
        old = claim("v1", "Graph retrieval reaches 0.62 F1.", metric_value=0.62)
        new = claim(
            "v2", "Graph retrieval reaches 0.55 F1 after correction.", metric_value=0.55,
            review_status=ReviewStatus.CANDIDATE, reviewed_by=None,
        )
        [candidate] = plan_supersession([old], [new])
        self.assertEqual(candidate.match_basis, "IDENTICAL_SCOPE")
        self.assertIn("0.62 to 0.55", candidate.note)

    def test_vanished_claim_is_surfaced_not_silently_dropped(self):
        old = claim("v1", "Graph retrieval degrades on temporal questions.", metric_value=None)
        [candidate] = plan_supersession([old], [])
        self.assertEqual(candidate.match_basis, "NO_MATCH")
        self.assertIsNone(candidate.replacement_claim_id)
        self.assertIn("Do not supersede it without checking", candidate.note)

    def test_vanished_claim_is_queued_high(self):
        old = claim("v1", "A finding that disappeared.", metric_value=None)
        [item] = build_supersession_queue(plan_supersession([old], []))
        self.assertEqual(item.priority, "HIGH")
        self.assertIn("SOURCE_UPDATED", item.reason)

    def test_unreviewed_claims_are_unaffected_by_a_source_change(self):
        old = claim(
            "v1", "Unreviewed claim.", metric_value=None,
            review_status=ReviewStatus.CANDIDATE, reviewed_by=None,
        )
        self.assertEqual(plan_supersession([old], []), [])

    def test_new_version_alone_does_not_supersede(self):
        old = claim("v1", "Graph retrieval reaches 0.62 F1.", metric_value=0.62)
        plan_supersession([old], [])
        self.assertIs(old.review_status, ReviewStatus.REVIEWED, "planning must not mutate")
        self.assertTrue(old.is_runtime_visible)

    def test_supersession_applies_only_with_a_reviewer(self):
        old = claim("v1", "x", metric_value=None)
        new = claim("v2", "y", metric_value=None)
        with self.assertRaises(ReviewError):
            apply_supersession(old, new, reviewer="")

    def test_retiring_without_replacement_requires_a_note(self):
        old = claim("v1", "x", metric_value=None)
        with self.assertRaises(ReviewError):
            apply_supersession(old, None, reviewer="alice")

    def test_applied_supersession_removes_runtime_visibility(self):
        old = claim("v1", "x", metric_value=None)
        new = claim("v2", "y", metric_value=None)
        apply_supersession(old, new, reviewer="alice")
        self.assertIs(old.review_status, ReviewStatus.SUPERSEDED)
        self.assertFalse(old.is_runtime_visible)
        self.assertEqual(old.superseded_by_claim_id, new.claim_id)


if __name__ == "__main__":
    unittest.main()


class TestSupersedeJob(unittest.TestCase):
    """The supersession planner must be reachable from the pipeline, not just tests."""

    def test_job_queues_rereview_without_superseding(self):
        from research_discovery.pipelines.supersede_job import run

        old = claim("v1", "Graph retrieval reaches 0.62 F1.", metric_value=0.62)
        new = claim(
            "v2", "Graph retrieval reaches 0.55 F1.", metric_value=0.55,
            review_status=ReviewStatus.CANDIDATE, reviewed_by=None,
        )
        run_log, items = run(CONFIG, [([old], [new])])
        self.assertEqual(len(items), 1)
        self.assertIn("SOURCE_UPDATED", items[0].reason)
        self.assertIs(old.review_status, ReviewStatus.REVIEWED, "must not supersede on its own")
        self.assertEqual(run_log.records_out, 1)

    def test_vanished_finding_is_counted_as_needing_attention(self):
        from research_discovery.pipelines.supersede_job import run

        old = claim("v1", "A finding that disappeared.", metric_value=None)
        run_log, items = run(CONFIG, [([old], [])])
        self.assertEqual(run_log.records_quarantined, 1)
        self.assertEqual(items[0].priority, "HIGH")
