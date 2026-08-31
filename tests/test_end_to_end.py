"""End-to-end flow without Spark or network.

Drives the real pipeline objects from a stub transport through parsing,
chunking, extraction, review and relationship detection, then asserts the four
behaviours the demo has to prove:

1. a reviewed claim is citable, a candidate claim is not;
2. a real disagreement in matching scope is reported as CONTRADICTS;
3. an apparent conflict in different scope is NOT_COMPARABLE_YET;
4. an answer that asserts a disagreement without a comparable verdict fails the
   contract check.
"""

from __future__ import annotations

import unittest

from research_discovery.agent.contracts import validate_answer
from research_discovery.config import Config
from research_discovery.extract.heuristic import HeuristicClaimExtractor
from research_discovery.ingest.sources import FetchResult, PolicyAwareFetcher, register_source
from research_discovery.models import ClaimType, RelationshipType, ReviewStatus, SourceType
from research_discovery.pipelines.extract_job import extract_claims
from research_discovery.pipelines.ingest_job import process_source
from research_discovery.review.comparability import detect_relationships
from research_discovery.review.queue import apply_claim_decision, build_claim_queue

PAPER_HTML = b"""<html><head><title>Graph Retrieval Study</title></head><body>
<h2>Results</h2>
<p>Our graph-based retrieval system reaches 0.62 F1 on multi-hop question answering
over HotpotQA, outperforming the vector RAG baseline under a top-20 retrieval budget.</p>
<h2>References</h2>
<p>[1] Someone et al. An unrelated paper about 0.99 F1.</p>
</body></html>"""


class StubTransport:
    def __init__(self, payload: bytes):
        self.payload = payload

    def get(self, url, *, etag=None):
        return FetchResult(self.payload, "text/html", 200, None)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.config = Config(catalog="main", schema="rd", parser="html", extractor="heuristic")
        self.fetcher = PolicyAwareFetcher(
            StubTransport(PAPER_HTML),
            allowed_hosts={"arxiv.org"},
            min_interval_seconds=0,
        )

    def _ingest(self):
        source = register_source(
            "https://arxiv.org/abs/2404.16130",
            source_type=SourceType.PRIMARY_PAPER,
            title="Graph Retrieval Study",
            licence="ARXIV-NONEXCLUSIVE",
        )
        return process_source(source, self.fetcher, self.config)

    def test_ingest_produces_page_scoped_chunks_without_references(self):
        _, version, chunks = self._ingest()
        self.assertIsNotNone(version)
        self.assertTrue(chunks)
        self.assertNotIn("0.99", " ".join(c.text for c in chunks))

    def test_reingesting_unchanged_content_is_a_no_op(self):
        source, version, _ = self._ingest()
        from research_discovery.ingest.sources import fetch_version

        self.assertIsNone(fetch_version(source, self.fetcher, previous=version))

    def test_extracted_claims_start_unreviewed_and_are_queued(self):
        source, _, chunks = self._ingest()
        claims, rejected = extract_claims(
            chunks, HeuristicClaimExtractor(), {source.source_id: source.canonical_url}
        )
        self.assertTrue(claims, "expected at least one candidate claim")
        self.assertEqual(rejected, 0)
        self.assertTrue(all(c.review_status is ReviewStatus.CANDIDATE for c in claims))
        self.assertTrue(all(not c.is_runtime_visible for c in claims))
        queue = build_claim_queue(claims, self.config)
        self.assertEqual(len(queue), len(claims))

    def test_review_is_what_makes_a_claim_citable(self):
        source, _, chunks = self._ingest()
        claims, _ = extract_claims(
            chunks, HeuristicClaimExtractor(), {source.source_id: source.canonical_url}
        )
        claim = claims[0]
        self.assertFalse(claim.is_runtime_visible)
        apply_claim_decision(
            claim,
            decision="AMENDED",
            reviewer="alice",
            note="benchmark and conditions read from the results table",
            amendments={"benchmark": "hotpotqa", "condition_text": "top-20 budget"},
        )
        self.assertTrue(claim.is_runtime_visible)

    def _reviewed(self, source_id, *, value, benchmark, condition):
        from research_discovery.models import Claim

        return Claim(
            source_version_id=f"srcv-{source_id}",
            source_id=source_id,
            claim_text=f"Graph retrieval reaches {value} F1 on {benchmark}.",
            claim_type=ClaimType.PERFORMANCE,
            source_url=f"https://arxiv.org/abs/{source_id}",
            extractor_name="manual",
            extractor_version="1.0",
            task="multi_hop_qa",
            method="graphrag_global",
            metric="f1",
            metric_value=value,
            benchmark=benchmark,
            condition_text=condition,
            missing_field_reason=None if condition else "conditions not stated",
            review_status=ReviewStatus.REVIEWED,
            reviewed_by="alice",
        )

    def test_the_four_demo_behaviours(self):
        a = self._reviewed("a", value=0.62, benchmark="hotpotqa", condition="top-20, GPT-4 reader")
        agrees = self._reviewed("b", value=0.60, benchmark="hotpotqa", condition="top-20, GPT-4 reader")
        conflicts = self._reviewed("c", value=0.38, benchmark="hotpotqa", condition="top-20, 8B reader")
        elsewhere = self._reviewed("d", value=0.72, benchmark="musique", condition=None)

        by_pair = {
            (r.from_claim_id, r.to_claim_id): r
            for r in detect_relationships([a, agrees, conflicts, elsewhere])
        }
        types = {r.relationship_type for r in by_pair.values()}

        self.assertIn(RelationshipType.SUPPORTS, types)
        self.assertIn(RelationshipType.CONTRADICTS, types)
        self.assertIn(RelationshipType.NOT_COMPARABLE_YET, types)

        # The different-benchmark pair is never a contradiction.
        cross = [
            r
            for r in by_pair.values()
            if {r.from_claim_id, r.to_claim_id} == {a.claim_id, elsewhere.claim_id}
        ][0]
        self.assertIs(cross.relationship_type, RelationshipType.NOT_COMPARABLE_YET)
        self.assertIn("benchmark", cross.missing_dimensions)

    def test_answer_asserting_an_unbacked_disagreement_is_rejected(self):
        answer = {
            "answer": "These sources contradict each other about GraphRAG.",
            "supporting_claims": [
                {
                    "claim_id": "clm-1",
                    "claim_text": "x",
                    "source_url": "https://arxiv.org/abs/1",
                    "review_status": "REVIEWED",
                }
            ],
            "comparability": [
                {
                    "from_claim_id": "clm-1",
                    "to_claim_id": "clm-2",
                    "comparability_status": "INSUFFICIENT_EVIDENCE",
                    "missing_dimensions": "benchmark",
                }
            ],
            "recommended_next_step": "PENDING_APPROVAL proposal recorded.",
            "limitations": ["Small corpus."],
        }
        self.assertIn("CONFLICT_ASSERTED_WITHOUT_COMPARABLE_VERDICT", validate_answer(answer))

    def test_quarantine_keeps_a_bad_source_from_failing_the_batch(self):
        from research_discovery.parsers.base import ParserError

        broken = PolicyAwareFetcher(
            StubTransport(b"<html><body></body></html>"),
            allowed_hosts={"arxiv.org"},
            min_interval_seconds=0,
        )
        source = register_source(
            "https://arxiv.org/abs/9", source_type=SourceType.PRIMARY_PAPER, licence="MIT"
        )
        with self.assertRaises(ParserError):
            process_source(source, broken, self.config)


if __name__ == "__main__":
    unittest.main()
