"""Identity, lifecycle and record-validation tests."""

from __future__ import annotations

import unittest

from research_discovery import ids
from research_discovery.models import (
    Chunk,
    Claim,
    ClaimRelationship,
    ClaimType,
    ComparabilityStatus,
    IngestionStatus,
    Proposal,
    RelationshipType,
    ReviewStatus,
    Source,
    SourceType,
    SourceVersion,
    can_transition,
)


class TestIds(unittest.TestCase):
    def test_ids_are_deterministic(self):
        self.assertEqual(ids.source_id("https://arxiv.org/abs/1"), ids.source_id("https://arxiv.org/abs/1"))

    def test_url_normalization_collapses_equivalent_urls(self):
        self.assertEqual(
            ids.source_id("https://ArXiv.org/abs/1/"),
            ids.source_id("https://arxiv.org:443/abs/1"),
        )

    def test_relationship_id_is_order_independent(self):
        self.assertEqual(
            ids.relationship_id("a", "b", "SUPPORTS"),
            ids.relationship_id("b", "a", "SUPPORTS"),
        )

    def test_relationship_id_differs_by_type(self):
        self.assertNotEqual(
            ids.relationship_id("a", "b", "SUPPORTS"),
            ids.relationship_id("a", "b", "CONTRADICTS"),
        )

    def test_claim_id_ignores_case_and_whitespace(self):
        self.assertEqual(
            ids.claim_id("v1", "GraphRAG  wins."),
            ids.claim_id("v1", "graphrag wins."),
        )


class TestLifecycle(unittest.TestCase):
    def test_forward_transition_allowed(self):
        self.assertTrue(can_transition(IngestionStatus.FETCHED, IngestionStatus.PARSED))

    def test_skipping_a_stage_is_rejected(self):
        self.assertFalse(can_transition(IngestionStatus.DISCOVERED, IngestionStatus.EXTRACTED))

    def test_quarantine_always_allowed(self):
        for state in IngestionStatus:
            self.assertTrue(can_transition(state, IngestionStatus.QUARANTINED))

    def test_source_advance_rejects_illegal_jump(self):
        source = Source("https://arxiv.org/abs/1", source_type=SourceType.PRIMARY_PAPER)
        with self.assertRaises(ValueError):
            source.advance(IngestionStatus.INDEXED)


class TestRecordValidation(unittest.TestCase):
    def _claim(self, **overrides):
        base = dict(
            source_version_id="srcv-1",
            source_id="src-1",
            claim_text="Graph retrieval reaches 0.62 F1 on HotpotQA.",
            claim_type=ClaimType.PERFORMANCE,
            source_url="https://arxiv.org/abs/1",
            extractor_name="manual",
            extractor_version="1.0",
            task="multi_hop_qa",
            method="graphrag_global",
            metric="f1",
            benchmark="hotpotqa",
            condition_text="Full dev set.",
        )
        base.update(overrides)
        return Claim(**base)

    def test_missing_scope_requires_a_reason(self):
        with self.assertRaises(ValueError) as ctx:
            self._claim(benchmark=None)
        self.assertIn("missing_field_reason", str(ctx.exception))

    def test_missing_scope_with_reason_is_accepted(self):
        claim = self._claim(benchmark=None, missing_field_reason="not stated in passage")
        self.assertEqual(claim.missing_scope_fields(), ("benchmark",))

    def test_reviewed_claim_requires_a_reviewer(self):
        with self.assertRaises(ValueError):
            self._claim(review_status=ReviewStatus.REVIEWED)

    def test_candidate_claim_is_not_runtime_visible(self):
        self.assertFalse(self._claim().is_runtime_visible)

    def test_confidence_range_enforced(self):
        with self.assertRaises(ValueError):
            self._claim(extraction_confidence=1.4)

    def test_contradiction_requires_comparable_scope(self):
        with self.assertRaises(ValueError):
            ClaimRelationship(
                from_claim_id="a",
                to_claim_id="b",
                relationship_type=RelationshipType.CONTRADICTS,
                comparability_status=ComparabilityStatus.INSUFFICIENT_EVIDENCE,
                detector_name="test",
            )

    def test_proposal_cannot_be_created_approved(self):
        with self.assertRaises(ValueError):
            Proposal(
                proposal_type="REVIEW_CLAIM",
                payload_json="{}",
                created_by="agent",
                status="APPROVED",
            )

    def test_empty_chunk_rejected(self):
        with self.assertRaises(ValueError):
            Chunk(source_version_id="v", source_id="s", chunk_index=0, text="   ")

    def test_source_version_requires_sha256(self):
        with self.assertRaises(ValueError):
            SourceVersion(source_id="s", content_hash="short")


if __name__ == "__main__":
    unittest.main()
