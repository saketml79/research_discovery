"""Comparability tests: the behaviour this build exists to guarantee."""

from __future__ import annotations

import unittest

from research_discovery.models import ClaimType, ComparabilityStatus, RelationshipType, ReviewStatus
from research_discovery.models import Claim
from research_discovery.review.comparability import (
    assess_comparability,
    build_relationship,
    detect_relationships,
)


def claim(**overrides) -> Claim:
    """Build a reviewed claim. The source version follows the source id, as it
    does in the real pipeline, so changing source_id yields a distinct claim."""
    source = overrides.get("source_id", "src-1")
    base = dict(
        source_version_id=f"srcv-{source}",
        source_id=source,
        claim_text="Graph retrieval reaches 0.62 F1 on HotpotQA.",
        claim_type=ClaimType.PERFORMANCE,
        source_url="https://example.org/a",
        extractor_name="manual",
        extractor_version="1.0",
        task="multi_hop_qa",
        method="graphrag_global",
        metric="f1",
        metric_value=0.62,
        benchmark="hotpotqa",
        condition_text="Full dev set, GPT-4 reader, top-20.",
        review_status=ReviewStatus.REVIEWED,
        reviewed_by="reviewer",
    )
    base.update(overrides)
    return Claim(**base)


class TestComparability(unittest.TestCase):
    def test_full_scope_overlap_is_comparable(self):
        verdict = assess_comparability(claim(), claim(source_id="src-2", metric_value=0.60))
        self.assertIs(verdict.status, ComparabilityStatus.COMPARABLE)
        self.assertEqual(verdict.score, 1.0)

    def test_missing_conditions_is_only_partially_comparable(self):
        other = claim(
            source_id="src-2",
            condition_text=None,
            missing_field_reason="conditions not stated",
        )
        verdict = assess_comparability(claim(), other)
        self.assertIs(verdict.status, ComparabilityStatus.PARTIALLY_COMPARABLE)
        self.assertIn("condition", verdict.missing_dimensions)

    def test_different_benchmark_is_insufficient_evidence(self):
        verdict = assess_comparability(claim(), claim(source_id="src-2", benchmark="musique"))
        self.assertIs(verdict.status, ComparabilityStatus.INSUFFICIENT_EVIDENCE)
        self.assertIn("benchmark", verdict.missing_dimensions)
        self.assertFalse(verdict.is_comparable)

    def test_null_scope_never_matches_null_scope(self):
        # Two claims that both fail to state a benchmark are NOT thereby
        # comparable. This is the subtle bug the whole design guards against.
        a = claim(benchmark=None, missing_field_reason="not stated")
        b = claim(source_id="src-2", benchmark=None, missing_field_reason="not stated")
        self.assertIs(assess_comparability(a, b).status, ComparabilityStatus.INSUFFICIENT_EVIDENCE)

    def test_case_and_whitespace_differences_still_match(self):
        verdict = assess_comparability(claim(), claim(source_id="src-2", benchmark="  HotpotQA "))
        self.assertIs(verdict.status, ComparabilityStatus.COMPARABLE)


class TestRelationshipClassification(unittest.TestCase):
    def test_close_values_in_same_scope_support_each_other(self):
        rel = build_relationship(claim(), claim(source_id="src-2", metric_value=0.60))
        self.assertIs(rel.relationship_type, RelationshipType.SUPPORTS)

    def test_far_values_in_same_scope_contradict(self):
        rel = build_relationship(claim(), claim(source_id="src-2", metric_value=0.38))
        self.assertIs(rel.relationship_type, RelationshipType.CONTRADICTS)
        self.assertIs(rel.comparability_status, ComparabilityStatus.COMPARABLE)

    def test_far_values_without_conditions_refine_rather_than_contradict(self):
        other = claim(
            source_id="src-2",
            metric_value=0.38,
            condition_text=None,
            missing_field_reason="conditions not stated",
        )
        rel = build_relationship(claim(), other)
        self.assertIs(rel.relationship_type, RelationshipType.REFINES)

    def test_incomparable_pair_is_not_comparable_yet(self):
        rel = build_relationship(claim(), claim(source_id="src-2", benchmark="musique"))
        self.assertIs(rel.relationship_type, RelationshipType.NOT_COMPARABLE_YET)
        self.assertIn("benchmark", rel.missing_dimensions)

    def test_intermediate_difference_is_a_refinement(self):
        rel = build_relationship(claim(), claim(source_id="src-2", metric_value=0.55))
        self.assertIs(rel.relationship_type, RelationshipType.REFINES)

    def test_prose_claims_without_numbers_never_contradict(self):
        a = claim(metric_value=None, claim_text="Graph retrieval improves multi-hop answers.")
        b = claim(
            source_id="src-2",
            metric_value=None,
            claim_text="Graph retrieval does not improve multi-hop answers.",
        )
        self.assertIs(build_relationship(a, b).relationship_type, RelationshipType.REFINES)


class TestDetection(unittest.TestCase):
    def test_claims_from_the_same_source_are_not_related(self):
        pool = [claim(), claim(claim_text="A second finding in the same paper.", metric_value=0.30)]
        self.assertEqual(detect_relationships(pool), [])

    def test_unreviewed_claims_are_excluded(self):
        pool = [
            claim(),
            claim(source_id="src-2", review_status=ReviewStatus.CANDIDATE, reviewed_by=None),
        ]
        self.assertEqual(detect_relationships(pool, reviewed_only=True), [])

    def test_relationships_are_deduplicated(self):
        pool = [claim(), claim(source_id="src-2", metric_value=0.60)]
        first = detect_relationships(pool)
        second = detect_relationships(list(reversed(pool)))
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].relationship_id, second[0].relationship_id)


if __name__ == "__main__":
    unittest.main()
