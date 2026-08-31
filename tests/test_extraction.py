"""Extraction, validation and anti-fabrication tests."""

from __future__ import annotations

import json
import unittest

from research_discovery.config import Config
from research_discovery.extract.base import (
    CandidateClaim,
    ExtractionError,
    to_claim,
    validate_candidate,
)
from research_discovery.extract.heuristic import HeuristicClaimExtractor
from research_discovery.extract.llm import LlmClaimExtractor
from research_discovery.extract.registry import get_extractor
from research_discovery.models import Chunk, ReviewStatus


CHUNK_TEXT = (
    "Our graph-based retrieval system reaches 0.62 F1 on the HotpotQA development set, "
    "outperforming the vector RAG baseline which reaches 0.51 F1 under the same top-20 "
    "retrieval budget."
)


def chunk(text: str = CHUNK_TEXT) -> Chunk:
    return Chunk(
        source_version_id="srcv-1",
        source_id="src-1",
        chunk_index=0,
        text=text,
        page_number=6,
    )


class StubChat:
    def __init__(self, payload, fail_times: int = 0):
        self.payload = payload
        self.fail_times = fail_times
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("transient")
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


class TestCandidateValidation(unittest.TestCase):
    def test_value_present_in_text_passes(self):
        candidate = CandidateClaim(
            claim_text="Graph retrieval reaches 0.62 F1.",
            claim_type="PERFORMANCE",
            task="multi_hop_qa",
            method="graphrag",
            metric="f1",
            metric_value=0.62,
            benchmark="hotpotqa",
            condition_text="top-20 budget",
        )
        self.assertEqual(validate_candidate(candidate, chunk()), [])

    def test_invented_number_is_rejected(self):
        candidate = CandidateClaim(
            claim_text="Graph retrieval reaches 0.87 F1.",
            claim_type="PERFORMANCE",
            task="multi_hop_qa",
            method="graphrag",
            metric="f1",
            metric_value=0.87,
            benchmark="hotpotqa",
            condition_text="top-20 budget",
        )
        self.assertIn("METRIC_VALUE_NOT_IN_SOURCE_TEXT", validate_candidate(candidate, chunk()))

    def test_percentage_form_of_a_ratio_is_accepted(self):
        text = "The system won 72% of comprehensiveness comparisons."
        candidate = CandidateClaim(
            claim_text="Won 72% of comparisons.",
            claim_type="PERFORMANCE",
            task="qfs",
            method="graphrag",
            metric="win_rate",
            metric_value=0.72,
            benchmark="podcast",
            condition_text="LLM judge",
        )
        self.assertEqual(validate_candidate(candidate, chunk(text)), [])

    def test_paraphrased_excerpt_is_rejected(self):
        candidate = CandidateClaim(
            claim_text="Graph retrieval wins.",
            claim_type="PERFORMANCE",
            task="t",
            method="m",
            metric="f1",
            benchmark="b",
            condition_text="c",
            evidence_excerpt="Our system is the best available today.",
        )
        self.assertIn("EXCERPT_NOT_VERBATIM", validate_candidate(candidate, chunk()))

    def test_verbatim_excerpt_is_accepted(self):
        candidate = CandidateClaim(
            claim_text="Graph retrieval wins.",
            claim_type="PERFORMANCE",
            task="t",
            method="m",
            metric="f1",
            benchmark="b",
            condition_text="c",
            evidence_excerpt="reaches 0.62 F1 on the HotpotQA development set",
        )
        self.assertEqual(validate_candidate(candidate, chunk()), [])

    def test_missing_scope_without_reason_is_rejected(self):
        candidate = CandidateClaim(claim_text="Something improved.", claim_type="PERFORMANCE")
        problems = validate_candidate(candidate, chunk())
        self.assertTrue(any(p.startswith("MISSING_FIELD_REASON_REQUIRED") for p in problems))


class TestToClaim(unittest.TestCase):
    def test_claim_is_always_created_as_candidate(self):
        candidate = CandidateClaim(
            claim_text="Graph retrieval reaches 0.62 F1.",
            claim_type="PERFORMANCE",
            task="multi_hop_qa",
            method="graphrag",
            metric="f1",
            metric_value=0.62,
            benchmark="hotpotqa",
            condition_text="top-20 budget",
            confidence=0.9,
        )
        result = to_claim(
            candidate,
            chunk(),
            source_url="https://example.org/a",
            extractor_name="llm",
            extractor_version="m/1",
        )
        self.assertIs(result.review_status, ReviewStatus.CANDIDATE)
        self.assertEqual(result.page_number, 6)
        self.assertFalse(result.is_runtime_visible)

    def test_invalid_candidate_never_becomes_a_claim(self):
        candidate = CandidateClaim(
            claim_text="Reaches 0.99 F1.",
            claim_type="PERFORMANCE",
            task="t",
            method="m",
            metric="f1",
            metric_value=0.99,
            benchmark="b",
            condition_text="c",
        )
        with self.assertRaises(ExtractionError):
            to_claim(
                candidate,
                chunk(),
                source_url="https://example.org/a",
                extractor_name="llm",
                extractor_version="m/1",
            )


class TestLlmExtractor(unittest.TestCase):
    def _extractor(self, payload, fail_times=0):
        return LlmClaimExtractor(
            StubChat(payload, fail_times), model_name="test-model", sleep=lambda _: None
        )

    def test_parses_a_well_formed_response(self):
        payload = {
            "claims": [
                {
                    "claim_text": "Graph retrieval reaches 0.62 F1 on HotpotQA.",
                    "claim_type": "PERFORMANCE",
                    "task": "multi_hop_qa",
                    "method": "graphrag",
                    "metric": "f1",
                    "metric_value": 0.62,
                    "benchmark": "hotpotqa",
                    "condition_text": "top-20 budget",
                    "confidence": 0.9,
                }
            ]
        }
        candidates = self._extractor(payload).extract(chunk())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].metric_value, 0.62)

    def test_tolerates_a_code_fence(self):
        payload = '```json\n{"claims": []}\n```'
        self.assertEqual(list(self._extractor(payload).extract(chunk())), [])

    def test_tolerates_invalid_backslash_escape(self):
        # A raw Windows-style path / LaTeX-like "\%" is invalid JSON on its
        # own but should be repaired rather than failing the whole response.
        payload = (
            '{"claims": [{"claim_text": "Cost drops by 40\\%, see C:\\Users\\x.", '
            '"claim_type": "PERFORMANCE", "confidence": 0.8}]}'
        )
        candidates = self._extractor(payload).extract(chunk())
        self.assertEqual(len(candidates), 1)

    def test_non_json_response_raises(self):
        with self.assertRaises(ExtractionError):
            self._extractor("I could not find any claims.").extract(chunk())

    def test_unknown_fields_are_rejected(self):
        payload = {
            "claims": [
                {
                    "claim_text": "x" * 20,
                    "claim_type": "PERFORMANCE",
                    "confidence": 0.5,
                    "made_up_field": 1,
                }
            ]
        }
        with self.assertRaises(ExtractionError):
            self._extractor(payload).extract(chunk())

    def test_retries_transient_failures_then_succeeds(self):
        client = StubChat({"claims": []}, fail_times=2)
        extractor = LlmClaimExtractor(client, model_name="test-model", sleep=lambda _: None)
        self.assertEqual(list(extractor.extract(chunk())), [])
        self.assertEqual(client.calls, 3)

    def test_gives_up_after_the_retry_budget(self):
        client = StubChat({"claims": []}, fail_times=99)
        extractor = LlmClaimExtractor(client, model_name="test-model", sleep=lambda _: None)
        with self.assertRaises(ExtractionError):
            extractor.extract(chunk())

    def test_version_records_model_and_prompt(self):
        self.assertIn("test-model", self._extractor({"claims": []}).version)

    def test_tolerates_a_bare_claims_array(self):
        payload = [
            {
                "claim_text": "Graph retrieval reaches 0.62 F1 on HotpotQA.",
                "claim_type": "PERFORMANCE",
                "confidence": 0.9,
            }
        ]
        candidates = self._extractor(payload).extract(chunk())
        self.assertEqual(len(candidates), 1)

    def test_tolerates_a_single_unwrapped_claim_object(self):
        payload = {
            "claim_text": "Graph retrieval reaches 0.62 F1 on HotpotQA.",
            "claim_type": "PERFORMANCE",
            "confidence": 0.9,
        }
        candidates = self._extractor(payload).extract(chunk())
        self.assertEqual(len(candidates), 1)

    def test_renames_claim_alias_to_claim_text(self):
        payload = {
            "claims": [
                {
                    "claim": "Graph retrieval reaches 0.62 F1 on HotpotQA.",
                    "claim_type": "PERFORMANCE",
                    "confidence": 0.9,
                }
            ]
        }
        candidates = self._extractor(payload).extract(chunk())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].claim_text, "Graph retrieval reaches 0.62 F1 on HotpotQA.")

    def test_flattens_nested_scope_object(self):
        payload = {
            "claims": [
                {
                    "claim_text": "Graph retrieval reaches 0.62 F1 on HotpotQA.",
                    "claim_type": "PERFORMANCE",
                    "confidence": 0.9,
                    "scope": {"task": "multi_hop_qa", "benchmark": "hotpotqa"},
                }
            ]
        }
        candidates = self._extractor(payload).extract(chunk())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].task, "multi_hop_qa")
        self.assertEqual(candidates[0].benchmark, "hotpotqa")

    def test_drops_no_claim_items_with_empty_claim_text(self):
        payload = {
            "claims": [
                {
                    "claim": None,
                    "missing_field_reason": "citation only",
                    "confidence": 1.0,
                }
            ]
        }
        self.assertEqual(list(self._extractor(payload).extract(chunk())), [])

    def test_tolerates_trailing_data_after_json(self):
        payload = '{"claims": []}\n{"claims": []}'
        self.assertEqual(list(self._extractor(payload).extract(chunk())), [])

    def test_drops_claim_text_shorter_than_schema_minimum(self):
        payload = {
            "claims": [
                {
                    "claim_text": "README.md",
                    "claim_type": "METHOD_DESCRIPTION",
                    "confidence": 1.0,
                }
            ]
        }
        self.assertEqual(list(self._extractor(payload).extract(chunk())), [])

    def test_batch_failure_is_isolated(self):
        client = StubChat("not json")
        extractor = LlmClaimExtractor(client, model_name="m", sleep=lambda _: None)
        results = extractor.extract_many([chunk(), chunk()])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(c.confidence == 0.0 for _, c in results))


class TestHeuristicExtractor(unittest.TestCase):
    def test_finds_a_result_sentence(self):
        candidates = HeuristicClaimExtractor().extract(chunk())
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].metric_value, 0.62)

    def test_ignores_prose_without_a_result_verb(self):
        text = "This section describes the architecture of the retrieval component in detail."
        self.assertEqual(list(HeuristicClaimExtractor().extract(chunk(text))), [])

    def test_always_reports_which_fields_it_could_not_fill(self):
        candidate = HeuristicClaimExtractor().extract(chunk())[0]
        self.assertIn("MISSING:", candidate.missing_field_reason or "")


class TestRegistry(unittest.TestCase):
    def test_heuristic_selected_by_config(self):
        extractor = get_extractor(Config(extractor="heuristic"))
        self.assertEqual(extractor.name, "heuristic")

    def test_unknown_extractor_raises(self):
        with self.assertRaises(KeyError):
            get_extractor(Config(extractor="telepathy"))


if __name__ == "__main__":
    unittest.main()
