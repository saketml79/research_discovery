"""Genie configuration, validation, deployment and answer-contract tests."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from research_discovery.agent.contracts import EXAMPLE_ANSWER, validate_answer
from research_discovery.agent.deploy import DeploymentResult, build_request, deploy
from research_discovery.agent.genie_config import (
    RUNTIME_FUNCTIONS,
    RUNTIME_VIEWS,
    build_serialized_space,
)
from research_discovery.agent.validate import ConfigValidationError, assert_valid, validate_space
from research_discovery.config import Config

CONFIG = Config(catalog="main", schema="research_discovery")


class TestGenieConfig(unittest.TestCase):
    def setUp(self):
        self.space = build_serialized_space(CONFIG)

    def test_all_identifiers_are_three_level(self):
        for entry in self.space["tables"] + self.space["functions"]:
            self.assertEqual(entry["identifier"].count("."), 2, entry)

    def test_only_views_are_attached(self):
        leaves = {e["identifier"].rsplit(".", 1)[-1] for e in self.space["tables"]}
        self.assertEqual(leaves, set(RUNTIME_VIEWS))
        self.assertNotIn("research_claim", leaves)

    def test_all_declared_functions_are_exposed(self):
        leaves = {e["identifier"].rsplit(".", 1)[-1] for e in self.space["functions"]}
        self.assertEqual(leaves, set(RUNTIME_FUNCTIONS))

    def test_at_least_five_examples_and_benchmarks(self):
        self.assertGreaterEqual(len(self.space["example_queries"]), 5)
        self.assertGreaterEqual(len(self.space["benchmarks"]), 5)

    def test_every_benchmark_states_pass_conditions(self):
        for benchmark in self.space["benchmarks"]:
            self.assertTrue(benchmark["pass_conditions"], benchmark["question"])

    def test_joins_declare_cardinality(self):
        for join in self.space["joins"]:
            self.assertIn(join["cardinality"], {"ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY"})

    def test_instructions_carry_the_refusal_rules(self):
        instructions = self.space["instructions"].lower()
        for phrase in ("insufficient evidence to compare", "compare_claims", "pending_approval"):
            self.assertIn(phrase, instructions)

    def test_config_is_json_serialisable(self):
        self.assertIsInstance(json.dumps(self.space), str)

    def test_schema_name_flows_into_identifiers(self):
        other = build_serialized_space(Config(catalog="c2", schema="s2"))
        self.assertTrue(all(e["identifier"].startswith("c2.s2.") for e in other["tables"]))


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.space = build_serialized_space(CONFIG)

    def test_generated_config_is_valid(self):
        self.assertEqual(validate_space(self.space), [])

    def test_attaching_a_base_table_is_caught(self):
        self.space["tables"].append({"identifier": "main.research_discovery.research_claim"})
        self.assertIn("BASE_TABLE_ATTACHED:main.research_discovery.research_claim", validate_space(self.space))

    def test_two_level_identifier_is_caught(self):
        self.space["tables"].append({"identifier": "research_discovery.v_x"})
        self.assertTrue(any(p.startswith("NOT_THREE_LEVEL") for p in validate_space(self.space)))

    def test_example_reading_a_base_table_is_caught(self):
        self.space["example_queries"].append(
            {"question": "q", "sql": "SELECT * FROM main.research_discovery.research_claim"}
        )
        self.assertIn("EXAMPLE_READS_BASE_TABLE:research_claim", validate_space(self.space))

    def test_view_named_like_a_base_table_is_not_flagged(self):
        # v_research_claim_current must not trip the research_claim rule.
        problems = validate_space(self.space)
        self.assertFalse([p for p in problems if p.startswith("EXAMPLE_READS_BASE_TABLE")])

    def test_stripped_instructions_are_caught(self):
        self.space["instructions"] = "Answer the user's questions."
        problems = validate_space(self.space)
        self.assertTrue(any(p.startswith("INSTRUCTIONS_MISSING_RULE") for p in problems))

    def test_benchmark_without_pass_conditions_is_caught(self):
        self.space["benchmarks"][0]["pass_conditions"] = []
        self.assertTrue(any(p.startswith("BENCHMARK_WITHOUT_PASS_CONDITIONS") for p in validate_space(self.space)))

    def test_assert_valid_raises_with_detail(self):
        self.space["instructions"] = ""
        with self.assertRaises(ConfigValidationError):
            assert_valid(self.space)


class StubGenieClient:
    def __init__(self, spaces=None):
        self.spaces = spaces or []
        self.created, self.updated = [], []

    def list_spaces(self):
        return self.spaces

    def create_space(self, payload):
        self.created.append(payload)
        return {"space_id": "space-new"}

    def update_space(self, space_id, payload):
        self.updated.append((space_id, payload))
        return {"space_id": space_id}


class TestDeployment(unittest.TestCase):
    def test_warehouse_id_is_required(self):
        with self.assertRaises(ValueError):
            build_request(CONFIG, "")

    def test_request_carries_serialized_space(self):
        payload = build_request(CONFIG, "wh-1")
        self.assertEqual(payload["warehouse_id"], "wh-1")
        self.assertIsInstance(payload["serialized_space"], str)
        wire = json.loads(payload["serialized_space"])
        self.assertEqual(wire["version"], 2)
        self.assertIn("v_research_claim_current", wire["instructions"]["text_instructions"][0]["content"][0])

    def test_first_deployment_creates(self):
        client = StubGenieClient()
        result = deploy(CONFIG, warehouse_id="wh-1", client=client)
        self.assertEqual(result.action, "CREATED")
        self.assertEqual(len(client.created), 1)

    def test_second_deployment_updates_in_place(self):
        client = StubGenieClient([{"space_id": "s1", "display_name": "Research Discovery Agent"}])
        result = deploy(CONFIG, warehouse_id="wh-1", client=client)
        self.assertEqual(result.action, "UPDATED")
        self.assertEqual(client.created, [])
        self.assertEqual(client.updated[0][0], "s1")

    def test_dry_run_calls_no_api(self):
        client = StubGenieClient()
        config = replace(CONFIG, dry_run=True)
        result = deploy(config, warehouse_id="wh-1", client=client, dry_run=True)
        self.assertEqual(result.action, "VALIDATED_ONLY")
        self.assertEqual(client.created, [])

    def test_result_is_a_dataclass(self):
        self.assertIsInstance(
            DeploymentResult("s", "CREATED", "Research Discovery Agent"), DeploymentResult
        )


class TestAnswerContract(unittest.TestCase):
    def test_example_answer_is_valid(self):
        self.assertEqual(validate_answer(EXAMPLE_ANSWER), [])

    def test_unreviewed_citation_is_caught(self):
        answer = json.loads(json.dumps(EXAMPLE_ANSWER))
        answer["supporting_claims"][0]["review_status"] = "CANDIDATE"
        self.assertTrue(any(p.startswith("UNREVIEWED_CLAIM_CITED") for p in validate_answer(answer)))

    def test_claim_not_retrieved_is_caught(self):
        problems = validate_answer(EXAMPLE_ANSWER, retrieved_claim_ids={"clm-other"})
        self.assertTrue(any(p.startswith("CLAIM_NOT_RETRIEVED") for p in problems))

    def test_disagreement_without_comparable_verdict_is_caught(self):
        answer = json.loads(json.dumps(EXAMPLE_ANSWER))
        answer["answer"] = "These two papers contradict each other on multi-hop QA."
        self.assertIn("CONFLICT_ASSERTED_WITHOUT_COMPARABLE_VERDICT", validate_answer(answer))

    def test_disagreement_with_comparable_verdict_passes(self):
        answer = json.loads(json.dumps(EXAMPLE_ANSWER))
        answer["answer"] = "These two reviewed results contradict each other on HotpotQA."
        answer["comparability"][0]["comparability_status"] = "COMPARABLE"
        self.assertEqual(validate_answer(answer), [])

    def test_consensus_from_one_source_is_caught(self):
        answer = json.loads(json.dumps(EXAMPLE_ANSWER))
        answer["answer"] = "There is a clear consensus that graph retrieval helps."
        self.assertIn("CONSENSUS_ASSERTED_FROM_FEWER_THAN_TWO_SOURCES", validate_answer(answer))

    def test_consensus_from_two_sources_passes(self):
        answer = json.loads(json.dumps(EXAMPLE_ANSWER))
        answer["answer"] = "There is a consensus across the reviewed sources."
        answer["supporting_claims"].append(
            {
                "claim_id": "clm-example00002",
                "claim_text": "A second reviewed finding.",
                "source_url": "https://arxiv.org/abs/2410.05779",
                "review_status": "REVIEWED",
            }
        )
        self.assertEqual(validate_answer(answer), [])

    def test_empty_limitations_is_caught(self):
        answer = json.loads(json.dumps(EXAMPLE_ANSWER))
        answer["limitations"] = []
        self.assertIn("EMPTY_LIMITATIONS", validate_answer(answer))

    def test_missing_required_field_is_caught(self):
        self.assertIn("MISSING_FIELD:answer", validate_answer({"supporting_claims": []}))


if __name__ == "__main__":
    unittest.main()
