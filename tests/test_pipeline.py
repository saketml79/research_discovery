"""Parsing, chunking, ingestion policy and review-boundary tests."""

from __future__ import annotations

import unittest

from research_discovery.chunking import chunk_document, split_long_text
from research_discovery.config import Config, ConfigError
from research_discovery.ingest.sources import (
    FetchResult,
    PolicyAwareFetcher,
    PolicyError,
    fetch_version,
    licence_permits_storage,
    register_source,
)
from research_discovery.models import BlockType, IngestionStatus, SourceType, SourceVersion
from research_discovery.parsers.base import ParsedBlock, ParsedDocument
from research_discovery.parsers.registry import get_parser, resolve_with_fallback
from research_discovery.parsers.text import HtmlParser, PlainTextParser
from research_discovery.review.proposals import ProposalValidationError, build_proposal
from research_discovery.review.queue import (
    ReviewError,
    apply_claim_decision,
    backlog,
    build_claim_queue,
)
from tests.test_comparability import claim as reviewed_claim
from research_discovery.models import Claim, ClaimType, ReviewStatus


class TestConfig(unittest.TestCase):
    def test_three_level_names(self):
        config = Config(catalog="main", schema="rd")
        self.assertEqual(config.table("research_claim"), "main.rd.research_claim")

    def test_injection_in_identifier_rejected(self):
        with self.assertRaises(ConfigError):
            Config(schema="rd; DROP TABLE x")

    def test_invalid_chunk_bounds_rejected(self):
        with self.assertRaises(ConfigError):
            Config(min_chunk_chars=3000, max_chunk_chars=1000)

    def test_from_args_coerces_types(self):
        config = Config.from_args({"catalog": "c", "dry_run": "true", "max_chunk_chars": "900"})
        self.assertTrue(config.dry_run)
        self.assertEqual(config.max_chunk_chars, 900)


class TestParsers(unittest.TestCase):
    def test_html_blocks_and_headings(self):
        html = b"""<html><head><title>T</title></head><body>
          <h2>Results</h2><p>Graph retrieval reaches 0.62 F1.</p>
          <script>ignore()</script><figcaption>Figure 1: overview</figcaption>
        </body></html>"""
        doc = HtmlParser().parse(html, source_uri="u", content_type="text/html")
        texts = [b.text for b in doc.blocks]
        self.assertIn("Graph retrieval reaches 0.62 F1.", texts)
        self.assertNotIn("ignore()", " ".join(texts))
        self.assertEqual(doc.blocks[0].section_title, "Results")
        self.assertEqual(doc.metadata["title"], "T")

    def test_figcaption_is_marked_as_a_caption(self):
        html = b"<html><body><figcaption>Figure 2: pipeline</figcaption></body></html>"
        doc = HtmlParser().parse(html, source_uri="u", content_type="text/html")
        self.assertIs(doc.blocks[0].block_type, BlockType.FIGURE_CAPTION)

    def test_plaintext_splits_on_blank_lines(self):
        doc = PlainTextParser().parse(b"one\n\ntwo", source_uri="u", content_type="text/plain")
        self.assertEqual(len(doc.blocks), 2)

    def test_fallback_reports_the_substitution(self):
        parser, warning = resolve_with_fallback("docling", "text/html")
        self.assertIsNotNone(parser)
        if warning is not None:
            self.assertTrue(warning.startswith("PARSER_FALLBACK_docling_TO_"))

    def test_unknown_parser_name_raises(self):
        with self.assertRaises(KeyError):
            get_parser("ocr9000")


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.config = Config(min_chunk_chars=40, max_chunk_chars=200)

    def _doc(self, blocks):
        return ParsedDocument(blocks=blocks, parser_name="test", parser_version="1")

    def test_chunk_never_spans_two_pages(self):
        doc = self._doc(
            [
                ParsedBlock("Short text on page one." * 2, page_number=1),
                ParsedBlock("Short text on page two." * 2, page_number=2),
            ]
        )
        chunks = chunk_document(doc, source_version_id="v", source_id="s", config=self.config)
        pages = {c.page_number for c in chunks}
        self.assertEqual(pages, {1, 2})
        for chunk in chunks:
            self.assertIsNotNone(chunk.page_number)

    def test_table_is_never_merged_with_prose(self):
        doc = self._doc(
            [
                ParsedBlock("Prose before the table which is fairly long.", page_number=1),
                ParsedBlock("F1 0.62 | 0.51", page_number=1, block_type=BlockType.TABLE),
            ]
        )
        chunks = chunk_document(doc, source_version_id="v", source_id="s", config=self.config)
        table = [c for c in chunks if c.block_type is BlockType.TABLE]
        self.assertEqual(len(table), 1)
        self.assertNotIn("Prose before", table[0].text)

    def test_references_are_excluded(self):
        doc = self._doc(
            [
                ParsedBlock("Real content that should survive chunking here.", page_number=1),
                ParsedBlock("[1] Someone et al. A paper.", page_number=9, block_type=BlockType.REFERENCES),
            ]
        )
        chunks = chunk_document(doc, source_version_id="v", source_id="s", config=self.config)
        self.assertTrue(all(c.block_type is not BlockType.REFERENCES for c in chunks))

    def test_chunk_indices_are_contiguous(self):
        doc = self._doc([ParsedBlock("Sentence number %d here." % i, page_number=1) for i in range(8)])
        chunks = chunk_document(doc, source_version_id="v", source_id="s", config=self.config)
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))

    def test_restricted_licence_truncates_and_marks(self):
        doc = self._doc([ParsedBlock("x" * 2000, page_number=1)])
        chunks = chunk_document(
            doc, source_version_id="v", source_id="s", config=self.config, storage_permitted=False
        )
        self.assertTrue(all(len(c.text) <= 400 for c in chunks))
        self.assertIn("TRUNCATED_LICENCE", chunks[0].extraction_warning)

    def test_split_long_text_respects_the_bound(self):
        text = " ".join(["This is a sentence."] * 60)
        for piece in split_long_text(text, 100):
            self.assertLessEqual(len(piece), 100)

    def test_split_long_text_loses_no_characters(self):
        # A sentence longer than the bound is split mid-token by design; the
        # invariant is that no character is dropped, not that words stay whole.
        text = "Alpha beta. Gamma delta. Epsilon zeta."
        pieces = split_long_text(text, 12)
        self.assertEqual("".join(pieces).replace(" ", ""), text.replace(" ", ""))

    def test_split_long_text_keeps_whole_sentences_when_they_fit(self):
        text = "Alpha beta. Gamma delta."
        self.assertEqual(split_long_text(text, 14), ["Alpha beta.", "Gamma delta."])


class StubTransport:
    def __init__(self, content=b"hello", status=200, etag=None):
        self.content, self.status, self.etag = content, status, etag
        self.calls = 0
        self.urls: list[str] = []

    def get(self, url, *, etag=None):
        self.calls += 1
        self.urls.append(url)
        return FetchResult(self.content, "text/html", self.status, self.etag)


class TestIngestionPolicy(unittest.TestCase):
    def test_host_not_on_allowlist_is_refused(self):
        fetcher = PolicyAwareFetcher(StubTransport(), allowed_hosts={"arxiv.org"})
        with self.assertRaises(PolicyError):
            fetcher.get("https://example.com/paper.pdf")

    def test_robots_disallow_is_refused(self):
        fetcher = PolicyAwareFetcher(
            StubTransport(), allowed_hosts={"arxiv.org"}, robots_check=lambda _: False
        )
        with self.assertRaises(PolicyError):
            fetcher.get("https://arxiv.org/abs/1")

    def test_rate_limit_sleeps_between_requests_to_one_host(self):
        slept = []
        clock = iter([0.0, 0.0, 0.1, 0.1, 0.2])
        fetcher = PolicyAwareFetcher(
            StubTransport(),
            allowed_hosts={"arxiv.org"},
            min_interval_seconds=1.0,
            clock=lambda: next(clock),
            sleep=slept.append,
        )
        fetcher.get("https://arxiv.org/abs/1")
        fetcher.get("https://arxiv.org/abs/2")
        self.assertTrue(slept and slept[0] > 0)

    def test_unknown_licence_does_not_permit_storage(self):
        self.assertFalse(licence_permits_storage(None))
        self.assertFalse(licence_permits_storage("METADATA_ONLY"))
        self.assertTrue(licence_permits_storage("cc-by-4.0"))

    def test_unchanged_content_produces_no_new_version(self):
        source = register_source(
            "https://arxiv.org/abs/1", source_type=SourceType.PRIMARY_PAPER, licence="MIT"
        )
        transport = StubTransport(content=b"same bytes")
        first, _ = fetch_version(source, transport)
        self.assertIsNotNone(first)
        second, content = fetch_version(source, transport, previous=first)
        self.assertIsNone(second)
        self.assertIsNone(content)

    def test_changed_content_increments_the_version(self):
        source = register_source(
            "https://arxiv.org/abs/1", source_type=SourceType.PRIMARY_PAPER, licence="MIT"
        )
        first, _ = fetch_version(source, StubTransport(content=b"v1"))
        second, content = fetch_version(source, StubTransport(content=b"v2"), previous=first)
        self.assertEqual(second.version_number, 2)
        self.assertNotEqual(first.source_version_id, second.source_version_id)
        self.assertEqual(content, b"v2")

    def test_304_means_unchanged(self):
        source = register_source(
            "https://arxiv.org/abs/1", source_type=SourceType.PRIMARY_PAPER, licence="MIT"
        )
        previous = SourceVersion(source_id=source.source_id, content_hash="0" * 64, etag="W/x")
        version, content = fetch_version(source, StubTransport(status=304), previous=previous)
        self.assertIsNone(version)
        self.assertIsNone(content)

    def test_arxiv_abstract_url_is_fetched_as_pdf(self):
        source = register_source(
            "https://arxiv.org/abs/2404.16130", source_type=SourceType.PRIMARY_PAPER, licence="MIT"
        )
        transport = StubTransport(content=b"%PDF-1.4 fake pdf bytes")
        fetch_version(source, transport)
        self.assertEqual(transport.urls, ["https://arxiv.org/pdf/2404.16130"])

    def test_metadata_only_source_is_not_storage_permitted(self):
        source = register_source(
            "https://arxiv.org/abs/1",
            source_type=SourceType.PRIMARY_PAPER,
            licence="METADATA_ONLY",
        )
        self.assertFalse(source.storage_permitted)
        self.assertIs(source.ingestion_status, IngestionStatus.DISCOVERED)


class TestReviewBoundary(unittest.TestCase):
    def _candidate(self, **kwargs) -> Claim:
        base = dict(
            source_version_id="v",
            source_id="s",
            claim_text="Graph retrieval reaches 0.62 F1.",
            claim_type=ClaimType.PERFORMANCE,
            source_url="https://example.org/a",
            extractor_name="llm",
            extractor_version="m/1",
            task="multi_hop_qa",
            method="graphrag",
            metric="f1",
            benchmark="hotpotqa",
            condition_text="top-20",
            extraction_confidence=0.9,
        )
        base.update(kwargs)
        return Claim(**base)

    def test_low_confidence_is_queued_high(self):
        items = build_claim_queue([self._candidate(extraction_confidence=0.2)], Config())
        self.assertEqual(items[0].priority, "HIGH")
        self.assertIn("LOW_CONFIDENCE", items[0].reason)

    def test_a_claim_is_queued_once_with_every_reason(self):
        claim = self._candidate(
            extraction_confidence=0.2,
            benchmark=None,
            missing_field_reason="not stated",
        )
        items = build_claim_queue([claim], Config())
        self.assertEqual(len(items), 1)
        self.assertIn("LOW_CONFIDENCE", items[0].reason)
        self.assertIn("MISSING_SCOPE", items[0].reason)

    def test_acceptance_requires_a_reviewer(self):
        with self.assertRaises(ReviewError):
            apply_claim_decision(self._candidate(), decision="ACCEPTED", reviewer="")

    def test_acceptance_makes_a_claim_runtime_visible(self):
        claim = apply_claim_decision(self._candidate(), decision="ACCEPTED", reviewer="alice")
        self.assertIs(claim.review_status, ReviewStatus.REVIEWED)
        self.assertTrue(claim.is_runtime_visible)

    def test_rejection_requires_a_note(self):
        with self.assertRaises(ReviewError):
            apply_claim_decision(self._candidate(), decision="REJECTED", reviewer="alice")

    def test_amendment_cannot_rewrite_the_claim_text(self):
        with self.assertRaises(ReviewError):
            apply_claim_decision(
                self._candidate(),
                decision="AMENDED",
                reviewer="alice",
                note="fixing",
                amendments={"claim_text": "something else entirely"},
            )

    def test_amendment_updates_scope_fields(self):
        claim = apply_claim_decision(
            self._candidate(),
            decision="AMENDED",
            reviewer="alice",
            note="benchmark was MuSiQue",
            amendments={"benchmark": "musique"},
        )
        self.assertEqual(claim.benchmark, "musique")
        self.assertIs(claim.review_status, ReviewStatus.REVIEWED)

    def test_a_decided_claim_cannot_be_decided_again(self):
        claim = apply_claim_decision(self._candidate(), decision="ACCEPTED", reviewer="alice")
        with self.assertRaises(ReviewError):
            apply_claim_decision(claim, decision="REJECTED", reviewer="bob", note="no")

    def test_backlog_counts_open_items_only(self):
        items = build_claim_queue([self._candidate(extraction_confidence=0.2)], Config())
        self.assertEqual(backlog(items)["HIGH"], 1)


class TestProposals(unittest.TestCase):
    def test_proposal_is_pending_approval(self):
        proposal = build_proposal(
            "REVIEW_CLAIM",
            {"claim_id": "clm-1", "requested_action": "verify benchmark"},
            created_by="agent",
            rationale="scope incomplete",
            known_claim_ids={"clm-1"},
        )
        self.assertEqual(proposal.status, "PENDING_APPROVAL")

    def test_citing_an_unretrieved_claim_is_refused(self):
        with self.assertRaises(ProposalValidationError) as ctx:
            build_proposal(
                "REVIEW_CLAIM",
                {"claim_id": "clm-hallucinated", "requested_action": "check"},
                created_by="agent",
                rationale="because",
                known_claim_ids={"clm-1"},
            )
        self.assertIn("clm-hallucinated", str(ctx.exception))

    def test_nested_claim_ids_are_checked(self):
        with self.assertRaises(ProposalValidationError):
            build_proposal(
                "OPEN_QUESTION",
                {"question_text": "q", "supporting_claim_ids": ["clm-1", "clm-ghost"]},
                created_by="agent",
                rationale="r",
                known_claim_ids={"clm-1"},
            )

    def test_missing_required_field_is_refused(self):
        with self.assertRaises(ProposalValidationError):
            build_proposal(
                "INGEST_SOURCE",
                {"canonical_url": "https://arxiv.org/abs/9"},
                created_by="agent",
                rationale="r",
            )

    def test_relative_url_is_refused(self):
        with self.assertRaises(ProposalValidationError):
            build_proposal(
                "INGEST_SOURCE",
                {"canonical_url": "/abs/9", "source_type": "PRIMARY_PAPER", "why_relevant": "x"},
                created_by="agent",
                rationale="r",
            )

    def test_rationale_is_mandatory(self):
        with self.assertRaises(ProposalValidationError):
            build_proposal(
                "REVIEW_CLAIM",
                {"claim_id": "clm-1", "requested_action": "x"},
                created_by="agent",
                rationale="   ",
            )


if __name__ == "__main__":
    unittest.main()
