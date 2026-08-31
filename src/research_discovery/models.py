"""Domain records.

Plain dataclasses with validation in ``__post_init__``. They are the contract
between parsers, extractors, review logic and the Delta writers, and they carry
no Spark dependency so the whole pipeline is unit-testable off-cluster.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .config import MAX_EVIDENCE_EXCERPT_CHARS
from . import ids


def utcnow() -> datetime:
    """Timezone-aware current UTC time. Never use ``datetime.now()`` directly."""
    return datetime.now(timezone.utc)


class SourceType(str, Enum):
    """Evidence tier of a source. Primary and secondary are never conflated."""

    PRIMARY_PAPER = "PRIMARY_PAPER"
    BENCHMARK_DOC = "BENCHMARK_DOC"
    REPOSITORY = "REPOSITORY"
    SECONDARY_BLOG = "SECONDARY_BLOG"
    TALK_TRANSCRIPT = "TALK_TRANSCRIPT"


class IngestionStatus(str, Enum):
    """Source lifecycle. Transitions are enforced by ``ALLOWED_TRANSITIONS``."""

    DISCOVERED = "DISCOVERED"
    FETCHED = "FETCHED"
    PARSED = "PARSED"
    CHUNKED = "CHUNKED"
    EXTRACTED = "EXTRACTED"
    REVIEWED = "REVIEWED"
    INDEXED = "INDEXED"
    QUARANTINED = "QUARANTINED"


#: Legal forward transitions. Any source may be quarantined from any state.
ALLOWED_TRANSITIONS: dict[IngestionStatus, frozenset[IngestionStatus]] = {
    IngestionStatus.DISCOVERED: frozenset({IngestionStatus.FETCHED}),
    IngestionStatus.FETCHED: frozenset({IngestionStatus.PARSED}),
    IngestionStatus.PARSED: frozenset({IngestionStatus.CHUNKED}),
    IngestionStatus.CHUNKED: frozenset({IngestionStatus.EXTRACTED}),
    IngestionStatus.EXTRACTED: frozenset({IngestionStatus.REVIEWED}),
    IngestionStatus.REVIEWED: frozenset({IngestionStatus.INDEXED}),
    IngestionStatus.INDEXED: frozenset({IngestionStatus.FETCHED}),  # refresh
    IngestionStatus.QUARANTINED: frozenset({IngestionStatus.FETCHED}),
}


def can_transition(current: IngestionStatus, target: IngestionStatus) -> bool:
    """True when ``current -> target`` is a legal lifecycle transition."""
    if target is IngestionStatus.QUARANTINED:
        return True
    return target in ALLOWED_TRANSITIONS.get(current, frozenset())


class ReviewStatus(str, Enum):
    """Review state of a claim or relationship."""

    CANDIDATE = "CANDIDATE"
    IN_REVIEW = "IN_REVIEW"
    REVIEWED = "REVIEWED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class ClaimType(str, Enum):
    """What kind of assertion a claim makes."""

    PERFORMANCE = "PERFORMANCE"
    LIMITATION = "LIMITATION"
    METHOD_DESCRIPTION = "METHOD_DESCRIPTION"
    RESOURCE_COST = "RESOURCE_COST"
    NEGATIVE_RESULT = "NEGATIVE_RESULT"
    RECOMMENDATION = "RECOMMENDATION"


class RelationshipType(str, Enum):
    """Typed edge between two claims."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REFINES = "REFINES"
    DUPLICATES = "DUPLICATES"
    NOT_COMPARABLE_YET = "NOT_COMPARABLE_YET"


class ComparabilityStatus(str, Enum):
    """Whether two claims may legitimately be compared at all."""

    COMPARABLE = "COMPARABLE"
    PARTIALLY_COMPARABLE = "PARTIALLY_COMPARABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class BlockType(str, Enum):
    """Structural role of a parsed block."""

    TEXT = "TEXT"
    TABLE = "TABLE"
    FIGURE_CAPTION = "FIGURE_CAPTION"
    ABSTRACT = "ABSTRACT"
    REFERENCES = "REFERENCES"


@dataclass(slots=True)
class Source:
    """A logical research source, identified by its canonical URL."""

    canonical_url: str
    source_type: SourceType
    title: str | None = None
    publisher: str | None = None
    authors: str | None = None
    published_at: datetime | None = None
    license: str | None = None
    storage_permitted: bool = False
    ingestion_status: IngestionStatus = IngestionStatus.DISCOVERED
    source_id: str = ""
    registered_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not self.canonical_url.startswith(("http://", "https://")):
            raise ValueError(f"canonical_url must be an absolute http(s) URL: {self.canonical_url!r}")
        self.source_type = SourceType(self.source_type)
        self.ingestion_status = IngestionStatus(self.ingestion_status)
        self.canonical_url = ids.normalize_url(self.canonical_url)
        self.source_id = self.source_id or ids.source_id(self.canonical_url)

    def advance(self, target: IngestionStatus) -> "Source":
        """Move the source to ``target``, rejecting illegal transitions."""
        if not can_transition(self.ingestion_status, target):
            raise ValueError(f"illegal transition {self.ingestion_status.value} -> {target.value}")
        self.ingestion_status = target
        self.updated_at = utcnow()
        return self


@dataclass(slots=True)
class SourceVersion:
    """One immutable fetched revision of a source."""

    source_id: str
    content_hash: str
    retrieved_at: datetime = field(default_factory=utcnow)
    version_number: int = 1
    raw_content_uri: str | None = None
    content_type: str | None = None
    byte_size: int | None = None
    http_status: int | None = None
    etag: str | None = None
    is_current: bool = True
    source_version_id: str = ""

    def __post_init__(self) -> None:
        if len(self.content_hash) != 64:
            raise ValueError("content_hash must be a hex SHA-256 digest")
        if self.version_number < 1:
            raise ValueError("version_number starts at 1")
        self.source_version_id = self.source_version_id or ids.source_version_id(
            self.source_id, self.content_hash
        )


@dataclass(slots=True)
class Chunk:
    """A parsed block of a document, carrying its page provenance."""

    source_version_id: str
    source_id: str
    chunk_index: int
    text: str
    block_type: BlockType = BlockType.TEXT
    page_number: int | None = None
    section_title: str | None = None
    parser_name: str = "unknown"
    parser_version: str = "0"
    extraction_warning: str | None = None
    lifecycle_state: str = "CHUNKED"
    chunk_id: str = ""
    content_hash: str = ""
    char_count: int = 0
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("chunk text must not be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        self.block_type = BlockType(self.block_type)
        self.char_count = len(self.text)
        self.content_hash = self.content_hash or ids.sha256_hex(ids.normalize_text(self.text))
        self.chunk_id = self.chunk_id or ids.chunk_id(self.source_version_id, self.chunk_index)


@dataclass(slots=True)
class Claim:
    """A structured, comparable assertion extracted from one source version.

    The scope fields (``task``, ``method``, ``metric``, ``benchmark``,
    ``condition_text``) are what make two claims comparable. When any is
    ``None``, ``missing_field_reason`` is mandatory: an absent field must be an
    explicit statement of ignorance, never a silent gap.
    """

    source_version_id: str
    source_id: str
    claim_text: str
    claim_type: ClaimType
    source_url: str
    extractor_name: str
    extractor_version: str
    chunk_id: str | None = None
    figure_id: str | None = None
    task: str | None = None
    method: str | None = None
    metric: str | None = None
    metric_value: float | None = None
    metric_unit: str | None = None
    benchmark: str | None = None
    condition_text: str | None = None
    evidence_excerpt: str | None = None
    page_number: int | None = None
    extraction_confidence: float | None = None
    missing_field_reason: str | None = None
    extracted_at: datetime = field(default_factory=utcnow)
    review_status: ReviewStatus = ReviewStatus.CANDIDATE
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None
    superseded_by_claim_id: str | None = None
    claim_id: str = ""

    SCOPE_FIELDS = ("task", "method", "metric", "benchmark", "condition_text")

    def __post_init__(self) -> None:
        if not self.claim_text.strip():
            raise ValueError("claim_text must not be empty")
        self.claim_type = ClaimType(self.claim_type)
        self.review_status = ReviewStatus(self.review_status)
        if self.extraction_confidence is not None and not 0.0 <= self.extraction_confidence <= 1.0:
            raise ValueError("extraction_confidence must be in [0, 1]")
        if self.evidence_excerpt:
            self.evidence_excerpt = self.evidence_excerpt[:MAX_EVIDENCE_EXCERPT_CHARS]
        if self.missing_scope_fields() and not self.missing_field_reason:
            raise ValueError(
                "missing_field_reason is required when a scope field is absent: "
                + ", ".join(self.missing_scope_fields())
            )
        if self.review_status is ReviewStatus.REVIEWED and not self.reviewed_by:
            raise ValueError("a REVIEWED claim must record reviewed_by")
        if self.chunk_id and self.figure_id:
            raise ValueError(
                "a claim is read from text or from a figure, not both; set chunk_id or figure_id"
            )
        if self.figure_id and self.extraction_confidence is None:
            # A visual interpretation without a stated confidence cannot be
            # reported honestly, so it may not be persisted at all.
            raise ValueError("a figure-derived claim must record extraction_confidence")
        self.claim_id = self.claim_id or ids.claim_id(self.source_version_id, self.claim_text)

    def missing_scope_fields(self) -> tuple[str, ...]:
        """Scope fields that the extractor could not fill."""
        return tuple(f for f in self.SCOPE_FIELDS if getattr(self, f) in (None, ""))

    @property
    def is_runtime_visible(self) -> bool:
        """True when this claim may back an affirmative statement."""
        return self.review_status is ReviewStatus.REVIEWED and self.superseded_by_claim_id is None


@dataclass(slots=True)
class ClaimRelationship:
    """A typed, comparability-gated edge between two claims."""

    from_claim_id: str
    to_claim_id: str
    relationship_type: RelationshipType
    comparability_status: ComparabilityStatus
    detector_name: str
    comparability_score: float | None = None
    missing_dimensions: str | None = None
    rationale: str | None = None
    review_status: ReviewStatus = ReviewStatus.CANDIDATE
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)
    relationship_id: str = ""

    def __post_init__(self) -> None:
        if self.from_claim_id == self.to_claim_id:
            raise ValueError("a claim cannot relate to itself")
        self.relationship_type = RelationshipType(self.relationship_type)
        self.comparability_status = ComparabilityStatus(self.comparability_status)
        self.review_status = ReviewStatus(self.review_status)
        if (
            self.relationship_type is RelationshipType.CONTRADICTS
            and self.comparability_status is ComparabilityStatus.INSUFFICIENT_EVIDENCE
        ):
            raise ValueError(
                "CONTRADICTS requires comparable scope; use NOT_COMPARABLE_YET instead"
            )
        self.relationship_id = self.relationship_id or ids.relationship_id(
            self.from_claim_id, self.to_claim_id, self.relationship_type.value
        )


@dataclass(slots=True)
class ReviewItem:
    """A request for a human decision."""

    target_type: str
    target_id: str
    reason: str
    priority: str = "NORMAL"
    status: str = "OPEN"
    assigned_to: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    resolved_at: datetime | None = None
    resolution_note: str | None = None
    review_id: str = ""

    def __post_init__(self) -> None:
        if self.target_type not in {"CLAIM", "RELATIONSHIP"}:
            raise ValueError("target_type must be CLAIM or RELATIONSHIP")
        if self.priority not in {"HIGH", "NORMAL", "LOW"}:
            raise ValueError("priority must be HIGH, NORMAL or LOW")
        if self.status not in {"OPEN", "ACCEPTED", "AMENDED", "REJECTED"}:
            raise ValueError("invalid review status")
        self.review_id = self.review_id or ids.review_id(self.target_type, self.target_id)


@dataclass(slots=True)
class Proposal:
    """An approval-gated recommendation. Never executed by the system."""

    proposal_type: str
    payload_json: str
    created_by: str
    rationale: str | None = None
    investigation_id: str | None = None
    status: str = "PENDING_APPROVAL"
    created_at: datetime = field(default_factory=utcnow)
    approved_by: str | None = None
    approved_at: datetime | None = None
    proposal_id: str = ""

    VALID_TYPES = frozenset(
        {"REVIEW_CLAIM", "INGEST_SOURCE", "RESOLVE_CONTRADICTION", "OPEN_QUESTION"}
    )

    def __post_init__(self) -> None:
        if self.proposal_type not in self.VALID_TYPES:
            raise ValueError(f"proposal_type must be one of {sorted(self.VALID_TYPES)}")
        if self.status != "PENDING_APPROVAL":
            raise ValueError("proposals may only be created as PENDING_APPROVAL")
        self.proposal_id = self.proposal_id or ids.stable_id(
            "prp", self.proposal_type, self.payload_json, self.created_at.isoformat()
        )


def to_row(record: Any) -> dict[str, Any]:
    """Convert a dataclass record to a plain dict with enum values unwrapped."""
    row = asdict(record)
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in row.items()}
