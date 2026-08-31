-- Research Discovery Engine :: core tables
-- Idempotent DDL. Every table is append-mostly with explicit provenance columns.
-- Column comments are load-bearing: the Genie Agent reads UC metadata at runtime.
-- Parameters: :catalog, :schema

USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA IDENTIFIER(:schema);

-- ---------------------------------------------------------------------------
-- 1. Source registry: one row per logical source (stable canonical URL).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_source (
  source_id           STRING  NOT NULL COMMENT 'Deterministic id derived from canonical_url. Stable across source versions.',
  canonical_url       STRING  NOT NULL COMMENT 'Stable, citable URL of the source. Used in every citation the agent returns.',
  title               STRING          COMMENT 'Source title as published.',
  source_type         STRING  NOT NULL COMMENT 'One of PRIMARY_PAPER, BENCHMARK_DOC, REPOSITORY, SECONDARY_BLOG, TALK_TRANSCRIPT. Primary vs secondary evidence must never be conflated.',
  publisher           STRING          COMMENT 'Publishing venue or organization (arXiv, ACL, vendor, author).',
  authors             STRING          COMMENT 'Comma-separated author list as published; not normalized to person entities.',
  published_at        TIMESTAMP       COMMENT 'Publication date reported by the source. NULL when the source does not state one.',
  license             STRING          COMMENT 'License or terms under which content may be stored (e.g. CC-BY-4.0, arXiv-nonexclusive, METADATA_ONLY).',
  storage_permitted   BOOLEAN NOT NULL COMMENT 'TRUE when full parsed text may be stored. FALSE means metadata plus short excerpts only.',
  ingestion_status    STRING  NOT NULL COMMENT 'Lifecycle state: DISCOVERED, FETCHED, PARSED, CHUNKED, EXTRACTED, REVIEWED, INDEXED, QUARANTINED.',
  registered_at       TIMESTAMP NOT NULL COMMENT 'When this source was first registered in the corpus.',
  updated_at          TIMESTAMP NOT NULL COMMENT 'Last state transition time for this source.',
  CONSTRAINT research_source_pk PRIMARY KEY (source_id)
)
USING DELTA
COMMENT 'Registry of research sources in the curated corpus. Grain: one row per canonical URL. Cite canonical_url, never a chunk id, in user-facing answers.'
TBLPROPERTIES (delta.enableChangeDataFeed = true);

-- ---------------------------------------------------------------------------
-- 2. Source versions: immutable snapshots. A changed content_hash is a new row.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_source_version (
  source_version_id   STRING  NOT NULL COMMENT 'Deterministic id from source_id + content_hash.',
  source_id           STRING  NOT NULL COMMENT 'FK to research_source.',
  version_number      INT     NOT NULL COMMENT 'Monotonic per source, starting at 1.',
  content_hash        STRING  NOT NULL COMMENT 'SHA-256 of the fetched bytes. Identical hash means the revision was already processed.',
  raw_content_uri     STRING          COMMENT 'UC volume path of the stored raw document. NULL when storage_permitted is FALSE.',
  content_type        STRING          COMMENT 'MIME type as fetched (application/pdf, text/html).',
  byte_size           BIGINT          COMMENT 'Size of the fetched payload in bytes.',
  http_status         INT             COMMENT 'HTTP status of the fetch, when fetched over HTTP.',
  etag                STRING          COMMENT 'ETag or Last-Modified value used for conditional refetch.',
  retrieved_at        TIMESTAMP NOT NULL COMMENT 'When these bytes were retrieved. This is the provenance timestamp for every derived record.',
  is_current          BOOLEAN NOT NULL COMMENT 'TRUE for the latest processed version of the source.',
  CONSTRAINT research_source_version_pk PRIMARY KEY (source_version_id),
  CONSTRAINT research_source_version_fk FOREIGN KEY (source_id) REFERENCES research_source (source_id)
)
USING DELTA
COMMENT 'Immutable fetched revisions of each source. Grain: one row per (source, content_hash). Never updated in place; a changed source creates a new version.';

-- ---------------------------------------------------------------------------
-- 3. Document chunks: retrieval corpus. Kept separate from extracted claims.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_chunk (
  chunk_id            STRING  NOT NULL COMMENT 'Deterministic id from source_version_id + chunk_index.',
  source_version_id   STRING  NOT NULL COMMENT 'FK to research_source_version.',
  source_id           STRING  NOT NULL COMMENT 'Denormalized FK to research_source for filter pushdown and index metadata.',
  chunk_index         INT     NOT NULL COMMENT 'Zero-based ordinal within the parsed document.',
  page_number         INT             COMMENT 'One-based page number. Never NULL for PDF-derived chunks; the page reference is required provenance.',
  section_title       STRING          COMMENT 'Nearest preceding heading, when the parser recovered document structure.',
  block_type          STRING  NOT NULL COMMENT 'TEXT, TABLE, FIGURE_CAPTION, ABSTRACT, REFERENCES. Tables and captions are never split from their page reference.',
  text                STRING  NOT NULL COMMENT 'Chunk text. Truncated to the permitted excerpt length when storage_permitted is FALSE.',
  char_count          INT     NOT NULL COMMENT 'Length of text in characters.',
  content_hash        STRING  NOT NULL COMMENT 'SHA-256 of the chunk text; used for idempotent re-parse.',
  parser_name         STRING  NOT NULL COMMENT 'Parser adapter that produced this chunk (pypdf, docling, ai_parse_document, html).',
  parser_version      STRING  NOT NULL COMMENT 'Version of the parser adapter. A parser upgrade is a re-parse, not a silent change.',
  extraction_warning  STRING          COMMENT 'Parser warning such as OCR_LOW_CONFIDENCE or LAYOUT_UNRECOVERED. Visible to reviewers.',
  lifecycle_state     STRING  NOT NULL COMMENT 'CHUNKED, INDEXED, SUPERSEDED or QUARANTINED.',
  created_at          TIMESTAMP NOT NULL COMMENT 'When the chunk was written.',
  CONSTRAINT research_chunk_pk PRIMARY KEY (chunk_id),
  CONSTRAINT research_chunk_fk FOREIGN KEY (source_version_id) REFERENCES research_source_version (source_version_id)
)
USING DELTA
COMMENT 'Parsed document chunks used as the retrieval corpus. Grain: one row per (source version, chunk index). Retrieval evidence lives here; comparable claims live in research_claim.'
TBLPROPERTIES (delta.enableChangeDataFeed = true);

-- ---------------------------------------------------------------------------
-- 4. Claims: the comparison unit. Scope fields are what make comparison legal.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_claim (
  claim_id            STRING  NOT NULL COMMENT 'Deterministic id from source_version_id + normalized claim_text.',
  source_version_id   STRING  NOT NULL COMMENT 'FK to research_source_version. Pins the claim to the exact revision it was read from.',
  source_id           STRING  NOT NULL COMMENT 'Denormalized FK to research_source.',
  chunk_id            STRING          COMMENT 'FK to the chunk the evidence excerpt came from. NULL for a claim read from a figure.',
  figure_id           STRING          COMMENT 'FK to research_figure when this claim was read from a chart or diagram rather than from text. A claim with a figure_id is a visual interpretation and must be reported with its confidence.',
  claim_text          STRING  NOT NULL COMMENT 'The asserted finding in one sentence, as stated by the source. Not the paper topic.',
  claim_type          STRING  NOT NULL COMMENT 'PERFORMANCE, LIMITATION, METHOD_DESCRIPTION, RESOURCE_COST, NEGATIVE_RESULT, RECOMMENDATION.',
  task                STRING          COMMENT 'Task the claim is about (multi_hop_qa, summarization). NULL means the source did not state one.',
  method              STRING          COMMENT 'Method or system the claim is about (graphrag_local, vector_rag).',
  metric              STRING          COMMENT 'Metric reported (comprehensiveness_win_rate, f1, recall_at_20).',
  metric_value        DOUBLE          COMMENT 'Numeric value as reported. Never inferred, never rounded from prose.',
  metric_unit         STRING          COMMENT 'Unit of metric_value (percent, ratio, usd, seconds).',
  benchmark           STRING          COMMENT 'Benchmark or dataset the result was measured on (hotpotqa, musique, podcast_transcripts).',
  condition_text      STRING          COMMENT 'Scope conditions: corpus size, model, budget, retriever settings. Comparability depends on this.',
  evidence_excerpt    STRING          COMMENT 'Short verbatim excerpt supporting the claim. Length-capped; never a full-text reproduction.',
  page_number         INT             COMMENT 'Page the excerpt came from.',
  source_url          STRING  NOT NULL COMMENT 'Canonical URL, denormalized so every claim is independently citable.',
  extractor_name      STRING  NOT NULL COMMENT 'Extractor adapter (llm, ai_extract, heuristic, manual).',
  extractor_version   STRING  NOT NULL COMMENT 'Version or model identifier of the extractor.',
  extraction_confidence DOUBLE        COMMENT 'Extractor self-reported confidence in [0,1]. A ranking signal for reviewers, not a truth value.',
  missing_field_reason STRING         COMMENT 'Why task/metric/benchmark/condition are NULL. Required when any is NULL.',
  extracted_at        TIMESTAMP NOT NULL COMMENT 'When extraction ran.',
  review_status       STRING  NOT NULL COMMENT 'CANDIDATE, IN_REVIEW, REVIEWED, REJECTED, SUPERSEDED. Only REVIEWED claims may support a consensus or contradiction statement.',
  reviewed_by         STRING          COMMENT 'Reviewer principal.',
  reviewed_at         TIMESTAMP       COMMENT 'Review decision time.',
  review_note         STRING          COMMENT 'Reviewer amendment or rejection rationale.',
  superseded_by_claim_id STRING       COMMENT 'Set when a newer source version replaced this claim.',
  CONSTRAINT research_claim_pk PRIMARY KEY (claim_id),
  CONSTRAINT research_claim_fk FOREIGN KEY (source_version_id) REFERENCES research_source_version (source_version_id)
)
USING DELTA
COMMENT 'Structured claims extracted from sources. Grain: one row per distinct claim per source version. A claim is unverified until review_status = REVIEWED.'
TBLPROPERTIES (delta.enableChangeDataFeed = true);

-- ---------------------------------------------------------------------------
-- 4b. Figures: evidence read from a chart, diagram or slide by a vision model.
--     Kept in its own table because a visual interpretation is a weaker kind of
--     evidence than stated text and must never be silently mixed with it.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_figure (
  figure_id           STRING  NOT NULL COMMENT 'Deterministic id from source_version_id + page_number + figure_index.',
  source_version_id   STRING  NOT NULL COMMENT 'FK to research_source_version.',
  source_id           STRING  NOT NULL COMMENT 'Denormalized FK to research_source.',
  page_number         INT     NOT NULL COMMENT 'One-based page the figure appears on. Never NULL: an image without a page reference is not citable evidence.',
  figure_index        INT     NOT NULL COMMENT 'Zero-based ordinal of the figure within the page.',
  figure_type         STRING  NOT NULL COMMENT 'CHART, DIAGRAM, TABLE_IMAGE, SCREENSHOT or SLIDE.',
  caption             STRING          COMMENT 'Caption text as printed, when the parser recovered one.',
  image_uri           STRING          COMMENT 'UC volume path of the cropped page image. NULL when the licence forbids storing it.',
  bounding_box        STRING          COMMENT 'JSON [x0,y0,x1,y1] in page coordinates, locating the figure for a reviewer.',
  extracted_text      STRING          COMMENT 'What the vision model read from the image: axis labels, series names, values.',
  extracted_entities  STRING          COMMENT 'JSON array of typed entities and relationships the model recovered.',
  extraction_confidence DOUBLE        COMMENT 'Vision model confidence in [0,1]. A chart reading is an interpretation; this is how sure the model is that it read the picture correctly, never that the underlying result is true.',
  vision_model        STRING  NOT NULL COMMENT 'Model or endpoint that produced the interpretation, with version. A model change is a re-extraction.',
  prompt_version      STRING  NOT NULL COMMENT 'Version of the vision prompt, so a reading can be replayed.',
  review_status       STRING  NOT NULL COMMENT 'CANDIDATE, REVIEWED or REJECTED. A figure reading is unverified until a human confirms it against the image.',
  reviewed_by         STRING          COMMENT 'Reviewer principal.',
  reviewed_at         TIMESTAMP       COMMENT 'Review decision time.',
  extracted_at        TIMESTAMP NOT NULL COMMENT 'When the vision extraction ran.',
  CONSTRAINT research_figure_pk PRIMARY KEY (figure_id),
  CONSTRAINT research_figure_fk FOREIGN KEY (source_version_id) REFERENCES research_source_version (source_version_id)
)
USING DELTA
COMMENT 'Figure and chart evidence recovered by a vision model. Grain: one row per figure per source version. A number read from a chart is a visual interpretation: it carries a confidence and a stored image reference so a reviewer can check it against the original.';

-- ---------------------------------------------------------------------------
-- 5. Claim relationships: typed, comparability-gated edges.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_claim_relationship (
  relationship_id     STRING  NOT NULL COMMENT 'Deterministic id from the ordered claim pair and relationship_type.',
  from_claim_id       STRING  NOT NULL COMMENT 'FK to research_claim.',
  to_claim_id         STRING  NOT NULL COMMENT 'FK to research_claim.',
  relationship_type   STRING  NOT NULL COMMENT 'SUPPORTS, CONTRADICTS, REFINES, DUPLICATES or NOT_COMPARABLE_YET. CONTRADICTS requires comparability_status = COMPARABLE.',
  comparability_status STRING NOT NULL COMMENT 'COMPARABLE, PARTIALLY_COMPARABLE or INSUFFICIENT_EVIDENCE.',
  comparability_score DOUBLE          COMMENT 'Weighted overlap of task, metric, benchmark and condition in [0,1]. A gate, not a probability of truth.',
  missing_dimensions  STRING          COMMENT 'Comma-separated scope fields absent on one or both claims (task, metric, benchmark, condition).',
  rationale           STRING          COMMENT 'Plain-language explanation of the edge and its comparability decision.',
  detector_name       STRING  NOT NULL COMMENT 'Component that proposed the edge (rule_comparability_v1, llm, manual).',
  created_at          TIMESTAMP NOT NULL COMMENT 'When the edge was proposed.',
  review_status       STRING  NOT NULL COMMENT 'CANDIDATE, REVIEWED or REJECTED.',
  reviewed_by         STRING          COMMENT 'Reviewer principal.',
  reviewed_at         TIMESTAMP       COMMENT 'Review decision time.',
  CONSTRAINT research_claim_rel_pk PRIMARY KEY (relationship_id)
)
USING DELTA
COMMENT 'Typed relationships between claims. Grain: one row per ordered claim pair and type. An apparent conflict without comparable scope is NOT_COMPARABLE_YET, never CONTRADICTS.';

-- ---------------------------------------------------------------------------
-- 6. Review queue: the boundary between candidate and runtime knowledge.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_review_queue (
  review_id           STRING  NOT NULL COMMENT 'Deterministic id from target_type + target_id + created_at.',
  target_type         STRING  NOT NULL COMMENT 'CLAIM or RELATIONSHIP.',
  target_id           STRING  NOT NULL COMMENT 'claim_id or relationship_id under review.',
  priority            STRING  NOT NULL COMMENT 'HIGH, NORMAL or LOW. HIGH is used for low-confidence extractions and contradiction candidates.',
  reason              STRING  NOT NULL COMMENT 'Why the item needs review (LOW_CONFIDENCE, MISSING_SCOPE, CONTRADICTION_CANDIDATE, SOURCE_UPDATED, PARSER_WARNING).',
  status              STRING  NOT NULL COMMENT 'OPEN, ACCEPTED, AMENDED, REJECTED.',
  assigned_to         STRING          COMMENT 'Reviewer principal, when assigned.',
  created_at          TIMESTAMP NOT NULL COMMENT 'When the item entered the queue.',
  resolved_at         TIMESTAMP       COMMENT 'When the reviewer decided.',
  resolution_note     STRING          COMMENT 'Reviewer note recorded with the decision.',
  CONSTRAINT research_review_queue_pk PRIMARY KEY (review_id)
)
USING DELTA
COMMENT 'Human review queue. Grain: one row per review request. Nothing reaches the agent runtime surface without an ACCEPTED or AMENDED decision here.';

-- ---------------------------------------------------------------------------
-- 7. Controlled vocabulary. Free-text scope fields are normalized against this.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_taxonomy (
  term_id             STRING  NOT NULL COMMENT 'Deterministic id from dimension + canonical_term.',
  dimension           STRING  NOT NULL COMMENT 'TASK, METHOD, METRIC, BENCHMARK or CONDITION.',
  canonical_term      STRING  NOT NULL COMMENT 'Preferred normalized term used in research_claim scope columns.',
  synonyms            STRING          COMMENT 'Comma-separated surface forms mapped to canonical_term.',
  definition          STRING          COMMENT 'Short definition so the agent and reviewers agree on the term.',
  parent_term_id      STRING          COMMENT 'Optional parent for hierarchical dimensions (e.g. a benchmark family).',
  CONSTRAINT research_taxonomy_pk PRIMARY KEY (term_id)
)
USING DELTA
COMMENT 'Controlled vocabulary for claim scope fields. Grain: one row per canonical term per dimension.';

-- ---------------------------------------------------------------------------
-- 8. Agent proposals. Write-only surface. No execution path exists.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_proposal (
  proposal_id         STRING  NOT NULL COMMENT 'Generated proposal id.',
  investigation_id    STRING          COMMENT 'Correlates the proposal with the question that produced it.',
  proposal_type       STRING  NOT NULL COMMENT 'REVIEW_CLAIM, INGEST_SOURCE, RESOLVE_CONTRADICTION, OPEN_QUESTION.',
  payload_json        STRING  NOT NULL COMMENT 'Structured proposal body. Validated against the tool contract before insert.',
  rationale           STRING          COMMENT 'Why the agent proposed this, with evidence references.',
  status              STRING  NOT NULL COMMENT 'Always PENDING_APPROVAL on insert. Only a human workflow may advance it.',
  created_by          STRING  NOT NULL COMMENT 'Agent or principal that created the proposal.',
  created_at          TIMESTAMP NOT NULL COMMENT 'Creation time.',
  approved_by         STRING          COMMENT 'Approver principal.',
  approved_at         TIMESTAMP       COMMENT 'Approval time.',
  CONSTRAINT agent_proposal_pk PRIMARY KEY (proposal_id)
)
USING DELTA
COMMENT 'Approval-gated proposals written by the agent. Inserts are PENDING_APPROVAL only; the agent has no tool that executes a proposal.';

-- CHECK constraints are not supported inline in CREATE TABLE on all SQL warehouse versions; add separately, idempotently.
ALTER TABLE agent_proposal DROP CONSTRAINT IF EXISTS agent_proposal_status_chk;
ALTER TABLE agent_proposal ADD CONSTRAINT agent_proposal_status_chk CHECK (status IN ('PENDING_APPROVAL', 'APPROVED', 'REJECTED'));

-- ---------------------------------------------------------------------------
-- 9. Pipeline run log. Makes ingestion behaviour queryable by the agent.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_run (
  run_id              STRING  NOT NULL COMMENT 'Generated run id.',
  stage               STRING  NOT NULL COMMENT 'REGISTER, FETCH, PARSE, CHUNK, EXTRACT, RELATE, INDEX.',
  status              STRING  NOT NULL COMMENT 'STARTED, SUCCEEDED, FAILED, PARTIAL.',
  records_in          BIGINT          COMMENT 'Records read.',
  records_out         BIGINT          COMMENT 'Records written.',
  records_quarantined BIGINT          COMMENT 'Records routed to QUARANTINED.',
  error_text          STRING          COMMENT 'Truncated error, when status is FAILED or PARTIAL.',
  started_at          TIMESTAMP NOT NULL COMMENT 'Stage start.',
  finished_at         TIMESTAMP       COMMENT 'Stage end.',
  CONSTRAINT pipeline_run_pk PRIMARY KEY (run_id)
)
USING DELTA
COMMENT 'Per-stage pipeline telemetry. Grain: one row per stage execution. Lets the agent report corpus freshness and review backlog honestly.';
