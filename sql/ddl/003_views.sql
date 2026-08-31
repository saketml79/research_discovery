-- Research Discovery Engine :: Genie runtime views
-- These four views ARE the agent's runtime surface. The agent is attached to the
-- views, not to the base tables, so unreviewed records cannot reach an answer.
-- Parameters: :catalog, :schema

USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA IDENTIFIER(:schema);

-- ---------------------------------------------------------------------------
-- v_research_claim_current: reviewed, non-superseded claims with source context.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_research_claim_current
COMMENT 'Reviewed, current claims with full citation context. Grain: one row per claim. ONLY these claims may be used for an affirmative research statement. Always return source_url, page_number and reviewed_at with any claim from this view.'
AS
SELECT
  c.claim_id,
  c.claim_text,
  c.claim_type,
  c.task,
  c.method,
  c.metric,
  c.metric_value,
  c.metric_unit,
  c.benchmark,
  c.condition_text,
  c.evidence_excerpt,
  c.page_number,
  c.source_id,
  c.source_url,
  s.title            AS source_title,
  s.source_type,
  s.publisher,
  s.published_at,
  sv.version_number  AS source_version_number,
  sv.retrieved_at,
  c.extraction_confidence,
  c.review_status,
  c.reviewed_at,
  c.reviewed_by
FROM research_claim        AS c
JOIN research_source_version AS sv ON c.source_version_id = sv.source_version_id
JOIN research_source       AS s  ON c.source_id = s.source_id
WHERE c.review_status = 'REVIEWED'
  AND c.superseded_by_claim_id IS NULL
  AND sv.is_current;

-- ---------------------------------------------------------------------------
-- v_research_claim_candidate: unreviewed claims, explicitly labelled.
-- Exposed so the agent can say "a candidate claim exists but is unreviewed"
-- instead of silently omitting it.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_research_claim_candidate
COMMENT 'Extracted but NOT yet reviewed claims. Grain: one row per candidate claim. Never present these as findings. They may only be named as unverified leads or as review recommendations.'
AS
SELECT
  c.claim_id,
  c.claim_text,
  c.task,
  c.method,
  c.metric,
  c.benchmark,
  c.condition_text,
  c.source_url,
  c.page_number,
  c.extraction_confidence,
  c.missing_field_reason,
  c.review_status,
  c.extracted_at,
  q.priority          AS review_priority,
  q.reason            AS review_reason,
  q.status            AS review_queue_status
FROM research_claim AS c
LEFT JOIN research_review_queue AS q
  ON q.target_type = 'CLAIM' AND q.target_id = c.claim_id AND q.status = 'OPEN'
WHERE c.review_status IN ('CANDIDATE', 'IN_REVIEW');

-- ---------------------------------------------------------------------------
-- v_claim_comparison: reviewed claim pairs with their comparability verdict.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_claim_comparison
COMMENT 'Pairs of reviewed claims with an explicit comparability verdict. Grain: one row per reviewed relationship. A pair whose comparability_status is INSUFFICIENT_EVIDENCE must be reported as "insufficient evidence to compare", naming missing_dimensions - it is NOT a disagreement.'
AS
SELECT
  r.relationship_id,
  r.relationship_type,
  r.comparability_status,
  r.comparability_score,
  r.missing_dimensions,
  r.rationale,
  a.claim_id      AS from_claim_id,
  a.claim_text    AS from_claim_text,
  a.metric_value  AS from_metric_value,
  a.source_url    AS from_source_url,
  a.source_type   AS from_source_type,
  b.claim_id      AS to_claim_id,
  b.claim_text    AS to_claim_text,
  b.metric_value  AS to_metric_value,
  b.source_url    AS to_source_url,
  b.source_type   AS to_source_type,
  COALESCE(a.task, b.task)           AS task,
  COALESCE(a.metric, b.metric)       AS metric,
  COALESCE(a.benchmark, b.benchmark) AS benchmark,
  r.reviewed_at
FROM research_claim_relationship AS r
JOIN v_research_claim_current AS a ON r.from_claim_id = a.claim_id
JOIN v_research_claim_current AS b ON r.to_claim_id   = b.claim_id
WHERE r.review_status = 'REVIEWED';

-- ---------------------------------------------------------------------------
-- v_research_open_questions: evidence-backed gaps, derived not invented.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_research_open_questions
COMMENT 'Open questions derived from the corpus itself. Grain: one row per gap. Each row is backed by counted claim evidence - the agent must never invent a research gap that does not appear here.'
AS
WITH unresolved AS (
  SELECT
    'UNRESOLVED_COMPARISON'                                   AS question_type,
    CONCAT('Claims about ', COALESCE(task, 'an unspecified task'),
           ' cannot be compared: missing ', missing_dimensions) AS question_text,
    task, metric, benchmark,
    COUNT(*)                                                  AS evidence_count,
    COLLECT_SET(from_claim_id)                                AS claim_ids
  FROM v_claim_comparison
  WHERE comparability_status = 'INSUFFICIENT_EVIDENCE'
  GROUP BY task, metric, benchmark, missing_dimensions
),
missing_scope AS (
  SELECT
    'MISSING_SCOPE'                                           AS question_type,
    CONCAT('Claims about ', COALESCE(method, 'an unspecified method'),
           ' report ', COALESCE(metric, 'no metric'),
           ' without a stated benchmark or condition')        AS question_text,
    task, metric, CAST(NULL AS STRING) AS benchmark,
    COUNT(*)                                                  AS evidence_count,
    COLLECT_SET(claim_id)                                     AS claim_ids
  FROM v_research_claim_current
  WHERE benchmark IS NULL OR condition_text IS NULL
  GROUP BY method, task, metric
),
single_source AS (
  SELECT
    'SINGLE_SOURCE_FINDING'                                   AS question_type,
    CONCAT('Only one source reports ', COALESCE(metric, 'this metric'),
           ' on ', COALESCE(benchmark, 'this benchmark'),
           ' - no independent replication in the corpus')     AS question_text,
    task, metric, benchmark,
    COUNT(*)                                                  AS evidence_count,
    COLLECT_SET(claim_id)                                     AS claim_ids
  FROM v_research_claim_current
  WHERE benchmark IS NOT NULL AND metric IS NOT NULL
  GROUP BY task, metric, benchmark
  HAVING COUNT(DISTINCT source_id) = 1
)
SELECT * FROM unresolved
UNION ALL SELECT * FROM missing_scope
UNION ALL SELECT * FROM single_source;

-- ---------------------------------------------------------------------------
-- v_source_coverage: corpus freshness and review backlog. Answers "what does
-- this corpus not cover", which is as important as what it does.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_source_coverage
COMMENT 'Corpus coverage, freshness and review backlog. Grain: one row per source_type. Use it to qualify any consensus statement with how much of the corpus is actually reviewed.'
AS
SELECT
  s.source_type,
  COUNT(DISTINCT s.source_id)                                                     AS source_count,
  COUNT(DISTINCT CASE WHEN s.ingestion_status = 'INDEXED' THEN s.source_id END)   AS indexed_source_count,
  COUNT(DISTINCT c.claim_id)                                                      AS claim_count,
  COUNT(DISTINCT CASE WHEN c.review_status = 'REVIEWED'  THEN c.claim_id END)     AS reviewed_claim_count,
  COUNT(DISTINCT CASE WHEN c.review_status = 'CANDIDATE' THEN c.claim_id END)     AS unreviewed_claim_count,
  COUNT(DISTINCT CASE WHEN ch.extraction_warning IS NOT NULL THEN ch.chunk_id END) AS chunks_with_parser_warning,
  MAX(sv.retrieved_at)                                                            AS most_recent_retrieval,
  MIN(sv.retrieved_at)                                                            AS oldest_retrieval
FROM research_source AS s
LEFT JOIN research_source_version AS sv ON s.source_id = sv.source_id AND sv.is_current
LEFT JOIN research_claim          AS c  ON c.source_id = s.source_id
LEFT JOIN research_chunk          AS ch ON ch.source_id = s.source_id
GROUP BY s.source_type;
