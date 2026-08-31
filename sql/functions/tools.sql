-- Research Discovery Engine :: Unity Catalog function tools
-- Every retrieval tool is READ-ONLY. The only writing tool inserts
-- PENDING_APPROVAL proposals and cannot mutate corpus or platform state.
-- Function comments are the tool descriptions the agent sees - keep them precise.
-- Parameters: :catalog, :schema

USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA IDENTIFIER(:schema);

-- ---------------------------------------------------------------------------
-- search_claims: scope-filtered retrieval over REVIEWED claims.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION search_claims(
  query_text STRING COMMENT 'Free-text terms matched against claim text, method and condition. Pass NULL to browse by scope only.',
  task_filter STRING COMMENT 'Canonical task term, or NULL for any task.',
  method_filter STRING COMMENT 'Canonical method term, or NULL for any method.',
  benchmark_filter STRING COMMENT 'Canonical benchmark term, or NULL for any benchmark.',
  max_results INT COMMENT 'Maximum rows to return. Use 10 unless the user asks for a full listing.'
)
RETURNS TABLE (
  claim_id STRING, claim_text STRING, claim_type STRING, task STRING, method STRING,
  metric STRING, metric_value DOUBLE, benchmark STRING, condition_text STRING,
  evidence_excerpt STRING, page_number INT, source_url STRING, source_type STRING,
  published_at TIMESTAMP, reviewed_at TIMESTAMP
)
COMMENT 'Retrieve REVIEWED research claims filtered by scope. Returns only claims that passed human review, each with its source URL, page number and review timestamp. Use this before making any factual research statement; cite every row you use.'
RETURN
  SELECT claim_id, claim_text, claim_type, task, method, metric, metric_value,
         benchmark, condition_text, evidence_excerpt, page_number, source_url,
         source_type, published_at, reviewed_at
  FROM (
    SELECT claim_id, claim_text, claim_type, task, method, metric, metric_value,
           benchmark, condition_text, evidence_excerpt, page_number, source_url,
           source_type, published_at, reviewed_at,
           ROW_NUMBER() OVER (ORDER BY published_at DESC NULLS LAST) AS rn
    FROM v_research_claim_current
    WHERE (search_claims.query_text IS NULL
           OR LOWER(claim_text)     LIKE '%' || LOWER(search_claims.query_text) || '%'
           OR LOWER(COALESCE(method, ''))         LIKE '%' || LOWER(search_claims.query_text) || '%'
           OR LOWER(COALESCE(condition_text, '')) LIKE '%' || LOWER(search_claims.query_text) || '%')
      AND (search_claims.task_filter      IS NULL OR task      = search_claims.task_filter)
      AND (search_claims.method_filter    IS NULL OR method    = search_claims.method_filter)
      AND (search_claims.benchmark_filter IS NULL OR benchmark = search_claims.benchmark_filter)
  )
  WHERE rn <= COALESCE(search_claims.max_results, 10);

-- ---------------------------------------------------------------------------
-- compare_claims: the comparability gate, as a deterministic function.
-- The agent must call this before using the words "contradict" or "disagree".
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION compare_claims(
  claim_id_a STRING COMMENT 'First claim id, from search_claims.',
  claim_id_b STRING COMMENT 'Second claim id, from search_claims.'
)
RETURNS TABLE (
  comparability_status STRING, comparability_score DOUBLE, missing_dimensions STRING,
  shared_task STRING, shared_metric STRING, shared_benchmark STRING,
  claim_a_text STRING, claim_a_value DOUBLE, claim_a_condition STRING, claim_a_url STRING,
  claim_b_text STRING, claim_b_value DOUBLE, claim_b_condition STRING, claim_b_url STRING,
  verdict_note STRING
)
COMMENT 'Decide whether two reviewed claims are comparable at all. Returns COMPARABLE, PARTIALLY_COMPARABLE or INSUFFICIENT_EVIDENCE plus the scope dimensions that are missing. You MUST call this before describing two claims as agreeing or disagreeing. When the status is INSUFFICIENT_EVIDENCE, report "insufficient evidence to compare" and name missing_dimensions.'
RETURN
  WITH a AS (SELECT * FROM v_research_claim_current WHERE claim_id = compare_claims.claim_id_a),
       b AS (SELECT * FROM v_research_claim_current WHERE claim_id = compare_claims.claim_id_b),
       dims AS (
         SELECT
           a.task IS NOT NULL AND a.task = b.task                     AS task_match,
           a.metric IS NOT NULL AND a.metric = b.metric               AS metric_match,
           a.benchmark IS NOT NULL AND a.benchmark = b.benchmark      AS benchmark_match,
           a.condition_text IS NOT NULL AND b.condition_text IS NOT NULL AS condition_present,
           a.claim_text AS a_text, a.metric_value AS a_value, a.condition_text AS a_cond, a.source_url AS a_url,
           b.claim_text AS b_text, b.metric_value AS b_value, b.condition_text AS b_cond, b.source_url AS b_url,
           a.task AS s_task, a.metric AS s_metric, a.benchmark AS s_benchmark
         FROM a CROSS JOIN b
       ),
       scored AS (
         SELECT *,
           0.30 * CAST(task_match AS DOUBLE)
         + 0.30 * CAST(metric_match AS DOUBLE)
         + 0.25 * CAST(benchmark_match AS DOUBLE)
         + 0.15 * CAST(condition_present AS DOUBLE) AS score,
           CONCAT_WS(',',
             CASE WHEN NOT task_match        THEN 'task'      END,
             CASE WHEN NOT metric_match      THEN 'metric'    END,
             CASE WHEN NOT benchmark_match   THEN 'benchmark' END,
             CASE WHEN NOT condition_present THEN 'condition' END) AS missing
         FROM dims
       )
  SELECT
    CASE WHEN task_match AND metric_match AND benchmark_match AND condition_present THEN 'COMPARABLE'
         WHEN task_match AND metric_match AND benchmark_match                       THEN 'PARTIALLY_COMPARABLE'
         ELSE 'INSUFFICIENT_EVIDENCE' END,
    score,
    NULLIF(missing, ''),
    s_task, s_metric, s_benchmark,
    a_text, a_value, a_cond, a_url,
    b_text, b_value, b_cond, b_url,
    CASE WHEN task_match AND metric_match AND benchmark_match AND condition_present
           THEN 'Same task, metric and benchmark with stated conditions on both sides: a difference in values is a real disagreement worth reporting.'
         WHEN task_match AND metric_match AND benchmark_match
           THEN 'Same task, metric and benchmark, but at least one claim does not state its conditions. Report the difference as conditional and name the missing conditions.'
         ELSE 'Scope does not overlap sufficiently. Report "insufficient evidence to compare" and name the missing dimensions. Do not call these claims contradictory.' END
  FROM scored;

-- ---------------------------------------------------------------------------
-- get_claim_evidence: full provenance chain for one claim.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_claim_evidence(
  claim_id STRING COMMENT 'Claim id whose evidence chain should be returned.'
)
RETURNS TABLE (
  claim_id STRING, claim_text STRING, evidence_excerpt STRING, page_number INT,
  section_title STRING, chunk_text STRING, parser_name STRING, extraction_warning STRING,
  source_title STRING, source_url STRING, source_type STRING, publisher STRING,
  published_at TIMESTAMP, retrieved_at TIMESTAMP, source_version_number INT,
  review_status STRING, reviewed_by STRING, reviewed_at TIMESTAMP, review_note STRING
)
COMMENT 'Return the full provenance chain for one claim: excerpt, originating chunk, parser and any parser warning, source metadata, retrieval time and review decision. Use it when a user asks to see the evidence behind a claim, or when a parser warning should qualify an answer.'
RETURN
  SELECT c.claim_id, c.claim_text, c.evidence_excerpt, c.page_number,
         ch.section_title, ch.text, ch.parser_name, ch.extraction_warning,
         s.title, s.canonical_url, s.source_type, s.publisher,
         s.published_at, sv.retrieved_at, sv.version_number,
         c.review_status, c.reviewed_by, c.reviewed_at, c.review_note
  FROM research_claim AS c
  JOIN research_source_version AS sv ON c.source_version_id = sv.source_version_id
  JOIN research_source        AS s  ON c.source_id = s.source_id
  LEFT JOIN research_chunk    AS ch ON c.chunk_id = ch.chunk_id
  WHERE c.claim_id = get_claim_evidence.claim_id;

-- ---------------------------------------------------------------------------
-- get_open_questions: evidence-backed gaps only.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_open_questions(
  topic_filter STRING COMMENT 'Optional task or method term to narrow the gaps. NULL returns all gaps.',
  max_results INT COMMENT 'Maximum rows. Use 5 unless the user asks for more.'
)
RETURNS TABLE (
  question_type STRING, question_text STRING, task STRING, metric STRING,
  benchmark STRING, evidence_count BIGINT, claim_ids ARRAY<STRING>
)
COMMENT 'Return open research questions derived from the corpus: unresolved comparisons, claims missing scope, and findings reported by only one source. Every row is backed by counted claims. Never state a research gap that this function did not return.'
RETURN
  SELECT question_type, question_text, task, metric, benchmark, evidence_count, claim_ids
  FROM (
    SELECT question_type, question_text, task, metric, benchmark, evidence_count, claim_ids,
           ROW_NUMBER() OVER (ORDER BY evidence_count DESC) AS rn
    FROM v_research_open_questions
    WHERE get_open_questions.topic_filter IS NULL
       OR LOWER(question_text) LIKE '%' || LOWER(get_open_questions.topic_filter) || '%'
       OR LOWER(COALESCE(task, '')) LIKE '%' || LOWER(get_open_questions.topic_filter) || '%'
  )
  WHERE rn <= COALESCE(get_open_questions.max_results, 5);

-- ---------------------------------------------------------------------------
-- get_corpus_coverage: honest scope of what the corpus can answer.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_corpus_coverage()
RETURNS TABLE (
  source_type STRING, source_count BIGINT, indexed_source_count BIGINT,
  claim_count BIGINT, reviewed_claim_count BIGINT, unreviewed_claim_count BIGINT,
  chunks_with_parser_warning BIGINT, most_recent_retrieval TIMESTAMP, oldest_retrieval TIMESTAMP
)
COMMENT 'Return corpus size, freshness and review backlog by source type. Call it when a user asks about consensus, coverage or how current the corpus is, and use it to qualify any synthesis with how much of the corpus is actually reviewed.'
RETURN SELECT * FROM v_source_coverage;

-- ---------------------------------------------------------------------------
-- search_passages: retrieval over the reviewed chunk corpus.
--
-- Chunks are the retrieval corpus; claims are the comparison unit. This tool
-- exists so a user can ask for supporting passages beyond the structured claim
-- search_passages is deliberately NOT defined here: it depends on whether an
-- AI Search / Vector Search endpoint is configured for this deployment. Deploy
-- 098_search_passages_vector.sql (requires a built index) or
-- 099_search_passages_fallback.sql (lexical scan, no endpoint needed) instead.
-- The bootstrap job runs the fallback by default.

-- ---------------------------------------------------------------------------
-- get_taxonomy: the controlled vocabulary, so the agent maps a user's wording
-- onto the terms the scope columns actually use instead of guessing.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_taxonomy(
  dimension_filter STRING COMMENT 'TASK, METHOD, METRIC, BENCHMARK or CONDITION. NULL returns every dimension.'
)
RETURNS TABLE (dimension STRING, canonical_term STRING, synonyms STRING, definition STRING)
COMMENT 'Return the controlled vocabulary for claim scope fields. Call it to translate a user''s wording into the canonical term before filtering, and to tell a user which terms the corpus actually indexes. A term absent here is a term the corpus does not cover.'
RETURN
  SELECT dimension, canonical_term, synonyms, definition
  FROM research_taxonomy
  WHERE get_taxonomy.dimension_filter IS NULL
     OR dimension = UPPER(get_taxonomy.dimension_filter)
  ORDER BY dimension, canonical_term;

-- ---------------------------------------------------------------------------
-- get_review_backlog: what is waiting on a human, and why.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_review_backlog(
  max_results INT COMMENT 'Maximum queue items. Use 10 unless the user asks for the full queue.'
)
RETURNS TABLE (
  review_id STRING, target_type STRING, target_id STRING, priority STRING,
  reason STRING, claim_text STRING, source_url STRING, created_at TIMESTAMP
)
COMMENT 'Return open review-queue items, highest priority first. Use it to answer "what needs review" and to ground a recommended next step in work that is actually outstanding. Items here are unreviewed by definition - never present their claim text as a finding.'
RETURN
  SELECT review_id, target_type, target_id, priority, reason, claim_text, source_url, created_at
  FROM (
    SELECT q.review_id, q.target_type, q.target_id, q.priority, q.reason,
           c.claim_text, c.source_url, q.created_at,
           ROW_NUMBER() OVER (
             ORDER BY CASE q.priority WHEN 'HIGH' THEN 0 WHEN 'NORMAL' THEN 1 ELSE 2 END, q.created_at
           ) AS rn
    FROM research_review_queue AS q
    LEFT JOIN research_claim AS c ON q.target_type = 'CLAIM' AND q.target_id = c.claim_id
    WHERE q.status = 'OPEN'
  )
  WHERE rn <= COALESCE(get_review_backlog.max_results, 10);

-- ---------------------------------------------------------------------------
-- get_figure_evidence: provenance for a claim read from a chart or diagram.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_figure_evidence(
  claim_id STRING COMMENT 'Claim whose figure-derived evidence should be returned.'
)
RETURNS TABLE (
  figure_id STRING, page_number INT, figure_index INT, caption STRING,
  image_uri STRING, extracted_text STRING, extraction_confidence DOUBLE,
  vision_model STRING, review_status STRING, source_url STRING
)
COMMENT 'Return the figure or chart a claim was read from, with its page number, stored image reference, the vision model used and that model''s confidence. A value read from a chart is a visual interpretation, not a stated number: always report the confidence and the review status alongside it.'
RETURN
  SELECT f.figure_id, f.page_number, f.figure_index, f.caption, f.image_uri,
         f.extracted_text, f.extraction_confidence, f.vision_model,
         f.review_status, s.canonical_url
  FROM research_claim AS c
  JOIN research_figure AS f ON c.figure_id = f.figure_id
  JOIN research_source AS s ON c.source_id = s.source_id
  WHERE c.claim_id = get_figure_evidence.claim_id;

-- ---------------------------------------------------------------------------
-- NOTE ON create_proposal
--
-- There is deliberately NO SQL function named create_proposal. A Unity Catalog
-- SQL function cannot write to a table, so a SQL "proposal tool" could only
-- return a string claiming a proposal was recorded while writing nothing - the
-- exact class of false statement this system exists to prevent.
--
-- The proposal tool is served by the custom MCP server in
-- src/research_discovery/mcp/server.py, which validates the payload and
-- performs a real INSERT of a PENDING_APPROVAL row. Attach it to the agent as
-- an MCP tool, not as a UC function.
-- ---------------------------------------------------------------------------
