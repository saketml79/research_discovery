-- Research Discovery Engine :: discovery tables
--
-- These four tables record how the corpus grows: the standing queries that are
-- swept on a schedule, the candidates those sweeps and live questions surface,
-- and the discovery runs themselves. They exist so that "why is this paper not
-- in the corpus?" always has a recorded answer.
--
-- Parameters: :catalog, :schema

USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA IDENTIFIER(:schema);

-- ---------------------------------------------------------------------------
-- Standing queries: how the corpus stays current without anyone watching.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_standing_query (
  query_id            STRING  NOT NULL COMMENT 'Deterministic id from query_text and topic.',
  query_text          STRING  NOT NULL COMMENT 'The search string sent to the scholarly metadata APIs.',
  topic               STRING  NOT NULL COMMENT 'Corpus topic this query maintains, e.g. graphrag_evaluation.',
  enabled             BOOLEAN NOT NULL COMMENT 'FALSE pauses the query without deleting its history.',
  recency_months      INT     NOT NULL COMMENT 'Only works published within this window are returned.',
  max_results         INT     NOT NULL COMMENT 'Cap on candidates per sweep, so one broad query cannot flood the review queue.',
  created_by          STRING  NOT NULL COMMENT 'Curator who added the query. Standing queries are a curation decision, not an agent decision.',
  created_at          TIMESTAMP NOT NULL COMMENT 'When the query was added.',
  last_run_at         TIMESTAMP       COMMENT 'When the sweep last ran this query.',
  CONSTRAINT research_standing_query_pk PRIMARY KEY (query_id)
)
USING DELTA
COMMENT 'Saved queries re-run by the scheduled discovery sweep. Grain: one row per query. Adding one is how a human tells the system which literature to keep watching.';

-- ---------------------------------------------------------------------------
-- Candidates: works the discovery APIs surfaced that are NOT in the corpus.
-- Metadata only. Nothing here has been fetched, parsed, or believed.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_source_candidate (
  candidate_id        STRING  NOT NULL COMMENT 'Deterministic id from the canonical URL.',
  canonical_url       STRING  NOT NULL COMMENT 'Stable URL of the discovered work.',
  title               STRING  NOT NULL COMMENT 'Title as reported by the discovery API.',
  provider            STRING  NOT NULL COMMENT 'Which API surfaced it: openalex, arxiv, semantic_scholar or rss.',
  external_id         STRING          COMMENT 'The provider''s own identifier, for re-querying.',
  doi                 STRING          COMMENT 'DOI when known. The primary cross-provider dedupe key.',
  source_type         STRING  NOT NULL COMMENT 'PRIMARY_PAPER, BENCHMARK_DOC, REPOSITORY, SECONDARY_BLOG or TALK_TRANSCRIPT.',
  authors             STRING          COMMENT 'Author list as reported.',
  venue               STRING          COMMENT 'Journal, conference or preprint server.',
  published_at        TIMESTAMP       COMMENT 'Publication date as reported by the provider.',
  abstract            STRING          COMMENT 'Abstract as distributed by the metadata API. Enough to judge relevance; never a substitute for the paper.',
  citation_count      INT             COMMENT 'Citation count at discovery time. A popularity signal, never a quality or truth signal.',
  is_open_access      BOOLEAN NOT NULL COMMENT 'Whether the provider reports an openly readable full text.',
  pdf_url             STRING          COMMENT 'Open-access PDF URL when one exists.',
  license             STRING          COMMENT 'Licence as reported. Governs whether full text may be stored.',
  fetchable           BOOLEAN NOT NULL COMMENT 'Whether this deployment may download the content: open access, allowlisted host, acceptable licence.',
  fetch_decision      STRING  NOT NULL COMMENT 'Plain-language reason for the fetchable verdict. This is what the agent reports when asked why a paper was not ingested.',
  relevance_score     DOUBLE          COMMENT 'Ranking score in [0,1] from term overlap, recency, citations, evidence tier and access. A triage aid, not a quality judgement.',
  matched_query       STRING          COMMENT 'The query that surfaced this candidate.',
  discovery_mode      STRING  NOT NULL COMMENT 'SCHEDULED_SWEEP or LIVE_QUESTION - whether a cron job or a user question surfaced it.',
  ingestion_speed     STRING  NOT NULL COMMENT 'METADATA_ONLY, PROVISIONAL or REVIEWED: how far this candidate has been taken.',
  status              STRING  NOT NULL COMMENT 'DISCOVERED, PROPOSED, APPROVED, INGESTED, REJECTED. Only APPROVED candidates are fetched by the ingestion job.',
  discovered_at       TIMESTAMP NOT NULL COMMENT 'When discovery surfaced it.',
  decided_by          STRING          COMMENT 'Who approved or rejected ingestion.',
  decided_at          TIMESTAMP       COMMENT 'When that decision was made.',
  CONSTRAINT research_source_candidate_pk PRIMARY KEY (candidate_id)
)
USING DELTA
COMMENT 'Works found by the discovery APIs but not in the corpus. Grain: one row per canonical URL. METADATA ONLY - nothing here has been fetched or extracted, so a candidate proves a work EXISTS and never what it found.';

-- ---------------------------------------------------------------------------
-- Discovery runs: an audit trail for corpus growth.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_discovery_run (
  discovery_run_id    STRING  NOT NULL COMMENT 'Generated run id.',
  query_text          STRING  NOT NULL COMMENT 'Query that was searched.',
  query_id            STRING          COMMENT 'FK to research_standing_query for a scheduled sweep; NULL for a live question.',
  discovery_mode      STRING  NOT NULL COMMENT 'SCHEDULED_SWEEP or LIVE_QUESTION.',
  providers_searched  STRING  NOT NULL COMMENT 'Comma-separated providers queried.',
  provider_errors     STRING          COMMENT 'JSON map of provider to error. A provider being down is recorded, not hidden: it means the sweep was incomplete.',
  candidates_found    INT     NOT NULL COMMENT 'New candidates after cross-provider dedupe.',
  candidates_fetchable INT    NOT NULL COMMENT 'How many may actually be downloaded.',
  already_known       INT     NOT NULL COMMENT 'Hits already in the corpus.',
  requested_by        STRING  NOT NULL COMMENT 'Scheduler, agent principal or user who triggered the run.',
  started_at          TIMESTAMP NOT NULL COMMENT 'Run start.',
  finished_at         TIMESTAMP       COMMENT 'Run end.',
  CONSTRAINT research_discovery_run_pk PRIMARY KEY (discovery_run_id)
)
USING DELTA
COMMENT 'Audit trail of discovery runs. Grain: one row per query execution. Lets the agent answer "when did we last look for this, and did every provider respond?"';

-- ---------------------------------------------------------------------------
-- Runtime views for the agent.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW v_source_candidate_current
COMMENT 'Discovered works NOT yet in the corpus. EVIDENCE TIER: EXTERNAL_CANDIDATE - metadata from a search API only. You may say such a work EXISTS and cite its URL, title, authors and date. You must NEVER state what it found, concluded or measured: nobody has read it. Use it to answer "what else is out there" and to propose ingestion.'
AS
SELECT
  candidate_id, canonical_url, title, source_type, authors, venue, published_at,
  abstract, citation_count, is_open_access, fetchable, fetch_decision,
  relevance_score, matched_query, discovery_mode, status, discovered_at,
  'EXTERNAL_CANDIDATE' AS evidence_tier
FROM research_source_candidate
WHERE status IN ('DISCOVERED', 'PROPOSED', 'APPROVED');

CREATE OR REPLACE VIEW v_corpus_gap
COMMENT 'Where discovery found relevant work the corpus does not hold. Grain: one row per topic and reason. Use it to answer honestly when the corpus cannot cover a question: it distinguishes "nobody has studied this" from "we have not ingested it yet" from "it exists but we may not fetch it".'
AS
SELECT
  matched_query                                  AS topic,
  COUNT(*)                                       AS candidate_count,
  SUM(CASE WHEN fetchable THEN 1 ELSE 0 END)     AS fetchable_count,
  SUM(CASE WHEN NOT fetchable THEN 1 ELSE 0 END) AS blocked_count,
  SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END) AS awaiting_ingestion,
  MAX(discovered_at)                             AS last_discovered_at,
  MAX(relevance_score)                           AS best_relevance,
  FIRST(fetch_decision)                          AS example_blocking_reason
FROM research_source_candidate
WHERE status IN ('DISCOVERED', 'PROPOSED', 'APPROVED')
GROUP BY matched_query;

CREATE OR REPLACE VIEW v_discovery_freshness
COMMENT 'When each standing query was last swept and what it returned. Grain: one row per standing query. Use it to qualify any "no source says X" answer with how recently the system last looked.'
AS
SELECT
  q.query_id, q.query_text, q.topic, q.enabled, q.recency_months,
  q.last_run_at,
  DATEDIFF(current_timestamp(), q.last_run_at) AS days_since_sweep,
  r.candidates_found                           AS last_run_candidates,
  r.provider_errors                            AS last_run_provider_errors
FROM research_standing_query AS q
LEFT JOIN research_discovery_run AS r
  ON r.query_id = q.query_id AND r.started_at = q.last_run_at;
