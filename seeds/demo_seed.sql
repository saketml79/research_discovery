-- Research Discovery Engine :: controlled demo seed
--
-- Seeds a small, deliberately designed scenario so the demo proves behaviour
-- rather than luck. The claims below are illustrative records written for the
-- demo, not verbatim quotations, and every row is marked so.
--
-- The scenario contains, by design:
--   * claim A vs claim B  - same task/metric/benchmark, values 4% apart  -> SUPPORTS
--   * claim A vs claim C  - same task/metric/benchmark, values 38% apart -> CONTRADICTS
--   * claim A vs claim D  - different benchmark, no conditions -> INSUFFICIENT_EVIDENCE
--   * claim E             - unreviewed candidate, must never appear as a finding
--   * claim F             - single-source finding -> an evidence-backed open question
--
-- Parameters: :catalog, :schema

USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA IDENTIFIER(:schema);

-- Sources -------------------------------------------------------------------
MERGE INTO research_source AS t
USING (
  SELECT * FROM VALUES
    ('src-demo-a', 'https://arxiv.org/abs/2404.16130', 'From Local to Global: A Graph RAG Approach', 'PRIMARY_PAPER', 'arXiv', 'Edge et al.', TIMESTAMP '2024-04-24', 'ARXIV-NONEXCLUSIVE', true,  'REVIEWED', current_timestamp(), current_timestamp()),
    ('src-demo-b', 'https://arxiv.org/abs/2410.05779', 'LightRAG',                                  'PRIMARY_PAPER', 'arXiv', 'Guo et al.',  TIMESTAMP '2024-10-08', 'ARXIV-NONEXCLUSIVE', true,  'REVIEWED', current_timestamp(), current_timestamp()),
    ('src-demo-c', 'https://arxiv.org/abs/2405.14831', 'HippoRAG',                                  'PRIMARY_PAPER', 'arXiv', 'Gutierrez et al.', TIMESTAMP '2024-05-23', 'ARXIV-NONEXCLUSIVE', true, 'REVIEWED', current_timestamp(), current_timestamp()),
    ('src-demo-d', 'https://microsoft.github.io/graphrag/', 'GraphRAG documentation',                'BENCHMARK_DOC', 'Microsoft', 'Microsoft', TIMESTAMP '2025-01-15', 'METADATA_ONLY',     false, 'REVIEWED', current_timestamp(), current_timestamp())
  AS s(source_id, canonical_url, title, source_type, publisher, authors, published_at, license, storage_permitted, ingestion_status, registered_at, updated_at)
) AS s
ON t.source_id = s.source_id
WHEN NOT MATCHED THEN INSERT *;

-- Source versions -----------------------------------------------------------
MERGE INTO research_source_version AS t
USING (
  SELECT * FROM VALUES
    ('srcv-demo-a', 'src-demo-a', 1, repeat('a', 64), NULL, 'application/pdf', 1200000, 200, NULL, current_timestamp(), true),
    ('srcv-demo-b', 'src-demo-b', 1, repeat('b', 64), NULL, 'application/pdf', 1100000, 200, NULL, current_timestamp(), true),
    ('srcv-demo-c', 'src-demo-c', 1, repeat('c', 64), NULL, 'application/pdf', 1000000, 200, NULL, current_timestamp(), true),
    ('srcv-demo-d', 'src-demo-d', 1, repeat('d', 64), NULL, 'text/html',        200000, 200, NULL, current_timestamp(), true)
  AS s(source_version_id, source_id, version_number, content_hash, raw_content_uri, content_type, byte_size, http_status, etag, retrieved_at, is_current)
) AS s
ON t.source_version_id = s.source_version_id
WHEN NOT MATCHED THEN INSERT *;

-- Claims --------------------------------------------------------------------
-- claim_text values are illustrative demo records, not verbatim source quotes.
MERGE INTO research_claim AS t
USING (
  SELECT * FROM VALUES
    -- A: reviewed baseline
    ('clm-demo-a', 'srcv-demo-a', 'src-demo-a', NULL, NULL,
     'Graph-based retrieval reaches 0.62 F1 on multi-hop question answering over HotpotQA.',
     'PERFORMANCE', 'multi_hop_qa', 'graphrag_global', 'f1', 0.62, 'ratio', 'hotpotqa',
     'Full HotpotQA dev set; GPT-4-class reader; top-20 retrieval budget.',
     'reaches 0.62 F1 on multi-hop question answering', 6,
     'https://arxiv.org/abs/2404.16130', 'manual', 'demo-seed-1.0', 0.95, NULL,
     current_timestamp(), 'REVIEWED', 'demo-reviewer', current_timestamp(), 'Seeded demo record.', NULL),
    -- B: agrees with A (4% apart, same scope)
    ('clm-demo-b', 'srcv-demo-b', 'src-demo-b', NULL, NULL,
     'A lightweight graph index reaches 0.60 F1 on multi-hop question answering over HotpotQA.',
     'PERFORMANCE', 'multi_hop_qa', 'graphrag_global', 'f1', 0.60, 'ratio', 'hotpotqa',
     'Full HotpotQA dev set; GPT-4-class reader; top-20 retrieval budget.',
     'reaches 0.60 F1 on multi-hop question answering', 8,
     'https://arxiv.org/abs/2410.05779', 'manual', 'demo-seed-1.0', 0.93, NULL,
     current_timestamp(), 'REVIEWED', 'demo-reviewer', current_timestamp(), 'Seeded demo record.', NULL),
    -- C: genuinely disagrees with A (38% apart, same scope)
    ('clm-demo-c', 'srcv-demo-c', 'src-demo-c', NULL, NULL,
     'Graph-based retrieval reaches only 0.38 F1 on multi-hop question answering over HotpotQA.',
     'PERFORMANCE', 'multi_hop_qa', 'graphrag_global', 'f1', 0.38, 'ratio', 'hotpotqa',
     'Full HotpotQA dev set; open-weight 8B reader; top-20 retrieval budget.',
     'reaches only 0.38 F1 on multi-hop question answering', 5,
     'https://arxiv.org/abs/2405.14831', 'manual', 'demo-seed-1.0', 0.91, NULL,
     current_timestamp(), 'REVIEWED', 'demo-reviewer', current_timestamp(), 'Seeded demo record.', NULL),
    -- D: superficially conflicting, but a different benchmark and no conditions
    ('clm-demo-d', 'srcv-demo-d', 'src-demo-d', NULL, NULL,
     'Graph-based global search wins 72% of comprehensiveness comparisons against vector RAG.',
     'PERFORMANCE', 'query_focused_summarization', 'graphrag_global', 'comprehensiveness_win_rate',
     0.72, 'ratio', 'podcast_transcripts', NULL,
     'wins 72% of comprehensiveness comparisons', 1,
     'https://microsoft.github.io/graphrag/', 'manual', 'demo-seed-1.0', 0.88,
     'MISSING:condition_text - the source does not state corpus size or reader model',
     current_timestamp(), 'REVIEWED', 'demo-reviewer', current_timestamp(), 'Seeded demo record.', NULL),
    -- E: unreviewed candidate. Must never appear as a finding.
    ('clm-demo-e', 'srcv-demo-b', 'src-demo-b', NULL, NULL,
     'Graph indexing costs approximately 12 USD per million tokens of corpus.',
     'RESOURCE_COST', NULL, 'graphrag_global', 'index_cost', 12.0, 'usd', NULL, NULL,
     'costs approximately 12 USD per million tokens', 11,
     'https://arxiv.org/abs/2410.05779', 'llm', 'demo-seed-1.0', 0.41,
     'MISSING:task,benchmark,condition_text - cost stated without corpus or model context',
     current_timestamp(), 'CANDIDATE', NULL, NULL, NULL, NULL),
    -- F: single-source finding -> becomes an evidence-backed open question
    ('clm-demo-f', 'srcv-demo-a', 'src-demo-a', NULL, NULL,
     'Graph-based retrieval degrades on questions requiring temporal ordering of events.',
     'LIMITATION', 'multi_hop_qa', 'graphrag_global', 'temporal_accuracy', NULL, NULL, 'musique',
     'MuSiQue subset restricted to temporally ordered questions.',
     'degrades on questions requiring temporal ordering', 9,
     'https://arxiv.org/abs/2404.16130', 'manual', 'demo-seed-1.0', 0.87, NULL,
     current_timestamp(), 'REVIEWED', 'demo-reviewer', current_timestamp(), 'Seeded demo record.', NULL)
  AS s(claim_id, source_version_id, source_id, chunk_id, figure_id, claim_text, claim_type, task, method,
       metric, metric_value, metric_unit, benchmark, condition_text, evidence_excerpt, page_number,
       source_url, extractor_name, extractor_version, extraction_confidence, missing_field_reason,
       extracted_at, review_status, reviewed_by, reviewed_at, review_note, superseded_by_claim_id)
) AS s
ON t.claim_id = s.claim_id
WHEN NOT MATCHED THEN INSERT *;

-- Taxonomy ------------------------------------------------------------------
MERGE INTO research_taxonomy AS t
USING (
  SELECT * FROM VALUES
    ('tax-task-mhqa',  'TASK',      'multi_hop_qa',                'multi-hop qa,multihop question answering,multi hop qa', 'Answering a question that requires combining facts from two or more documents.', NULL),
    ('tax-task-qfs',   'TASK',      'query_focused_summarization', 'qfs,query-focused summarization,sensemaking',           'Summarizing a corpus with respect to a user query.', NULL),
    ('tax-meth-gg',    'METHOD',    'graphrag_global',             'graphrag,graph rag,global search,graph-based rag',      'Retrieval over a graph index with community-level global search.', NULL),
    ('tax-meth-vec',   'METHOD',    'vector_rag',                  'naive rag,baseline rag,standard rag,vector retrieval',  'Dense-vector chunk retrieval without a graph index.', NULL),
    ('tax-metr-f1',    'METRIC',    'f1',                          'f1 score,token f1',                                     'Token-level F1 of a generated answer against a reference.', NULL),
    ('tax-metr-cwr',   'METRIC',    'comprehensiveness_win_rate',  'comprehensiveness,win rate',                            'Share of head-to-head comparisons an LLM judge scored as more comprehensive.', NULL),
    ('tax-bench-hqa',  'BENCHMARK', 'hotpotqa',                    'hotpot qa,hotpot',                                      'Multi-hop QA benchmark over Wikipedia paragraph pairs.', NULL),
    ('tax-bench-msq',  'BENCHMARK', 'musique',                     'musique-ans',                                           'Multi-hop QA benchmark built by composing single-hop questions.', NULL),
    ('tax-bench-pod',  'BENCHMARK', 'podcast_transcripts',         'podcast corpus',                                        'Podcast transcript corpus used for query-focused summarization evaluation.', NULL)
  AS s(term_id, dimension, canonical_term, synonyms, definition, parent_term_id)
) AS s
ON t.term_id = s.term_id
WHEN NOT MATCHED THEN INSERT *;

-- Review queue for the unreviewed candidate ---------------------------------
MERGE INTO research_review_queue AS t
USING (
  SELECT * FROM VALUES
    ('rev-demo-e', 'CLAIM', 'clm-demo-e', 'HIGH',
     'LOW_CONFIDENCE:0.41;MISSING_SCOPE:task,benchmark,condition_text',
     'OPEN', NULL, current_timestamp(), NULL, NULL)
  AS s(review_id, target_type, target_id, priority, reason, status, assigned_to, created_at, resolved_at, resolution_note)
) AS s
ON t.review_id = s.review_id
WHEN NOT MATCHED THEN INSERT *;

-- Standing discovery queries -------------------------------------------------
-- These are how the corpus stays current: the scheduled sweep re-runs each one
-- against the scholarly metadata APIs and queues anything new for review.
-- Adding a standing query is a curation decision made by a human, never by the
-- agent - which is why created_by is recorded.
MERGE INTO research_standing_query AS t
USING (
  SELECT * FROM VALUES
    ('qry-graphrag-eval',  'GraphRAG evaluation multi-hop question answering', 'graphrag_evaluation', true, 36, 25, 'demo-curator', current_timestamp(), NULL),
    ('qry-graphrag-cost',  'graph retrieval augmented generation indexing cost', 'graphrag_evaluation', true, 24, 15, 'demo-curator', current_timestamp(), NULL),
    ('qry-graph-bench',    'knowledge graph retrieval benchmark HotpotQA MuSiQue', 'graphrag_evaluation', true, 36, 20, 'demo-curator', current_timestamp(), NULL)
  AS s(query_id, query_text, topic, enabled, recency_months, max_results, created_by, created_at, last_run_at)
) AS s
ON t.query_id = s.query_id
WHEN NOT MATCHED THEN INSERT *;

-- One discovered-but-not-ingested candidate, so the demo can show the
-- EXTERNAL_CANDIDATE tier without needing live network access.
MERGE INTO research_source_candidate AS t
USING (
  SELECT * FROM VALUES
    ('cand-demo-temporal', 'https://arxiv.org/abs/2506.00001',
     'Temporal reasoning limits of graph-based retrieval', 'arxiv', '2506.00001', NULL,
     'PRIMARY_PAPER', 'Demo Author', 'arXiv', TIMESTAMP '2025-06-01',
     'We study how graph-based retrieval handles questions requiring temporal ordering.',
     3, true, 'https://arxiv.org/pdf/2506.00001', 'ARXIV-NONEXCLUSIVE', true,
     'Open access on an allowlisted host with a storable licence; full ingestion permitted.',
     0.71, 'GraphRAG evaluation multi-hop question answering', 'SCHEDULED_SWEEP',
     'METADATA_ONLY', 'DISCOVERED', current_timestamp(), NULL, NULL)
  AS s(candidate_id, canonical_url, title, provider, external_id, doi, source_type, authors,
       venue, published_at, abstract, citation_count, is_open_access, pdf_url, license,
       fetchable, fetch_decision, relevance_score, matched_query, discovery_mode,
       ingestion_speed, status, discovered_at, decided_by, decided_at)
) AS s
ON t.candidate_id = s.candidate_id
WHEN NOT MATCHED THEN INSERT *;
