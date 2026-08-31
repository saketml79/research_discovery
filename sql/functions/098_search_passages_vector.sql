-- Vector-search-backed definition of search_passages. OPTIONAL - deploy this
-- INSTEAD OF 099_search_passages_fallback.sql only when var.ai_search_endpoint
-- is configured and index_job has built research_chunk_index. CREATE FUNCTION
-- validates the index eagerly, so deploying this without a real index fails
-- deployment; that is why it is not run by the default bootstrap job.
--
-- Parameters: :catalog, :schema

USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA IDENTIFIER(:schema);

CREATE OR REPLACE FUNCTION search_passages(
  query_text STRING COMMENT 'Natural-language description of the passage to find.',
  max_results INT COMMENT 'Maximum passages to return. Use 5 unless the user asks for more.'
)
RETURNS TABLE (
  chunk_id STRING, text STRING, page_number INT, section_title STRING,
  source_url STRING, source_title STRING, source_type STRING,
  parser_name STRING, extraction_warning STRING, search_score DOUBLE
)
COMMENT 'Retrieve supporting passages from REVIEWED sources. A passage is evidence context, NOT a finding: it has not been through claim review, so never state a result that rests only on a passage. Cite the source_url and page_number, and surface extraction_warning when one is present. Prefer search_claims for anything you intend to assert.'
RETURN
  SELECT chunk_id, text, page_number, section_title, source_url, source_title,
         source_type, parser_name, extraction_warning, search_score
  FROM (
    SELECT r.chunk_id, r.text, r.page_number, r.section_title,
           s.canonical_url AS source_url, s.title AS source_title, s.source_type,
           r.parser_name, r.extraction_warning, r.search_score,
           ROW_NUMBER() OVER (ORDER BY r.search_score DESC) AS rn
    FROM VECTOR_SEARCH(
           -- :catalog/:schema are substituted as literals when this script runs, so the
           -- index name stays a foldable constant. current_catalog()/current_schema()
           -- are non-deterministic at analysis time and VECTOR_SEARCH rejects them.
           -- num_results is capped at a fixed literal for the same reason; the
           -- caller's requested max_results is applied afterwards.
           index => :catalog || '.' || :schema || '.research_chunk_index',
           query_text => search_passages.query_text,
           num_results => 50
         ) AS r
    JOIN research_source AS s ON r.source_id = s.source_id
    WHERE s.ingestion_status IN ('REVIEWED', 'INDEXED')
  )
  WHERE rn <= COALESCE(search_passages.max_results, 5);
