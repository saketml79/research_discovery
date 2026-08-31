-- Fallback definition of search_passages for deployments without AI Search.
--
-- Serves the same contract as the VECTOR_SEARCH version in tools.sql using a
-- lexical scan, so the agent's tool surface is identical whether or not vector
-- retrieval is configured. Deploy this INSTEAD OF the tools.sql definition when
-- var.ai_search_endpoint is empty; the bootstrap job selects one or the other.
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
COMMENT 'Retrieve supporting passages from REVIEWED sources. A passage is evidence context, NOT a finding: it has not been through claim review, so never state a result that rests only on a passage. Cite the source_url and page_number, and surface extraction_warning when one is present. Prefer search_claims for anything you intend to assert. NOTE: this deployment uses lexical matching, not semantic search, so recall is limited to literal term overlap - say so if a user asks why a passage was not found.'
RETURN
  SELECT c.chunk_id, c.text, c.page_number, c.section_title,
         s.canonical_url, s.title, s.source_type,
         c.parser_name, c.extraction_warning,
         -- Crude lexical score: term coverage normalized by chunk length, so a
         -- short passage containing the terms outranks a long one that mentions
         -- them once. Not comparable to a vector similarity score.
         CAST(
           SIZE(FILTER(SPLIT(LOWER(search_passages.query_text), '\\s+'),
                       t -> LENGTH(t) > 2 AND LOWER(c.text) LIKE '%' || t || '%'))
           AS DOUBLE
         ) / GREATEST(SQRT(c.char_count), 1.0) AS search_score
  FROM research_chunk AS c
  JOIN research_source AS s ON c.source_id = s.source_id
  JOIN research_source_version AS v
    ON c.source_version_id = v.source_version_id AND v.is_current
  WHERE s.ingestion_status IN ('REVIEWED', 'INDEXED')
    AND c.lifecycle_state IN ('CHUNKED', 'INDEXED')
    AND SIZE(FILTER(SPLIT(LOWER(search_passages.query_text), '\\s+'),
                    t -> LENGTH(t) > 2 AND LOWER(c.text) LIKE '%' || t || '%')) > 0
  ORDER BY search_score DESC
  LIMIT COALESCE(search_passages.max_results, 5);
