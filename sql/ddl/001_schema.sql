-- Research Discovery Engine :: schema and volume
-- Idempotent. Executed by the `bootstrap` job task before any pipeline runs.
-- Parameters: :catalog, :schema, :volume

CREATE SCHEMA IF NOT EXISTS IDENTIFIER(:catalog || '.' || :schema)
COMMENT 'Research Discovery Engine: governed source, chunk, claim and review records for a curated research corpus.';

CREATE VOLUME IF NOT EXISTS IDENTIFIER(:catalog || '.' || :schema || '.' || :volume)
COMMENT 'Raw, immutable source documents (PDF / HTML snapshots) keyed by source_version_id.';
