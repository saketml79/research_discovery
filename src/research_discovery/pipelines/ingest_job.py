"""Stage 1-3: register sources, fetch versions, parse and chunk.

Idempotent by construction. A source whose bytes have not changed produces no
new version and therefore no new chunks; a re-run over unchanged input merges
identical rows on identical ids.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from ..chunking import chunk_document
from ..config import Config
from ..ingest.sources import FetchError, HttpFetcher, load_seed_sources
from ..ingest import sources as source_ops
from ..io import delta
from ..models import Chunk, IngestionStatus, Source, SourceVersion
from ..parsers.base import ParserError
from ..parsers.registry import resolve_with_fallback
from .common import RunLog, configure_logging, get_spark, parse_args, stage

logger = logging.getLogger(__name__)


def process_source(
    source: Source,
    fetcher: HttpFetcher,
    config: Config,
    *,
    previous_version: SourceVersion | None = None,
) -> tuple[Source, SourceVersion | None, list[Chunk]]:
    """Fetch, parse and chunk one source.

    Returns:
        The updated source, the new version (``None`` when unchanged) and the
        chunks derived from it.

    Raises:
        FetchError: The fetch failed or was refused by policy.
        ParserError: No usable parser, or the document had no extractable text.
    """
    version, raw = source_ops.fetch_version(
        source, fetcher, previous=previous_version, volume_path=config.volume_path
    )
    if version is None:
        return source, None, []

    source.advance(IngestionStatus.FETCHED)

    content_type = version.content_type or "application/octet-stream"
    parser, fallback_warning = resolve_with_fallback(config.parser, content_type)
    document = parser.parse(raw, source_uri=source.canonical_url, content_type=content_type)
    if document.is_empty:
        raise ParserError(f"parser produced no text for {source.canonical_url}")

    source.advance(IngestionStatus.PARSED)

    chunks = chunk_document(
        document,
        source_version_id=version.source_version_id,
        source_id=source.source_id,
        config=config,
        storage_permitted=source.storage_permitted,
        parser_warning=fallback_warning,
    )
    source.advance(IngestionStatus.CHUNKED)
    return source, version, chunks


def run(
    config: Config,
    fetcher: HttpFetcher,
    seed_path: Path,
    *,
    spark: Any = None,
    limit: int = 0,
) -> RunLog:
    """Run the ingest stage over the seed manifest."""
    with stage("PARSE", logger) as run_log:
        sources = load_seed_sources(seed_path)
        if limit:
            sources = sources[:limit]
        run_log.records_in = len(sources)

        written_sources: list[Source] = []
        written_versions: list[SourceVersion] = []
        written_chunks: list[Chunk] = []

        for source in sources:
            try:
                updated, version, chunks = process_source(source, fetcher, config)
            except (FetchError, ParserError) as exc:
                # One bad source must not cost the batch. Quarantine and move on;
                # the run log and v_source_coverage make the gap visible.
                logger.warning("quarantining %s: %s", source.canonical_url, exc)
                source.advance(IngestionStatus.QUARANTINED)
                written_sources.append(source)
                run_log.records_quarantined += 1
                continue
            written_sources.append(updated)
            if version is not None:
                written_versions.append(version)
                written_chunks.extend(chunks)

        run_log.records_out = len(written_chunks)

        if spark is not None:
            delta.upsert(spark, config.table("research_source"), written_sources, dry_run=config.dry_run)
            delta.upsert(
                spark, config.table("research_source_version"), written_versions, dry_run=config.dry_run
            )
            delta.upsert(spark, config.table("research_chunk"), written_chunks, dry_run=config.dry_run)
    return run_log


def main(argv: Sequence[str] | None = None) -> int:
    """Job entry point."""
    config, args = parse_args(list(argv) if argv is not None else None)
    configure_logging(args.log_level)

    seed_path = Path(args.seed_path or "seeds/sources.csv")
    if not seed_path.exists():
        logger.error("seed manifest not found: %s", seed_path)
        return 2

    from ..ingest.http import UrlLibFetcher  # noqa: PLC0415 - optional transport

    fetcher = source_ops.PolicyAwareFetcher(UrlLibFetcher())
    spark = None if config.dry_run else get_spark()
    run_log = run(config, fetcher, seed_path, spark=spark, limit=args.limit)
    if spark is not None:
        delta.append(spark, config.table("pipeline_run"), [run_log], dry_run=config.dry_run)
    return 0 if run_log.status in {"SUCCEEDED", "PARTIAL"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
