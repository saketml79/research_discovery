"""Stage 4-5: extract candidate claims and build the review queue.

Every claim leaves this job as ``CANDIDATE``. Nothing here can make a claim
runtime-visible; only ``review_job`` applying a human decision can.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..config import Config
from ..extract.base import ClaimExtractor, ExtractionError, to_claim
from ..extract.registry import get_extractor
from ..io import delta
from ..models import Chunk, Claim, ReviewItem
from ..review.queue import build_claim_queue
from .common import RunLog, configure_logging, get_spark, parse_args, stage

logger = logging.getLogger(__name__)


def extract_claims(
    chunks: Sequence[Chunk],
    extractor: ClaimExtractor,
    source_urls: dict[str, str],
) -> tuple[list[Claim], int]:
    """Extract claims from chunks, counting the candidates that failed validation.

    Args:
        chunks: Chunks to extract from.
        extractor: Configured extractor.
        source_urls: ``source_id -> canonical_url``, so each claim is
            independently citable without a join.

    Returns:
        The valid claims and the number of rejected candidates.
    """
    claims: list[Claim] = []
    rejected = 0
    for chunk, candidate in extractor.extract_many(chunks):
        url = source_urls.get(chunk.source_id, "")
        if not url:
            logger.warning("no source URL for %s; skipping claim", chunk.source_id)
            rejected += 1
            continue
        try:
            claims.append(
                to_claim(
                    candidate,
                    chunk,
                    source_url=url,
                    extractor_name=extractor.name,
                    extractor_version=extractor.version,
                )
            )
        except (ExtractionError, ValueError) as exc:
            # A candidate that fails validation - an invented number, a
            # paraphrased "verbatim" excerpt - is dropped, not repaired.
            logger.warning("rejected candidate from chunk %s: %s", chunk.chunk_id, exc)
            rejected += 1
    return claims, rejected


def run(
    config: Config,
    chunks: Sequence[Chunk],
    source_urls: dict[str, str],
    *,
    extractor: ClaimExtractor | None = None,
    spark: Any = None,
) -> tuple[RunLog, list[Claim], list[ReviewItem]]:
    """Run extraction and queue building over ``chunks``."""
    with stage("EXTRACT", logger) as run_log:
        run_log.records_in = len(chunks)
        active = extractor or get_extractor(config)
        claims, rejected = extract_claims(chunks, active, source_urls)
        queue = build_claim_queue(claims, config)
        run_log.records_out = len(claims)
        run_log.records_quarantined = rejected

        if spark is not None:
            delta.upsert(spark, config.table("research_claim"), claims, dry_run=config.dry_run)
            delta.upsert(spark, config.table("research_review_queue"), queue, dry_run=config.dry_run)
    return run_log, claims, queue


def _load_chunks(spark: Any, config: Config, limit: int) -> tuple[list[Chunk], dict[str, str]]:
    """Load chunks awaiting extraction plus their source URLs."""
    chunk_table = config.table("research_chunk")
    claim_table = config.table("research_claim")
    query = f"""
        SELECT c.*
        FROM {chunk_table} AS c
        LEFT ANTI JOIN (SELECT DISTINCT chunk_id FROM {claim_table}) AS done
          ON c.chunk_id = done.chunk_id
        WHERE c.lifecycle_state = 'CHUNKED'
        {f'LIMIT {int(limit)}' if limit else ''}
    """
    rows = spark.sql(query).collect()
    chunks = [
        Chunk(
            source_version_id=r["source_version_id"],
            source_id=r["source_id"],
            chunk_index=r["chunk_index"],
            text=r["text"],
            block_type=r["block_type"],
            page_number=r["page_number"],
            section_title=r["section_title"],
            parser_name=r["parser_name"],
            parser_version=r["parser_version"],
            extraction_warning=r["extraction_warning"],
            chunk_id=r["chunk_id"],
        )
        for r in rows
    ]
    url_rows = spark.sql(
        f"SELECT source_id, canonical_url FROM {config.table('research_source')}"
    ).collect()
    return chunks, {r["source_id"]: r["canonical_url"] for r in url_rows}


def main(argv: Sequence[str] | None = None) -> int:
    """Job entry point."""
    config, args = parse_args(list(argv) if argv is not None else None)
    configure_logging(args.log_level)
    spark = get_spark()
    chunks, urls = _load_chunks(spark, config, args.limit)
    if not chunks:
        logger.info("no chunks awaiting extraction")
        return 0
    run_log, _, _ = run(config, chunks, urls, spark=spark)
    delta.append(spark, config.table("pipeline_run"), [run_log], dry_run=config.dry_run)
    return 0 if run_log.status in {"SUCCEEDED", "PARTIAL"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
