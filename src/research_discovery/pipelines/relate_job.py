"""Stage 6: build comparability-gated relationships between reviewed claims.

Runs only over reviewed claims. Relating candidates would let two unverified
extractions manufacture a disagreement that no source actually made.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..config import Config
from ..io import delta
from ..models import Claim, ClaimRelationship, ClaimType, ReviewStatus
from ..review.comparability import detect_relationships, summarize
from ..review.queue import build_relationship_queue
from .common import RunLog, configure_logging, get_spark, parse_args, stage

logger = logging.getLogger(__name__)


def run(
    config: Config, claims: Sequence[Claim], *, spark: Any = None
) -> tuple[RunLog, list[ClaimRelationship]]:
    """Detect and persist claim relationships."""
    with stage("RELATE", logger) as run_log:
        run_log.records_in = len(claims)
        relationships = detect_relationships(claims, reviewed_only=True)
        queue = build_relationship_queue(relationships)
        run_log.records_out = len(relationships)
        logger.info("relationship mix: %s", summarize(relationships))

        if spark is not None:
            delta.upsert(
                spark,
                config.table("research_claim_relationship"),
                relationships,
                dry_run=config.dry_run,
            )
            delta.upsert(spark, config.table("research_review_queue"), queue, dry_run=config.dry_run)
    return run_log, relationships


def _load_reviewed_claims(spark: Any, config: Config) -> list[Claim]:
    """Load reviewed, non-superseded claims."""
    rows = spark.sql(
        f"""
        SELECT * FROM {config.table('research_claim')}
        WHERE review_status = 'REVIEWED' AND superseded_by_claim_id IS NULL
        """
    ).collect()
    return [
        Claim(
            source_version_id=r["source_version_id"],
            source_id=r["source_id"],
            claim_text=r["claim_text"],
            claim_type=ClaimType(r["claim_type"]),
            source_url=r["source_url"],
            extractor_name=r["extractor_name"],
            extractor_version=r["extractor_version"],
            chunk_id=r["chunk_id"],
            task=r["task"],
            method=r["method"],
            metric=r["metric"],
            metric_value=r["metric_value"],
            metric_unit=r["metric_unit"],
            benchmark=r["benchmark"],
            condition_text=r["condition_text"],
            evidence_excerpt=r["evidence_excerpt"],
            page_number=r["page_number"],
            extraction_confidence=r["extraction_confidence"],
            missing_field_reason=r["missing_field_reason"],
            review_status=ReviewStatus.REVIEWED,
            reviewed_by=r["reviewed_by"],
            reviewed_at=r["reviewed_at"],
            claim_id=r["claim_id"],
        )
        for r in rows
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Job entry point."""
    config, args = parse_args(list(argv) if argv is not None else None)
    configure_logging(args.log_level)
    spark = get_spark()
    claims = _load_reviewed_claims(spark, config)
    if len(claims) < 2:
        logger.info("fewer than two reviewed claims; nothing to relate")
        return 0
    run_log, _ = run(config, claims, spark=spark)
    delta.append(spark, config.table("pipeline_run"), [run_log], dry_run=config.dry_run)
    return 0 if run_log.status in {"SUCCEEDED", "PARTIAL"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
