"""Shared pipeline plumbing: logging, argument parsing and run telemetry."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..config import Config
from ..ids import stable_id
from ..models import utcnow


def configure_logging(level: str = "INFO") -> None:
    """Configure structured-ish stdout logging once per job."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def parse_args(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    """Parse standard job arguments into a ``Config`` plus job-specific flags."""
    parser = argparse.ArgumentParser(description="Research Discovery pipeline task")
    parser.add_argument("--catalog")
    parser.add_argument("--schema")
    parser.add_argument("--volume")
    parser.add_argument("--parser")
    parser.add_argument("--extractor")
    parser.add_argument("--extraction-model", dest="extraction_model")
    parser.add_argument("--ai-search-endpoint", dest="ai_search_endpoint")
    parser.add_argument("--warehouse-id", dest="warehouse_id")
    parser.add_argument("--seed-path", dest="seed_path")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    provided = {k: v for k, v in vars(args).items() if v not in (None, False, 0, "")}
    config = Config.from_args(provided)
    return config, args


@dataclass(slots=True)
class RunLog:
    """Telemetry for one pipeline stage, persisted to ``pipeline_run``."""

    stage: str
    status: str = "STARTED"
    records_in: int = 0
    records_out: int = 0
    records_quarantined: int = 0
    error_text: str | None = None
    started_at: Any = field(default_factory=utcnow)
    finished_at: Any = None
    run_id: str = ""

    def __post_init__(self) -> None:
        self.run_id = self.run_id or stable_id("run", self.stage, self.started_at.isoformat())


@contextmanager
def stage(name: str, logger: logging.Logger) -> Iterator[RunLog]:
    """Record a stage, marking it FAILED and re-raising on error.

    The run row is written by the caller so the writer stays injectable; the
    context manager only guarantees the status is accurate.
    """
    record = RunLog(stage=name)
    logger.info("stage %s started (run_id=%s)", name, record.run_id)
    try:
        yield record
    except Exception as exc:
        record.status = "FAILED"
        record.error_text = f"{type(exc).__name__}: {exc}"[:2000]
        record.finished_at = utcnow()
        logger.exception("stage %s failed", name)
        raise
    else:
        if record.status == "STARTED":
            record.status = "PARTIAL" if record.records_quarantined else "SUCCEEDED"
        record.finished_at = utcnow()
        logger.info(
            "stage %s %s: in=%d out=%d quarantined=%d",
            name,
            record.status,
            record.records_in,
            record.records_out,
            record.records_quarantined,
        )


def get_spark() -> Any:
    """Return the active ``SparkSession``.

    Raises:
        RuntimeError: Spark is unavailable, i.e. the job is running off-cluster.
    """
    try:
        from pyspark.sql import SparkSession  # noqa: PLC0415 - lazy backend
    except ImportError as exc:  # pragma: no cover - clusters always have Spark
        raise RuntimeError("pyspark is unavailable; run this task on a Databricks cluster") from exc
    return SparkSession.builder.getOrCreate()
