"""Run the benchmark suite against the deployed Genie Agent via the API.

This is the acceptance criterion "the same test prompts pass from the Genie UI
and the Genie Agents API", made executable. Run it after every deployment; a
non-zero exit means a behavioural property regressed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from ..agent.benchmark import report, run_benchmarks, to_json
from ..agent.client import GenieAgentClient, WorkspaceGenieConversationApi
from .common import configure_logging, parse_args

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = Path("benchmark-report.json")


def main(argv: Sequence[str] | None = None) -> int:
    """Job entry point. Returns 1 when any benchmark fails its automated checks."""
    config, args = parse_args(list(argv) if argv is not None else None)
    configure_logging(args.log_level)

    space_id = config.extra.get("space_id", "")
    if not space_id:
        logger.error("--space-id is required; pass the deployed Genie space id")
        return 2

    client = GenieAgentClient(WorkspaceGenieConversationApi(), space_id)
    results = run_benchmarks(client, config)

    print(report(results))
    DEFAULT_REPORT_PATH.write_text(to_json(results), encoding="utf-8")
    logger.info("wrote %s", DEFAULT_REPORT_PATH)

    failed = [r for r in results if not r.passed]
    if failed:
        logger.error("%d benchmark(s) failed automated checks", len(failed))
        return 1
    logger.info("all %d benchmarks passed automated checks", len(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
