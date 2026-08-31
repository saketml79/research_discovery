"""Stage 8: validate and deploy the Genie Agent configuration.

Validation runs first and unconditionally, so a bad configuration fails the job
before it can reach the workspace.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from ..agent.deploy import WorkspaceGenieClient, deploy
from ..agent.genie_config import build_serialized_space, write_config
from ..agent.validate import assert_valid
from .common import configure_logging, parse_args

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("genie/research-agent.json")


def main(argv: Sequence[str] | None = None) -> int:
    """Job entry point."""
    config, args = parse_args(list(argv) if argv is not None else None)
    configure_logging(args.log_level)

    warehouse_id = args.warehouse_id or config.extra.get("warehouse_id", "")
    config = replace(config, extra={**config.extra, "warehouse_id": warehouse_id})

    space = build_serialized_space(config)
    assert_valid(space)
    logger.info("Genie configuration validated (%d tables, %d functions, %d benchmarks)",
                len(space["tables"]), len(space["functions"]), len(space["benchmarks"]))

    output = write_config(config, DEFAULT_OUTPUT)
    logger.info("wrote %s (%d bytes)", output, output.stat().st_size)

    if not warehouse_id:
        logger.warning("no --warehouse-id supplied; validated configuration only")
        return 0

    result = deploy(
        config,
        warehouse_id=warehouse_id,
        client=WorkspaceGenieClient(),
        dry_run=config.dry_run,
    )
    logger.info("deployment %s: space_id=%s", result.action, result.space_id or "(none)")
    print(json.dumps({"action": result.action, "space_id": result.space_id}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
