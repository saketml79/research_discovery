#!/usr/bin/env python3
"""Regenerate the committed Genie Agent configuration.

Run after changing anything in ``agent/genie_config.py``. CI fails when the
committed file differs from what this script produces.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research_discovery.agent.genie_config import write_config
from research_discovery.agent.validate import assert_valid, validate_space
from research_discovery.agent.genie_config import build_serialized_space
from research_discovery.config import Config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="main")
    parser.add_argument("--schema", default="research_discovery")
    parser.add_argument("--out", default="genie/research-agent.json")
    args = parser.parse_args()

    config = Config(catalog=args.catalog, schema=args.schema)
    assert_valid(build_serialized_space(config))
    path = write_config(config, Path(args.out))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
