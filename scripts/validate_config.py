#!/usr/bin/env python3
"""Validate the Genie Agent configuration. Exits non-zero on any problem."""

from __future__ import annotations

import sys

from research_discovery.agent.genie_config import build_serialized_space
from research_discovery.agent.validate import validate_space
from research_discovery.config import Config


def main() -> int:
    space = build_serialized_space(Config())
    problems = validate_space(space)
    if problems:
        print("Genie configuration is NOT deployable:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"Genie configuration is valid: {len(space['tables'])} views, "
        f"{len(space['functions'])} functions, {len(space['example_queries'])} examples, "
        f"{len(space['benchmarks'])} benchmarks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
