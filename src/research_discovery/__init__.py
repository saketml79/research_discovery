"""Research Discovery Engine.

A governed, claims-aware research workspace for Databricks: sources are
registered and versioned, parsed into page-scoped chunks, converted into
structured claims with explicit scope, gated behind human review, and served to
a Genie Agent that may only cite reviewed claims and may only call two claims
contradictory when their scope actually overlaps.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
