"""Extractor selection."""

from __future__ import annotations

from typing import Any, Callable

from ..config import Config
from .ai_extract import AiExtractClaimExtractor
from .base import ClaimExtractor
from .heuristic import HeuristicClaimExtractor
from .llm import LlmClaimExtractor, ServingEndpointChatClient


def get_extractor(
    config: Config,
    *,
    chat_client: Any | None = None,
    sql_runner: Callable[[str, dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> ClaimExtractor:
    """Build the extractor named by ``config.extractor``.

    Args:
        config: Deployment configuration.
        chat_client: Overrides the serving-endpoint client (used in tests).
        sql_runner: Required for the ``ai_extract`` adapter.

    Raises:
        KeyError: The configured extractor is unknown.
    """
    name = config.extractor
    if name == "heuristic":
        return HeuristicClaimExtractor()
    if name == "ai_extract":
        return AiExtractClaimExtractor(sql_runner)
    if name == "llm":
        client = chat_client or ServingEndpointChatClient(config.extraction_model)
        return LlmClaimExtractor(client, model_name=config.extraction_model)
    raise KeyError(f"unknown extractor {name!r}; known: llm, ai_extract, heuristic")
