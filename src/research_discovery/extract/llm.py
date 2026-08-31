"""LLM claim extractor backed by a Databricks model-serving endpoint.

The chat client is injected, so the extractor is fully testable with a stub and
carries no hard dependency on ``databricks-sdk`` at import time. Retries use
bounded exponential backoff and only cover transport-shaped failures; a schema
violation is a data problem and is never retried into a different answer.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable, Protocol, Sequence

from ..models import Chunk
from .base import CandidateClaim, ClaimExtractor, ExtractionError, ExtractorUnavailableError
from .prompts import CLAIM_RESPONSE_SCHEMA, EXTRACTION_PROMPT_VERSION, build_messages

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_MAX_ATTEMPTS = 3
_BASE_BACKOFF_SECONDS = 0.5


class ChatClient(Protocol):
    """Minimal chat interface an extractor needs."""

    def complete(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        """Return the assistant's message content as text."""


class ServingEndpointChatClient:
    """Adapter over a Databricks serving endpoint via the workspace SDK."""

    def __init__(self, endpoint: str, workspace_client: Any | None = None) -> None:
        self._endpoint = endpoint
        self._client = workspace_client

    def _resolve(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from databricks.sdk import WorkspaceClient  # noqa: PLC0415 - lazy backend
        except ImportError as exc:  # pragma: no cover - exercised on clusters
            raise ExtractorUnavailableError(
                "databricks-sdk is not installed; install it or select extractor=heuristic"
            ) from exc
        self._client = WorkspaceClient()
        return self._client

    def complete(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        client = self._resolve()
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole  # noqa: PLC0415

        role_map = {
            "system": ChatMessageRole.SYSTEM,
            "user": ChatMessageRole.USER,
            "assistant": ChatMessageRole.ASSISTANT,
        }
        response = client.serving_endpoints.query(
            name=self._endpoint,
            messages=[ChatMessage(role=role_map[m["role"]], content=m["content"]) for m in messages],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1500),
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ExtractionError("serving endpoint returned no choices")
        return choices[0].message.content or ""


class LlmClaimExtractor(ClaimExtractor):
    """Extracts claims by prompting a chat model with a strict JSON schema."""

    name = "llm"

    def __init__(
        self,
        client: ChatClient,
        *,
        model_name: str,
        max_claims_per_chunk: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args:
        client: Chat backend.
        model_name: Recorded as part of ``extractor_version`` for provenance.
        max_claims_per_chunk: Upper bound requested from the model.
        sleep: Injection point so retry backoff is instant under test.
        """
        self._client = client
        self._model_name = model_name
        self._max_claims = max_claims_per_chunk
        self._sleep = sleep

    @property
    def version(self) -> str:  # type: ignore[override]
        """Model identity plus prompt version: both change extraction behaviour."""
        return f"{self._model_name}/{EXTRACTION_PROMPT_VERSION}"

    def extract(self, chunk: Chunk) -> Sequence[CandidateClaim]:
        messages = build_messages(
            chunk_text=chunk.text,
            source_title=chunk.section_title,
            source_type="UNKNOWN",
            source_url="",
            section_title=chunk.section_title,
            page_number=chunk.page_number,
            max_claims=self._max_claims,
        )
        raw = self._complete_with_retry(messages)
        payload = _parse_json(raw)
        _validate_schema(payload)

        candidates: list[CandidateClaim] = []
        for item in payload["claims"][: self._max_claims]:
            candidates.append(
                CandidateClaim(
                    claim_text=str(item["claim_text"]).strip(),
                    claim_type=str(item["claim_type"]),
                    task=_clean(item.get("task")),
                    method=_clean(item.get("method")),
                    metric=_clean(item.get("metric")),
                    metric_value=_as_float(item.get("metric_value")),
                    metric_unit=_clean(item.get("metric_unit")),
                    benchmark=_clean(item.get("benchmark")),
                    condition_text=_clean(item.get("condition_text")),
                    evidence_excerpt=_clean(item.get("evidence_excerpt")),
                    confidence=_as_float(item.get("confidence")),
                    missing_field_reason=_clean(item.get("missing_field_reason")),
                )
            )
        return candidates

    def _complete_with_retry(self, messages: Sequence[dict[str, str]]) -> str:
        last: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return self._client.complete(messages, temperature=0.0)
            except ExtractorUnavailableError:
                raise
            except Exception as exc:  # transport-shaped failure: retry
                last = exc
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                backoff = _BASE_BACKOFF_SECONDS * (2**attempt)
                self._sleep(backoff + random.uniform(0, backoff / 2))
        raise ExtractionError(f"model call failed after {_MAX_ATTEMPTS} attempts: {last}") from last


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse the model response, tolerating a surrounding code fence."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise ExtractionError(f"model response was not JSON: {text[:200]!r}") from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"model response was not valid JSON: {exc}") from exc


def _validate_schema(payload: dict[str, Any]) -> None:
    """Validate the response against ``CLAIM_RESPONSE_SCHEMA``.

    Uses ``jsonschema`` when available and falls back to explicit structural
    checks, so validation is never skipped just because a library is absent.
    """
    try:
        import jsonschema  # noqa: PLC0415 - optional dependency
    except ImportError:
        if not isinstance(payload.get("claims"), list):
            raise ExtractionError("response is missing a 'claims' array") from None
        allowed = set(CLAIM_RESPONSE_SCHEMA["properties"]["claims"]["items"]["properties"])
        for item in payload["claims"]:
            if not isinstance(item, dict):
                raise ExtractionError("each claim must be an object")
            for required in ("claim_text", "claim_type", "confidence"):
                if required not in item:
                    raise ExtractionError(f"claim is missing required field {required!r}")
            unknown = set(item) - allowed
            if unknown:
                raise ExtractionError(f"claim has unknown fields: {sorted(unknown)}")
        return
    try:
        jsonschema.validate(payload, CLAIM_RESPONSE_SCHEMA)
    except jsonschema.ValidationError as exc:  # pragma: no cover - lib-dependent
        raise ExtractionError(f"response failed schema validation: {exc.message}") from exc


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
