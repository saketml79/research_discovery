"""Deploy the Genie Agent configuration through the Genie Agents API.

Idempotent: the agent is looked up by display name and updated when it exists,
created otherwise, so re-running a deployment does not accumulate agents. The
configuration is validated before any API call - an invalid config never reaches
the workspace.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import Config
from .genie_config import AGENT_DISPLAY_NAME, build_serialized_space
from .validate import assert_valid

logger = logging.getLogger(__name__)


class GenieApiClient(Protocol):
    """The subset of the Genie Agents API this deployment needs."""

    def list_spaces(self) -> list[dict[str, Any]]:
        """Return the spaces visible to the caller."""

    def create_space(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a space and return the created object."""

    def update_space(self, space_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Replace a space's configuration."""


@dataclass(frozen=True, slots=True)
class DeploymentResult:
    """Outcome of a deployment attempt."""

    space_id: str
    action: str  # CREATED | UPDATED | VALIDATED_ONLY
    display_name: str


class WorkspaceGenieClient:
    """``GenieApiClient`` backed by the Databricks SDK.

    The SDK's Genie surface is still evolving; calls go through
    ``api_client.do`` so this adapter does not break when a typed helper is
    renamed. Endpoint paths are in one place and easy to update.
    """

    SPACES_PATH = "/api/2.0/genie/spaces"

    def __init__(self, workspace_client: Any | None = None) -> None:
        self._client = workspace_client

    def _resolve(self) -> Any:
        if self._client is not None:
            return self._client
        from databricks.sdk import WorkspaceClient  # noqa: PLC0415 - lazy backend

        self._client = WorkspaceClient()
        return self._client

    def list_spaces(self) -> list[dict[str, Any]]:
        response = self._resolve().api_client.do("GET", self.SPACES_PATH)
        return list(response.get("spaces", []) if isinstance(response, dict) else [])

    def create_space(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._resolve().api_client.do("POST", self.SPACES_PATH, body=payload)

    def update_space(self, space_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._resolve().api_client.do(
            "PATCH", f"{self.SPACES_PATH}/{space_id}", body=payload
        )


def build_request(config: Config, warehouse_id: str) -> dict[str, Any]:
    """Build the API request body, validating the space first.

    Raises:
        ConfigValidationError: The configuration is not deployable.
        ValueError: ``warehouse_id`` is empty.
    """
    if not warehouse_id:
        raise ValueError("warehouse_id is required to deploy a Genie Agent")
    space = build_serialized_space(config)
    space["warehouse_id"] = warehouse_id
    assert_valid(space)
    return {
        "display_name": space["display_name"],
        "description": space["description"],
        "warehouse_id": warehouse_id,
        "serialized_space": json.dumps(space),
    }


def deploy(
    config: Config,
    *,
    warehouse_id: str,
    client: GenieApiClient,
    dry_run: bool = False,
) -> DeploymentResult:
    """Create or update the Research Discovery Genie Agent.

    Args:
        config: Deployment configuration.
        warehouse_id: SQL warehouse the agent runs its queries on.
        client: Genie API client.
        dry_run: Validate and log the request without calling the API.

    Returns:
        What was done, and to which space.
    """
    payload = build_request(config, warehouse_id)

    if dry_run:
        logger.info("dry run: validated Genie configuration for %s", AGENT_DISPLAY_NAME)
        return DeploymentResult(space_id="", action="VALIDATED_ONLY", display_name=AGENT_DISPLAY_NAME)

    existing = next(
        (s for s in client.list_spaces() if s.get("display_name") == AGENT_DISPLAY_NAME), None
    )
    if existing:
        space_id = str(existing.get("space_id") or existing.get("id"))
        client.update_space(space_id, payload)
        logger.info("updated Genie space %s", space_id)
        return DeploymentResult(space_id, "UPDATED", AGENT_DISPLAY_NAME)

    created = client.create_space(payload)
    space_id = str(created.get("space_id") or created.get("id", ""))
    logger.info("created Genie space %s", space_id)
    return DeploymentResult(space_id, "CREATED", AGENT_DISPLAY_NAME)
