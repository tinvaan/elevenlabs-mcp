"""Shared fixtures and helpers for integration tests."""

import os
import pytest
import vcr

from typing import Any
from scrub import Scrub


@pytest.fixture(name="env")
def mcp_env() -> tuple[str, str]:
    """Return validated integration env vars needed by MCP integration tests."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        pytest.fail("ELEVENLABS_API_KEY is not set")

    target = os.environ.get("ELEVENLABS_TEST_AGENT")
    if not target:
        pytest.fail("ELEVENLABS_TEST_AGENT is not set")

    return api_key, target


@pytest.fixture(name="agents")
def mcp_agents(env: tuple[str, str]) -> Any:
    """Import and return the elevenlabs_mcp.agents module."""
    _ = env
    try:
        from elevenlabs_mcp import agents as agents_module
    except ImportError as err:
        pytest.fail(f"Unable to import elevenlabs_mcp.agents: {err}")

    return agents_module


@pytest.fixture(name="server")
def mcp_server(env: tuple[str, str]) -> Any:
    """Import and return the elevenlabs_mcp.server module."""
    _ = env
    try:
        from elevenlabs_mcp import server as server_module
    except ImportError as err:
        pytest.fail(f"Unable to import elevenlabs_mcp.server: {err}")

    return server_module


@pytest.fixture(name="vcr")
def vcr_mocks_fixture() -> vcr.VCR:
    """Build a configured VCR instance for integration cassettes."""
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    return vcr.VCR(**{
        "record_mode": "once",
        "filter_headers": ["authorization", "xi-api-key", "x-api-key"],
        "before_record_request": lambda request: Scrub(api_key).request(request),
        "before_record_response": lambda response: Scrub(api_key).response(response),
    })
