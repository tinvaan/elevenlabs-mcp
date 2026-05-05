"""Shared fixtures and helpers for integration tests."""

import os
import pytest
import vcr

from typing import Any
from scrub import Scrub


@pytest.fixture(name="ctx")
def context() -> tuple[str, Any]:
    """Return integration context after validating required env vars."""
    if not os.environ.get("ELEVENLABS_API_KEY"):
        pytest.skip("ELEVENLABS_API_KEY is not set")

    agent = os.environ.get("ELEVENLABS_TEST_AGENT")
    if not agent:
        pytest.skip("ELEVENLABS_TEST_AGENT is not set")

    from elevenlabs_mcp import agents
    return agent, agents

@pytest.fixture(name="vcr")
def vcr_mocks_fixture() -> vcr.VCR:
    """Build a configured VCR instance for integration cassettes."""
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    kwargs: dict[str, Any] = {
        "record_mode": "once",
        "filter_headers": ["authorization", "xi-api-key", "x-api-key"],
        "before_record_request": lambda request: Scrub(api_key).request(request),
        "before_record_response": lambda response: Scrub(api_key).response(response),
    }
    return vcr.VCR(**kwargs)
