"""Integration tests for conversational AI agent tools."""

import os
from pathlib import Path
from typing import Any, cast

import pytest
import vcr


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


class Scrub:
    """Scrub API key values from recorded request/response payloads."""

    def __init__(self, apikey: str):
        """Initialize the scrubber with the key to redact."""
        self.key = apikey

    def request(self, req: Any) -> Any:
        """Scrub API key from a VCR request body."""
        if not self.key or not req.body:
            return req

        if isinstance(req.body, bytes):
            req.body = req.body.replace(
                self.key.encode("utf-8"), b"<ELEVENLABS_API_KEY>"
            )
        elif isinstance(req.body, str):
            req.body = req.body.replace(self.key, "<ELEVENLABS_API_KEY>")

        return req

    def response(self, res: Any) -> Any:
        """Scrub API key from a VCR response body."""
        if not self.key:
            return res

        body = res.get("body", {}).get("string")
        if isinstance(body, bytes):
            res["body"]["string"] = body.replace(
                self.key.encode("utf-8"), b"<ELEVENLABS_API_KEY>"
            )
        elif isinstance(body, str):
            res["body"]["string"] = body.replace(self.key, "<ELEVENLABS_API_KEY>")
        return res


def mocks() -> vcr.VCR:
    """Build a configured VCR instance for integration cassettes."""
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    kwargs: dict[str, Any] = {
        "record_mode": "once",
        "filter_headers": ["authorization", "xi-api-key", "x-api-key"],
        "before_record_request": lambda request: Scrub(api_key).request(request),
        "before_record_response": lambda response: Scrub(api_key).response(response),
    }
    return vcr.VCR(**kwargs)


def test_add_knowledge_base_to_agent_text(ctx: tuple[str, Any]):
    """Create a text knowledge base document and attach it to an agent."""
    target, agents = ctx
    cassette = (
        Path(__file__).parent
        / "cassettes"
        / "test_add_knowledge_base_to_agent_text_integration.yaml"
    )
    with cast(Any, mocks().use_cassette(str(cassette))):
        response = agents.add_knowledge_base_to_agent(
            agent_id=target,
            knowledge_base_name="integration-kb-text",
            text="This is a test knowledge base entry recorded with VCR.",
        )

    assert response.type == "text"
    assert "Knowledge base created with ID:" in response.text
    assert f"added to agent {target} successfully." in response.text
