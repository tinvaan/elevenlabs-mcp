"""Integration tests for conversational AI agent tools."""

from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize(
    ("kind", "casette"),
    [
        ("text", "test_add_knowledge_base_to_agent_text_integration.yaml"),
        ("url", "test_add_knowledge_base_to_agent_url_integration.yaml"),
        ("file", "test_add_knowledge_base_to_agent_file_integration.yaml"),
    ],
    ids=["text", "url", "file"],
)
def test_add_knowledge_base_to_agent(
    ctx: tuple[str, Any],
    vcr: Any,
    tmp_path: Path,
    kind: str,
    casette: str,
):
    """Create and attach a knowledge base document for each supported input type."""
    target, agents = ctx
    payload: dict[str, str] = {
        "knowledge_base_name": f"integration-kb-{kind}",
    }

    if kind == "url":
        payload["url"] = "https://example.com"
    elif kind == "text":
        payload["text"] = "This is a test knowledge base entry recorded with VCR."
    else:
        input_file = tmp_path / "integration-kb-file.txt"
        input_file.write_text(
            "This is a test file knowledge base entry recorded with VCR."
        )
        payload["input_file_path"] = str(input_file)

    cassette = Path(__file__).parent / "cassettes" / casette
    with vcr.use_cassette(str(cassette)):
        response = agents.add_knowledge_base_to_agent(agent_id=target, **payload)

    assert response.type == "text"
    assert "Knowledge base created with ID:" in response.text
    assert f"added to agent {target} successfully." in response.text
