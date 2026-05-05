"""Integration tests for conversational AI agent tools."""

import re
import pytest

from pathlib import Path
from typing import Any


def _extract_id(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    assert match is not None
    return match.group(1)


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
    knowledge_base_id: str | None = None
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
        try:
            response = agents.add_knowledge_base_to_agent(agent_id=target, **payload)
            knowledge_base_id = _extract_id(r"Knowledge base created with ID: (\S+)", response.text)

            assert response.type == "text"
            assert "Knowledge base created with ID:" in response.text
            assert f"added to agent {target} successfully." in response.text
        finally:
            if knowledge_base_id:
                agents.client.conversational_ai.knowledge_base.documents.delete(
                    documentation_id=knowledge_base_id,
                    force=True,
                )


def test_create_agent(ctx: tuple[str, Any], vcr: Any):
    """Create a dedicated integration-test agent and delete it after assertion."""
    _, agents = ctx
    created_agent_id: str | None = None
    cassette = Path(__file__).parent / "cassettes" / "test_create_agent_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        try:
            response = agents.create_agent(
                name="integration-test-agent",
                first_message="Hi, this is an integration test agent.",
                system_prompt="You are an integration test agent.",
            )
            created_agent_id = _extract_id(r"Agent ID: (\S+),", response.text)

            assert response.type == "text"
            assert "Agent created successfully:" in response.text
            assert "Name: integration-test-agent" in response.text
            assert f"Agent ID: {created_agent_id}" in response.text
        finally:
            if created_agent_id:
                agents.client.conversational_ai.agents.delete(agent_id=created_agent_id)


def test_list_agents(ctx: tuple[str, Any], vcr: Any):
    """List available agents in the workspace."""
    _, agents = ctx
    cassette = Path(__file__).parent / "cassettes" / "test_list_agents_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        response = agents.list_agents()

    assert response.type == "text"
    assert "Available agents:" in response.text


def test_get_agent(ctx: tuple[str, Any], vcr: Any):
    """Get details for the target test agent."""
    target, agents = ctx
    cassette = Path(__file__).parent / "cassettes" / "test_get_agent_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        response = agents.get_agent(agent_id=target)

    assert response.type == "text"
    assert "Agent Details:" in response.text
    assert f"Agent ID: {target}" in response.text


def test_list_workspace_tools(ctx: tuple[str, Any], vcr: Any):
    """List all workspace tools."""
    _, agents = ctx
    cassette = Path(__file__).parent / "cassettes" / "test_list_workspace_tools_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        response = agents.list_workspace_tools()

    assert response.type == "text"
    assert "Workspace Tools (" in response.text or "No tools found in workspace." in response.text


@pytest.mark.parametrize(
    ("kind", "casette"),
    [
        ("client", "test_get_tool_client_integration.yaml"),
        ("webhook", "test_get_tool_webhook_integration.yaml"),
    ],
    ids=["client", "webhook"],
)
def test_get_tool(ctx: tuple[str, Any], vcr: Any, kind: str, casette: str):
    """Create and fetch details for each supported tool type."""
    _, agents = ctx
    cassette = Path(__file__).parent / "cassettes" / casette

    tool_id: str | None = None
    with vcr.use_cassette(str(cassette)):
        try:
            fn = agents.create_client_tool if kind == "client" else agents.create_webhook_tool
            created = fn(config={
                "name": f"integration-get-tool-{kind}",
                "description": f"Integration test tool for get_tool {kind}",
                **({"url": "https://example.com/tool-get"} if kind == "webhook" else {}),
                **({"method": "GET"} if kind == "webhook" else {})
            })
            tool_id = _extract_id(r"Tool ID: (\S+)", created.text)

            response = agents.get_tool(tool_id=tool_id)
            assert response.type == "text"
            assert f"Tool ID: {tool_id}" in response.text
            assert f"Type: {kind}" in response.text
            assert f"Name: integration-get-tool-{kind}" in response.text
            if kind == "webhook":
                assert "URL: https://example.com/tool-get" in response.text
                assert "Method: GET" in response.text
        finally:
            if tool_id:
                agents.delete_tool(tool_id=tool_id)


@pytest.mark.parametrize(
    ("kind", "casette"),
    [
        ("webhook", "test_create_webhook_tool_integration.yaml"),
        ("client", "test_create_client_tool_integration.yaml"),
    ],
    ids=["webhook", "client"],
)
def test_create_tool(ctx: tuple[str, Any], vcr: Any, kind: str, casette: str):
    """Create each supported workspace tool type."""
    _, agents = ctx
    cassette = Path(__file__).parent / "cassettes" / casette
    tool_id: str | None = None

    with vcr.use_cassette(str(cassette)):
        try:
            fn = agents.create_client_tool if kind == "client" else agents.create_webhook_tool
            created = fn(config={
                "name": f"integration-{kind}-tool",
                "description": f"Integration {kind} tool",
                **({"url": "https://example.com/hook"} if kind == "webhook" else {}),
                **({"method": "GET"} if kind == "webhook" else {}),
            })

            assert created.type == "text"
            assert f"Successfully created {kind} tool." in created.text
            tool_id = _extract_id(r"Tool ID: (\S+)", created.text)
        finally:
            if tool_id:
                agents.delete_tool(tool_id=tool_id)


@pytest.mark.parametrize(
    ("kind", "casette"),
    [
        ("webhook", "test_update_webhook_tool_integration.yaml"),
        ("client", "test_update_client_tool_integration.yaml"),
    ],
    ids=["webhook", "client"],
)
def test_update_tool(ctx: tuple[str, Any], vcr: Any, kind: str, casette: str):
    """Update each supported workspace tool type."""
    _, agents = ctx
    cassette = Path(__file__).parent / "cassettes" / casette
    tool_id: str | None = None

    with vcr.use_cassette(str(cassette)):
        try:
            fn = agents.create_client_tool if kind == "client" else agents.create_webhook_tool
            created = fn(config={
                "name": f"integration-{kind}-tool",
                "description": f"Integration {kind} tool",
                **({"method": "GET"} if kind == "webhook" else {}),
                **({"url": "https://example.com/hook"} if kind == "webhook" else {}),
            })
            tool_id = _extract_id(r"Tool ID: (\S+)", created.text)

            fn = agents.update_client_tool if kind == "client" else agents.update_webhook_tool
            updated = fn(tool_id=tool_id, config={
                "name": f"integration-{kind}-tool-updated",
                "description": f"Integration {kind} tool updated",
                **({"method": "GET"} if kind == "webhook" else {}),
                **({"expects_response": True} if kind == "client" else {}),
                **({"url": "https://example.com/hook-updated"} if kind == "webhook" else {}),
            })

            assert updated.type == "text"
            assert f"Successfully updated {kind} tool." in updated.text
            assert f"Tool ID: {tool_id}" in updated.text
            assert f"Name: integration-{kind}-tool-updated" in updated.text
        finally:
            if tool_id:
                agents.delete_tool(tool_id=tool_id)


@pytest.mark.parametrize("scenario", ["regular", "attached"], ids=["regular", "attached"])
@pytest.mark.parametrize("kind", ["client", "webhook"], ids=["client", "webhook"])
def test_delete_tool(ctx: tuple[str, Any], vcr: Any, scenario: str, kind: str):
    """Delete tools normally and when they are attached to an agent."""
    target, agents = ctx
    tool_id: str | None = None
    removed, deleted = False, False
    cassette = Path(__file__).parent / "cassettes" / (
        f"test_delete_tool_{kind}_{scenario}_integration.yaml"
    )

    with vcr.use_cassette(str(cassette)):
        try:
            fn = agents.create_client_tool if kind == "client" else agents.create_webhook_tool
            created = fn(config={
                "name": f"integration-{kind}-delete-tool",
                "description": f"Integration delete {kind} tool",
                **({"url": "https://example.com/delete-hook"} if kind == "webhook" else {}),
                **({"method": "GET"} if kind == "webhook" else {}),
            })
            tool_id = _extract_id(r"Tool ID: (\S+)", created.text)

            if scenario == "attached":
                agents.add_tool_to_agent(agent_id=target, tool_id=tool_id)
                blocked = agents.delete_tool(tool_id=tool_id)
                assert blocked.type == "text"
                assert "Tool currently in use by the following agents:" in blocked.text

                removed = agents.remove_tool_from_agent(agent_id=target, tool_id=tool_id)
                assert removed.type == "text"
                assert f"Successfully removed tool {tool_id} from agent {target}." in removed.text
                removed = True

            deleted = agents.delete_tool(tool_id=tool_id)
            assert deleted.type == "text"
            assert f"Successfully deleted tool with ID: {tool_id}" in deleted.text
            deleted = True
        finally:
            if tool_id:
                if not removed and scenario == "attached":
                    agents.remove_tool_from_agent(agent_id=target, tool_id=tool_id)

                if not deleted:
                    agents.delete_tool(tool_id=tool_id)


@pytest.mark.parametrize(
    ("kind", "casette"),
    [
        ("client", "test_add_tool_to_agent_client_integration.yaml"),
        ("webhook", "test_add_tool_to_agent_webhook_integration.yaml"),
    ],
    ids=["client", "webhook"],
)
def test_add_tool_to_agent(ctx: tuple[str, Any], vcr: Any, kind: str, casette: str):
    """Create a tool and attach it to the test agent."""
    target, agents = ctx
    attached = False
    tool_id: str | None = None
    cassette = Path(__file__).parent / "cassettes" / casette

    with vcr.use_cassette(str(cassette)):
        try:
            fn = agents.create_client_tool if kind == "client" else agents.create_webhook_tool
            created = fn(config={
                "name": f"integration-{kind}-attach-tool",
                "description": f"Integration attach {kind} tool",
                **({"url": "https://example.com/attach-hook"} if kind == "webhook" else {}),
                **({"method": "GET"} if kind == "webhook" else {}),
            })
            tool_id = _extract_id(r"Tool ID: (\S+)", created.text)

            added = agents.add_tool_to_agent(agent_id=target, tool_id=tool_id)
            assert added.type == "text"
            assert f"Successfully added tool {tool_id} to agent {target}." in added.text
            attached = True
        finally:
            if tool_id:
                if attached:
                    agents.remove_tool_from_agent(agent_id=target, tool_id=tool_id)
                agents.delete_tool(tool_id=tool_id)


@pytest.mark.parametrize(
    ("kind", "casette"),
    [
        ("client", "test_remove_tool_from_agent_client_integration.yaml"),
        ("webhook", "test_remove_tool_from_agent_webhook_integration.yaml"),
    ],
    ids=["client", "webhook"],
)
def test_remove_tool_from_agent(ctx: tuple[str, Any], vcr: Any, kind: str, casette: str):
    """Create a tool, attach it, then remove it from the test agent."""
    target, agents = ctx
    removed = False
    tool_id: str | None = None
    cassette = Path(__file__).parent / "cassettes" / casette

    with vcr.use_cassette(str(cassette)):
        try:
            fn = agents.create_client_tool if kind == "client" else agents.create_webhook_tool
            created = fn(config={
                "name": f"integration-{kind}-remove-tool",
                "description": f"Integration remove {kind} tool",
                **({"url": "https://example.com/remove-hook"} if kind == "webhook" else {}),
                **({"method": "GET"} if kind == "webhook" else {}),
            })
            tool_id = _extract_id(r"Tool ID: (\S+)", created.text)
            agents.add_tool_to_agent(agent_id=target, tool_id=tool_id)

            removed = agents.remove_tool_from_agent(agent_id=target, tool_id=tool_id)
            assert removed.type == "text"
            assert f"Successfully removed tool {tool_id} from agent {target}." in removed.text
            removed = True
        finally:
            if tool_id and not removed:
                agents.remove_tool_from_agent(agent_id=target, tool_id=tool_id)
            if tool_id:
                agents.delete_tool(tool_id=tool_id)


@pytest.mark.parametrize(
    ("kind", "casette"),
    [
        ("client", "test_get_tool_dependent_agents_client_integration.yaml"),
        ("webhook", "test_get_tool_dependent_agents_webhook_integration.yaml"),
    ],
    ids=["client", "webhook"],
)
def test_get_tool_dependent_agents(ctx: tuple[str, Any], vcr: Any, kind: str, casette: str):
    """List dependent agents for each supported attached tool type."""
    target, agents = ctx
    tool_id: str | None = None
    cassette = Path(__file__).parent / "cassettes" / casette

    with vcr.use_cassette(str(cassette)):
        try:
            fn = agents.create_client_tool if kind == "client" else agents.create_webhook_tool
            created = fn(config={
                "name": f"integration-{kind}-tool-deps",
                "description": f"Integration dependent agents {kind} case",
                **({"url": "https://example.com/dependent-hook"} if kind == "webhook" else {}),
                **({"method": "GET"} if kind == "webhook" else {}),
            })
            tool_id = _extract_id(r"Tool ID: (\S+)", created.text)
            agents.add_tool_to_agent(agent_id=target, tool_id=tool_id)

            response = agents.get_tool_dependent_agents(tool_id=tool_id, page_size=10)
            assert response.type == "text"
            assert "Agents using this tool (" in response.text and target in response.text
        finally:
            if tool_id:
                agents.remove_tool_from_agent(agent_id=target, tool_id=tool_id)
                agents.delete_tool(tool_id=tool_id)
