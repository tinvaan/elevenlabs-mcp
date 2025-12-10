"""
Unit tests for elevenlabs_mcp/agents.py

These tests mock the ElevenLabs client to test tool logic without making API calls.
"""

import pytest

from unittest.mock import patch

from conftest import create_mock
from elevenlabs_mcp.agents import (
    get_tool,
    get_agent,
    list_agents,
    create_agent,
    list_workspace_tools,
    create_webhook_tool,
    create_client_tool,
    update_tool,
    delete_tool,
    add_tool_to_agent,
    remove_tool_from_agent,
    get_tool_dependent_agents,
    add_knowledge_base_to_agent
)
from elevenlabs_mcp.utils import ElevenLabsMcpError

class TestListAgents:
    """Tests for the list_agents function."""

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.agents.list",
        return_value=create_mock(agents=[]),
    )
    def test_list_agents_empty(self, mock_list, mock_client):
        """Test list_agents with no agents."""
        result = list_agents()

        assert result.type == "text"
        assert "No agents found" in result.text

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.agents.list",
        return_value=create_mock(
            agents=[
                create_mock(
                    name="Test Agent",
                    agent_id="agent123",
                )
            ]
        ),
    )
    def test_list_agents_with_data(self, mock_list, mock_client):
        """Test list_agents with agent data."""
        result = list_agents()

        assert result.type == "text"
        assert "Test Agent" in result.text
        assert "agent123" in result.text


class TestGetAgent:
    """Tests for the get_agent function."""

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.agents.get",
        return_value=create_mock(
            name="Test Agent",
            agent_id="agent123",
            conversation_config=create_mock(
                tts=create_mock(voice_id="voice123")
            ),
            metadata=create_mock(created_at_unix_secs=1700000000),
        ),
    )
    def test_get_agent(self, mock_get, mock_client):
        """Test getting a specific agent."""
        result = get_agent("agent123")

        assert result.type == "text"
        assert "Test Agent" in result.text
        assert "agent123" in result.text
        assert "voice123" in result.text
        mock_get.assert_called_once_with(agent_id="agent123")

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.agents.get",
        return_value=create_mock(
            name="Test Agent",
            agent_id="agent123",
            conversation_config=create_mock(tts=None),
            metadata=create_mock(created_at_unix_secs=1700000000),
        ),
    )
    def test_get_agent_no_tts(self, mock_get, mock_client):
        """Test getting an agent without TTS config."""
        result = get_agent("agent123")

        assert result.type == "text"
        assert "None" in result.text


class TestListWorkspaceTools:
    """Tests for the list_workspace_tools function."""

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.list",
        return_value=create_mock(tools=[]),
    )
    def test_list_workspace_tools_empty(self, mock_list, mock_client):
        """Test list_workspace_tools with no tools."""
        result = list_workspace_tools()

        assert result.type == "text"
        assert "No tools found" in result.text

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.list",
        return_value=create_mock(
            tools=[
                create_mock(
                    id="tool123",
                    tool_config=create_mock(
                        type="webhook",
                        name="Test Tool",
                    ),
                )
            ]
        ),
    )
    def test_list_workspace_tools_with_data(self, mock_list, mock_client):
        """Test list_workspace_tools with tool data."""
        result = list_workspace_tools()

        assert result.type == "text"
        assert "Test Tool" in result.text
        assert "tool123" in result.text
        assert "webhook" in result.text


class TestGetTool:
    """Tests for the get_tool function."""

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.get",
        return_value=create_mock(
            id="tool123",
            tool_config=create_mock(
                type="webhook",
                name="Webhook Tool",
                description="A webhook tool",
                api_schema=create_mock(
                    url="https://example.com/api",
                    method="POST",
                ),
                response_timeout_secs=30,
            ),
        ),
    )
    def test_get_tool_webhook(self, mock_get, mock_client):
        """Test getting a webhook tool."""
        result = get_tool("tool123")

        assert result.type == "text"
        assert "tool123" in result.text
        assert "webhook" in result.text
        assert "https://example.com/api" in result.text
        assert "POST" in result.text

    @patch("elevenlabs_mcp.agents.client.conversational_ai.tools.get")
    def test_get_tool_client(self, mock_get, mock_client):
        """Test getting a client tool."""
        mock_config = create_mock(
            type="client",
            name="Client Tool",
            description="A client tool",
            response_timeout_secs=None,
        )
        del mock_config.api_schema  # Client tools don't have api_schema

        mock_get.return_value = create_mock(
            id="tool456",
            tool_config=mock_config,
        )

        result = get_tool("tool456")

        assert result.type == "text"
        assert "tool456" in result.text
        assert "client" in result.text


class TestCreateWebhookTool:
    """Tests for the create_webhook_tool function."""

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.create",
        return_value=create_mock(id="newtool123"),
    )
    def test_create_webhook_tool_simple(self, mock_create, mock_client):
        """Test creating a simple webhook tool."""
        result = create_webhook_tool(
            name="Weather API",
            description="Get weather data",
            url="https://api.weather.com/v1",
            method="GET",
        )

        assert result.type == "text"
        assert "newtool123" in result.text
        assert "Weather API" in result.text
        mock_create.assert_called_once()

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.create",
        return_value=create_mock(id="newtool456"),
    )
    def test_create_webhook_tool_with_params(self, mock_create, mock_client):
        """Test creating a webhook tool with parameters (no headers)."""
        result = create_webhook_tool(
            name="User API",
            description="User management",
            url="https://api.example.com/users/{user_id}",
            method="POST",
            path_params={"user_id": "The user ID"},
            query_params={"include": "Include related data"},
            request_body_properties={"name": "User name", "email": "User email"},
            request_body_required=["name"],
        )

        assert result.type == "text"
        assert "newtool456" in result.text

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.create",
        side_effect=Exception("API Error"),
    )
    def test_create_webhook_tool_failure(self, mock_create, mock_client):
        """Test webhook tool creation failure."""
        with pytest.raises(ElevenLabsMcpError):
            create_webhook_tool(
                name="Failed Tool",
                description="This will fail",
                url="https://api.example.com",
            )


class TestCreateClientTool:
    """Tests for the create_client_tool function."""

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.create",
        return_value=create_mock(id="clienttool123"),
    )
    def test_create_client_tool_simple(self, mock_create, mock_client):
        """Test creating a simple client tool."""
        result = create_client_tool(
            name="Navigate",
            description="Navigate to a page",
        )

        assert result.type == "text"
        assert "clienttool123" in result.text
        assert "Navigate" in result.text

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.create",
        return_value=create_mock(id="clienttool456"),
    )
    def test_create_client_tool_with_params(self, mock_create, mock_client):
        """Test creating a client tool with parameters."""
        result = create_client_tool(
            name="Form Submit",
            description="Submit a form",
            expects_response=True,
            parameters_properties={"form_id": "The form ID", "data": "Form data"},
            parameters_required=["form_id"],
            response_timeout_secs=10,
        )

        assert result.type == "text"
        assert "clienttool456" in result.text
        assert "True" in result.text


class TestUpdateTool:
    """Tests for the update_tool function."""

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.update",
        return_value=create_mock(
            id="tool123",
            tool_config=create_mock(name="New Name"),
        ),
    )
    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.get",
        return_value=create_mock(
            id="tool123",
            tool_config=create_mock(
                type="webhook",
                name="Old Name",
                description="Old description",
                api_schema=create_mock(
                    url="https://old.api.com",
                    method="GET",
                    request_headers=None,
                    path_params_schema=None,
                    query_params_schema=None,
                    request_body_schema=None,
                ),
                response_timeout_secs=30,
            ),
        ),
    )
    def test_update_webhook_tool(self, mock_get, mock_update, mock_client):
        """Test updating a webhook tool."""
        result = update_tool(tool_id="tool123", name="New Name")

        assert result.type == "text"
        assert "tool123" in result.text
        mock_update.assert_called_once()

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.update",
        return_value=create_mock(
            id="tool456",
            tool_config=create_mock(name="New Client"),
        ),
    )
    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.get",
        return_value=create_mock(
            id="tool456",
            tool_config=create_mock(
                type="client",
                name="Old Client",
                description="Old description",
                expects_response=False,
                parameters=None,
                response_timeout_secs=None,
            ),
        ),
    )
    def test_update_client_tool(self, mock_get, mock_update, mock_client):
        """Test updating a client tool."""
        result = update_tool(tool_id="tool456", description="New description")

        assert result.type == "text"
        mock_update.assert_called_once()


class TestDeleteTool:
    """Tests for the delete_tool function."""

    @patch("elevenlabs_mcp.agents.client.conversational_ai.tools.delete", return_value=None)
    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.get_dependent_agents",
        return_value=create_mock(agents=[]),
    )
    def test_delete_tool_success(self, mock_get_dependent, mock_delete, mock_client):
        """Test successful tool deletion."""
        result = delete_tool("tool123")

        assert result.type == "text"
        assert "Successfully deleted" in result.text
        assert "tool123" in result.text

    @patch("elevenlabs_mcp.agents.client.conversational_ai.tools.delete")
    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.get_dependent_agents",
        return_value=create_mock(
            agents=[create_mock(name="Dependent Agent")]
        ),
    )
    def test_delete_tool_with_dependents(self, mock_get_dependent, mock_delete, mock_client):
        """Test deleting a tool with dependent agents."""
        result = delete_tool("tool123")

        assert result.type == "text"
        assert "Cannot delete" in result.text
        assert "Dependent Agent" in result.text
        mock_delete.assert_not_called()


class TestAddToolToAgent:
    """Tests for the add_tool_to_agent function."""

    @patch("elevenlabs_mcp.agents.client.conversational_ai.agents.update", return_value=create_mock())
    @patch("elevenlabs_mcp.agents.client.conversational_ai.agents.get")
    def test_add_tool_to_agent(self, mock_get, mock_update, mock_client):
        """Test adding a tool to an agent."""
        mock_get.return_value = create_mock(
            conversation_config=create_mock(
                agent=create_mock(tool_ids=[])
            )
        )

        result = add_tool_to_agent(agent_id="agent123", tool_id="tool456")

        assert result.type == "text"
        assert "Successfully added" in result.text
        assert "tool456" in result.text
        assert "agent123" in result.text

    @patch("elevenlabs_mcp.agents.client.conversational_ai.agents.update")
    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.agents.get",
        return_value=create_mock(
            conversation_config=create_mock(
                agent=create_mock(tool_ids=["tool456"])
            )
        ),
    )
    def test_add_tool_already_attached(self, mock_get, mock_update, mock_client):
        """Test adding a tool that's already attached."""
        result = add_tool_to_agent(agent_id="agent123", tool_id="tool456")

        assert result.type == "text"
        assert "already attached" in result.text
        mock_update.assert_not_called()


class TestRemoveToolFromAgent:
    """Tests for the remove_tool_from_agent function."""

    @patch("elevenlabs_mcp.agents.client.conversational_ai.agents.update", return_value=create_mock())
    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.agents.get",
        return_value=create_mock(
            conversation_config=create_mock(
                agent=create_mock(prompt=create_mock(tool_ids=["tool456", "tool789"]))
            )
        ),
    )
    def test_remove_tool_from_agent(self, mock_get, mock_update, mock_client):
        """Test removing a tool from an agent."""
        result = remove_tool_from_agent(agent_id="agent123", tool_id="tool456")

        assert result.type == "text"
        assert "Successfully removed" in result.text

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.agents.get",
        return_value=create_mock(
            conversation_config=create_mock(
                agent=create_mock(prompt=create_mock(tool_ids=[]))
            )
        ),
    )
    def test_remove_tool_not_attached(self, mock_get, mock_client):
        """Test removing a tool that's not attached."""
        result = remove_tool_from_agent(agent_id="agent123", tool_id="tool456")

        assert result.type == "text"
        assert "not attached" in result.text


class TestGetToolDependentAgents:
    """Tests for the get_tool_dependent_agents function."""

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.get_dependent_agents",
        return_value=create_mock(agents=[]),
    )
    def test_get_dependent_agents_empty(self, mock_get_dependent, mock_client):
        """Test getting dependent agents with no results."""
        result = get_tool_dependent_agents("tool123")

        assert result.type == "text"
        assert "No agents are using this tool" in result.text

    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.tools.get_dependent_agents",
        return_value=create_mock(
            agents=[
                create_mock(
                    name="Using Agent",
                    id="agent123",
                    type="available",
                )
            ]
        ),
    )
    def test_get_dependent_agents_with_data(self, mock_get_dependent, mock_client):
        """Test getting dependent agents with results."""
        result = get_tool_dependent_agents("tool123")

        assert result.type == "text"
        assert "1 total" in result.text


class TestCreateAgent:
    """Tests for the create_agent function."""

    @patch("elevenlabs_mcp.agents.create_platform_settings", return_value={})
    @patch("elevenlabs_mcp.agents.create_conversation_config", return_value={})
    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.agents.create",
        return_value=create_mock(agent_id="newagent123"),
    )
    def test_create_agent(self, mock_create, mock_config, mock_platform, mock_client):
        """Test creating an agent."""
        result = create_agent(
            name="New Agent",
            first_message="Hello!",
            system_prompt="You are helpful.",
        )

        assert result.type == "text"
        assert "New Agent" in result.text
        assert "newagent123" in result.text


class TestAddKnowledgeBaseToAgent:
    """Tests for the add_knowledge_base_to_agent function."""

    @patch("elevenlabs_mcp.agents.client.conversational_ai.agents.update", return_value=create_mock())
    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.agents.get",
        return_value=create_mock(
            conversation_config=create_mock(
                agent={"prompt": {"knowledge_base": []}}
            )
        ),
    )
    @patch(
        "elevenlabs_mcp.agents.client.conversational_ai.knowledge_base.documents.create_from_file",
        return_value=create_mock(id="kb456"),
    )
    def test_add_knowledge_base_with_text(self, mock_create_kb, mock_get, mock_update, mock_client):
        """Test adding knowledge base from text."""
        result = add_knowledge_base_to_agent(
            agent_id="agent123",
            knowledge_base_name="Text KB",
            text="This is some knowledge base content.",
        )

        assert result.type == "text"
        assert "kb456" in result.text
