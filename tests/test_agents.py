"""
Unit tests for elevenlabs_mcp/agents.py

These tests mock the ElevenLabs client to test tool logic without making API calls.
"""

import pytest

from unittest.mock import Mock, patch

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

    def test_list_agents_empty(self, mock_client):
        """Test list_agents with no agents."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            patched_client.conversational_ai.agents.list.return_value = Mock(agents=[])
            result = list_agents()

            assert result.type == "text"
            assert "No agents found" in result.text

    def test_list_agents_with_data(self, mock_client):
        """Test list_agents with agent data."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_agent = Mock()
            mock_agent.name = "Test Agent"
            mock_agent.agent_id = "agent123"

            patched_client.conversational_ai.agents.list.return_value = Mock(agents=[mock_agent])
            result = list_agents()

            assert result.type == "text"
            assert "Test Agent" in result.text
            assert "agent123" in result.text


class TestGetAgent:
    """Tests for the get_agent function."""

    def test_get_agent(self, mock_client):
        """Test getting a specific agent."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_tts = Mock()
            mock_tts.voice_id = "voice123"

            mock_metadata = Mock()
            mock_metadata.created_at_unix_secs = 1700000000

            mock_response = Mock()
            mock_response.name = "Test Agent"
            mock_response.agent_id = "agent123"
            mock_response.conversation_config = Mock(tts=mock_tts)
            mock_response.metadata = mock_metadata

            patched_client.conversational_ai.agents.get.return_value = mock_response

            result = get_agent("agent123")

            assert result.type == "text"
            assert "Test Agent" in result.text
            assert "agent123" in result.text
            assert "voice123" in result.text
            patched_client.conversational_ai.agents.get.assert_called_once_with(agent_id="agent123")

    def test_get_agent_no_tts(self, mock_client):
        """Test getting an agent without TTS config."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_metadata = Mock()
            mock_metadata.created_at_unix_secs = 1700000000

            mock_response = Mock()
            mock_response.name = "Test Agent"
            mock_response.agent_id = "agent123"
            mock_response.conversation_config = Mock(tts=None)
            mock_response.metadata = mock_metadata

            patched_client.conversational_ai.agents.get.return_value = mock_response
            result = get_agent("agent123")

            assert result.type == "text"
            assert "None" in result.text


class TestListWorkspaceTools:
    """Tests for the list_workspace_tools function."""

    def test_list_workspace_tools_empty(self, mock_client):
        """Test list_workspace_tools with no tools."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            patched_client.conversational_ai.tools.list.return_value = Mock(tools=[])
            result = list_workspace_tools()

            assert result.type == "text"
            assert "No tools found" in result.text

    def test_list_workspace_tools_with_data(self, mock_client):
        """Test list_workspace_tools with tool data."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_tool_config = Mock()
            mock_tool_config.type = "webhook"
            mock_tool_config.name = "Test Tool"

            mock_tool = Mock()
            mock_tool.id = "tool123"
            mock_tool.tool_config = mock_tool_config

            patched_client.conversational_ai.tools.list.return_value = Mock(tools=[mock_tool])
            result = list_workspace_tools()

            assert result.type == "text"
            assert "Test Tool" in result.text
            assert "tool123" in result.text
            assert "webhook" in result.text


class TestGetTool:
    """Tests for the get_tool function."""

    def test_get_tool_webhook(self, mock_client):
        """Test getting a webhook tool."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_api_schema = Mock()
            mock_api_schema.url = "https://example.com/api"
            mock_api_schema.method = "POST"

            mock_config = Mock()
            mock_config.type = "webhook"
            mock_config.name = "Webhook Tool"
            mock_config.description = "A webhook tool"
            mock_config.api_schema = mock_api_schema
            mock_config.response_timeout_secs = 30

            mock_response = Mock()
            mock_response.id = "tool123"
            mock_response.tool_config = mock_config

            patched_client.conversational_ai.tools.get.return_value = mock_response
            result = get_tool("tool123")

            assert result.type == "text"
            assert "tool123" in result.text
            assert "webhook" in result.text
            assert "https://example.com/api" in result.text
            assert "POST" in result.text

    def test_get_tool_client(self, mock_client):
        """Test getting a client tool."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_config = Mock()
            mock_config.type = "client"
            mock_config.name = "Client Tool"
            mock_config.description = "A client tool"
            mock_config.response_timeout_secs = None
            del mock_config.api_schema  # Client tools don't have api_schema

            mock_response = Mock()
            mock_response.id = "tool456"
            mock_response.tool_config = mock_config

            patched_client.conversational_ai.tools.get.return_value = mock_response
            result = get_tool("tool456")

            assert result.type == "text"
            assert "tool456" in result.text
            assert "client" in result.text


class TestCreateWebhookTool:
    """Tests for the create_webhook_tool function."""

    def test_create_webhook_tool_simple(self, mock_client):
        """Test creating a simple webhook tool."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_response = Mock()
            mock_response.id = "newtool123"

            patched_client.conversational_ai.tools.create.return_value = mock_response
            result = create_webhook_tool(
                name="Weather API",
                description="Get weather data",
                url="https://api.weather.com/v1",
                method="GET",
            )

            assert result.type == "text"
            assert "newtool123" in result.text
            assert "Weather API" in result.text
            patched_client.conversational_ai.tools.create.assert_called_once()

    def test_create_webhook_tool_with_params(self, mock_client):
        """Test creating a webhook tool with parameters (no headers)."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_response = Mock()
            mock_response.id = "newtool456"

            patched_client.conversational_ai.tools.create.return_value = mock_response
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

    def test_create_webhook_tool_failure(self, mock_client):
        """Test webhook tool creation failure."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            patched_client.conversational_ai.tools.create.side_effect = Exception(
                "API Error"
            )

            with pytest.raises(ElevenLabsMcpError):
                create_webhook_tool(
                    name="Failed Tool",
                    description="This will fail",
                    url="https://api.example.com",
                )


class TestCreateClientTool:
    """Tests for the create_client_tool function."""

    def test_create_client_tool_simple(self, mock_client):
        """Test creating a simple client tool."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_response = Mock()
            mock_response.id = "clienttool123"

            patched_client.conversational_ai.tools.create.return_value = mock_response

            result = create_client_tool(
                name="Navigate",
                description="Navigate to a page",
            )

            assert result.type == "text"
            assert "clienttool123" in result.text
            assert "Navigate" in result.text

    def test_create_client_tool_with_params(self, mock_client):
        """Test creating a client tool with parameters."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_response = Mock()
            mock_response.id = "clienttool456"

            patched_client.conversational_ai.tools.create.return_value = mock_response

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

    def test_update_webhook_tool(self, mock_client):
        """Test updating a webhook tool."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_existing_schema = Mock()
            mock_existing_schema.url = "https://old.api.com"
            mock_existing_schema.method = "GET"
            mock_existing_schema.request_headers = None
            mock_existing_schema.path_params_schema = None
            mock_existing_schema.query_params_schema = None
            mock_existing_schema.request_body_schema = None

            mock_existing_config = Mock()
            mock_existing_config.type = "webhook"
            mock_existing_config.name = "Old Name"
            mock_existing_config.description = "Old description"
            mock_existing_config.api_schema = mock_existing_schema
            mock_existing_config.response_timeout_secs = 30

            mock_existing = Mock()
            mock_existing.id = "tool123"
            mock_existing.tool_config = mock_existing_config

            mock_updated_config = Mock()
            mock_updated_config.name = "New Name"

            mock_updated = Mock()
            mock_updated.id = "tool123"
            mock_updated.tool_config = mock_updated_config

            patched_client.conversational_ai.tools.get.return_value = mock_existing
            patched_client.conversational_ai.tools.update.return_value = mock_updated

            result = update_tool(tool_id="tool123", name="New Name")

            assert result.type == "text"
            assert "tool123" in result.text
            patched_client.conversational_ai.tools.update.assert_called_once()

    def test_update_client_tool(self, mock_client):
        """Test updating a client tool."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_existing_config = Mock()
            mock_existing_config.type = "client"
            mock_existing_config.name = "Old Client"
            mock_existing_config.description = "Old description"
            mock_existing_config.expects_response = False
            mock_existing_config.parameters = None
            mock_existing_config.response_timeout_secs = None

            mock_existing = Mock()
            mock_existing.id = "tool456"
            mock_existing.tool_config = mock_existing_config

            mock_updated_config = Mock()
            mock_updated_config.name = "New Client"

            mock_updated = Mock()
            mock_updated.id = "tool456"
            mock_updated.tool_config = mock_updated_config

            patched_client.conversational_ai.tools.get.return_value = mock_existing
            patched_client.conversational_ai.tools.update.return_value = mock_updated

            result = update_tool(tool_id="tool456", description="New description")

            assert result.type == "text"
            patched_client.conversational_ai.tools.update.assert_called_once()


class TestDeleteTool:
    """Tests for the delete_tool function."""

    def test_delete_tool_success(self, mock_client):
        """Test successful tool deletion."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            patched_client.conversational_ai.tools.get_dependent_agents.return_value = Mock(agents=[])
            patched_client.conversational_ai.tools.delete.return_value = None

            result = delete_tool("tool123")

            assert result.type == "text"
            assert "Successfully deleted" in result.text
            assert "tool123" in result.text

    def test_delete_tool_with_dependents(self, mock_client):
        """Test deleting a tool with dependent agents."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_agent = Mock()
            mock_agent.name = "Dependent Agent"

            patched_client.conversational_ai.tools.get_dependent_agents.return_value = Mock(agents=[mock_agent])
            result = delete_tool("tool123")

            assert result.type == "text"
            assert "Cannot delete" in result.text
            assert "Dependent Agent" in result.text
            patched_client.conversational_ai.tools.delete.assert_not_called()


class TestAddToolToAgent:
    """Tests for the add_tool_to_agent function."""

    def test_add_tool_to_agent(self, mock_client):
        """Test adding a tool to an agent."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_prompt_config = Mock()
            mock_prompt_config.tool_ids = []

            mock_agent = Mock()
            mock_agent.conversation_config = Mock(agent=mock_prompt_config)

            patched_client.conversational_ai.agents.get.return_value = mock_agent
            patched_client.conversational_ai.agents.update.return_value = Mock()

            result = add_tool_to_agent(agent_id="agent123", tool_id="tool456")

            assert result.type == "text"
            assert "Successfully added" in result.text
            assert "tool456" in result.text
            assert "agent123" in result.text

    def test_add_tool_already_attached(self, mock_client):
        """Test adding a tool that's already attached."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_prompt_config = Mock()
            mock_prompt_config.tool_ids = ["tool456"]

            mock_agent = Mock()
            mock_agent.conversation_config = Mock(agent=mock_prompt_config)

            patched_client.conversational_ai.agents.get.return_value = mock_agent

            result = add_tool_to_agent(agent_id="agent123", tool_id="tool456")

            assert result.type == "text"
            assert "already attached" in result.text
            patched_client.conversational_ai.agents.update.assert_not_called()


class TestRemoveToolFromAgent:
    """Tests for the remove_tool_from_agent function."""

    def test_remove_tool_from_agent(self, mock_client):
        """Test removing a tool from an agent."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_prompt_config = Mock()
            mock_prompt_config.tool_ids = ["tool456", "tool789"]

            mock_agent = Mock()
            mock_agent.conversation_config = Mock(agent=mock_prompt_config)

            patched_client.conversational_ai.agents.get.return_value = mock_agent
            patched_client.conversational_ai.agents.update.return_value = Mock()

            result = remove_tool_from_agent(agent_id="agent123", tool_id="tool456")

            assert result.type == "text"
            assert "Successfully removed" in result.text

    def test_remove_tool_not_attached(self, mock_client):
        """Test removing a tool that's not attached."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_prompt_config = Mock()
            mock_prompt_config.tool_ids = []

            mock_agent = Mock()
            mock_agent.conversation_config = Mock(agent=mock_prompt_config)

            patched_client.conversational_ai.agents.get.return_value = mock_agent
            result = remove_tool_from_agent(agent_id="agent123", tool_id="tool456")

            assert result.type == "text"
            assert "not attached" in result.text


class TestGetToolDependentAgents:
    """Tests for the get_tool_dependent_agents function."""

    def test_get_dependent_agents_empty(self, mock_client):
        """Test getting dependent agents with no results."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            patched_client.conversational_ai.tools.get_dependent_agents.return_value = Mock(agents=[])
            result = get_tool_dependent_agents("tool123")

            assert result.type == "text"
            assert "No agents are using this tool" in result.text

    def test_get_dependent_agents_with_data(self, mock_client):
        """Test getting dependent agents with results."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_agent = Mock()
            mock_agent.name = "Using Agent"
            mock_agent.id = "agent123"
            mock_agent.type = "available"

            patched_client.conversational_ai.tools.get_dependent_agents.return_value = Mock(agents=[mock_agent])
            result = get_tool_dependent_agents("tool123")

            assert result.type == "text"
            assert "1 total" in result.text


class TestCreateAgent:
    """Tests for the create_agent function."""

    def test_create_agent(self, mock_client):
        """Test creating an agent."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            with patch("elevenlabs_mcp.agents.create_conversation_config") as mock_config:
                with patch("elevenlabs_mcp.agents.create_platform_settings") as mock_platform:
                    mock_config.return_value = {}
                    mock_platform.return_value = {}

                    mock_response = Mock()
                    mock_response.agent_id = "newagent123"

                    patched_client.conversational_ai.agents.create.return_value = mock_response
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

    def test_add_knowledge_base_with_text(self, mock_client):
        """Test adding knowledge base from text."""
        with patch("elevenlabs_mcp.agents.client") as patched_client:
            mock_kb_response = Mock()
            mock_kb_response.id = "kb456"

            mock_agent_config = {"prompt": {"knowledge_base": []}}
            mock_agent = Mock()
            mock_agent.conversation_config = Mock(agent=mock_agent_config)

            patched_client.conversational_ai.knowledge_base.documents.create_from_file.return_value = (mock_kb_response)
            patched_client.conversational_ai.agents.get.return_value = mock_agent
            patched_client.conversational_ai.agents.update.return_value = Mock()

            result = add_knowledge_base_to_agent(
                agent_id="agent123",
                knowledge_base_name="Text KB",
                text="This is some knowledge base content.",
            )

            assert result.type == "text"
            assert "kb456" in result.text
