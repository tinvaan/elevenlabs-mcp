"""
Unit tests for elevenlabs_mcp/server.py

These tests mock the ElevenLabs client to test tool logic without making API calls.
"""

import pytest

from unittest.mock import Mock, patch

from conftest import create_mock
from elevenlabs_mcp.server import (
    format_diarized_transcript,
    search_voices,
    list_models,
    get_voice,
    list_phone_numbers,
    get_conversation,
    list_conversations,
    search_voice_library,
    get_elevenlabs_resource,
    _is_broken_pipe_error
)


class TestFormatDiarizedTranscript:
    """Tests for the format_diarized_transcript function."""

    def test_simple_diarization(self, mock_client):
        """Test formatting transcript with speaker labels."""

        mock_transcription = Mock()
        mock_transcription.text = "Hello world"
        mock_transcription.words = [
            Mock(speaker_id="speaker_1", text="Hello", type="word"),
            Mock(speaker_id="speaker_1", text="world", type="word"),
        ]

        result = format_diarized_transcript(mock_transcription)
        assert "SPEAKER 1:" in result
        assert "Hello" in result
        assert "world" in result

    def test_multiple_speakers(self, mock_client):
        """Test formatting with multiple speakers."""

        mock_transcription = Mock()
        mock_transcription.text = "Hi there How are you"
        mock_transcription.words = [
            Mock(speaker_id="speaker_1", text="Hi", type="word"),
            Mock(speaker_id="speaker_1", text="there", type="word"),
            Mock(speaker_id="speaker_2", text="How", type="word"),
            Mock(speaker_id="speaker_2", text="are", type="word"),
            Mock(speaker_id="speaker_2", text="you", type="word"),
        ]

        result = format_diarized_transcript(mock_transcription)
        assert "SPEAKER 1:" in result
        assert "SPEAKER 2:" in result
        assert "Hi there" in result
        assert "How are you" in result

    def test_no_words_fallback(self, mock_client):
        """Test fallback to text when words are not available."""

        mock_transcription = Mock()
        mock_transcription.text = "Fallback text"
        mock_transcription.words = None

        result = format_diarized_transcript(mock_transcription)
        assert result == "Fallback text"

    def test_dict_words(self, mock_client):
        """Test handling words as dictionaries."""

        mock_transcription = Mock()
        mock_transcription.text = "Dict words"
        mock_transcription.words = [
            {"speaker_id": "speaker_1", "text": "Dict", "type": "word"},
            {"speaker_id": "speaker_1", "text": "words", "type": "word"},
        ]

        result = format_diarized_transcript(mock_transcription)
        assert "SPEAKER 1:" in result
        assert "Dict" in result


class TestSearchVoices:
    """Tests for the search_voices function."""

    @patch(
        "elevenlabs_mcp.server.client.voices.search",
        return_value=create_mock(
            voices=[
                create_mock(
                    voice_id="voice123",
                    name="Test Voice",
                    category="generated",
                )
            ]
        ),
    )
    def test_search_voices_returns_list(self, mock_search, mock_client):
        """Test that search_voices returns a list of McpVoice objects."""
        result = search_voices(search="test")

        assert len(result) == 1
        assert result[0].id == "voice123"
        assert result[0].name == "Test Voice"
        assert result[0].category == "generated"

    @patch(
        "elevenlabs_mcp.server.client.voices.search",
        return_value=create_mock(voices=[]),
    )
    def test_search_voices_empty(self, mock_search, mock_client):
        """Test search_voices with no results."""
        result = search_voices(search="nonexistent")

        assert len(result) == 0


class TestListModels:
    """Tests for the list_models function."""

    @patch(
        "elevenlabs_mcp.server.client.models.list",
        return_value=[
            create_mock(
                model_id="model123",
                name="Test Model",
                languages=[create_mock(language_id="en", name="English")],
            )
        ],
    )
    def test_list_models(self, mock_list):
        """Test that list_models returns model information."""
        result = list_models()

        assert len(result) == 1
        assert result[0].id == "model123"
        assert result[0].name == "Test Model"
        assert len(result[0].languages) == 1
        assert result[0].languages[0].language_id == "en"


class TestGetVoice:
    """Tests for the get_voice function."""

    @patch(
        "elevenlabs_mcp.server.client.voices.get",
        return_value=create_mock(
            voice_id="voice123",
            name="Test Voice",
            category="professional",
            fine_tuning=create_mock(state={"status": "ready"}),
        ),
    )
    def test_get_voice(self, mock_get, mock_client):
        """Test getting a specific voice."""
        result = get_voice("voice123")

        assert result.id == "voice123"
        assert result.name == "Test Voice"
        mock_get.assert_called_once_with(voice_id="voice123")


class TestListPhoneNumbers:
    """Tests for the list_phone_numbers function."""

    @patch(
        "elevenlabs_mcp.server.client.conversational_ai.phone_numbers.list",
        return_value=[],
    )
    def test_list_phone_numbers_empty(self, mock_list, mock_client):
        """Test list_phone_numbers with no phone numbers."""
        result = list_phone_numbers()

        assert result.type == "text"
        assert "No phone numbers found" in result.text

    @patch(
        "elevenlabs_mcp.server.client.conversational_ai.phone_numbers.list",
        return_value=[
            create_mock(
                phone_number="+1234567890",
                phone_number_id="phone123",
                provider="twilio",
                label="Main",
                assigned_agent=create_mock(
                    agent_name="Test Agent",
                    agent_id="agent123",
                ),
            )
        ],
    )
    def test_list_phone_numbers_with_data(self, mock_list, mock_client):
        """Test list_phone_numbers with phone numbers."""
        result = list_phone_numbers()

        assert result.type == "text"
        assert "+1234567890" in result.text
        assert "phone123" in result.text
        assert "twilio" in result.text


class TestGetConversation:
    """Tests for the get_conversation function."""

    @patch(
        "elevenlabs_mcp.server.client.conversational_ai.conversations.get",
        return_value=create_mock(
            conversation_id="conv123",
            status="completed",
            agent_id="agent123",
            transcript=[
                create_mock(role="agent", message="Hello", tool_calls=None, feedback=None)
            ],
            metadata=None,
            analysis=None,
        ),
    )
    def test_get_conversation(self, mock_get, mock_client):
        """Test getting a conversation."""
        result = get_conversation("conv123")

        assert result.type == "text"
        assert "conv123" in result.text
        assert "completed" in result.text

    @patch(
        "elevenlabs_mcp.server.client.conversational_ai.conversations.get",
        return_value=create_mock(
            conversation_id="conv123",
            status="completed",
            agent_id="agent123",
            transcript=[],
            metadata=create_mock(call_duration_secs=120),
            analysis=None,
        ),
    )
    def test_get_conversation_with_metadata(self, mock_get, mock_client):
        """Test getting a conversation with metadata."""
        result = get_conversation("conv123")

        assert "120 seconds" in result.text


class TestListConversations:
    """Tests for the list_conversations function."""

    @patch(
        "elevenlabs_mcp.server.client.conversational_ai.conversations.list",
        return_value=create_mock(conversations=[]),
    )
    def test_list_conversations_empty(self, mock_list, mock_client):
        """Test list_conversations with no conversations."""
        result = list_conversations()

        assert result.type == "text"
        assert "No conversations found" in result.text

    @patch(
        "elevenlabs_mcp.server.client.conversational_ai.conversations.list",
        return_value=create_mock(
            conversations=[
                create_mock(
                    conversation_id="conv123",
                    status="completed",
                    agent_name="Test Agent",
                    agent_id="agent123",
                    start_time_unix_secs=1700000000,
                    call_duration_secs=60,
                    message_count=5,
                    call_successful=True,
                )
            ],
            has_more=False,
            next_cursor=None,
        ),
    )
    def test_list_conversations_with_data(self, mock_list, mock_client):
        """Test list_conversations with data."""
        result = list_conversations()

        assert result.type == "text"
        assert "conv123" in result.text
        assert "completed" in result.text


class TestSearchVoiceLibrary:
    """Tests for the search_voice_library function."""

    @patch(
        "elevenlabs_mcp.server.client.voices.get_shared",
        return_value=create_mock(voices=[]),
    )
    def test_search_voice_library_empty(self, mock_get_shared, mock_client):
        """Test search with no results."""
        result = search_voice_library(search="nonexistent")

        assert result.type == "text"
        assert "No shared voices found" in result.text

    @patch(
        "elevenlabs_mcp.server.client.voices.get_shared",
        return_value=create_mock(
            voices=[
                create_mock(
                    name="Test Voice",
                    voice_id="voice123",
                    category="professional",
                    verified_languages=[],
                )
            ]
        ),
    )
    def test_search_voice_library_with_results(self, mock_get_shared, mock_client):
        """Test search with results."""
        result = search_voice_library(search="test")

        assert result.type == "text"
        assert "Test Voice" in result.text
        assert "voice123" in result.text


class TestGetElevenlabsResource:
    """Tests for the get_elevenlabs_resource function."""

    def test_get_resource_file_not_found(self, mock_client, temp_dir):
        """Test accessing a non-existent resource."""
        with patch("elevenlabs_mcp.server.base_path", str(temp_dir)):
            with pytest.raises(FileNotFoundError):
                get_elevenlabs_resource("nonexistent.mp3")


class TestIsBrokenPipeError:
    """Tests for the _is_broken_pipe_error helper."""

    def test_broken_pipe_error(self, mock_client):
        """Test detecting BrokenPipeError."""
        assert _is_broken_pipe_error(BrokenPipeError()) is True

    def test_other_error(self, mock_client):
        """Test non-BrokenPipeError."""
        assert _is_broken_pipe_error(ValueError("test")) is False

    def test_exception_group_all_broken_pipe(self, mock_client):
        """Test ExceptionGroup with all BrokenPipeErrors."""
        group = BaseExceptionGroup("test", [BrokenPipeError(), BrokenPipeError()])
        assert _is_broken_pipe_error(group) is True

    def test_exception_group_mixed(self, mock_client):
        """Test ExceptionGroup with mixed exceptions."""
        group = BaseExceptionGroup("test", [BrokenPipeError(), ValueError("test")])
        assert _is_broken_pipe_error(group) is False
