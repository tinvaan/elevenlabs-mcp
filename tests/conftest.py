import os
import pytest
import tempfile

from pathlib import Path
from unittest.mock import patch, MagicMock, Mock


def create_mock(**attrs):
    """Helper to create a Mock with attributes set.

    This is useful for creating mocks in @patch decorators where you need
    to set attributes on the mock object. Regular Mock(**kwargs) doesn't
    set attributes - it uses kwargs for configuration.

    Example:
        @patch("module.function", return_value=[
            create_mock(name="Test", value=123)
        ])
    """
    mock = Mock()
    mock.configure_mock(**attrs)
    return mock


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_audio_file(temp_dir):
    audio_file = temp_dir / "test.mp3"
    audio_file.touch()
    return audio_file


@pytest.fixture
def sample_video_file(temp_dir):
    video_file = temp_dir / "test.mp4"
    video_file.touch()
    return video_file


@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables required for module import."""
    with patch.dict(os.environ, {
        "ELEVENLABS_API_KEY": "test_key",
        "ELEVENLABS_MCP_OUTPUT_MODE": "files",
    }):
        yield


@pytest.fixture
def mock_client():
    """Create a mock ElevenLabs client."""
    with patch("elevenlabs_mcp.mcp.ElevenLabs") as mock_elevenlabs:
        mock_instance = MagicMock()
        mock_elevenlabs.return_value = mock_instance
        yield mock_instance
