import pytest
from pathlib import Path
import tempfile


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


def pytest_ignore_collect(collection_path, config):
    requested = [str(arg).replace("\\", "/").rstrip("/") for arg in config.args]
    if any(path.endswith("tests/integration") for path in requested):
        return False

    normalized = str(collection_path).replace("\\", "/")
    return "tests/integration/" in normalized or normalized.endswith("tests/integration")
