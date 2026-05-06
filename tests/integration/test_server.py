"""Integration tests for core server tools."""

import base64
import json
import pytest
import re

from pathlib import Path
from typing import Any

from elevenlabs_mcp.utils import ElevenLabsMcpError


def test_text_to_speech(server: Any, vcr: Any, tmp_path: Path):
    """Generate speech audio from a short text prompt."""
    cassette = Path(__file__).parent / "cassettes" / "test_text_to_speech_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        with pytest.raises(ElevenLabsMcpError, match="Text is required"):
            server.text_to_speech(text="")

        with pytest.raises(ElevenLabsMcpError, match="cannot both be provided"):
            server.text_to_speech(
                text="Hello",
                voice_id="voice-id",
                voice_name="Voice Name",
            )

        response = server.text_to_speech(
            text="Hello from integration tests.",
            output_directory=str(tmp_path),
        )
        assert (
            "File saved as:" in response.text
            if response.type == "text"
            else response.type == "resource"
        )


def test_speech_to_text(server: Any, vcr: Any, tmp_path: Path):
    """Transcribe generated speech and return transcript directly."""
    cassette = Path(__file__).parent / "cassettes" / "test_speech_to_text_integration.yaml"
    audiofile = tmp_path / "stt_source_audio.mp3"

    with vcr.use_cassette(str(cassette)):
        with pytest.raises(
            ElevenLabsMcpError,
            match="Must save transcript to file or return it to the client directly",
        ):
            server.speech_to_text(
                input_file_path="/does/not/exist.mp3",
                save_transcript_to_file=False,
                return_transcript_to_client_directly=False,
            )

        tts = server.text_to_speech(
            text="Hello from speech to text integration test.",
            output_directory=str(tmp_path),
        )
        if tts.type == "text":
            match = re.search(r"File saved as: (.+?)\. Voice used:", tts.text)
            assert match is not None
            audiofile = Path(match.group(1))
        else:
            blob = getattr(tts.resource, "blob", None)
            assert blob is not None
            audiofile.write_bytes(base64.b64decode(blob))

        response = server.speech_to_text(
            input_file_path=str(audiofile),
            save_transcript_to_file=False,
            return_transcript_to_client_directly=True,
        )

    assert response.type == "text"
    assert response.text.strip() != ""


def test_text_to_sound_effects(server: Any, vcr: Any, tmp_path: Path):
    """Generate a sound effect from a text prompt."""
    cassette = Path(__file__).parent / "cassettes" / "test_text_to_sound_effects_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        with pytest.raises(ElevenLabsMcpError, match="Duration must be between 0.5 and 5 seconds"):
            server.text_to_sound_effects(text="A short beep", duration_seconds=0.1)

        response = server.text_to_sound_effects(
            text="A short digital notification ping",
            duration_seconds=1.0,
            output_directory=str(tmp_path),
        )
        assert (
            "File saved as:" in response.text
            if response.type == "text"
            else response.type == "resource"
        )


def test_search_voices(server: Any, vcr: Any):
    """Search voices in the account voice library."""
    cassette = Path(__file__).parent / "cassettes" / "test_search_voices_integration.yaml"
    with vcr.use_cassette(str(cassette)):
        response = server.search_voices()

    assert isinstance(response, list)
    assert all(voice.id and voice.name for voice in response)

    cassette = Path(__file__).parent / "cassettes" / "test_search_voices_filtered_integration.yaml"
    with vcr.use_cassette(str(cassette)):
        response = server.search_voices(search="adam", sort="name", sort_direction="asc")

    assert isinstance(response, list)
    assert all(voice.id and voice.name for voice in response)


def test_list_models(server: Any, vcr: Any):
    """List available ElevenLabs models."""
    cassette = Path(__file__).parent / "cassettes" / "test_list_models_integration.yaml"
    with vcr.use_cassette(str(cassette)):
        response = server.list_models()

    assert isinstance(response, list)
    assert len(response) > 0
    assert all(model.id and model.name for model in response)


def test_get_voice(server: Any, vcr: Any):
    """Get details for one voice from the account library."""
    cassette = Path(__file__).parent / "cassettes" / "test_get_voice_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        voices = server.search_voices()
        if not voices:
            pytest.skip("No voices available in this workspace")

        voice_id = voices[0].id
        response = server.get_voice(voice_id=voice_id)

    assert response.id == voice_id
    assert bool(response.name)
    assert bool(response.category)


def test_voice_clone_validation(server: Any):
    """Reject voice clone requests when input files are missing."""
    with pytest.raises(ElevenLabsMcpError):
        server.voice_clone(name="integration-clone", files=["/does/not/exist.mp3"])


def test_isolate_audio_validation(server: Any):
    """Reject isolate-audio requests for missing files."""
    with pytest.raises(ElevenLabsMcpError):
        server.isolate_audio(input_file_path="/does/not/exist.mp3")


def test_check_subscription(server: Any, vcr: Any):
    """Fetch subscription details for the configured ElevenLabs account."""
    cassette = Path(__file__).parent / "cassettes" / "test_check_subscription_integration.yaml"
    with vcr.use_cassette(str(cassette)):
        response = server.check_subscription()

    assert response.type == "text"
    assert json.loads(response.text).get('status') == "active"


def test_get_conversation(env: tuple[str, str], server: Any, vcr: Any):
    """Fetch one conversation details payload when available."""
    _, target = env
    cassette = Path(__file__).parent / "cassettes" / "test_get_conversation_integration.yaml"
    with vcr.use_cassette(str(cassette)):
        listed = server.list_conversations(
            agent_id=target,
            page_size=1,
            max_length=20000,
        )
        if "No conversations found." in listed.text:
            # TODO: Add a cassette with actual conversations and assert responses
            pytest.skip("No conversations available for the configured test agent")

        match = re.search(r"Conversation ID: (\S+)", listed.text)
        assert match is not None
        conversation_id = match.group(1)

        response = server.get_conversation(conversation_id=conversation_id)
        assert response.type == "text"
        assert "Conversation Details:" in response.text
        assert f"ID: {conversation_id}" in response.text
        assert "Transcript:" in response.text


@pytest.mark.parametrize(
    ("cassette", "page_size", "use_target"),
    [
        ("test_list_conversations_integration.yaml", 5, True),
        ("test_list_conversations_unfiltered_integration.yaml", 3, False),
        ("test_list_conversations_page_size_cap_integration.yaml", 999, True),
    ],
    ids=["target-filtered", "unfiltered", "page-size-cap"],
)
def test_list_conversations(
    env: tuple[str, str],
    server: Any,
    vcr: Any,
    cassette: str,
    page_size: int,
    use_target: bool,
):
    """List conversations across common query variants."""
    _, target = env

    with vcr.use_cassette(str(Path(__file__).parent / "cassettes" / cassette)):
        response = server.list_conversations(**{
            "page_size": page_size,
            **({"agent_id": target} if use_target else {})
        })
        assert response.type == "text"
        assert "No conversations found." in response.text or "Showing " in response.text

    if "No conversations found" in response.text:
        # TODO: Add a cassette with actual conversations and assert responses
        pytest.skip("No conversations available for configured test agent")


@pytest.mark.parametrize(
    (
        "cassette",
        "voice_name",
        "use_real_voice",
        "use_generated_source_audio",
        "expect_error",
    ),
    [
        (
            "test_speech_to_speech_validation_integration.yaml",
            "__integration_missing_voice_name__",
            False,
            False,
            True,
        ),
        (
            "test_speech_to_speech_missing_input_file_integration.yaml",
            None,
            True,
            False,
            True,
        ),
        (
            "test_speech_to_speech_integration.yaml",
            None,
            True,
            True,
            False,
        ),
    ],
    ids=["missing-voice", "missing-input-file", "success"],
)
def test_speech_to_speech(
    server: Any,
    vcr: Any,
    tmp_path: Path,
    cassette: str,
    voice_name: str | None,
    use_real_voice: bool,
    use_generated_source_audio: bool,
    expect_error: bool,
):
    """Validate speech-to-speech failure paths and successful conversion."""
    source_audio = tmp_path / "sts_source_audio.mp3"

    with vcr.use_cassette(str(Path(__file__).parent / "cassettes" / cassette)):
        if use_real_voice:
            voices = server.search_voices()
            if not voices:
                pytest.skip("No voices available in this workspace")
            voice_name = voices[0].name

        assert voice_name is not None

        if use_generated_source_audio:
            tts = server.text_to_speech(
                text="Hello from speech to speech integration test.",
                output_directory=str(tmp_path),
            )
            if tts.type == "text":
                match = re.search(r"File saved as: (.+?)\. Voice used:", tts.text)
                assert match is not None
                source_audio = Path(match.group(1))
            else:
                blob = getattr(tts.resource, "blob", None)
                assert blob is not None
                source_audio.write_bytes(base64.b64decode(blob))
        else:
            source_audio = Path("/does/not/exist.mp3")

        if expect_error:
            with pytest.raises(ElevenLabsMcpError):
                server.speech_to_speech(
                    input_file_path=str(source_audio),
                    voice_name=voice_name,
                    output_directory=str(tmp_path),
                )
            return

        response = server.speech_to_speech(
            input_file_path=str(source_audio),
            voice_name=voice_name,
            output_directory=str(tmp_path),
        )
        assert (
            "File saved as:" in response.text
            if response.type == "text"
            else response.type == "resource"
        )


@pytest.mark.parametrize(
    ("cassette", "voice_description", "text", "expect_error"),
    [
        (None, "", None, True),
        (
            "test_text_to_voice_auto_text_integration.yaml",
            "A warm and clear podcast narrator voice",
            None,
            False,
        ),
        (
            "test_text_to_voice_with_text_integration.yaml",
            "A warm and clear podcast narrator voice",
            (
                "Hello from text to voice integration tests. This paragraph is intentionally "
                "long so it satisfies the API minimum length requirement of one hundred "
                "characters for custom text input."
            ),
            False,
        ),
    ],
    ids=["empty-description", "auto-generated-text", "provided-text"],
)
def test_text_to_voice(
    server: Any,
    vcr: Any,
    cassette: str | None,
    voice_description: str,
    text: str | None,
    expect_error: bool,
):
    """Validate text-to-voice across validation and generation paths."""
    if expect_error:
        with pytest.raises(ElevenLabsMcpError, match="Voice description is required"):
            server.text_to_voice(voice_description=voice_description, text=text)
        return

    assert cassette is not None
    with vcr.use_cassette(str(Path(__file__).parent / "cassettes" / cassette)):
        response = server.text_to_voice(voice_description=voice_description, text=text)

    if isinstance(response, list):
        assert len(response) == 3
        assert all(getattr(item, "type", None) == "resource" for item in response)
    else:
        assert response.type == "text"
        assert "Generated voice IDs are:" in response.text


@pytest.mark.parametrize(
    ("cassette", "use_valid_preview", "expect_error"),
    [
        ("test_create_voice_from_preview_invalid_id_integration.yaml", False, True),
        ("test_create_voice_from_preview_integration.yaml", True, False),
    ],
    ids=["invalid-generated-voice-id", "success"],
)
def test_create_voice_from_preview(
    server: Any,
    vcr: Any,
    cassette: str,
    use_valid_preview: bool,
    expect_error: bool,
):
    """Create voice from preview for success and invalid-id scenarios."""
    created_voice_id: str | None = None
    generated_voice_id = "invalid-generated-voice-id"

    with vcr.use_cassette(str(Path(__file__).parent / "cassettes" / cassette)):
        try:
            if expect_error:
                with pytest.raises(Exception):
                    server.create_voice_from_preview(
                        generated_voice_id=generated_voice_id,
                        voice_name="integration-invalid-preview",
                        voice_description="integration invalid preview",
                    )
                return

            if use_valid_preview:
                previews = server.client.text_to_voice.create_previews(
                    voice_description="A calm, warm narrator voice for integration tests.",
                    auto_generate_text=True,
                )
                assert previews.previews
                generated_voice_id = previews.previews[0].generated_voice_id

            response = server.create_voice_from_preview(
                generated_voice_id=generated_voice_id,
                voice_name="integration-created-preview-voice",
                voice_description="Created during integration tests",
            )

            assert response.type == "text"
            assert "Success. Voice created:" in response.text
            match = re.search(r"ID:(\S+)", response.text)
            assert match is not None
            created_voice_id = match.group(1)
        finally:
            if created_voice_id:
                server.client.voices.delete(voice_id=created_voice_id)


@pytest.mark.parametrize(
    ("scenario", "provider", "provider_label", "cassette"),
    [
        (
            "invalid-phone-id",
            None,
            None,
            "test_make_outbound_call_invalid_phone_number_integration.yaml",
        ),
        ("provider-route", "twilio", "Twilio", None),
        ("provider-route", "sip_trunk", "SIP trunk", None),
    ],
    ids=["invalid-phone-id", "twilio-route", "sip-trunk-route"],
)
def test_make_outbound_call(
    server: Any,
    env: tuple[str, str],
    vcr: Any,
    monkeypatch: Any,
    scenario: str,
    provider: str | None,
    provider_label: str | None,
    cassette: str | None,
):
    """Validate outbound-call error handling and provider routing."""
    _, target = env

    if scenario == "invalid-phone-id":
        assert cassette is not None
        with vcr.use_cassette(str(Path(__file__).parent / "cassettes" / cassette)):
            with pytest.raises(ElevenLabsMcpError, match="Phone number with ID"):
                server.make_outbound_call(
                    agent_id=target,
                    agent_phone_number_id="invalid-phone-id",
                    to_number="+15555555555",
                )
        return

    assert provider is not None
    assert provider_label is not None

    class _Phone:
        def __init__(self, provider_name: str):
            self.provider = provider_name

    calls: list[str] = []

    def _fake_twilio(**kwargs: Any):
        calls.append("twilio")
        return {"provider": "twilio", "kwargs": kwargs}

    def _fake_sip(**kwargs: Any):
        calls.append("sip_trunk")
        return {"provider": "sip_trunk", "kwargs": kwargs}

    monkeypatch.setattr(server, "_get_phone_number_by_id", lambda _id: _Phone(provider))
    monkeypatch.setattr(server.client.conversational_ai.twilio, "outbound_call", _fake_twilio)
    monkeypatch.setattr(server.client.conversational_ai.sip_trunk, "outbound_call", _fake_sip)

    response = server.make_outbound_call(
        agent_id="agent-id",
        agent_phone_number_id="phone-id",
        to_number="+15555555555",
    )

    assert response.type == "text"
    assert f"Outbound call initiated via {provider_label}" in response.text
    assert calls == [provider]


def test_search_voice_library(server: Any, vcr: Any):
    """Search public shared voice library."""
    cassette = Path(__file__).parent / "cassettes" / "test_search_voice_library_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        response = server.search_voice_library(page=0, page_size=3, search="english")

    assert response.type == "text"
    assert (
        "Shared Voices:" in response.text or
        "No shared voices found with the specified criteria." in response.text
    )


def test_list_phone_numbers(server: Any, vcr: Any):
    """List account phone numbers for conversational AI."""
    cassette = Path(__file__).parent / "cassettes" / "test_list_phone_numbers_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        response = server.list_phone_numbers()

    assert response.type == "text"
    if "No phone numbers found." in response.text:
        pytest.skip("No phone numbers available in this workspace.")
    assert "Phone Numbers:" in response.text


def test_play_audio(server: Any, tmp_path: Path, monkeypatch: Any):
    """Validate play-audio for both invalid and valid local files."""
    with pytest.raises(ElevenLabsMcpError):
        server.play_audio(input_file_path="/does/not/exist.mp3")

    audiofile = tmp_path / "sample.mp3"
    audio = b"ID3\x00\x00\x00integration-test"
    audiofile.write_bytes(audio)

    captured: dict[str, Any] = {}

    def fake_play(data: bytes, use_ffmpeg: bool = False):
        captured["data"] = data
        captured["use_ffmpeg"] = use_ffmpeg

    monkeypatch.setattr(server, "play", fake_play)
    response = server.play_audio(input_file_path=str(audiofile))

    assert captured["data"] == audio
    assert captured["use_ffmpeg"] is False
    assert response.type == "text"
    assert str(audiofile) in response.text


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        (
            {"prompt": None, "composition_plan": None},
            "Either prompt or composition_plan must be provided",
        ),
        (
            {"prompt": "test", "composition_plan": object()},
            "Only one of prompt or composition_plan must be provided",
        ),
        (
            {"prompt": None, "composition_plan": object(), "music_length_ms": 12000},
            "music_length_ms cannot be used if composition_plan is provided",
        ),
    ],
    ids=["missing-inputs", "both-inputs", "length-with-plan"],
)
def test_compose_music_validation(server: Any, kwargs: dict[str, Any], error_match: str):
    """Reject invalid compose-music argument combinations."""
    with pytest.raises(ElevenLabsMcpError, match=error_match):
        server.compose_music(**kwargs)


def test_create_composition_plan(server: Any, vcr: Any):
    """Create a composition plan from a text prompt."""
    cassette = Path(__file__).parent / "cassettes" / "test_create_composition_plan_integration.yaml"

    with vcr.use_cassette(str(cassette)):
        response = server.create_composition_plan(prompt="Calm ambient piano with soft strings")

    assert response is not None
    if hasattr(response, "model_dump"):
        payload = response.model_dump(exclude_none=True)
        assert isinstance(payload, dict)
        assert len(payload) > 0
    elif isinstance(response, dict):
        assert len(response) > 0
    else:
        assert hasattr(response, "__dict__")
        assert len(response.__dict__) > 0
