"""Scrubbing helpers for integration VCR cassettes."""

import json
import re
from typing import Any


class Scrub:
    """Scrub API key values from recorded request/response payloads."""

    KEYS: dict[str, str] = {
        "creator_email": "test@example.com",
        "creator_name": "Test User",
    }
    PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "test@example.com"),
        (re.compile(r"\btool_[a-z0-9]+\b"), "tool_0001test_tool"),
        (re.compile(r"\btest_[a-z0-9]+\b"), "test_0001test_case"),
        (re.compile(r"\bagtbrch_[a-z0-9]+\b"), "agtbrch_0001agent_branch"),
        (re.compile(r"\bagtvrsn_[a-z0-9]+\b"), "agtvrsn_0001agent_version"),
        (re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE), "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    )

    def __init__(self, apikey: str):
        """Initialize the scrubber with the key to redact."""
        self.key = apikey

    def _scrub_text(self, text: str) -> str:
        """Scrub API keys and sensitive patterns from text values."""
        redacted = text
        if self.key:
            redacted = redacted.replace(self.key, "<ELEVENLABS_API_KEY>")

        for pattern, replacement in Scrub.PATTERNS:
            redacted = pattern.sub(replacement, redacted)

        return redacted

    def _scrub_json_values(self, obj: Any) -> Any:
        """Recursively scrub only JSON values, not keys."""
        if isinstance(obj, dict):
            cleaned: dict[str, Any] = {}
            for key, value in obj.items():
                if key in Scrub.KEYS and isinstance(value, str):
                    cleaned[key] = Scrub.KEYS[key]
                else:
                    cleaned[key] = self._scrub_json_values(value)
            return cleaned

        if isinstance(obj, list):
            return [self._scrub_json_values(item) for item in obj]

        if isinstance(obj, str):
            return self._scrub_text(obj)

        return obj

    def _scrub_blob(self, value: str | bytes) -> str | bytes:
        """Scrub text/bytes; for JSON, scrub values only."""
        raw = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value

        try:
            parsed = json.loads(raw)
            scrubbed = json.dumps(self._scrub_json_values(parsed), separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            scrubbed = self._scrub_text(raw)

        return scrubbed.encode("utf-8") if isinstance(value, bytes) else scrubbed

    def request(self, req: Any) -> Any:
        """Scrub API key and sensitive metadata from a VCR request."""
        if getattr(req, "body", None):
            req.body = self._scrub_blob(req.body)

        if getattr(req, "uri", None):
            req.uri = self._scrub_text(req.uri)

        return req

    def response(self, res: Any) -> Any:
        """Scrub API key and sensitive metadata from a VCR response."""
        body = res.get("body", {}).get("string")
        if isinstance(body, (bytes, str)):
            res["body"]["string"] = self._scrub_blob(body)

        headers = res.get("headers", {})
        for header, values in headers.items():
            if isinstance(values, list):
                headers[header] = [
                    self._scrub_text(value) if isinstance(value, str) else value
                    for value in values
                ]

        return res
