import json
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from zahn.llm import call_ollama, parse_binary_response
from zahn.models import BinaryLLMResponse


# Simulates the truncation seen in the wild: closing } never generated
_VALID_BINARY = """{
  "label": "yes",
  "detected_language": "en",
  "excerpt": "case is late",
  "reasoning": "The customer is frustrated about a delayed case."
}"""
_TRUNCATED_ENDS_QUOTE = _VALID_BINARY.rstrip().rstrip("}")
_TRUNCATED_ENDS_COMMA = _VALID_BINARY.rstrip().rstrip("}").rstrip() + ","

VALID_BINARY_JSON = _VALID_BINARY


class TestParseBinaryResponse:
    def test_parses_yes(self):
        result = parse_binary_response(VALID_BINARY_JSON)
        assert isinstance(result, BinaryLLMResponse)
        assert result.label == "yes"

    def test_parses_no(self):
        raw = VALID_BINARY_JSON.replace('"yes"', '"no"')
        result = parse_binary_response(raw)
        assert result.label == "no"

    def test_no_label_allows_empty_excerpt(self):
        raw = VALID_BINARY_JSON.replace('"yes"', '"no"').replace('"case is late"', '""')
        result = parse_binary_response(raw)
        assert result.label == "no"
        assert result.excerpt == ""

    def test_strips_surrounding_text(self):
        raw = f"Here is the output:\n{VALID_BINARY_JSON}\nEnd."
        result = parse_binary_response(raw)
        assert result.label == "yes"

    def test_strips_markdown_fences(self):
        raw = f"```json\n{VALID_BINARY_JSON}\n```"
        result = parse_binary_response(raw)
        assert result.label == "yes"

    def test_raises_on_no_json(self):
        with pytest.raises(json.JSONDecodeError):
            parse_binary_response("no json here at all")

    def test_raises_on_invalid_label(self):
        bad = VALID_BINARY_JSON.replace('"yes"', '"frustration"')
        with pytest.raises(ValidationError):
            parse_binary_response(bad)

    def test_raises_on_old_ternary_label(self):
        # LLM might revert to 3-way labels if prompt is ambiguous
        for old_label in ("frustration", "satisfaction", "neutral"):
            bad = VALID_BINARY_JSON.replace('"yes"', f'"{old_label}"')
            with pytest.raises(ValidationError):
                parse_binary_response(bad)

    def test_raises_on_capitalised_label(self):
        # "Yes" and "NO" are not valid — must be exact lowercase
        for bad_label in ("Yes", "No", "YES", "NO"):
            bad = VALID_BINARY_JSON.replace('"yes"', f'"{bad_label}"')
            with pytest.raises(ValidationError):
                parse_binary_response(bad)

    def test_raises_on_invalid_language(self):
        bad = VALID_BINARY_JSON.replace('"en"', '"de"')
        with pytest.raises(ValidationError, match="detected_language must be one of"):
            parse_binary_response(bad)

    def test_raises_on_empty_excerpt_when_yes(self):
        bad = VALID_BINARY_JSON.replace('"case is late"', '"  "')
        with pytest.raises(ValidationError):
            parse_binary_response(bad)

    def test_raises_on_empty_reasoning(self):
        bad = VALID_BINARY_JSON.replace(
            '"The customer is frustrated about a delayed case."', '"  "'
        )
        with pytest.raises(ValidationError):
            parse_binary_response(bad)

    def test_excerpt_whitespace_stripped(self):
        padded = VALID_BINARY_JSON.replace('"case is late"', '"  case is late  "')
        result = parse_binary_response(padded)
        assert result.excerpt == "case is late"

    def test_all_fields_populated(self):
        result = parse_binary_response(VALID_BINARY_JSON)
        assert result.label == "yes"
        assert result.detected_language == "en"
        assert result.excerpt == "case is late"
        assert "frustrated" in result.reasoning


class TestCallOllama:
    def test_calls_correct_endpoint(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": VALID_BINARY_JSON}
        mock_response.raise_for_status = MagicMock()

        config = MagicMock()
        config.ollama_base_url = "http://localhost:11434"
        config.ollama_model = "qwen2.5:7b"
        config.ollama_timeout = 30

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = call_ollama("test prompt", config)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/api/generate" in call_args[0][0]
        assert result == VALID_BINARY_JSON

    def test_trailing_slash_handled(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "{}"}
        mock_response.raise_for_status = MagicMock()

        config = MagicMock()
        config.ollama_base_url = "http://localhost:11434/"
        config.ollama_model = "qwen2.5:7b"
        config.ollama_timeout = 30

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            call_ollama("test", config)

        url_called = mock_client.post.call_args[0][0]
        assert not url_called.count("//api")  # no double slashes


# ---------------------------------------------------------------------------
# parse_binary_response — truncation recovery
# ---------------------------------------------------------------------------

class TestParseBinaryResponseRecovery:
    def test_recovers_truncated_ending_with_quote(self):
        result = parse_binary_response(_TRUNCATED_ENDS_QUOTE)
        assert result.label == "yes"

    def test_recovers_truncated_ending_with_trailing_comma(self):
        result = parse_binary_response(_TRUNCATED_ENDS_COMMA)
        assert result.label == "yes"

    def test_recovery_preserves_all_fields(self):
        result = parse_binary_response(_TRUNCATED_ENDS_QUOTE)
        assert result.excerpt == "case is late"
        assert result.detected_language == "en"
        assert "frustrated" in result.reasoning

    def test_reraises_when_unrecoverable(self):
        with pytest.raises(json.JSONDecodeError):
            parse_binary_response("no json here at all")

    def test_reraises_when_recovery_also_fails(self):
        # Ends with a quote but the completed JSON still has an invalid label
        with pytest.raises(Exception):
            parse_binary_response('{"label": "bad_value"')
