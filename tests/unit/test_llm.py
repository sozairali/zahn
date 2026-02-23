import json
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from zahn.llm import call_ollama, parse_llm_response
from zahn.models import LLMResponse


VALID_JSON = """{
  "label": "frustration",
  "detected_language": "en",
  "excerpt": "case is late",
  "reasoning": "The customer is frustrated about a delayed case."
}"""


class TestParseOllamaResponse:
    def test_parses_valid_json(self):
        result = parse_llm_response(VALID_JSON)
        assert isinstance(result, LLMResponse)
        assert result.label == "frustration"

    def test_strips_surrounding_text(self):
        raw = f"Sure, here is the result:\n{VALID_JSON}\nHope that helps!"
        result = parse_llm_response(raw)
        assert result.label == "frustration"

    def test_strips_markdown_fences(self):
        raw = f"```json\n{VALID_JSON}\n```"
        result = parse_llm_response(raw)
        assert result.label == "frustration"

    def test_raises_on_no_json(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_response("I don't know what to say here.")

    def test_raises_on_invalid_label(self):
        bad = VALID_JSON.replace('"frustration"', '"angry"')
        with pytest.raises(ValidationError):
            parse_llm_response(bad)

    def test_raises_on_empty_excerpt(self):
        bad = VALID_JSON.replace('"case is late"', '"  "')
        with pytest.raises(ValidationError):
            parse_llm_response(bad)

    def test_excerpt_whitespace_stripped(self):
        padded = VALID_JSON.replace('"case is late"', '"  case is late  "')
        result = parse_llm_response(padded)
        assert result.excerpt == "case is late"

    def test_satisfaction_label(self):
        raw = VALID_JSON.replace('"frustration"', '"satisfaction"')
        result = parse_llm_response(raw)
        assert result.label == "satisfaction"

    def test_neutral_label(self):
        raw = VALID_JSON.replace('"frustration"', '"neutral"')
        result = parse_llm_response(raw)
        assert result.label == "neutral"


class TestCallOllama:
    def test_calls_correct_endpoint(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": VALID_JSON}
        mock_response.raise_for_status = MagicMock()

        config = MagicMock()
        config.ollama_base_url = "http://localhost:11434"
        config.ollama_model = "llama3.2:3b"
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
        assert result == VALID_JSON

    def test_trailing_slash_handled(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "{}"}
        mock_response.raise_for_status = MagicMock()

        config = MagicMock()
        config.ollama_base_url = "http://localhost:11434/"
        config.ollama_model = "llama3.2:3b"
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
