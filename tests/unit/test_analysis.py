import pytest
from unittest.mock import patch, MagicMock

from zahn.analysis import analyze_message
from zahn.models import SentimentJob, SentimentResult


VALID_RAW = """{
  "label": "frustration",
  "excerpt": "extremely late",
  "reasoning": "The customer is frustrated about a late case."
}"""


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.ollama_base_url = "http://localhost:11434"
    config.ollama_model = "llama3.2:3b"
    config.ollama_timeout = 30
    return config


class TestAnalyzeMessage:
    def test_returns_sentiment_result(self, sample_job, mock_config):
        with patch("zahn.analysis.call_ollama", return_value=VALID_RAW):
            result = analyze_message(sample_job, mock_config)

        assert isinstance(result, SentimentResult)
        assert result.job_id == sample_job.id
        assert result.label == "frustration"

    def test_excerpt_from_llm(self, sample_job, mock_config):
        with patch("zahn.analysis.call_ollama", return_value=VALID_RAW):
            result = analyze_message(sample_job, mock_config)
        assert result.excerpt == "extremely late"

    def test_raw_llm_response_stored(self, sample_job, mock_config):
        with patch("zahn.analysis.call_ollama", return_value=VALID_RAW):
            result = analyze_message(sample_job, mock_config)
        assert result.raw_llm_response == VALID_RAW

    def test_prompt_includes_message(self, sample_job, mock_config):
        captured = {}

        def fake_ollama(prompt, cfg):
            captured["prompt"] = prompt
            return VALID_RAW

        with patch("zahn.analysis.call_ollama", side_effect=fake_ollama):
            analyze_message(sample_job, mock_config)

        assert sample_job.message_text in captured["prompt"]

    def test_ollama_error_propagates(self, sample_job, mock_config):
        import httpx
        with patch("zahn.analysis.call_ollama", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(httpx.TimeoutException):
                analyze_message(sample_job, mock_config)
