import pytest
from unittest.mock import patch, MagicMock

import httpx

from zahn.analysis import analyze_message, run_classifier, ClassifierResult
from zahn.models import SentimentJob, SentimentResult
from zahn.prompt import build_frustration_prompt


FRUSTRATION_RAW = """{
  "label": "yes",
  "detected_language": "en",
  "excerpt": "extremely late",
  "reasoning": "The customer is frustrated about a late case."
}"""

SATISFACTION_RAW = """{
  "label": "no",
  "detected_language": "en",
  "excerpt": "extremely late",
  "reasoning": "No positive signals present."
}"""

SAT_YES_RAW = """{
  "label": "yes",
  "detected_language": "en",
  "excerpt": "great quality",
  "reasoning": "Customer praised quality unprompted."
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
        with patch("zahn.analysis.call_ollama", side_effect=[FRUSTRATION_RAW, SATISFACTION_RAW]):
            result = analyze_message(sample_job, mock_config)

        assert isinstance(result, SentimentResult)
        assert result.job_id == sample_job.id

    def test_frustration_label_from_first_call(self, sample_job, mock_config):
        with patch("zahn.analysis.call_ollama", side_effect=[FRUSTRATION_RAW, SATISFACTION_RAW]):
            result = analyze_message(sample_job, mock_config)
        assert result.frustration_label == "yes"

    def test_satisfaction_label_from_second_call(self, sample_job, mock_config):
        with patch("zahn.analysis.call_ollama", side_effect=[FRUSTRATION_RAW, SATISFACTION_RAW]):
            result = analyze_message(sample_job, mock_config)
        assert result.satisfaction_label == "no"

    def test_both_yes_possible(self, mixed_job, mock_config):
        # Core independence guarantee: both dimensions can fire simultaneously.
        # mixed_job message contains both "extremely late" (frustration excerpt)
        # and "great quality" (satisfaction excerpt) so both validators pass.
        with patch("zahn.analysis.call_ollama", side_effect=[FRUSTRATION_RAW, SAT_YES_RAW]):
            result = analyze_message(mixed_job, mock_config)
        assert result.frustration_label == "yes"
        assert result.satisfaction_label == "yes"

    def test_both_no_possible(self, sample_job, mock_config):
        frust_no = FRUSTRATION_RAW.replace('"yes"', '"no"')
        with patch("zahn.analysis.call_ollama", side_effect=[frust_no, SATISFACTION_RAW]):
            result = analyze_message(sample_job, mock_config)
        assert result.frustration_label == "no"
        assert result.satisfaction_label == "no"

    def test_frustration_excerpt_stored(self, sample_job, mock_config):
        with patch("zahn.analysis.call_ollama", side_effect=[FRUSTRATION_RAW, SATISFACTION_RAW]):
            result = analyze_message(sample_job, mock_config)
        assert result.frustration_excerpt == "extremely late"

    def test_satisfaction_excerpt_stored(self, satisfied_job, mock_config):
        frust_no = FRUSTRATION_RAW.replace('"yes"', '"no"')
        with patch("zahn.analysis.call_ollama", side_effect=[frust_no, SAT_YES_RAW]):
            result = analyze_message(satisfied_job, mock_config)
        assert result.satisfaction_excerpt == "great quality"

    def test_raw_responses_stored(self, sample_job, mock_config):
        with patch("zahn.analysis.call_ollama", side_effect=[FRUSTRATION_RAW, SATISFACTION_RAW]):
            result = analyze_message(sample_job, mock_config)
        assert result.raw_frustration_response == FRUSTRATION_RAW
        assert result.raw_satisfaction_response == SATISFACTION_RAW

    def test_detected_language_from_frustration_call(self, sample_job, mock_config):
        # Language detection comes from the first (frustration) call
        with patch("zahn.analysis.call_ollama", side_effect=[FRUSTRATION_RAW, SATISFACTION_RAW]):
            result = analyze_message(sample_job, mock_config)
        assert result.detected_language == "en"

    def test_two_ollama_calls_made(self, sample_job, mock_config):
        with patch("zahn.analysis.call_ollama", side_effect=[FRUSTRATION_RAW, SATISFACTION_RAW]) as mock_llm:
            analyze_message(sample_job, mock_config)
        assert mock_llm.call_count == 2

    def test_both_prompts_contain_message(self, sample_job, mock_config):
        captured = []

        def fake_ollama(prompt, cfg):
            captured.append(prompt)
            return FRUSTRATION_RAW if len(captured) == 1 else SATISFACTION_RAW

        with patch("zahn.analysis.call_ollama", side_effect=fake_ollama):
            analyze_message(sample_job, mock_config)

        assert sample_job.message_text in captured[0]
        assert sample_job.message_text in captured[1]

    def test_two_prompts_are_different(self, sample_job, mock_config):
        # The frustration and satisfaction prompts must be distinct — if they were
        # the same, the second call would be meaningless.
        captured = []

        def fake_ollama(prompt, cfg):
            captured.append(prompt)
            return FRUSTRATION_RAW if len(captured) == 1 else SATISFACTION_RAW

        with patch("zahn.analysis.call_ollama", side_effect=fake_ollama):
            analyze_message(sample_job, mock_config)

        assert captured[0] != captured[1]

    def test_first_call_error_propagates(self, sample_job, mock_config):
        import httpx
        with patch("zahn.analysis.call_ollama", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(httpx.TimeoutException):
                analyze_message(sample_job, mock_config)

    def test_second_call_error_propagates(self, sample_job, mock_config):
        # A failure on the satisfaction call must also bubble up — not be silently swallowed.
        # Provide enough side-effects to cover all 3 satisfaction retry attempts.
        import httpx
        timeout = httpx.TimeoutException("timeout on sat call")
        with patch(
            "zahn.analysis.call_ollama",
            side_effect=[FRUSTRATION_RAW, timeout, timeout, timeout],
        ):
            with pytest.raises(httpx.TimeoutException):
                analyze_message(sample_job, mock_config)

    def test_frustration_reasoning_stored(self, sample_job, mock_config):
        with patch("zahn.analysis.call_ollama", side_effect=[FRUSTRATION_RAW, SATISFACTION_RAW]):
            result = analyze_message(sample_job, mock_config)
        assert "frustrated" in result.frustration_reasoning

    def test_satisfaction_reasoning_stored(self, satisfied_job, mock_config):
        frust_no = FRUSTRATION_RAW.replace('"yes"', '"no"')
        with patch("zahn.analysis.call_ollama", side_effect=[frust_no, SAT_YES_RAW]):
            result = analyze_message(satisfied_job, mock_config)
        assert "quality" in result.satisfaction_reasoning


# ---------------------------------------------------------------------------
# run_classifier
# ---------------------------------------------------------------------------

class TestRunClassifier:
    def test_returns_classifier_result(self, mock_config):
        with patch("zahn.analysis.call_ollama", return_value=FRUSTRATION_RAW):
            result = run_classifier("case is late", build_frustration_prompt, mock_config)
        assert isinstance(result, ClassifierResult)

    def test_success_populates_all_fields(self, mock_config):
        with patch("zahn.analysis.call_ollama", return_value=FRUSTRATION_RAW):
            result = run_classifier("extremely late", build_frustration_prompt, mock_config)
        assert result.label == "yes"
        assert result.excerpt == "extremely late"
        assert result.detected_language == "en"
        assert result.raw == FRUSTRATION_RAW
        assert result.parse_error == ""
        assert result.exception is None

    def test_never_raises_on_parse_failure(self, mock_config):
        # Atomicity guarantee: run_classifier must return, never raise
        with patch("zahn.analysis.call_ollama", return_value="not json"):
            run_classifier("msg", build_frustration_prompt, mock_config)  # must not raise

    def test_never_raises_on_http_failure(self, mock_config):
        error = httpx.HTTPStatusError(
            "conn refused",
            request=httpx.Request("POST", "http://localhost"),
            response=httpx.Response(500),
        )
        with patch("zahn.analysis.call_ollama", side_effect=error):
            result = run_classifier("msg", build_frustration_prompt, mock_config)
        assert result.exception is not None

    def test_total_failure_sets_parse_error(self, mock_config):
        with patch("zahn.analysis.call_ollama", return_value="not json"):
            result = run_classifier("msg", build_frustration_prompt, mock_config)
        assert result.parse_error != ""
        assert result.label == ""

    def test_total_failure_stores_exception(self, mock_config):
        with patch("zahn.analysis.call_ollama", return_value="not json"):
            result = run_classifier("msg", build_frustration_prompt, mock_config)
        assert result.exception is not None

    def test_retries_on_parse_failure(self, mock_config):
        with patch("zahn.analysis.call_ollama", side_effect=["not json", FRUSTRATION_RAW]) as mock_call:
            result = run_classifier("extremely late", build_frustration_prompt, mock_config)
        assert result.label == "yes"
        assert mock_call.call_count == 2

    def test_exhausts_max_attempts(self, mock_config):
        with patch("zahn.analysis.call_ollama", return_value="not json") as mock_call:
            run_classifier("msg", build_frustration_prompt, mock_config, max_attempts=3)
        assert mock_call.call_count == 3

    def test_respects_custom_max_attempts(self, mock_config):
        with patch("zahn.analysis.call_ollama", return_value="not json") as mock_call:
            run_classifier("msg", build_frustration_prompt, mock_config, max_attempts=1)
        assert mock_call.call_count == 1

    def test_excerpt_retry_counted_on_hallucination(self, mock_config):
        # First response has an excerpt not present in the message → excerpt retry
        # Second response has a valid excerpt
        hallucinated = FRUSTRATION_RAW.replace('"extremely late"', '"invented excerpt xyz"')
        with patch("zahn.analysis.call_ollama", side_effect=[hallucinated, FRUSTRATION_RAW]):
            result = run_classifier("extremely late", build_frustration_prompt, mock_config)
        assert result.label == "yes"
        assert result.excerpt_retries == 1

    def test_excerpt_retries_zero_on_clean_success(self, mock_config):
        with patch("zahn.analysis.call_ollama", return_value=FRUSTRATION_RAW):
            result = run_classifier("extremely late", build_frustration_prompt, mock_config)
        assert result.excerpt_retries == 0

    def test_excerpt_retries_counted_on_total_failure(self, mock_config):
        # All responses have hallucinated excerpts — excerpt_retries reflects total count
        hallucinated = FRUSTRATION_RAW.replace('"extremely late"', '"invented excerpt xyz"')
        with patch("zahn.analysis.call_ollama", return_value=hallucinated):
            result = run_classifier("short msg", build_frustration_prompt, mock_config, max_attempts=3)
        assert result.excerpt_retries == 3

    def test_no_label_response_skips_excerpt_validation(self, mock_config):
        # label=no → validate_excerpt is a no-op → no ExcerptValidationError
        no_response = FRUSTRATION_RAW.replace('"yes"', '"no"').replace(
            '"extremely late"', '"something not in message"'
        )
        with patch("zahn.analysis.call_ollama", return_value=no_response):
            result = run_classifier("unrelated message", build_frustration_prompt, mock_config)
        assert result.label == "no"
        assert result.exception is None
