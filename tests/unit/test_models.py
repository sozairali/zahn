import pytest
from pydantic import ValidationError
from zahn.models import SentimentJob, LLMResponse, SentimentResult


class TestSentimentJob:
    def test_minimal(self):
        job = SentimentJob(id=1, message_text="hello")
        assert job.attempts == 0
        assert job.language_hint is None

    def test_full(self):
        job = SentimentJob(
            id=42,
            message_text="test message",
            source_record_id=100,
            source_record_type="Conversation",
            language_hint="en",
            attempts=1,
        )
        assert job.source_record_type == "Conversation"


class TestLLMResponse:
    def test_valid_frustration(self, frustration_llm_response):
        assert frustration_llm_response.label == "frustration"

    def test_valid_labels(self):
        for label in ("frustration", "satisfaction", "neutral"):
            r = LLMResponse(
                label=label, excerpt="some text", reasoning="some reason", detected_language="en"
            )
            assert r.label == label

    def test_invalid_label(self):
        with pytest.raises(ValidationError, match="label must be one of"):
            LLMResponse(label="angry", excerpt="text", reasoning="reason", detected_language="en")

    def test_empty_excerpt_rejected(self):
        with pytest.raises(ValidationError, match="excerpt must not be empty"):
            LLMResponse(label="neutral", excerpt="   ", reasoning="reason", detected_language="en")

    def test_excerpt_stripped(self):
        r = LLMResponse(
            label="neutral", excerpt="  hello world  ", reasoning="reason", detected_language="en"
        )
        assert r.excerpt == "hello world"

    def test_empty_reasoning_rejected(self):
        with pytest.raises(ValidationError, match="reasoning must not be empty"):
            LLMResponse(label="neutral", excerpt="text", reasoning="  ", detected_language="en")

    def test_empty_label_rejected(self):
        with pytest.raises(ValidationError):
            LLMResponse(label="", excerpt="text", reasoning="reason", detected_language="en")


class TestSentimentResult:
    def test_valid(self):
        result = SentimentResult(
            job_id=1,
            label="frustration",
            excerpt="case is late",
            reasoning="Delayed delivery caused frustration.",
            raw_llm_response='{"label":"frustration"}',
        )
        assert result.job_id == 1
        assert result.label == "frustration"

    def test_invalid_label(self):
        with pytest.raises(ValidationError):
            SentimentResult(
                job_id=1,
                label="unknown",
                excerpt="text",
                reasoning="reason",
                raw_llm_response="{}",
            )
