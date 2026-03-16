import pytest
from pydantic import ValidationError
from zahn.models import SentimentJob, BinaryLLMResponse, SentimentResult


class TestSentimentJob:
    def test_minimal(self):
        job = SentimentJob(id=1, message_text="hello")
        assert job.attempts == 0

    def test_full(self):
        job = SentimentJob(
            id=42,
            message_text="test message",
            source_record_id=100,
            source_record_type="Conversation",
            attempts=1,
        )
        assert job.source_record_type == "Conversation"


class TestBinaryLLMResponse:
    def test_valid_yes(self):
        r = BinaryLLMResponse(
            label="yes", excerpt="case was late", reasoning="Delayed delivery.", detected_language="en"
        )
        assert r.label == "yes"

    def test_valid_no(self):
        r = BinaryLLMResponse(
            label="no", excerpt="all good", reasoning="No issues found.", detected_language="fr"
        )
        assert r.label == "no"

    def test_no_label_allows_empty_excerpt(self):
        r = BinaryLLMResponse(
            label="no", excerpt="", reasoning="No issues found.", detected_language="en"
        )
        assert r.excerpt == ""

    def test_invalid_label_word_rejected(self):
        with pytest.raises(ValidationError, match="label must be one of"):
            BinaryLLMResponse(
                label="frustration", excerpt="text", reasoning="reason", detected_language="en"
            )

    def test_empty_label_rejected(self):
        with pytest.raises(ValidationError):
            BinaryLLMResponse(label="", excerpt="text", reasoning="reason", detected_language="en")

    def test_case_sensitive_yes_rejected(self):
        # LLMs sometimes capitalise — "Yes" must be rejected, not silently accepted
        with pytest.raises(ValidationError):
            BinaryLLMResponse(label="Yes", excerpt="text", reasoning="reason", detected_language="en")

    def test_case_sensitive_no_rejected(self):
        with pytest.raises(ValidationError):
            BinaryLLMResponse(label="NO", excerpt="text", reasoning="reason", detected_language="en")

    def test_invalid_language_rejected(self):
        with pytest.raises(ValidationError, match="detected_language must be one of"):
            BinaryLLMResponse(label="yes", excerpt="text", reasoning="reason", detected_language="de")

    def test_empty_excerpt_rejected_when_yes(self):
        with pytest.raises(ValidationError, match="excerpt must not be empty"):
            BinaryLLMResponse(label="yes", excerpt="  ", reasoning="reason", detected_language="en")

    def test_excerpt_stripped(self):
        r = BinaryLLMResponse(
            label="yes", excerpt="  case late  ", reasoning="Delayed.", detected_language="en"
        )
        assert r.excerpt == "case late"

    def test_whitespace_reasoning_rejected(self):
        with pytest.raises(ValidationError, match="reasoning must not be empty"):
            BinaryLLMResponse(label="no", excerpt="text", reasoning="   ", detected_language="en")

    def test_empty_reasoning_rejected(self):
        with pytest.raises(ValidationError, match="reasoning must not be empty"):
            BinaryLLMResponse(label="no", excerpt="text", reasoning="", detected_language="en")

    def test_valid_languages(self):
        for lang in ("en", "fr", "es"):
            r = BinaryLLMResponse(
                label="no", excerpt="text", reasoning="reason", detected_language=lang
            )
            assert r.detected_language == lang


class TestSentimentResult:
    def _valid_result(self, **overrides):
        defaults = dict(
            job_id=1,
            frustration_label="yes",
            satisfaction_label="no",
            detected_language="en",
            frustration_excerpt="case is late",
            frustration_reasoning="Delayed delivery caused frustration.",
            satisfaction_excerpt="no positive signals",
            satisfaction_reasoning="No satisfaction indicators present.",
            raw_frustration_response='{"label":"yes"}',
            raw_satisfaction_response='{"label":"no"}',
        )
        defaults.update(overrides)
        return SentimentResult(**defaults)

    def test_valid_frustrated_not_satisfied(self):
        result = self._valid_result(frustration_label="yes", satisfaction_label="no")
        assert result.frustration_label == "yes"
        assert result.satisfaction_label == "no"

    def test_valid_satisfied_not_frustrated(self):
        result = self._valid_result(frustration_label="no", satisfaction_label="yes")
        assert result.frustration_label == "no"
        assert result.satisfaction_label == "yes"

    def test_both_yes_allowed(self):
        # Core independence test: customer can be simultaneously frustrated AND satisfied
        result = self._valid_result(
            frustration_label="yes",
            satisfaction_label="yes",
            satisfaction_excerpt="great quality",
            satisfaction_reasoning="Customer praised quality.",
        )
        assert result.frustration_label == "yes"
        assert result.satisfaction_label == "yes"

    def test_both_no_allowed(self):
        # Neutral/routine note: neither dimension fires
        result = self._valid_result(frustration_label="no", satisfaction_label="no")
        assert result.frustration_label == "no"
        assert result.satisfaction_label == "no"

    def test_invalid_frustration_label(self):
        with pytest.raises(ValidationError):
            self._valid_result(frustration_label="frustration")

    def test_empty_frustration_label_rejected(self):
        with pytest.raises(ValidationError):
            self._valid_result(frustration_label="")

    def test_invalid_satisfaction_label(self):
        with pytest.raises(ValidationError):
            self._valid_result(satisfaction_label="neutral")

    def test_empty_satisfaction_label_rejected(self):
        with pytest.raises(ValidationError):
            self._valid_result(satisfaction_label="")

    def test_detected_language_required(self):
        with pytest.raises(ValidationError):
            self._valid_result(detected_language=None)

    def test_job_id_stored(self):
        result = self._valid_result(job_id=99)
        assert result.job_id == 99
