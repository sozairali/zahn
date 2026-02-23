import pytest
from zahn.models import SentimentJob, LLMResponse


@pytest.fixture
def sample_job() -> SentimentJob:
    return SentimentJob(
        id=1,
        message_text="This case is extremely late and I had to redo it three times!",
        attempts=0,
    )


@pytest.fixture
def frustration_llm_response() -> LLMResponse:
    return LLMResponse(
        label="frustration",
        excerpt="extremely late and I had to redo it three times",
        reasoning="The customer expresses frustration about a delayed case requiring multiple remakes.",
        detected_language="en",
    )
