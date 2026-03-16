import pytest
from zahn.models import SentimentJob


@pytest.fixture
def sample_job() -> SentimentJob:
    return SentimentJob(
        id=1,
        message_text="This case is extremely late and I had to redo it three times!",
        attempts=0,
    )


@pytest.fixture
def satisfied_job() -> SentimentJob:
    """A job whose message is unambiguously positive — no complaints, no delays."""
    return SentimentJob(
        id=2,
        message_text="Great quality as always, margins were perfect.",
        attempts=0,
    )


@pytest.fixture
def mixed_job() -> SentimentJob:
    """A job whose message contains both a frustration signal and a satisfaction signal.

    Used to verify that the two classifiers are independent — both can fire 'yes'
    simultaneously on the same note.
    """
    return SentimentJob(
        id=3,
        message_text="Great quality as always, but this case was extremely late.",
        attempts=0,
    )
