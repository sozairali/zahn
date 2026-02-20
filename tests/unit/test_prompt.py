import pytest
from zahn.prompt import build_prompt, build_domain_context, load_keywords, _KeywordEntry

import pathlib

CSV_PATH = str(
    pathlib.Path(__file__).parent.parent.parent
    / "data/raw/Frustration Finder Query Generator - Sheet1.csv"
)

DOMAIN_CTX = "  [emotion]  frustrated(9), angry(9)\n  [logistics]  late(5)"
MESSAGE = "This case is late and I am frustrated."


class TestBuildPrompt:
    def test_returns_string(self):
        result = build_prompt(MESSAGE, DOMAIN_CTX)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_message(self):
        result = build_prompt(MESSAGE, DOMAIN_CTX)
        assert MESSAGE in result

    def test_contains_domain_context(self):
        result = build_prompt(MESSAGE, DOMAIN_CTX)
        assert "frustrated(9)" in result
        assert "late(5)" in result

    def test_contains_json_schema_hint(self):
        result = build_prompt(MESSAGE, DOMAIN_CTX)
        assert '"label"' in result
        assert '"excerpt"' in result
        assert '"reasoning"' in result
        assert '"detected_language"' in result

    def test_contains_valid_labels(self):
        result = build_prompt(MESSAGE, DOMAIN_CTX)
        assert "frustration" in result
        assert "satisfaction" in result
        assert "neutral" in result

    def test_contains_special_rules(self):
        result = build_prompt(MESSAGE, DOMAIN_CTX)
        assert "remake" in result.lower()

    def test_no_markdown_fences_instruction(self):
        result = build_prompt(MESSAGE, DOMAIN_CTX)
        assert "markdown" in result.lower()

    def test_multilingual_mention(self):
        result = build_prompt(MESSAGE, DOMAIN_CTX)
        assert "fr" in result
        assert "es" in result

    def test_different_messages_differ(self):
        r1 = build_prompt("Message A", DOMAIN_CTX)
        r2 = build_prompt("Message B", DOMAIN_CTX)
        assert r1 != r2

    def test_empty_domain_context(self):
        result = build_prompt(MESSAGE, "")
        assert MESSAGE in result

    def test_message_with_curly_braces(self):
        # Curly braces in the message must not raise KeyError
        result = build_prompt("My crown {size} is wrong", DOMAIN_CTX)
        assert "{size}" in result

    def test_message_with_format_placeholders(self):
        # More aggressive: message looks like a Python format string
        result = build_prompt("{label} is {0} percent wrong", DOMAIN_CTX)
        assert "{label}" in result
        assert "{0}" in result


class TestBuildDomainContext:
    @pytest.fixture
    def sample_keywords(self):
        return [
            _KeywordEntry(language="en", keyword="late", keyword_type="logistics", score=5),
            _KeywordEntry(language="en", keyword="frustrated", keyword_type="emotion", score=9),
            _KeywordEntry(language="en", keyword="remakes", keyword_type="quality", score=6),
            _KeywordEntry(language="fr", keyword="furieux", keyword_type="emotion", score=10),
        ]

    def test_returns_string(self, sample_keywords):
        ctx = build_domain_context(sample_keywords)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_contains_keyword_types(self, sample_keywords):
        ctx = build_domain_context(sample_keywords)
        assert "emotion" in ctx
        assert "logistics" in ctx
        assert "quality" in ctx

    def test_contains_scores(self, sample_keywords):
        ctx = build_domain_context(sample_keywords)
        assert "(9)" in ctx
        assert "(5)" in ctx

    def test_fr_keywords_tagged(self, sample_keywords):
        ctx = build_domain_context(sample_keywords)
        assert "/fr" in ctx

    def test_en_keywords_not_tagged(self, sample_keywords):
        ctx = build_domain_context(sample_keywords)
        assert "late/en" not in ctx

    def test_empty_list(self):
        assert build_domain_context([]) == ""


class TestLoadKeywords:
    def test_loads_from_csv(self):
        keywords = load_keywords(CSV_PATH)
        assert len(keywords) > 0

    def test_returns_keyword_entries(self):
        keywords = load_keywords(CSV_PATH)
        assert all(isinstance(k, _KeywordEntry) for k in keywords)

    def test_includes_en_and_fr(self):
        keywords = load_keywords(CSV_PATH)
        langs = {k.language for k in keywords}
        assert "en" in langs
        assert "fr" in langs

    def test_scores_are_integers(self):
        keywords = load_keywords(CSV_PATH)
        assert all(isinstance(k.score, int) for k in keywords)

    def test_keywords_not_empty(self):
        keywords = load_keywords(CSV_PATH)
        assert all(k.keyword for k in keywords)
