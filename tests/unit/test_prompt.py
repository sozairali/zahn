from zahn.prompt import build_prompt

MESSAGE = "This case is late and I am frustrated."


class TestBuildPrompt:
    def test_returns_string(self):
        result = build_prompt(MESSAGE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_message(self):
        result = build_prompt(MESSAGE)
        assert MESSAGE in result

    def test_contains_json_schema_hint(self):
        result = build_prompt(MESSAGE)
        assert '"label"' in result
        assert '"excerpt"' in result
        assert '"reasoning"' in result
        assert '"detected_language"' in result

    def test_contains_valid_labels(self):
        result = build_prompt(MESSAGE)
        assert "frustration" in result
        assert "satisfaction" in result
        assert "neutral" in result

    def test_contains_special_rules(self):
        result = build_prompt(MESSAGE)
        assert "remake" in result.lower()

    def test_no_markdown_fences_instruction(self):
        result = build_prompt(MESSAGE)
        assert "markdown" in result.lower()

    def test_multilingual_mention(self):
        result = build_prompt(MESSAGE)
        assert "fr" in result
        assert "es" in result

    def test_different_messages_differ(self):
        assert build_prompt("Message A") != build_prompt("Message B")

    def test_message_with_curly_braces(self):
        result = build_prompt("My crown {size} is wrong")
        assert "{size}" in result

    def test_message_with_format_placeholders(self):
        result = build_prompt("{label} is {0} percent wrong")
        assert "{label}" in result
        assert "{0}" in result
