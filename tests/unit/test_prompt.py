from zahn.prompt import build_frustration_prompt, build_satisfaction_prompt, _SHARED_CONTEXT

MESSAGE = "This case is late and I am frustrated."


class TestBuildFrustrationPrompt:
    def test_returns_string(self):
        result = build_frustration_prompt(MESSAGE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_message(self):
        result = build_frustration_prompt(MESSAGE)
        assert MESSAGE in result

    def test_placeholder_not_left_in_output(self):
        # The {message_text} literal must not appear in the rendered prompt
        result = build_frustration_prompt(MESSAGE)
        assert "{message_text}" not in result

    def test_contains_json_schema_hint(self):
        result = build_frustration_prompt(MESSAGE)
        assert '"label"' in result
        assert '"detected_language"' in result
        assert '"excerpt"' in result
        assert '"reasoning"' in result

    def test_contains_yes_no_labels(self):
        result = build_frustration_prompt(MESSAGE)
        assert "yes" in result
        assert "no" in result

    def test_does_not_contain_old_label_schema(self):
        # The output schema must not reference the old 3-way labels
        result = build_frustration_prompt(MESSAGE)
        assert '"frustration|satisfaction|neutral"' not in result

    def test_contains_frustration_specific_rules(self):
        result = build_frustration_prompt(MESSAGE)
        assert "remake" in result.lower()

    def test_no_markdown_fences_instruction(self):
        result = build_frustration_prompt(MESSAGE)
        assert "markdown" in result.lower()

    def test_contains_shared_context(self):
        result = build_frustration_prompt(MESSAGE)
        assert _SHARED_CONTEXT in result

    def test_multilingual_mention(self):
        result = build_frustration_prompt(MESSAGE)
        assert "fr" in result
        assert "es" in result

    def test_different_messages_differ(self):
        assert build_frustration_prompt("Message A") != build_frustration_prompt("Message B")

    def test_message_with_curly_braces(self):
        result = build_frustration_prompt("My crown {size} is wrong")
        assert "{size}" in result

    def test_message_with_format_placeholders(self):
        result = build_frustration_prompt("{label} is {0} percent wrong")
        assert "{label}" in result
        assert "{0}" in result

    def test_empty_message_handled(self):
        # An empty message must not raise; the prompt is still valid
        result = build_frustration_prompt("")
        assert isinstance(result, str)
        assert "{message_text}" not in result


class TestBuildSatisfactionPrompt:
    def test_returns_string(self):
        result = build_satisfaction_prompt(MESSAGE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_message(self):
        result = build_satisfaction_prompt(MESSAGE)
        assert MESSAGE in result

    def test_placeholder_not_left_in_output(self):
        result = build_satisfaction_prompt(MESSAGE)
        assert "{message_text}" not in result

    def test_contains_json_schema_hint(self):
        result = build_satisfaction_prompt(MESSAGE)
        assert '"label"' in result
        assert '"detected_language"' in result
        assert '"excerpt"' in result
        assert '"reasoning"' in result

    def test_contains_yes_no_labels(self):
        result = build_satisfaction_prompt(MESSAGE)
        assert "yes" in result
        assert "no" in result

    def test_does_not_contain_old_label_schema(self):
        result = build_satisfaction_prompt(MESSAGE)
        assert '"frustration|satisfaction|neutral"' not in result

    def test_contains_satisfaction_specific_rules(self):
        # Satisfaction prompt must mention the positive-only condition
        result = build_satisfaction_prompt(MESSAGE)
        assert "genuinely positive" in result.lower()

    def test_no_markdown_fences_instruction(self):
        result = build_satisfaction_prompt(MESSAGE)
        assert "markdown" in result.lower()

    def test_contains_shared_context(self):
        result = build_satisfaction_prompt(MESSAGE)
        assert _SHARED_CONTEXT in result

    def test_multilingual_mention(self):
        result = build_satisfaction_prompt(MESSAGE)
        assert "fr" in result
        assert "es" in result

    def test_different_messages_differ(self):
        assert build_satisfaction_prompt("Message A") != build_satisfaction_prompt("Message B")

    def test_message_with_curly_braces(self):
        result = build_satisfaction_prompt("My crown {size} is wrong")
        assert "{size}" in result

    def test_message_with_format_placeholders(self):
        result = build_satisfaction_prompt("{label} is {0} percent wrong")
        assert "{label}" in result
        assert "{0}" in result

    def test_empty_message_handled(self):
        result = build_satisfaction_prompt("")
        assert isinstance(result, str)
        assert "{message_text}" not in result


class TestPromptsAreDistinct:
    def test_frustration_and_satisfaction_differ(self):
        frust = build_frustration_prompt(MESSAGE)
        sat = build_satisfaction_prompt(MESSAGE)
        assert frust != sat

    def test_share_common_context(self):
        frust = build_frustration_prompt(MESSAGE)
        sat = build_satisfaction_prompt(MESSAGE)
        assert _SHARED_CONTEXT in frust
        assert _SHARED_CONTEXT in sat

    def test_frustration_does_not_contain_satisfaction_task(self):
        # "genuinely positive" is a satisfaction-specific instruction, not frustration
        result = build_frustration_prompt(MESSAGE)
        assert "genuinely positive" not in result

    def test_satisfaction_does_not_contain_frustration_specific_instruction(self):
        # The "DO NOT CANCEL FRUSTRATION" block belongs only to the frustration prompt
        result = build_satisfaction_prompt(MESSAGE)
        assert "DO NOT CANCEL FRUSTRATION" not in result
