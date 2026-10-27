from types import SimpleNamespace

import pytest

from app.parsing.jd_parser import JobDescription, bullet_requirements, parse_job_description

JD_TEXT = """ML Engineer
Lingua Labs

Responsibilities
- Fine-tune transformer models

Requirements
- 2+ years of experience in machine learning
- Strong Python and PyTorch
- Experience with vector databases

Nice to have
- Kubernetes
- LangChain
"""


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_messages = None

    def chat_completion(self, messages, **kwargs):
        self.last_messages = messages
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def parse(reply, text=JD_TEXT):
    return parse_job_description(text, FakeClient(reply))


def test_bullet_requirements_uses_headings_and_ignores_responsibilities():
    assert bullet_requirements(JD_TEXT) == [
        ("required", "2+ years of experience in machine learning"),
        ("required", "Strong Python and PyTorch"),
        ("required", "Experience with vector databases"),
        ("preferred", "Kubernetes"),
        ("preferred", "LangChain"),
    ]


def test_good_items_are_kept_and_missed_bullets_are_added():
    jd = parse('{"title": "ML Engineer", "required": ["Python", "PyTorch"], "preferred": ["Kubernetes", "LangChain"]}')
    assert jd.title == "ML Engineer"
    assert jd.required == ["Python", "PyTorch",
                           "2+ years of experience in machine learning", "Experience with vector databases"]
    assert jd.preferred == ["Kubernetes", "LangChain"]


def test_values_copied_from_the_format_example_are_removed():
    # Model ne example ki "SQL" aur "Docker" copy kar di, jo JD mein hai hi nahi
    jd = parse('{"required": ["Python", "SQL"], "preferred": ["Docker", "Kubernetes"]}')
    assert "SQL" not in jd.required
    assert "Docker" not in jd.preferred
    assert "Python" in jd.required and "Kubernetes" in jd.preferred


def test_placeholder_values_are_removed():
    jd = parse('{"required": ["<requirement>", "Python"], "preferred": ["<requirement>"]}')
    assert not any("<" in x for x in jd.required + jd.preferred)


def test_model_that_returns_nothing_still_gets_all_bullets():
    jd = parse('{"required": [], "preferred": []}')
    assert len(jd.required) == 3
    assert jd.preferred == ["Kubernetes", "LangChain"]


def test_short_item_must_match_whole_word():
    text = "Requirements\n- Good communication skills\n"
    jd = parse('{"required": ["Go"]}', text)       # "Go" sirf "Good" ke andar hai
    assert jd.required == ["Good communication skills"]


def test_nulls_are_tolerated():
    jd = parse('{"title": null, "required": ["Python"], "preferred": null}')
    assert "Python" in jd.required and jd.title == ""


def test_empty_jd_raises_before_calling_llm():
    client = FakeClient("{}")
    with pytest.raises(ValueError):
        parse_job_description("   \n ", client)
    assert client.calls == 0


def test_retry_after_invalid_json():
    client = FakeClient("not json", '{"required": ["Python"]}')
    jd = parse_job_description(JD_TEXT, client)
    assert client.calls == 2 and "Python" in jd.required


def test_very_long_text_is_truncated():
    client = FakeClient("{}")
    parse_job_description("Requirements\n- " + "x" * 50_000, client)
    assert len(client.last_messages[-1]["content"]) < 7000


def test_model_can_be_built_directly():
    assert JobDescription().required == []

from app.parsing.jd_parser import parse_job_description_rules  # noqa: E402


def test_rules_parser_works_without_any_llm():
    jd = parse_job_description_rules(JD_TEXT)
    assert jd.title == "ML Engineer"
    assert jd.required == ["2+ years of experience in machine learning", "Strong Python and PyTorch",
                           "Experience with vector databases"]
    assert jd.preferred == ["Kubernetes", "LangChain"]


def test_rules_parser_rejects_empty_text():
    with pytest.raises(ValueError):
        parse_job_description_rules("  \n ")
