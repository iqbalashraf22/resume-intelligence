from types import SimpleNamespace

import pytest

from app.parsing.jd_parser import JobDescription, parse_job_description

GOOD = '{"title": "ML Engineer", "required": ["Python", "PyTorch"], "preferred": ["Kubernetes"]}'


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


def test_valid_json_is_parsed():
    jd = parse_job_description("We need an ML engineer", FakeClient(GOOD))
    assert jd.title == "ML Engineer"
    assert jd.required == ["Python", "PyTorch"]
    assert jd.preferred == ["Kubernetes"]


def test_duplicates_blanks_and_overlap_are_removed():
    reply = '{"required": ["Python", "python", " ", "SQL"], "preferred": ["sql", "Docker", "Docker"]}'
    jd = parse_job_description("text", FakeClient(reply))
    assert jd.required == ["Python", "SQL"]
    assert jd.preferred == ["Docker"]


def test_nulls_are_tolerated():
    jd = parse_job_description("text", FakeClient('{"title": null, "required": ["Go"], "preferred": null}'))
    assert jd.title == "" and jd.preferred == []


def test_empty_jd_raises_before_calling_llm():
    client = FakeClient(GOOD)
    with pytest.raises(ValueError):
        parse_job_description("   \n ", client)
    assert client.calls == 0


def test_retry_after_invalid_json():
    client = FakeClient("not json", GOOD)
    jd = parse_job_description("text", client)
    assert client.calls == 2 and jd.required == ["Python", "PyTorch"]


def test_very_long_text_is_truncated():
    client = FakeClient(GOOD)
    parse_job_description("x" * 50_000, client)
    user_message = client.last_messages[-1]["content"]
    assert len(user_message) < 7000


def test_model_can_be_built_directly():
    assert JobDescription().required == []