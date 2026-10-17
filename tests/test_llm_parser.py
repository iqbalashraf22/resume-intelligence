from types import SimpleNamespace

import pytest

from app.parsing.llm_parser import build_input, extract_json, parse_sections
from app.parsing.pipeline import parse_resume

GOOD = '{"skills": ["Python", "SQL"], "experience": [{"title": "Analyst", "company": "Acme", "bullets": ["Built dashboards"]}]}'


class FakeClient:
    """Asli Hugging Face ki jagah: har call par agla jawab deta hai."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def chat_completion(self, messages, **kwargs):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


SECTIONS = {"header": "John Doe\njohn@example.com", "skills": "Python, SQL",
            "experience": "Analyst at Acme\n- Built dashboards"}


def test_extract_json_strips_fences_and_chatter():
    raw = "Sure! Here you go:\n```json\n" + GOOD + "\n```\nHope it helps."
    assert extract_json(raw) == GOOD


def test_build_input_excludes_header():
    text = build_input(SECTIONS)
    assert "john@example.com" not in text
    assert "## SKILLS" in text and "## EXPERIENCE" in text


def test_build_input_without_sections_raises():
    with pytest.raises(ValueError):
        build_input({"header": "John Doe"})


def test_valid_json_is_parsed():
    body = parse_sections(SECTIONS, FakeClient(GOOD))
    assert body.skills == ["Python", "SQL"]
    assert body.experience[0].company == "Acme"


def test_fenced_json_is_parsed():
    body = parse_sections(SECTIONS, FakeClient("```json\n" + GOOD + "\n```"))
    assert body.skills == ["Python", "SQL"]


def test_nulls_and_numbers_are_tolerated():
    reply = '{"skills": ["Python"], "education": [{"degree": "B.Tech", "year": 2022, "field": null}]}'
    body = parse_sections(SECTIONS, FakeClient(reply))
    assert body.education[0].year == "2022"
    assert body.education[0].field == ""


def test_retry_after_invalid_json():
    client = FakeClient("this is not json", GOOD)
    body = parse_sections(SECTIONS, client)
    assert client.calls == 2
    assert body.skills == ["Python", "SQL"]


def test_gives_up_after_retries():
    client = FakeClient("not json at all")
    with pytest.raises(ValueError):
        parse_sections(SECTIONS, client)
    assert client.calls == 3          # 1 koshish + 2 retries


def test_api_error_is_reported_clearly():
    class Broken:
        def chat_completion(self, **kwargs):
            raise ConnectionError("no internet")

    with pytest.raises(RuntimeError, match="Hugging Face"):
        parse_sections(SECTIONS, Broken())


def test_full_pipeline_with_fake_llm(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("John Doe\njohn@example.com\n\nSKILLS\nPython, SQL\n\nEXPERIENCE\nAnalyst at Acme\n- Built dashboards\n")
    resume = parse_resume(str(f), FakeClient(GOOD))
    assert resume.contact.email == "john@example.com"
    assert resume.contact.name == "John Doe"
    assert resume.skills == ["Python", "SQL"]