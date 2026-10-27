import numpy as np
import pytest

from app.feedback import build_feedback, weak_bullets
from app.matching import match_resume
from app.parsing.jd_parser import JobDescription

VOCAB = ["docker", "pasta"]
SYNONYMS = {"kubernetes": "docker", "containers": "docker"}


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            t = t.lower()
            for k, v in SYNONYMS.items():
                t = t.replace(k, v)
            vec = [float(t.count(w)) for w in VOCAB]
            vec.append(1.0 if sum(vec) == 0 else 0.0)
            out.append(vec)
        v = np.array(out)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


RESUME = {
    "header": "Alice",
    "skills": "Python, Docker",
    "experience": ("Backend Engineer | Acme | 2021 - Present\n"
                   "- Deployed docker containers to production\n"
                   "- Reduced latency by 40% using caching\n"
                   "- Mentored new hires"),
    "projects": "Search Tool | Python\n- Built a semantic search tool for internal documents",
}


def feedback(required=(), preferred=(), **kw):
    jd = JobDescription(required=list(required), preferred=list(preferred))
    match = match_resume(RESUME, jd, model=FakeModel())
    return build_feedback(RESUME, match, model=FakeModel(), **kw)


def gaps(items):
    return [i for i in items if i.kind == "gap"]


def test_weak_bullets_flags_lines_without_numbers():
    found = weak_bullets(RESUME)
    texts = [t for _, t in found]
    assert "Deployed docker containers to production" in texts
    assert "Built a semantic search tool for internal documents" in texts
    assert not any("40%" in t for t in texts)          # number hai
    assert not any("Mentored" in t for t in texts)     # bahut chhoti line
    assert not any("|" in t for t in texts)            # entry ki heading line nahi


def test_weak_bullets_ignores_header_and_other_sections():
    assert weak_bullets({"header": "Some long header line without numbers here", "skills": "Python Docker Git SQL R"}) == []


def test_gap_with_related_experience_has_evidence():
    g = gaps(feedback(required=["Kubernetes"]))[0]
    assert g.title == "Missing: Kubernetes" and g.priority == "high"
    assert g.evidence and g.evidence[0].score >= 0.3
    assert {e.section for e in g.evidence} <= {"skills", "experience"}
    assert "related experience" in g.detail


def test_gap_without_related_experience_says_so():
    g = gaps(feedback(required=["Pasta making"]))[0]
    assert g.evidence == []
    assert "nothing in your resume relates" in g.detail


def test_required_gaps_come_before_preferred_and_priority_differs():
    items = gaps(feedback(required=["Kubernetes"], preferred=["Pasta making"]))
    assert [i.priority for i in items] == ["high", "medium"]


def test_matched_requirements_give_no_gap():
    assert gaps(feedback(required=["Python", "Docker"])) == []


def test_max_gaps_is_respected():
    items = gaps(feedback(required=["Pasta making", "Pasta sauce", "Pasta dough"], max_gaps=2))
    assert len(items) == 2


def test_weak_bullet_items_are_added_with_evidence():
    items = [i for i in feedback(required=["Python"]) if i.kind == "weak_bullet"]
    assert items and all(i.evidence and i.evidence[0].text in i.detail for i in items)


def test_related_min_filters_evidence():
    g = gaps(feedback(required=["Kubernetes"], related_min=1.1))[0]
    assert g.evidence == []


def test_feedback_serializes():
    data = [i.model_dump() for i in feedback(required=["Kubernetes"])]
    assert data[0]["kind"] == "gap" and "evidence" in data[0]
