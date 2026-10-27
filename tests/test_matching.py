import numpy as np
import pytest

from app.matching import degree_level, match_resume
from app.parsing.jd_parser import JobDescription

VOCAB = ["pasta", "communication", "budget"]
SYNONYMS = {"talking": "communication", "clients": "communication"}


class FakeModel:
    """Sirf 'semantic' requirements ke liye: chhote nakli vectors."""

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
    "header": "Alice\nalice@example.com",
    "summary": "Strong communication skills",
    "skills": "Python, Docker, MLflow",
    "experience": "Built a vector database search using LangChain",
    "education": "M.Tech in Artificial Intelligence\nB.Tech in Computer Science",
}


def run(required=(), preferred=(), resume=RESUME, threshold=0.45):
    jd = JobDescription(required=list(required), preferred=list(preferred))
    return match_resume(resume, jd, model=FakeModel(), threshold=threshold)


def by_label(result):
    return {d.requirement: d for d in result.details}


def test_skill_match_has_evidence():
    d = run(required=["Python"]).details[0]
    assert d.matched and d.exact and d.score == 1.0
    assert d.section == "skills" and "Python" in d.evidence


def test_alias_and_plural_are_found():
    d = run(required=["Experience with vector databases"]).details[0]
    assert d.requirement == "Vector databases" and d.matched and d.section == "experience"


def test_missing_skill():
    r = run(required=["Kubernetes"])
    assert r.missing_required == ["Kubernetes"] and r.percent == 0.0
    assert r.details[0].score == 0.0 and not r.details[0].exact


def test_and_bullet_becomes_separate_skills():
    d = by_label(run(required=["Strong Python and Kubernetes"]))
    assert set(d) == {"Python", "Kubernetes"}
    assert d["Python"].matched and not d["Kubernetes"].matched


def test_or_bullet_needs_only_one_skill():
    r = run(required=["Kubernetes or Docker", "Kubernetes or Terraform"])
    d = by_label(r)
    assert d["Kubernetes or Docker"].matched
    assert not d["Kubernetes or Terraform"].matched


def test_degree_levels():
    assert degree_level("Master's degree in a related field") == 2
    assert degree_level("M.Tech in AI") == 2
    assert degree_level("B.Tech in Computer Science") == 1
    assert degree_level("PhD preferred") == 3
    assert degree_level("Scrum Master and Python") == 0
    assert degree_level("tell me about it") == 0


def test_master_requirement_checked_against_resume_degree():
    assert run(required=["Master's degree in a related field"]).details[0].matched
    only_bachelor = dict(RESUME, education="B.Tech in Computer Science")
    assert not run(required=["Master's degree in a related field"], resume=only_bachelor).details[0].matched


def test_higher_degree_satisfies_lower_requirement():
    assert run(required=["Bachelor's degree in CS"]).details[0].matched
    assert not run(required=["PhD in a related field"]).details[0].matched


def test_semantic_requirement_uses_embeddings():
    r = by_label(run(required=["Good talking with clients", "Pasta making"]))
    assert r["Good talking with clients"].matched and not r["Good talking with clients"].exact
    assert not r["Pasta making"].matched


def test_threshold_applies_to_semantic_only():
    r = run(required=["Good talking with clients", "Python"], threshold=1.01)
    d = by_label(r)
    assert not d["Good talking with clients"].matched
    assert d["Python"].matched


def test_same_skill_is_counted_once():
    r = run(required=["Python"], preferred=["Experience with Python"])
    assert len(r.details) == 1


def test_weights_required_counts_double():
    r = run(required=["Python", "Kubernetes"], preferred=["Docker"])
    assert r.percent == pytest.approx(60.0)       # (2*1 + 1*1) / (2*2 + 1)
    assert r.missing_required == ["Kubernetes"] and r.missing_preferred == []


def test_skill_only_jd_never_loads_the_embedding_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("embedding model should not be used")

    monkeypatch.setattr("app.matching.embed", boom)
    r = match_resume(RESUME, JobDescription(required=["Python", "Master's degree"], preferred=["Kubernetes"]))
    assert r.percent == pytest.approx(80.0)       # (2 + 2) / (2 + 2 + 1)


def test_go_is_not_found_in_google():
    resume = {"experience": "Worked at Google on search"}
    assert not run(required=["Go"], resume=resume).details[0].matched


def test_no_requirements_gives_zero():
    assert run().percent == 0.0


def test_resume_with_no_content_misses_everything():
    r = run(required=["Python", "Pasta making"], preferred=["Docker"], resume={"header": "Alice"})
    assert r.percent == 0.0
    assert r.missing_required == ["Python", "Pasta making"] and r.missing_preferred == ["Docker"]


def test_result_serializes_missing_lists():
    data = run(required=["Python", "Kubernetes"]).model_dump()
    assert data["missing_required"] == ["Kubernetes"]
    assert "details" in data and "percent" in data
