import numpy as np
import pytest

from app.vectorstore import add_resume, get_collection, make_chunks, search

WORDS = ["python", "sql", "docker", "pasta"]


class FakeModel:
    """Keywords gin kar vector banata hai, taaki search ka nateeja pehle se pata ho."""

    def encode(self, texts, normalize_embeddings=True):
        vecs = np.array([[t.lower().count(w) + 0.01 for w in WORDS] for t in texts], dtype=float)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def col(tmp_path):
    return get_collection(path=tmp_path)


SECTIONS_A = {"header": "Alice\nalice@example.com", "skills": "Python, SQL", "experience": "Wrote python code"}
SECTIONS_B = {"header": "Bob", "skills": "Docker, Kubernetes", "experience": "Deployed docker containers"}


def test_make_chunks_skips_header_and_labels_sections():
    chunks = make_chunks(SECTIONS_A)
    assert [c["section"] for c in chunks] == ["skills", "experience"]
    assert all("alice@example.com" not in c["text"] for c in chunks)


def test_long_section_is_split_on_line_boundaries():
    text = "\n".join(f"- bullet number {i} with some words" for i in range(40))
    chunks = make_chunks({"experience": text}, max_chars=200)
    assert len(chunks) > 3
    assert all(len(c["text"]) <= 200 for c in chunks)
    assert "\n".join(c["text"] for c in chunks) == text


def test_add_and_search_finds_the_right_resume(col):
    add_resume("alice", SECTIONS_A, col, model=FakeModel())
    add_resume("bob", SECTIONS_B, col, model=FakeModel())
    hits = search("docker", k=3, collection=col, model=FakeModel())
    assert hits[0]["resume_id"] == "bob"            # docker wala resume sabse upar
    assert hits[-1]["resume_id"] == "alice"         # python wala sabse neeche
    assert hits[-1]["score"] < hits[0]["score"]


def test_search_can_be_limited_to_one_resume(col):
    add_resume("alice", SECTIONS_A, col, model=FakeModel())
    add_resume("bob", SECTIONS_B, col, model=FakeModel())
    hits = search("docker", k=5, resume_id="alice", collection=col, model=FakeModel())
    assert hits and all(h["resume_id"] == "alice" for h in hits)


def test_adding_same_resume_twice_does_not_duplicate(col):
    add_resume("alice", SECTIONS_A, col, model=FakeModel())
    add_resume("alice", SECTIONS_A, col, model=FakeModel())
    assert col.count() == 2


def test_search_on_empty_store_returns_nothing(col):
    assert search("python", collection=col, model=FakeModel()) == []


def test_results_have_expected_fields(col):
    add_resume("alice", SECTIONS_A, col, model=FakeModel())
    hit = search("python", k=1, collection=col, model=FakeModel())[0]
    assert set(hit) == {"text", "section", "resume_id", "score"}