import numpy as np
import pytest

from app.embeddings import cosine_similarity, embed


class FakeModel:
    """Asli model ki jagah: har text ko chhota sa vector deta hai (download nahi chahiye)."""

    def encode(self, texts, normalize_embeddings=True):
        vecs = np.array([[3.0, 4.0] if "ml" in t else [4.0, -3.0] for t in texts])
        if normalize_embeddings:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


def test_cosine_identical_is_one():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_opposite_is_minus_one():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_unrelated_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_ignores_length():
    assert cosine_similarity([1, 1], [10, 10]) == pytest.approx(1.0)


def test_cosine_zero_vector_is_safe():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_embed_accepts_one_string_or_list():
    one = embed("ml pipelines", model=FakeModel())
    many = embed(["ml pipelines", "pasta"], model=FakeModel())
    assert one.shape == (1, 2)
    assert many.shape == (2, 2)


def test_embed_vectors_are_normalized():
    v = embed(["ml pipelines", "pasta"], model=FakeModel())
    assert np.linalg.norm(v, axis=1) == pytest.approx([1.0, 1.0])


def test_similar_texts_score_higher_with_fake_model():
    v = embed(["ml pipelines", "ml models", "pasta"], model=FakeModel())
    assert cosine_similarity(v[0], v[1]) > cosine_similarity(v[0], v[2])