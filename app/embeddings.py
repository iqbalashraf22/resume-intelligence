import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Model ki files project ke andar rahengi (project D par hai, toh C bharega nahi)
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "models"

_model = None


def get_model():
    """Model pehli baar load hota hai (download bhi), phir yaad rakha jata hai."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            os.getenv("EMBED_MODEL", DEFAULT_MODEL), cache_folder=str(CACHE_DIR)
        )
    return _model


def embed(texts, model=None) -> np.ndarray:
    """Text (ya texts ki list) ko vectors mein badlo. Shape: (kitne texts, 384)."""
    if isinstance(texts, str):
        texts = [texts]
    model = model or get_model()
    return np.asarray(model.encode(texts, normalize_embeddings=True))


def cosine_similarity(a, b) -> float:
    """Do vectors kitne milte-julte hain: 1 = bilkul same meaning, 0 = koi sambandh nahi."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0