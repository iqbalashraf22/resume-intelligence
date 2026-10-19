from pathlib import Path

import chromadb
from chromadb.config import Settings

from .embeddings import embed

DB_DIR = Path(__file__).resolve().parent.parent / ".cache" / "chroma"
COLLECTION = "resume_chunks"
MAX_CHARS = 500


def get_collection(path=DB_DIR, name: str = COLLECTION):
    """Vector database kholo (nahi hai toh bana do). Data disk par save rehta hai."""
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def make_chunks(sections: dict, max_chars: int = MAX_CHARS) -> list[dict]:
    """Har section ko chhote tukdon (chunks) mein todo. Lines ke beech se nahi katte."""
    chunks = []
    for section, text in sections.items():
        if section == "header":                    # naam/phone/email search ke liye nahi
            continue
        current = ""
        for line in text.split("\n"):
            if current and len(current) + len(line) + 1 > max_chars:
                chunks.append({"section": section, "text": current.strip()})
                current = ""
            current += line + "\n"
        if current.strip():
            chunks.append({"section": section, "text": current.strip()})
    return chunks


def add_resume(resume_id: str, sections: dict, collection=None, model=None) -> int:
    """Resume ke chunks vector store mein daalo. Dobara chalane par purane chunks badal jate hain."""
    collection = collection or get_collection()
    collection.delete(where={"resume_id": resume_id})

    chunks = make_chunks(sections)
    if not chunks:
        return 0

    # "SKILLS: Python, SQL" jaisa label embedding ko batata hai ki yeh kis section ka hai
    vectors = embed([f"{c['section']}: {c['text']}" for c in chunks], model=model)
    collection.add(
        ids=[f"{resume_id}:{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=vectors.tolist(),
        metadatas=[{"resume_id": resume_id, "section": c["section"]} for c in chunks],
    )
    return len(chunks)


def search(query: str, k: int = 3, resume_id: str | None = None, collection=None, model=None) -> list[dict]:
    """Query se sabse milte-julte chunks dhundo. score: 1 = bilkul same matlab."""
    collection = collection or get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=embed(query, model=model).tolist(),
        n_results=k,
        where={"resume_id": resume_id} if resume_id else None,
    )
    hits = []
    for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({
            "text": text,
            "section": meta["section"],
            "resume_id": meta["resume_id"],
            "score": round(1 - dist, 3),           # cosine distance -> similarity
        })
    return hits