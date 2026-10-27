import os
import re

import numpy as np
from pydantic import BaseModel, Field

from .embeddings import embed
from .matching import MatchResult
from .vectorstore import make_chunks

DEFAULT_RELATED_MIN = 0.30   # isse kam similarity wale chunk ko "related" nahi maante


class Evidence(BaseModel):
    section: str
    text: str
    score: float = 1.0


class FeedbackItem(BaseModel):
    kind: str                  # "gap" (JD ki requirement missing) ya "weak_bullet" (resume ki kamzor line)
    priority: str              # "high", "medium"
    title: str
    detail: str
    evidence: list[Evidence] = Field(default_factory=list)


def weak_bullets(sections: dict, limit: int = 5) -> list[tuple[str, str]]:
    """Experience/projects ki woh lines jinme koi number nahi (koi measurable result nahi)."""
    found = []
    for name in ("experience", "projects"):
        for line in sections.get(name, "").split("\n"):
            text = re.sub(r"^[-*]\s+", "", line.strip())
            if len(text.split()) >= 5 and "|" not in text and not re.search(r"\d", text):
                found.append((name, text))
    return found[:limit]


def _gap_item(match, evidence: list[Evidence]) -> FeedbackItem:
    req = match.requirement
    if evidence:
        detail = (f"The job asks for {req}, but your resume doesn't mention it. You have related experience "
                  f"(see evidence). If you have really used {req}, add a bullet that names it and the result; "
                  f"if not, don't claim it - say you are learning it, or build a small project with it.")
    else:
        detail = (f"The job asks for {req}, but nothing in your resume relates to it. "
                  f"Only add it if it is true; otherwise treat it as a skill to learn.")
    return FeedbackItem(
        kind="gap",
        priority="high" if match.kind == "required" else "medium",
        title=f"Missing: {req}",
        detail=detail,
        evidence=evidence,
    )


def build_feedback(sections: dict, match: MatchResult, model=None, max_gaps: int = 6,
                   max_bullets: int = 5, related_min: float | None = None) -> list[FeedbackItem]:
    """Resume + match result se feedback ki list banao. Har sujhav ke saath resume ka asli saboot."""
    if related_min is None:
        related_min = float(os.getenv("RELATED_MIN", DEFAULT_RELATED_MIN))

    missing = [d for d in match.details if not d.matched]
    missing.sort(key=lambda d: d.kind != "required")          # required pehle, order wahi rehta hai
    missing = missing[:max_gaps]

    chunks = make_chunks(sections)
    sims = None
    if missing and chunks:
        chunk_vecs = embed([f"{c['section']}: {c['text']}" for c in chunks], model=model)
        req_vecs = embed([d.requirement for d in missing], model=model)
        sims = req_vecs @ chunk_vecs.T

    items = []
    for i, d in enumerate(missing):
        evidence = []
        if sims is not None:
            for j in np.argsort(-sims[i])[:2]:
                if sims[i][j] >= related_min:
                    evidence.append(Evidence(section=chunks[j]["section"], text=chunks[j]["text"][:200],
                                             score=round(float(sims[i][j]), 3)))
        items.append(_gap_item(d, evidence))

    for section, text in weak_bullets(sections, max_bullets):
        items.append(FeedbackItem(
            kind="weak_bullet", priority="medium", title="Add a measurable result",
            detail=f'This line has no numbers: "{text}". If you know them, add a number such as a percentage, '
                   f"time saved, users or volume.",
            evidence=[Evidence(section=section, text=text)],
        ))
    return items
