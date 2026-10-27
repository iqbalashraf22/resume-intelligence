import os
import re

import numpy as np
from pydantic import BaseModel, Field, computed_field

from .embeddings import embed
from .parsing.jd_parser import JobDescription
from .skills import extract_skills, has_skill
from .vectorstore import make_chunks

DEFAULT_THRESHOLD = 0.45   # sirf "semantic" requirements ke liye (neeche dekho)
REQUIRED_WEIGHT = 2        # zaroori requirement ka wazan
PREFERRED_WEIGHT = 1       # bonus requirement ka wazan

# Degree ka level: 3 = PhD, 2 = Master, 1 = Bachelor
DEGREE_LEVELS = [
    (3, re.compile(r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctoral\b", re.IGNORECASE)),
    (2, re.compile(r"\bmaster'?s\b|\bmasters\b|\bmaster\s+(?:degree|of|in)\b|\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bmba\b|\bm\.e\.",
                   re.IGNORECASE)),
    (1, re.compile(r"\bbachelor'?s?\b|\bb\.?\s?tech\b|\bb\.e\.|\bb\.?\s?sc\b|\bbca\b|\bb\.?\s?com\b|\bundergraduate\b",
                   re.IGNORECASE)),
]


def degree_level(text: str) -> int:
    for level, pattern in DEGREE_LEVELS:
        if pattern.search(text):
            return level
    return 0


class RequirementMatch(BaseModel):
    requirement: str
    kind: str              # "required" ya "preferred"
    score: float           # 1.0 = rule se mila, 0.0 = nahi mila, beech ka = embedding similarity
    matched: bool
    exact: bool = False    # True = skill/degree rule se pakda (andaaze se nahi)
    section: str = ""      # resume ka kaunsa section proof hai
    evidence: str = ""     # us section ka text (citation ke liye)


class MatchResult(BaseModel):
    percent: float = 0.0
    details: list[RequirementMatch] = Field(default_factory=list)

    @computed_field
    @property
    def missing_required(self) -> list[str]:
        return [d.requirement for d in self.details if d.kind == "required" and not d.matched]

    @computed_field
    @property
    def missing_preferred(self) -> list[str]:
        return [d.requirement for d in self.details if d.kind == "preferred" and not d.matched]


def _plan(items: list[tuple[str, str]]) -> list[dict]:
    """Har requirement ko check karne ka tarika tay karo:
    degree  -> degree ka level dekho
    skill   -> taxonomy ke skills nikaalo ('A and B' = do alag, 'A or B' = koi ek chalega)
    semantic-> koi skill nahi mila, embedding se andaaza lagao
    """
    plan, seen = [], set()

    def add(label, kind, mode, **extra):
        if label.lower() not in seen:               # ek hi skill do baar na gine
            seen.add(label.lower())
            plan.append({"label": label, "kind": kind, "mode": mode, **extra})

    for req, kind in items:
        level = degree_level(req)
        skills = [] if level else extract_skills(req)
        if level:
            add(req, kind, "degree", level=level)
        elif len(skills) > 1 and re.search(r"\bor\b", req, re.IGNORECASE):
            add(" or ".join(skills), kind, "skill", skills=skills, any_of=True)
        elif skills:
            for skill in skills:
                add(skill, kind, "skill", skills=[skill], any_of=True)
        else:
            add(req, kind, "semantic")
    return plan


def _summarize(details: list[RequirementMatch]) -> MatchResult:
    weights = [REQUIRED_WEIGHT if d.kind == "required" else PREFERRED_WEIGHT for d in details]
    total = sum(weights)
    got = sum(w for w, d in zip(weights, details) if d.matched)
    return MatchResult(percent=round(100 * got / total, 1) if total else 0.0, details=details)


def match_resume(sections: dict, jd: JobDescription, model=None, threshold: float | None = None) -> MatchResult:
    """Resume (sections) ko JD ki requirements se match karo."""
    if threshold is None:
        threshold = float(os.getenv("MATCH_THRESHOLD", DEFAULT_THRESHOLD))

    items = [(r, "required") for r in jd.required] + [(r, "preferred") for r in jd.preferred]
    plan = _plan(items)
    if not plan:
        return MatchResult()

    chunks = make_chunks(sections)

    def result(step, matched, score, chunk_index=None, exact=False):
        chunk = chunks[chunk_index] if chunk_index is not None else None
        return RequirementMatch(
            requirement=step["label"], kind=step["kind"], score=round(score, 3), matched=matched, exact=exact,
            section=chunk["section"] if chunk else "", evidence=chunk["text"][:200] if chunk else "",
        )

    details: list[RequirementMatch | None] = [None] * len(plan)
    semantic = []

    for i, step in enumerate(plan):
        if step["mode"] == "skill":
            hit = None
            for skill in step["skills"]:                       # koi ek mil jaye toh kaafi (any_of)
                idx = next((j for j, c in enumerate(chunks) if has_skill(c["text"], skill)), None)
                if idx is not None:
                    hit = idx
                    break
            details[i] = result(step, hit is not None, 1.0 if hit is not None else 0.0, hit, exact=hit is not None)
        elif step["mode"] == "degree":
            levels = [degree_level(c["text"]) for c in chunks]
            best = max(levels, default=0)
            ok = best >= step["level"]
            details[i] = result(step, ok, 1.0 if ok else 0.0, levels.index(best) if ok else None, exact=ok)
        else:
            semantic.append(i)

    if semantic:                                               # embedding tabhi chalao jab zaroorat ho
        if not chunks:
            for i in semantic:
                details[i] = result(plan[i], False, 0.0)
        else:
            chunk_vecs = embed([f"{c['section']}: {c['text']}" for c in chunks], model=model)
            req_vecs = embed([plan[i]["label"] for i in semantic], model=model)
            sims = req_vecs @ chunk_vecs.T
            for row, i in enumerate(semantic):
                best = int(np.argmax(sims[row]))
                score = float(sims[row][best])
                details[i] = result(plan[i], score >= threshold, score, best)

    return _summarize(details)
