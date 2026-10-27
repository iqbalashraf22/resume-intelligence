import re

from pydantic import Field, model_validator

from .llm_parser import ask_json
from .schema import Lenient

MAX_JD_CHARS = 6000   # bahut lamba text chhote model ko confuse karta hai

JD_SYSTEM_PROMPT = (
    "You extract requirements from job descriptions. Return ONLY a valid JSON object, "
    "with no explanation and no markdown. Rules: copy requirements only from the job "
    "description text; the format example shows structure only, never copy its values; "
    "never invent anything; each requirement is ONE short phrase (a skill, a tool, or a "
    "qualification) written as it appears in the text; required = must-have items; "
    "preferred = nice-to-have or bonus items; do not repeat items; include EVERY requirement "
    'listed in the text. Use "" or [] when something is missing.'
)

JD_FORMAT_EXAMPLE = '{"title": "<job title>", "required": ["<requirement>", "<requirement>"], "preferred": ["<requirement>"]}'

BULLET = re.compile(r"^[-*\u2022]\s+(.*)$")
PREFERRED_HEAD = re.compile(r"nice to have|preferred|bonus|good to have|desirable|\bplus\b", re.IGNORECASE)
REQUIRED_HEAD = re.compile(r"require|qualif|must|what you.?ll need|looking for|skills", re.IGNORECASE)
PLACEHOLDER = re.compile(r"^<.*>$")


class JobDescription(Lenient):
    title: str = ""
    required: list[str] = Field(default_factory=list)
    preferred: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def tidy(self):
        """Khali aur duplicate items hatao; required wala item preferred mein dobara na ho."""
        def unique(items, exclude=()):
            seen = {x.lower() for x in exclude}
            out = []
            for item in items:
                item = item.strip()
                if item and item.lower() not in seen:
                    seen.add(item.lower())
                    out.append(item)
            return out

        self.required = unique(self.required)
        self.preferred = unique(self.preferred, exclude=self.required)
        return self


def _contains(phrase: str, text: str) -> bool:
    """phrase poora shabd/phrase ki tarah text mein hai? ("go" "good" mein nahi milta)"""
    return bool(re.search(r"(?<!\w)" + re.escape(phrase.strip()) + r"(?!\w)", text, re.IGNORECASE))


def bullet_requirements(text: str) -> list[tuple[str, str]]:
    """Bina LLM ke: 'Requirements' ke neeche ke bullets = required, 'Nice to have' ke neeche = preferred."""
    mode, items = None, []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = BULLET.match(line)
        if m:
            if mode:
                items.append((mode, m.group(1).strip()))
        elif len(line) <= 50:                       # chhoti line = heading
            if PREFERRED_HEAD.search(line):
                mode = "preferred"
            elif REQUIRED_HEAD.search(line):
                mode = "required"
            else:
                mode = None                         # Responsibilities, About... ke bullets chhod do
    return items


def repair(jd: JobDescription, text: str) -> JobDescription:
    """LLM ki galtiyan theek karo:
    1) jo item JD ke text mein hai hi nahi (example se copy, ya invent kiya) hata do
    2) jo bullet koi item cover nahi karta, use requirement bana ke jod do
    """
    required = [i for i in jd.required if not PLACEHOLDER.match(i.strip()) and _contains(i, text)]
    preferred = [i for i in jd.preferred if not PLACEHOLDER.match(i.strip()) and _contains(i, text)]
    kept = required + preferred

    for kind, bullet in bullet_requirements(text):
        if not any(_contains(item, bullet) for item in kept):
            (required if kind == "required" else preferred).append(bullet)

    return JobDescription(title=jd.title, required=required, preferred=preferred)


def parse_job_description(text: str, client=None) -> JobDescription:
    text = text.strip()
    if not text:
        raise ValueError("Job description khali hai.")
    text = text[:MAX_JD_CHARS]
    user_prompt = f"Return JSON in this format:\n{JD_FORMAT_EXAMPLE}\n\nJob description:\n{text}"
    return repair(ask_json(JD_SYSTEM_PROMPT, user_prompt, JobDescription, client), text)

def parse_job_description_rules(text: str) -> JobDescription:
    """LLM ke bina: title = pehli line, requirements = 'Requirements' / 'Nice to have' ke neeche ke bullets."""
    text = text.strip()
    if not text:
        raise ValueError("Job description khali hai.")
    text = text[:MAX_JD_CHARS]
    title = text.split("\n", 1)[0].strip()
    return repair(JobDescription(title=title), text)
