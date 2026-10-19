from pydantic import Field, model_validator

from .llm_parser import ask_json
from .schema import Lenient

MAX_JD_CHARS = 6000   # bahut lamba text chhote model ko confuse karta hai

JD_SYSTEM_PROMPT = (
    "You extract requirements from job descriptions. Return ONLY a valid JSON object, "
    "with no explanation and no markdown. Rules: use only information that is in the text; "
    "never invent anything; each requirement is ONE short phrase (a skill, a tool, or a "
    "qualification, for example 'Python', 'Kubernetes', '3+ years of backend experience'); "
    "required = must-have items; preferred = nice-to-have or bonus items; do not repeat items; "
    'include EVERY requirement listed in the text. Use "" or [] when something is missing.'
)

JD_FORMAT_EXAMPLE = '{"title": "", "required": ["Python", "SQL"], "preferred": ["Docker"]}'


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


def parse_job_description(text: str, client=None) -> JobDescription:
    text = text.strip()
    if not text:
        raise ValueError("Job description khali hai.")
    user_prompt = (
        f"Return JSON in this format:\n{JD_FORMAT_EXAMPLE}\n\n"
        f"Job description:\n{text[:MAX_JD_CHARS]}"
    )
    return ask_json(JD_SYSTEM_PROMPT, user_prompt, JobDescription, client)