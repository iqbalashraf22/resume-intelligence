from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Lenient(BaseModel):
    """LLM kabhi null ya number bhej deta hai; usse crash hone se bachao."""

    model_config = ConfigDict(coerce_numbers_to_str=True)   # 2022 -> "2022"

    @model_validator(mode="before")
    @classmethod
    def drop_nulls(cls, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}  # null -> default
        return data


class ContactInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    links: list[str] = Field(default_factory=list)

    @field_validator("email")
    @classmethod
    def clean_email(cls, v: str) -> str:
        return v.strip().lower()


class Experience(Lenient):
    title: str = ""
    company: str = ""
    start_date: str = ""
    end_date: str = ""
    bullets: list[str] = Field(default_factory=list)


class Education(Lenient):
    degree: str = ""
    field: str = ""
    institution: str = ""
    year: str = ""


class Project(Lenient):
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class ResumeBody(Lenient):
    """Resume ka woh hissa jo LLM nikalta hai."""

    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class ParsedResume(ResumeBody):
    """Poora resume = contact (regex) + body (LLM)."""

    contact: ContactInfo = Field(default_factory=ContactInfo)