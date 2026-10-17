import os

from .schema import ResumeBody

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SECTIONS_FOR_LLM = ("skills", "experience", "education", "projects", "certifications")
MAX_RETRIES = 2   # pehli koshish ke baad 2 aur koshish

SYSTEM_PROMPT = (
    "You are a resume parser. Read the resume text and return ONLY a valid JSON object, "
    "with no explanation and no markdown. Rules: use only information that is in the text; "
    'never invent anything; use "" for missing text and [] for missing lists; '
    "keep bullet points exactly as written; skills must be a flat list of single skill names. "
    "Include EVERY job, EVERY education entry and EVERY project in the text: one JSON object "
    "for each, and never stop after the first one. For a project, put a one-sentence summary "
    "of its bullet points in description."
)

FORMAT_EXAMPLE = """{
  "skills": ["Python", "SQL"],
  "experience": [
    {"title": "", "company": "", "start_date": "", "end_date": "", "bullets": [""]}
  ],
  "education": [
    {"degree": "", "field": "", "institution": "", "year": ""}
  ],
  "projects": [
    {"name": "", "description": "", "technologies": [""]}
  ],
  "certifications": [""]
}"""


def get_client():
    """LLM client banao. .env mein LLM_BACKEND=ollama ya hf (default hf) se chuno."""
    from dotenv import load_dotenv

    load_dotenv()
    if os.getenv("LLM_BACKEND", "hf").lower() == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            api_key=os.getenv("OLLAMA_API_KEY") or None,   # cloud ke liye zaroori, local ke liye nahi
        )

    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN nahi mila. Project folder ki .env file mein daalo.")
    return InferenceClient(model=os.getenv("HF_MODEL", DEFAULT_MODEL), token=token)

def build_input(sections: dict) -> str:
    """Sirf kaam ke sections bhejo. Header (naam, phone, email) LLM ko nahi jata."""
    parts = [f"## {name.upper()}\n{sections[name]}" for name in SECTIONS_FOR_LLM if name in sections]
    if not parts:
        raise ValueError("Resume mein koi parse karne layak section nahi mila.")
    return "\n\n".join(parts)


def extract_json(text: str) -> str:
    """Model kabhi ```json ... ``` ya extra baatein likh deta hai; sirf { ... } rakho."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def parse_sections(sections: dict, client=None) -> ResumeBody:
    client = client or get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Return JSON in this format. Every list can hold many objects: repeat the object once for each entry in the text.\n{FORMAT_EXAMPLE}\n\n"
                                    f"Resume text:\n{build_input(sections)}"},
    ]

    last_error = None
    for _ in range(MAX_RETRIES + 1):
        try:
            response = client.chat_completion(messages=messages, max_tokens=2000, temperature=0)
        except Exception as e:   # token / model / network ki galti, retry se theek nahi hogi
            raise RuntimeError(f"Hugging Face call fail hui: {e}") from e

        raw = response.choices[0].message.content or ""
        try:
            return ResumeBody.model_validate_json(extract_json(raw))
        except ValueError as e:   # galat JSON ya galat schema (pydantic ka error bhi ValueError hai)
            last_error = e
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That output was invalid: {str(e)[:500]}\n"
                                            "Return ONLY the corrected JSON object."},
            ]

    raise ValueError(f"LLM {MAX_RETRIES + 1} koshishon mein sahi JSON nahi de paya: {last_error}")