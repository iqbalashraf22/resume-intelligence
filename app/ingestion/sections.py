import re

# Standard section naam -> resumes mein use hone wale alag-alag naam
SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "objective",
                "career objective", "about me"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "work history", "internships"],
    "education": ["education", "academic background", "academics",
                  "educational qualifications"],
    "skills": ["skills", "technical skills", "core competencies",
               "technologies", "key skills"],
    "projects": ["projects", "personal projects", "academic projects",
                 "key projects"],
    "certifications": ["certifications", "certificates", "licenses"],
    "achievements": ["achievements", "awards", "honors", "accomplishments"],
}

# Ulta karo: "work experience" -> "experience" (jaldi dhundne ke liye)
ALIAS_TO_SECTION = {
    alias: section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}


def heading_name(line: str):
    """Agar line heading hai toh standard section naam do, warna None."""
    if not line or len(line) > 40:           # headings chhoti hoti hain
        return None
    # "Technical Skills:" -> "technical skills" (colon, numbers hata do)
    cleaned = re.sub(r"[^a-z& ]", "", line.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return ALIAS_TO_SECTION.get(cleaned)


def split_sections(text: str) -> dict:
    sections = {"header": []}                # naam/email pehli heading se pehle hota hai
    current = "header"

    for line in text.split("\n"):
        name = heading_name(line.strip())
        if name:
            current = name
            sections.setdefault(current, [])  # naya dabba, agar pehle se nahi hai
        else:
            sections[current].append(line)

    # Khali dabbe hata do, lines ko wapas text bana do
    result = {}
    for name, lines in sections.items():
        content = "\n".join(lines).strip()
        if content:
            result[name] = content
    return result