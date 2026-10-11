import re

from app.ingestion.sections import ALIAS_TO_SECTION

from .schema import ContactInfo

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"[+(]?\d[\d ()\-]{7,}\d")
LINK_RE = re.compile(r"https?://\S+|www\.\S+|(?:linkedin\.com|github\.com)/\S+", re.IGNORECASE)
NAME_WORD_RE = re.compile(r"^[A-Z][A-Za-z.'\-]*$")


def find_email(text: str) -> str:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else ""


def find_phone(text: str) -> str:
    text = EMAIL_RE.sub(" ", text)               # email ke andar ke numbers phone na ban jayein
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 13:              # "2015 - 2019" (8 digits) reject ho jayega
            return m.group(0).strip()
    return ""


def find_links(text: str) -> list[str]:
    links = []
    for m in LINK_RE.finditer(text):
        link = m.group(0).rstrip(".,;)|")
        if link not in links:
            links.append(link)
    return links


def looks_like_name(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 40:
        return False
    if re.search(r"[\d@|,:/()]", line):
        return False
    heading = re.sub(r"[^a-z& ]", "", line.lower()).strip()
    if heading in ALIAS_TO_SECTION:              # "Work Experience" naam nahi hai
        return False
    words = line.split()
    return 2 <= len(words) <= 4 and all(NAME_WORD_RE.match(w) for w in words)


def find_name(text: str, email: str) -> str:
    lines = [l.strip() for l in text.split("\n")]
    candidates = [l for l in lines if looks_like_name(l)]

    # 1) Email ke naam se match karo: rahul.verma@... -> "Rahul Verma"
    if email:
        local = email.split("@")[0].lower()
        tokens = {t for t in re.split(r"[^a-z]+", local) if len(t) >= 3}
        for c in candidates:
            words = {w.lower().strip(".'-") for w in c.split()}
            if tokens & words:
                return c

    # 2) Nahi mila toh shuru ki lines mein pehla naam jaisa text
    for line in lines[:10]:
        if looks_like_name(line):
            return line
    return ""


def extract_contacts(text: str) -> ContactInfo:
    email = find_email(text)
    return ContactInfo(
        name=find_name(text, email),
        email=email,
        phone=find_phone(text),
        links=find_links(text),
    )