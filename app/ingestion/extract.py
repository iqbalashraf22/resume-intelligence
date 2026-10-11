import re
from pathlib import Path

import pymupdf  # PyMuPDF
from docx import Document

SUPPORTED = {".pdf", ".docx", ".txt"}


def read_pdf(path: Path) -> str:
    with pymupdf.open(path) as doc:
        pages = [page.get_text("text", sort=True) for page in doc]
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:                  # kai resumes tables mein skills rakhte hain
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")           # invisible spaces hatao
    text = re.sub(r"[•●▪◦►]", "-", text)       # sab bullets ko "-" banao
    text = re.sub(r"[ \t]+", " ", text)        # kai spaces ko ek karo
    text = re.sub(r" *\n *", "\n", text)       # line ke aas-paas ke spaces hatao
    text = re.sub(r"\n{3,}", "\n\n", text)     # zyada khali lines kam karo
    return text.strip()


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")

    readers = {".pdf": read_pdf, ".docx": read_docx, ".txt": read_txt}
    return clean_text(readers[ext](path))