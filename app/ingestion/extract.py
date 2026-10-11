import re
from pathlib import Path

import pymupdf  # PyMuPDF
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

SUPPORTED = {".pdf", ".docx", ".txt"}


def read_pdf(path: Path) -> str:
    with pymupdf.open(path) as doc:
        # sort=True hata diya: usse two-column resumes ki lines mix ho jati thi
        pages = [page.get_text("text") for page in doc]
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    doc = Document(path)
    parts = []
    for child in doc.element.body.iterchildren():   # document ke order mein
        if child.tag.endswith("}p"):                # paragraph
            parts.append(Paragraph(child, doc).text)
        elif child.tag.endswith("}tbl"):            # table
            table = Table(child, doc)
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
    return "\n".join(parts)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[•●▪◦►]", "-", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")

    readers = {".pdf": read_pdf, ".docx": read_docx, ".txt": read_txt}
    return clean_text(readers[ext](path))