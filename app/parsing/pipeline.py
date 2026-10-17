from app.ingestion.extract import extract_text
from app.ingestion.sections import split_sections

from .contacts import extract_contacts
from .llm_parser import parse_sections
from .schema import ParsedResume


def parse_resume(file_path: str, client=None) -> ParsedResume:
    text = extract_text(file_path)
    sections = split_sections(text)
    contact = extract_contacts(text)             # regex: free aur pakka
    body = parse_sections(sections, client)      # LLM: messy hissa
    return ParsedResume(contact=contact, **body.model_dump())