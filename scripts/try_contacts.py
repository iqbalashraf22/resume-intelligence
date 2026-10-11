import sys
from app.ingestion.extract import extract_text
from app.parsing.contacts import extract_contacts
from app.parsing.schema import ParsedResume

text = extract_text(sys.argv[1])
resume = ParsedResume(contact=extract_contacts(text))
print(resume.model_dump_json(indent=2))