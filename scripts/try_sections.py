import sys
from app.ingestion.extract import extract_text
from app.ingestion.sections import split_sections

sections = split_sections(extract_text(sys.argv[1]))

for name, content in sections.items():
    print(f"\n===== {name.upper()} =====")
    print(content[:200])

print("\nSections found:", list(sections))