import sys

from app.feedback import build_feedback
from app.ingestion.extract import extract_text
from app.ingestion.sections import split_sections
from app.matching import match_resume
from app.parsing.jd_parser import parse_job_description_rules

sections = split_sections(extract_text(sys.argv[1]))
jd = parse_job_description_rules(extract_text(sys.argv[2]))
match = match_resume(sections, jd)

print(f"Match: {match.percent}%\n")
for item in build_feedback(sections, match):
    print(f"[{item.priority.upper()}] {item.title}")
    print(f"  {item.detail}")
    for ev in item.evidence:
        print(f"  evidence ({ev.score:.2f}) [{ev.section}] {ev.text[:90].replace(chr(10), ' ')}")
    print()
