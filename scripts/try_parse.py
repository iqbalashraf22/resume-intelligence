import sys
from app.parsing.pipeline import parse_resume

resume = parse_resume(sys.argv[1])
print(resume.model_dump_json(indent=2))