import sys

from app.ingestion.extract import extract_text
from app.parsing.jd_parser import parse_job_description

jd = parse_job_description(extract_text(sys.argv[1]))
print(jd.model_dump_json(indent=2))