import sys
from pathlib import Path

from app.ingestion.extract import extract_text
from app.ingestion.sections import split_sections
from app.vectorstore import add_resume, get_collection, search

SAMPLES = Path("data/sample_resumes")
DEFAULT_QUERIES = [
    "Python and machine learning experience",
    "containerization and deployment",
    "experience with natural language processing",
]

collection = get_collection()
for f in sorted(SAMPLES.glob("*")):
    if f.suffix.lower() in {".pdf", ".docx", ".txt"}:
        n = add_resume(f.stem, split_sections(extract_text(str(f))), collection)
        print(f"Added {f.stem}: {n} chunks")

for query in sys.argv[1:] or DEFAULT_QUERIES:
    print(f"\n=== {query} ===")
    for hit in search(query, k=3, collection=collection):
        snippet = hit["text"].replace("\n", " ")[:90]
        print(f"{hit['score']:.2f}  {hit['resume_id']:<28} [{hit['section']}]  {snippet}")