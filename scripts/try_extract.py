import sys
from app.ingestion.extract import extract_text

text = extract_text(sys.argv[1])
print(text)
print(f"\n--- {len(text)} characters extracted ---")