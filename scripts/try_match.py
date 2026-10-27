import os
import sys
from pathlib import Path

from app.ingestion.extract import extract_text
from app.ingestion.sections import split_sections
from app.matching import match_resume
from app.parsing.jd_parser import JobDescription, parse_job_description, parse_job_description_rules

RESUMES = Path("data/sample_resumes")
JDS = Path("data/sample_jds")
CACHE = Path(".cache/jd_parsed")      # JD ek baar parse hoti hai, phir yahan se aati hai
EXTS = {".pdf", ".docx", ".txt"}
USE_LLM = os.getenv("JD_USE_LLM", "1") != "0"   # JD_USE_LLM=0 -> LLM ke bina, sirf bullets se


def files(folder: Path):
    return sorted(f for f in folder.glob("*") if f.suffix.lower() in EXTS)


def load_sections(path: Path) -> dict:
    return split_sections(extract_text(str(path)))


def load_jd(path: Path) -> JobDescription:
    cached = CACHE / f"{path.stem}{'' if USE_LLM else '.rules'}.json"
    if cached.exists() and cached.stat().st_mtime >= path.stat().st_mtime:
        return JobDescription.model_validate_json(cached.read_text(encoding="utf-8"))
    text = extract_text(str(path))
    if USE_LLM:
        print(f"Parsing {path.name} with the LLM (pehli baar, thoda time lagega)...")
        jd = parse_job_description(text)
    else:
        jd = parse_job_description_rules(text)
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(jd.model_dump_json(indent=2), encoding="utf-8")
    return jd


if len(sys.argv) >= 3:
    # Detail: python -m scripts.try_match <resume file> <jd file> [threshold]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None
    result = match_resume(load_sections(Path(sys.argv[1])), load_jd(Path(sys.argv[2])), threshold=threshold)
    print(f"\nMatch: {result.percent}%\n")
    for d in result.details:
        print(f"{'YES' if d.matched else 'NO ':<3} {d.kind:<9} {d.score:.2f} {'exact' if d.exact else '     '}  "
              f"{d.requirement:<48} [{d.section}]")
    print("\nMissing (required): ", result.missing_required or "none")
    print("Missing (preferred):", result.missing_preferred or "none")
else:
    # Table: har resume x har JD
    jds = [(f, load_jd(f)) for f in files(JDS)]
    print(f"\n{'resume':<28}" + "".join(f"{f.stem[:14]:>16}" for f, _ in jds))
    for r in files(RESUMES):
        sections = load_sections(r)
        row = "".join(f"{match_resume(sections, jd).percent:>15.1f}%" for _, jd in jds)
        print(f"{r.stem:<28}{row}")
