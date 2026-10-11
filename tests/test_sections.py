from pathlib import Path

import pytest

from app.ingestion.extract import extract_text
from app.ingestion.sections import heading_name, split_sections

SAMPLE = """John Doe
john@example.com

SKILLS:
Python, SQL

Work Experience
Data Analyst at Acme
- Built dashboards

Education
B.Tech in CS
"""


def test_detects_sections():
    sections = split_sections(SAMPLE)
    assert set(sections) == {"header", "skills", "experience", "education"}


def test_content_goes_to_right_section():
    sections = split_sections(SAMPLE)
    assert "john@example.com" in sections["header"]
    assert "Python" in sections["skills"]
    assert "Python" not in sections["experience"]


def test_heading_with_colon_and_caps():
    assert heading_name("TECHNICAL SKILLS:") == "skills"
    assert heading_name("Internships") == "experience"


def test_long_line_is_not_a_heading():
    line = "Skills in Python and machine learning for building production systems"
    assert heading_name(line) is None


# ---- Ab 5 asli sample resumes par test ----

SAMPLES = Path("data/sample_resumes")

EXPECTED = {
    "resume_1_priya_sharma.pdf": {"header", "summary", "skills", "experience", "projects", "education", "certifications"},
    "resume_2_rahul_verma.pdf": {"header", "summary", "skills", "experience", "projects", "education", "certifications"},
    "resume_3_ananya_iyer.docx": {"header", "summary", "skills", "experience", "education", "projects"},
    "resume_4_vikram_singh.docx": {"header", "summary", "skills", "experience", "education", "projects", "certifications"},
    "resume_5_neha_gupta.txt": {"header", "summary", "skills", "experience", "education", "projects", "achievements"},
}


@pytest.mark.parametrize("filename, expected", EXPECTED.items())
def test_sample_resume_sections(filename, expected):
    path = SAMPLES / filename
    if not path.exists():
        pytest.skip(f"{filename} not found")
    sections = split_sections(extract_text(str(path)))
    assert set(sections) == expected


@pytest.mark.parametrize("filename", EXPECTED)
def test_python_is_in_skills(filename):
    path = SAMPLES / filename
    if not path.exists():
        pytest.skip(f"{filename} not found")
    sections = split_sections(extract_text(str(path)))
    assert "Python" in sections["skills"]