from pathlib import Path

import pytest

from app.ingestion.extract import extract_text
from app.parsing.contacts import (
    extract_contacts, find_email, find_links, find_name, find_phone, looks_like_name,
)
from app.parsing.schema import ContactInfo, ParsedResume


def test_find_email():
    assert find_email("Contact: Priya.Sharma@Example.com | Pune") == "Priya.Sharma@Example.com"
    assert find_email("no email here") == ""


def test_find_phone_formats():
    assert find_phone("Call +91 98765 43210 now") == "+91 98765 43210"
    assert find_phone("(415) 555-0132") == "(415) 555-0132"


def test_date_range_is_not_a_phone():
    assert find_phone("B.Tech | 2018 - 2022\nJun 2022 - Present") == ""


def test_digits_inside_email_are_not_a_phone():
    assert find_phone("ps1234567890@example.com") == ""


def test_find_links_and_dedupe():
    text = "github.com/abc | linkedin.com/in/abc, https://abc.dev. github.com/abc"
    assert find_links(text) == ["github.com/abc", "linkedin.com/in/abc", "https://abc.dev"]


def test_looks_like_name():
    assert looks_like_name("Priya Sharma")
    assert looks_like_name("PRIYA SHARMA")
    assert not looks_like_name("Work Experience")       # section heading
    assert not looks_like_name("john@example.com")
    assert not looks_like_name("Python, SQL")
    assert not looks_like_name("CONTACT")                # ek hi word


def test_name_found_even_when_not_first_line():
    text = "CONTACT\nrahul.verma@example.com\n\nRahul Verma\nBackend Engineer"
    assert find_name(text, "rahul.verma@example.com") == "Rahul Verma"


def test_schema_defaults_and_email_validator():
    assert ContactInfo(email="  A@B.COM ").email == "a@b.com"
    r = ParsedResume()
    assert r.skills == [] and r.contact.name == ""


SAMPLES = Path("data/sample_resumes")

EXPECTED = {
    "resume_1_priya_sharma.pdf": ("Priya Sharma", "priya.sharma@example.com", "+91 98765 43210",
                                  ["linkedin.com/in/priya-sharma-demo", "github.com/priya-demo"]),
    "resume_2_rahul_verma.pdf": ("Rahul Verma", "rahul.verma@example.com", "+91 91234 56780",
                                 ["github.com/rahul-demo"]),
    "resume_3_ananya_iyer.docx": ("Ananya Iyer", "ananya.iyer@example.com", "+91 99887 76655",
                                  ["linkedin.com/in/ananya-iyer-demo"]),
    "resume_4_vikram_singh.docx": ("Vikram Singh", "vikram.singh@example.com", "+91 90000 11122",
                                   ["github.com/vikram-demo"]),
    "resume_5_neha_gupta.txt": ("Neha Gupta", "neha.gupta@example.com", "+91 98111 22334",
                                ["github.com/neha-demo"]),
}


@pytest.mark.parametrize("filename, expected", EXPECTED.items())
def test_sample_resume_contacts(filename, expected):
    path = SAMPLES / filename
    if not path.exists():
        pytest.skip(f"{filename} not found")
    c = extract_contacts(extract_text(str(path)))
    assert (c.name, c.email, c.phone, c.links) == expected