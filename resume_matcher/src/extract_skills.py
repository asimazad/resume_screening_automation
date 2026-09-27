"""
Skill + entity extraction.

Important design choice (and why): off-the-shelf spaCy NER has NO "SKILL"
entity type, so asking it to "extract skills" directly will silently return
nothing useful. Instead we:
  1. Use a skills taxonomy (skills_taxonomy.txt) for phrase matching -> skills.
  2. Use spaCy NER only for what it's actually trained to find: ORG, DATE, GPE.
This split matters more for your final report than the code itself.
"""

import re
import spacy
from pathlib import Path

_TAXONOMY_PATH = Path(__file__).parent.parent / "resources" / "skills_taxonomy.txt"


def load_taxonomy(path: Path = _TAXONOMY_PATH) -> list[str]:
    with open(path, encoding="utf-8") as f:
        skills = [line.strip().lower() for line in f if line.strip()]
    # match longer phrases first so "machine learning" isn't shadowed by "learning"
    return sorted(skills, key=len, reverse=True)


_NLP = spacy.load("en_core_web_sm")
_TAXONOMY = load_taxonomy()


def extract_skills(text: str, taxonomy: list[str] = _TAXONOMY) -> list[str]:
    """Dictionary/phrase matching against the skills taxonomy.
    Word-boundary regex avoids 'r' matching inside 'react', etc."""
    text_lower = text.lower()
    found = []
    for skill in taxonomy:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def extract_entities(text: str) -> dict:
    """spaCy NER for what it's good at: organizations, dates, locations.
    NOT used for skills -- see module docstring."""
    doc = _NLP(text)
    ents = {"ORG": [], "DATE": [], "GPE": []}
    for ent in doc.ents:
        if ent.label_ in ents:
            ents[ent.label_].append(ent.text)
    return {k: sorted(set(v)) for k, v in ents.items()}


def extract_profile(text: str) -> dict:
    """Combined profile: skills (dictionary match) + entities (spaCy NER)."""
    return {
        "skills": extract_skills(text),
        "entities": extract_entities(text),
    }


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(Path(__file__).parent.parent / "data" / "sample_resumes.csv")
    for _, row in df.iterrows():
        profile = extract_profile(row["resume_text"])
        print(f"{row['resume_id']} ({row['category']}): {profile['skills']}")
