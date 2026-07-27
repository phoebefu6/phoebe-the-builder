from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Entity:
    text: str
    label: str        # PERSON | ORG | MONEY | DATE | EMAIL | PERCENT
    start: int
    end: int


_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|k|m|bn))?\b", re.I)
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s?%")
_DATE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.I,
)
_ORG_SUFFIX = re.compile(
    r"\b([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*)*)\s+"
    r"(Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|Co\.?|Group|University|Institute|Labs?|Technologies|Systems)\b"
)
# capitalized 2-3 word sequences that look like names (crude PERSON heuristic)
_CAP_SEQ = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")

# common non-person capitalized words to exclude from PERSON matches
_STOP_CAPS = {
    "The", "This", "That", "These", "Those", "We", "It", "In", "On", "At", "For",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}


def _overlaps(a_start: int, a_end: int, spans: list) -> bool:
    return any(a_start < e and s < a_end for s, e in spans)


def heuristic_ner(text: str) -> list[Entity]:
    """Extract entities with regex + capitalization rules. No model, no dependencies.

    Precise on structured entities (email, money, %, dates, Inc/LLC orgs); best-effort on PERSON.
    """
    ents: list[Entity] = []
    taken: list = []

    def add(pattern: re.Pattern, label: str, group: int = 0):
        for m in pattern.finditer(text):
            s, e = m.start(group), m.end(group)
            if not _overlaps(s, e, taken):
                ents.append(Entity(m.group(group).strip(), label, s, e))
                taken.append((s, e))

    add(_EMAIL, "EMAIL")
    add(_MONEY, "MONEY")
    add(_PERCENT, "PERCENT")
    add(_DATE, "DATE")
    # ORG (whole match including suffix)
    for m in _ORG_SUFFIX.finditer(text):
        s, e = m.start(), m.end()
        if not _overlaps(s, e, taken):
            ents.append(Entity(m.group(0).strip(), "ORG", s, e))
            taken.append((s, e))
    # PERSON: capitalized multi-word sequences not already taken and not stopword-led
    for m in _CAP_SEQ.finditer(text):
        s, e = m.start(1), m.end(1)
        first = m.group(1).split()[0]
        if first in _STOP_CAPS:
            continue
        if not _overlaps(s, e, taken):
            ents.append(Entity(m.group(1), "PERSON", s, e))
            taken.append((s, e))
    return sorted(ents, key=lambda x: x.start)


def llm_ner(text: str, api_key: str) -> list[Entity]:
    """Use Claude for higher-recall NER (locations, roles, paraphrased orgs)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": (
            "Extract named entities. Return ONLY JSON: "
            '{"entities": [{"text": "...", "label": "PERSON|ORG|GPE|MONEY|DATE|EMAIL|PERCENT"}]}\n\n' + text
        )}],
    )
    data = json.loads(resp.content[0].text.strip())
    ents = []
    for e in data.get("entities", []):
        t = e.get("text", "")
        idx = text.find(t)
        ents.append(Entity(t, e.get("label", "MISC"), idx, idx + len(t)))
    return ents


def extract_entities(text: str, api_key: Optional[str] = None) -> list[Entity]:
    """Extract entities. Uses Claude if a key is available, else heuristics."""
    if not text.strip():
        return []
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    return llm_ner(text, api_key) if api_key else heuristic_ner(text)


def group_by_label(ents: list[Entity]) -> dict:
    out: dict = {}
    for e in ents:
        out.setdefault(e.label, [])
        if e.text not in out[e.label]:
            out[e.label].append(e.text)
    return out


SAMPLE_TEXT = (
    "On March 15, 2026, Acme Analytics Inc. announced that Sarah Chen would join as CTO. "
    "The company raised $12 million in a round led by Beta Ventures LLC, a 40% increase over "
    "its previous valuation. Questions can be sent to press@acme.com. "
    "Michael Torres, formerly of Globex Corporation, will lead the new AI Labs division."
)
