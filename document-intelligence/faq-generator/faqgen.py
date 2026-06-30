from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FAQItem:
    """A single generated question/answer pair grounded in the source doc."""

    question: str
    answer: str
    source_chunk: str = ""  # the doc passage the answer is grounded in


@dataclass
class FAQSet:
    items: list[FAQItem] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Frequently Asked Questions", ""]
        for it in self.items:
            lines.append(f"### {it.question}")
            lines.append(it.answer)
            lines.append("")
        return "\n".join(lines).strip()


# ----------------------------- chunking -----------------------------

def chunk_document(text: str, max_chars: int = 600) -> list[str]:
    """Split a doc into retrievable chunks on blank lines / headings, capped at max_chars.

    Keeps a heading attached to the paragraph that follows it so each chunk is self-contained."""
    blocks = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append(block)
        else:
            # break long block on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", block)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) > max_chars and buf:
                    chunks.append(buf.strip())
                    buf = s
                else:
                    buf = f"{buf} {s}".strip()
            if buf:
                chunks.append(buf.strip())
    return chunks


# ------------------------- lightweight retrieval -------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def retrieve(query: str, chunks: list[str], top_k: int = 1) -> list[str]:
    """Rank chunks by token-overlap (a tiny TF-style scorer) and return the top_k.

    Stands in for a vector DB: zero dependencies, good enough to ground a short answer."""
    q = set(_tokens(query))
    if not q:
        return chunks[:top_k]
    scored = []
    for ch in chunks:
        ch_tokens = _tokens(ch)
        if not ch_tokens:
            continue
        overlap = sum(1 for t in ch_tokens if t in q)
        score = overlap / (len(ch_tokens) ** 0.5)  # length-normalized
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for score, ch in scored[:top_k] if score > 0] or chunks[:top_k]


# --------------------------- heuristic FAQ ---------------------------

_HEADING = re.compile(r"^\s*(#{1,4}\s+.+|[A-Z][A-Za-z0-9 ,/&'-]{2,60}:)\s*$", re.M)


def _clean_heading(h: str) -> str:
    return re.sub(r"^#+\s*", "", h.strip()).rstrip(":").strip()


def _strip_leading_heading(passage: str) -> str:
    """Drop a leading markdown/colon heading line so the answer is just the body text."""
    lines = passage.strip().splitlines()
    if lines and (_HEADING.match(lines[0] + "\n") or lines[0].lstrip().startswith("#")):
        lines = lines[1:]
    return "\n".join(lines).strip()


def heuristic_faq(text: str, max_items: int = 8) -> FAQSet:
    """Build an FAQ with no LLM: turn each section heading into a question, answer with the
    retrieved passage. Deterministic, free, good enough to seed a support FAQ."""
    chunks = chunk_document(text)
    headings = [_clean_heading(m.group(0)) for m in _HEADING.finditer(text)]

    items: list[FAQItem] = []
    seen: set[str] = set()
    for h in headings:
        if not h or h.lower() in seen:
            continue
        seen.add(h.lower())
        question = h if h.endswith("?") else f"What do I need to know about {h}?"
        passage = retrieve(h, chunks, top_k=1)[0]
        answer = _strip_leading_heading(passage)
        if len(answer) < 15:  # title-only chunk, nothing useful to answer with
            continue
        items.append(FAQItem(question=question, answer=answer, source_chunk=passage))
        if len(items) >= max_items:
            break

    # Fallback: no headings found -> use leading sentence of each chunk
    if not items:
        for ch in chunks[:max_items]:
            first = re.split(r"(?<=[.!?])\s+", ch)[0]
            items.append(FAQItem(
                question=f"What does this cover: \"{first[:60]}...\"?",
                answer=ch,
                source_chunk=ch,
            ))
    return FAQSet(items=items)


# ----------------------------- LLM FAQ -----------------------------

def llm_faq(text: str, api_key: str, max_items: int = 8) -> FAQSet:
    """Use Claude to generate natural support questions, each answered ONLY from the doc."""
    import anthropic

    chunks = chunk_document(text)
    context = "\n\n---\n\n".join(f"[chunk {i}] {c}" for i, c in enumerate(chunks))
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a support lead. From the documentation below, write up to {max_items} "
                    "FAQ entries customers would actually ask. Answer each ONLY using facts in the "
                    "doc - never invent details. Quote the most relevant chunk text in source_chunk. "
                    "Respond with ONLY valid JSON, no markdown fences:\n"
                    '{"items": [{"question": "...", "answer": "...", "source_chunk": "..."}]}\n\n'
                    f"Documentation:\n{context}"
                ),
            }
        ],
    )
    data = json.loads(response.content[0].text.strip())
    return FAQSet(items=[
        FAQItem(
            question=it.get("question", ""),
            answer=it.get("answer", ""),
            source_chunk=it.get("source_chunk", ""),
        )
        for it in data.get("items", [])
    ])


def generate_faq(text: str, api_key: Optional[str] = None, max_items: int = 8) -> FAQSet:
    """Generate an FAQ from documentation.

    Uses Claude if an API key is available, else deterministic heuristics."""
    if not text.strip():
        return FAQSet()
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return heuristic_faq(text, max_items=max_items)
    return llm_faq(text, api_key, max_items=max_items)


SAMPLE_DOC = """\
# Acme Cloud - Getting Started

## Account Setup
To create an account, visit app.acme.cloud and sign up with your work email. New accounts
get a 14-day free trial of the Pro plan. No credit card is required to start the trial.

## Billing
We bill monthly on the date you upgraded. You can switch between Starter, Pro, and Team
plans at any time; changes are prorated. To cancel, go to Settings > Billing > Cancel.
Cancellations take effect at the end of the current billing period.

## Data Limits
The Starter plan includes 10 GB of storage and 100k API calls per month. Pro raises this
to 100 GB and 1M API calls. If you exceed your limit, requests are throttled, not blocked,
and we email the account owner.

## Security
All data is encrypted at rest with AES-256 and in transit with TLS 1.3. We are SOC 2 Type
II certified. Two-factor authentication is available to all plans and required on Team.

## Support
Starter plans get email support with a 48-hour response target. Pro and Team get priority
support with a 4-hour target and access to live chat during business hours.
"""
