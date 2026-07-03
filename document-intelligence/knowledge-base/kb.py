from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """One retrievable passage, tagged with the source doc it came from (for citations)."""

    doc: str
    text: str
    vector: dict = field(default_factory=dict)


@dataclass
class Answer:
    """An answer grounded in KB chunks, with the sources it was drawn from."""

    text: str
    sources: list[str] = field(default_factory=list)
    passages: list[Chunk] = field(default_factory=list)


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def chunk_document(doc_name: str, text: str, max_chars: int = 500) -> list[tuple[str, str]]:
    """Split a doc into (doc_name, passage) pairs on blank lines, length-capped."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            out.append((doc_name, block))
        else:
            buf = ""
            for s in re.split(r"(?<=[.!?])\s+", block):
                if len(buf) + len(s) > max_chars and buf:
                    out.append((doc_name, buf.strip()))
                    buf = s
                else:
                    buf = f"{buf} {s}".strip()
            if buf:
                out.append((doc_name, buf.strip()))
    return out


class KnowledgeBase:
    """A tiny RAG knowledge base: ingest docs, embed to TF-IDF vectors, retrieve + answer.

    Persistable to JSON so institutional knowledge survives after the author leaves."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.idf: dict[str, float] = {}

    # ------------------------- ingest / index -------------------------

    def add_document(self, doc_name: str, text: str) -> int:
        pairs = chunk_document(doc_name, text)
        for doc, passage in pairs:
            self.chunks.append(Chunk(doc=doc, text=passage))
        self._reindex()
        return len(pairs)

    def _reindex(self) -> None:
        n = len(self.chunks)
        df: Counter = Counter()
        for ch in self.chunks:
            for term in set(_tokens(ch.text)):
                df[term] += 1
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        for ch in self.chunks:
            ch.vector = self._vectorize(ch.text)

    def _vectorize(self, text: str) -> dict[str, float]:
        counts = Counter(t for t in _tokens(text) if t in self.idf)
        if not counts:
            return {}
        total = sum(counts.values())
        return {t: (c / total) * self.idf[t] for t, c in counts.items()}

    # ------------------------- retrieve / answer -------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[float, Chunk]]:
        qv = self._vectorize(query)
        scored = [(_cosine(qv, ch.vector), ch) for ch in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(s, ch) for s, ch in scored[:top_k] if s > 0]

    def ask(self, query: str, api_key: Optional[str] = None, top_k: int = 3) -> Answer:
        """Answer a question from the KB. Claude synthesizes if a key is present, else an
        extractive answer stitched from the top passages. Always returns citations."""
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return Answer(text="No relevant information found in the knowledge base.")

        passages = [ch for _, ch in hits]
        sources = list(dict.fromkeys(ch.doc for ch in passages))

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # extractive fallback: the single best passage, cited
            best = passages[0]
            return Answer(
                text=f"{best.text}",
                sources=sources,
                passages=passages,
            )
        return self._llm_answer(query, passages, sources, api_key)

    def _llm_answer(self, query: str, passages: list[Chunk], sources: list[str], api_key: str) -> Answer:
        import anthropic

        context = "\n\n".join(f"[{p.doc}] {p.text}" for p in passages)
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=600,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Answer the question using ONLY the knowledge-base passages below. "
                        "If the answer isn't in them, say so. Cite the source doc names you used. "
                        f"Question: {query}\n\nPassages:\n{context}"
                    ),
                }
            ],
        )
        return Answer(text=response.content[0].text.strip(), sources=sources, passages=passages)

    # ------------------------- persistence -------------------------

    def save(self, path: str) -> None:
        data = {"chunks": [{"doc": c.doc, "text": c.text} for c in self.chunks]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        kb = cls()
        with open(path) as f:
            data = json.load(f)
        kb.chunks = [Chunk(doc=c["doc"], text=c["text"]) for c in data["chunks"]]
        kb._reindex()
        return kb


def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


SAMPLE_DOCS = {
    "onboarding.md": (
        "New engineers should request access to the staging VPN on day one via the IT portal. "
        "Access is approved by the team lead and usually granted within 4 hours.\n\n"
        "The dev environment is spun up with `make dev`. If it fails, delete the .cache folder "
        "and retry - this is the most common first-day issue."
    ),
    "deploys.md": (
        "We deploy to production every Tuesday and Thursday at 10am. Deploys are frozen the "
        "last week of each quarter.\n\n"
        "To roll back a bad deploy, run `deploy rollback <version>`. The previous three versions "
        "are always kept warm. Notify #eng-oncall before any rollback."
    ),
    "incidents.md": (
        "Sev-1 incidents page the on-call engineer immediately via PagerDuty. The on-call must "
        "acknowledge within 5 minutes.\n\n"
        "Every Sev-1 requires a post-mortem within 48 hours, written in the incidents Notion and "
        "reviewed in the Friday reliability sync. Blameless format only."
    ),
}
