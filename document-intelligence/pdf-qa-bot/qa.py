from __future__ import annotations

import os
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    page: int


def chunk_text(pages: list[str], chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split page text into overlapping chunks so each fits an LLM context window."""
    chunks: list[Chunk] = []
    for page_num, page_text in enumerate(pages, start=1):
        cleaned = re.sub(r"\s+", " ", page_text).strip()
        if not cleaned:
            continue
        start = 0
        while start < len(cleaned):
            end = start + chunk_size
            chunks.append(Chunk(text=cleaned[start:end], page=page_num))
            start += chunk_size - overlap
    return chunks


def extract_pages(pdf_path: str) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def retrieve(chunks: list[Chunk], question: str, top_k: int = 3) -> list[Chunk]:
    """TF-IDF retrieval: rank chunks by cosine similarity to the question."""
    if not chunks:
        return []
    corpus = [c.text for c in chunks] + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked_idx = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in ranked_idx if sims[i] > 0]


def answer_question(chunks: list[Chunk], question: str, api_key: str | None = None) -> str:
    """Answer using retrieved chunks. Calls Claude if an API key is available, else
    falls back to returning the most relevant excerpt directly (extractive mode)."""
    relevant = retrieve(chunks, question)
    if not relevant:
        return "No relevant content found in the document for that question."

    context = "\n\n".join(f"[Page {c.page}] {c.text}" for c in relevant)
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Answer the question using only the context below. "
                        "Cite page numbers. If the answer isn't in the context, say so.\n\n"
                        f"Context:\n{context}\n\nQuestion: {question}"
                    ),
                }
            ],
        )
        return response.content[0].text

    best = relevant[0]
    return f"[Extractive mode, no API key set]\n(Page {best.page}) {best.text}"
