# Semantic Search Engine

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/semantic-search/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/semantic-search/demo.ipynb)

> Search by meaning, not just exact words — what Ctrl+F can't do.

## Business Impact
- **Before:** Users keyword-search the wiki/help center, miss the right doc because the wording differs, and file a ticket anyway.
- **After:** A vector store ranks docs by relevance — "get my money back" finds the refund policy even with no shared words (with dense embeddings).
- **Estimated ROI:** fewer "I couldn't find it" tickets; faster self-serve answers.

## Tech Stack
Python · TF-IDF lexical vectorizer (zero-dependency default) · in-memory cosine vector store · Voyage AI dense embeddings (optional upgrade) · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Semantic and keyword (Ctrl+F) results show side by side. Toggle Voyage dense embeddings when `VOYAGE_API_KEY` is set (`pip install voyageai`).

## How it works
1. **Embed** each doc — default is a TF-IDF vector (term importance via IDF), swappable for dense Voyage vectors.
2. **Store** vectors in memory (a vector DB without the DB).
3. **Rank** by cosine similarity at query time.
4. **Contrast** against a substring keyword baseline so the difference is visible.

**Honest limit:** TF-IDF is still lexical — "keep my files safe" scores 0 against an encryption doc because no tokens overlap. Dense embeddings (Voyage `voyage-3`) close that gap; the vector-store and search code stay identical — only `embed_fn` changes.

## Learning Connection
Built while studying **Embeddings & Vector DBs** (Anthropic).
Applies: dense vs lexical retrieval, cosine ranking, and the embed → store → rank architecture that every RAG system shares.

## Impact Note
- **Who benefits:** support, docs/DevRel, anyone with a searchable knowledge base.
- **Potential risks:** semantic search can surface confidently-wrong "close" matches; show scores and let users see the source. TF-IDF mode won't bridge true synonyms — don't market it as full semantic search without dense embeddings behind it.
