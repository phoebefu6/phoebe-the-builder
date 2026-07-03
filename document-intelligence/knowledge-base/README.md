# Knowledge Base Builder

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/knowledge-base/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/knowledge-base/demo.ipynb)

> Turn scattered docs into a searchable, cited knowledge base — before the expert leaves.

**Capstone of Month 4 — Document Intelligence.** Composes chunking, retrieval, grounded answers, and persistence into one RAG system.

## Business Impact
- **Before:** Institutional knowledge lives in one person's head and walks out when they leave.
- **After:** Ingest docs/runbooks/wiki pages once; anyone asks a question and gets a **cited** answer — and the whole KB serializes to a file that outlives its author.
- **Estimated ROI:** faster onboarding, fewer "ask the one person who knows" bottlenecks, resilient to turnover.

## Tech Stack
Python · full RAG pipeline (chunk → TF-IDF embed → cosine retrieve → cited answer) · JSON persistence · Claude API (optional synthesized answers) · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Ingest new docs live, ask questions, and see the cited passages behind every answer. Set an `ANTHROPIC_API_KEY` for synthesized answers.

## How it works
1. **Chunk** each doc into passages, tagged with the source doc name.
2. **Embed** to TF-IDF vectors and index (a vector store without the DB).
3. **Retrieve** the top-k passages by cosine similarity.
4. **Answer** with citations — extractive by default (best cited passage); Claude mode stitches multiple passages into one fluent, still-cited answer.
5. **Persist** to JSON — knowledge captured once survives every departure.

## Learning Connection
Built while studying **RAG & Vector DBs** (Anthropic) — the capstone that ties Month 4 together (chunking from Day 34, retrieval from Day 35, grounded answers from Day 31).
Applies: the embed → store → retrieve → ground → cite → persist pipeline every production RAG system shares.

## Impact Note
- **Who benefits:** engineering, ops, support, any team with tribal knowledge to capture.
- **Potential risks:** answers are only as good and current as the ingested docs — stale docs give stale answers, and TF-IDF retrieval can miss synonym-only matches (swap in dense embeddings for that). Always show citations so readers can verify, and keep the KB fresh.
