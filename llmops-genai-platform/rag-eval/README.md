# RAG Evaluation Harness

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/llmops-genai-platform/rag-eval/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=llmops-genai-platform/rag-eval/demo.ipynb)

> You changed the chunking or swapped the embedding model. Did RAG get **better** or **worse**? "It feels better" is not an answer - this harness gives you numbers.

## Business Impact
- **Before:** Every RAG tweak (chunk size, embedding model, top-k, reranker) is a leap of faith. A "fix" quietly makes retrieval worse and nobody notices until users complain.
- **After:** A gold eval set + ranking-aware metrics turn every change into a measurable before/after. You ship the change only when the delta is green.
- **Estimated ROI:** Kills the guess-and-check loop - hours per iteration saved, and far fewer silent RAG regressions reaching production.

## Tech Stack
Python (stdlib `math` + `re` + `collections` for a TF-IDF cosine retriever) · pandas · matplotlib · Streamlit · Docker

No API keys, no vector DB, no network. The lexical retriever is a stand-in you swap for your real one - the metrics don't care how retrieval works, only that it returns a ranked list of `doc_id`s.

## What it does
- **Ranking-aware retrieval metrics** - `hit@k`, `recall@k`, `precision@k`, `MRR`, `nDCG@k` over a gold eval set
- **Answer-faithfulness proxy** - checks the retrieved context actually contains the gold answer string (a cheap, deterministic stand-in for LLM-graded faithfulness, no API key)
- **Silent-miss surfacing** - a query with zero lexical overlap scores 0 everywhere instead of erroring; the harness exposes retrieval misses, it doesn't hide them
- **A/B diff** - run a baseline vs a candidate config and read the per-metric delta (🟢 up / 🔴 down)
- **precision/recall vs k** chart - see exactly where widening `k` stops helping

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs (per-question table, aggregate metrics, precision/recall-vs-k chart, A/B diff), or click the Colab/Binder badges above to run it live.

For the Streamlit app (Corpus · Eval set · Run · A/B tabs):
```bash
pip install -r requirements.txt
streamlit run app.py
```

Or run the harness headless:
```bash
python harness.py
```

## How to use it on your own RAG
1. Replace `SAMPLE_DOCS` with your corpus (`doc_id -> text`).
2. Replace `SAMPLE_CASES` with your gold set - each question and the `doc_id`(s) a good retriever should return.
3. Swap `LexicalRetriever` for anything exposing `.retrieve(query, k) -> [doc_id, ...]` (your vector store, hybrid retriever, reranker chain).
4. Call `evaluate(cases, your_retriever.retrieve, docs, k=...)` and `compare(baseline, candidate)`.

## Learning Connection
Built while studying **RAG evaluation & LLMOps** (retrieval metrics, offline eval harnesses).
Applies: ranking metrics (recall/precision/MRR/nDCG), eval-set design, regression-testing a retrieval pipeline.

## Impact Note
- **Who benefits:** anyone shipping RAG who needs to justify a retrieval change with evidence instead of vibes.
- **Potential risks:** metrics are only as good as the gold eval set - a small or biased eval set gives false confidence. The lexical retriever is a demo stand-in, not a production retriever. Treat `answer_hit` as a smoke test, not a real faithfulness grade.
