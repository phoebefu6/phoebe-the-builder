# API Pagination Extractor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/api-paginator/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/api-paginator/demo.ipynb)

> Paginated API pulls are hand-coded each time — one generic extractor speaks offset, page, cursor, and link-follow, with 429 retry/backoff built in.

## Business Impact
- **Before:** Every new SaaS integration means another hand-rolled while-loop; each one handles (or forgets) rate limits, stop conditions, and runaway pages differently.
- **After:** `paginate(fetch, strategy)` — the next paginated API is a one-line strategy change, with completeness provable from the built-in stats.
- **Estimated ROI:** Hours per integration, plus no more silent partial extracts from mishandled stop conditions.

## Tech Stack
Python 3.10+ (pure stdlib core — inject any HTTP client), Streamlit, pandas, matplotlib. Mock API with injected 429s makes the whole thing testable offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **One interface** — `paginate(fetch_page, strategy, ...)` where `fetch_page(params) -> (json, status)` wraps requests/httpx/anything.
2. **Four dialects** — `offset` (`?offset&limit`), `page` (`?page&per_page`), `cursor` (follow `next_cursor` until null), `link` (follow `next_url`, Link-header style).
3. **Rate limits** — 429 responses honor `Retry-After`, back off exponentially, capped per page; `max_pages` guards against runaway loops.
4. **Accountable** — stats track requests, retries, 429s, and items per page; the notebook asserts all four strategies return the identical, complete, duplicate-free record set.

Demo: 950 records, a 429 injected every 7th call — all four strategies: 950/950, ordered, no dupes, 11 requests (10 pages + 1 retry).

## Learning Connection
Built while studying API extraction patterns (Month 7: Data Engineering Pro).
Applies: strategy-pattern design, backoff/retry semantics, and provable extraction completeness.

## Impact Note
- **Who benefits:** Data engineers wiring SaaS sources without an off-the-shelf connector; anyone who has shipped a partial extract without knowing.
- **Potential risks:** APIs that mutate between pages can still skip/duplicate records (a snapshot or cursor-stability guarantee is the API's job); respect each vendor's rate-limit terms.
