from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class FetchStats:
    requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    items: int = 0
    pages: List[int] = field(default_factory=list)  # items per successful page


class RetryError(RuntimeError):
    pass


def paginate(fetch_page: Callable[[Dict], Tuple[Dict, int]], strategy: str,
             page_size: int = 100, max_pages: int = 1000, max_retries: int = 4,
             items_key: str = "items", cursor_key: str = "next_cursor",
             sleep_fn: Callable[[float], None] = lambda s: None) -> Tuple[List[Dict], FetchStats]:
    """Generic paginated extractor.

    fetch_page(params) -> (json_dict, status_code). Strategies:
      - offset:  params {offset, limit}; stop when a page returns < limit items
      - page:    params {page, per_page}; stop when a page returns < per_page items
      - cursor:  params {cursor, limit}; follow json[cursor_key] until null
      - link:    params {url}; follow json['next_url'] until null (Link-header style)

    429 responses honor Retry-After via sleep_fn and retry with exponential backoff,
    up to max_retries per page. max_pages caps runaway loops.
    """
    if strategy not in ("offset", "page", "cursor", "link"):
        raise ValueError(f"unknown strategy: {strategy}")

    stats = FetchStats()
    items: List[Dict] = []
    offset, page_num = 0, 1
    cursor: Optional[str] = None
    url: Optional[str] = "/v1/records"

    def call(params: Dict) -> Dict:
        delay = 0.5
        for attempt in range(max_retries + 1):
            stats.requests += 1
            body, status = fetch_page(params)
            if status == 200:
                return body
            if status == 429:
                stats.rate_limited += 1
                stats.retries += 1
                sleep_fn(float(body.get("retry_after", delay)))
                delay *= 2
                continue
            raise RetryError(f"HTTP {status} on {params}")
        raise RetryError(f"gave up after {max_retries} retries on {params}")

    for _ in range(max_pages):
        if strategy == "offset":
            body = call({"offset": offset, "limit": page_size})
        elif strategy == "page":
            body = call({"page": page_num, "per_page": page_size})
        elif strategy == "cursor":
            body = call({"cursor": cursor, "limit": page_size})
        else:
            if url is None:
                break
            body = call({"url": url})

        batch = body.get(items_key, [])
        items.extend(batch)
        stats.items += len(batch)
        stats.pages.append(len(batch))

        if strategy == "offset":
            if len(batch) < page_size:
                break
            offset += page_size
        elif strategy == "page":
            if len(batch) < page_size:
                break
            page_num += 1
        elif strategy == "cursor":
            cursor = body.get(cursor_key)
            if not cursor:
                break
        else:
            url = body.get("next_url")
            if not url:
                break
    return items, stats


class MockAPI:
    """In-memory API serving the same dataset via all four pagination styles,
    with configurable flaky 429s — so the extractor is testable offline."""

    def __init__(self, n_records: int = 950, flaky_429_every: int = 0, seed: int = 5) -> None:
        import random
        self.records = [{"id": i, "name": f"rec-{i:04d}", "value": round(i * 1.7, 1)}
                        for i in range(1, n_records + 1)]
        self._rng = random.Random(seed)
        self.flaky_429_every = flaky_429_every
        self._calls = 0

    def _maybe_429(self) -> bool:
        self._calls += 1
        return self.flaky_429_every > 0 and self._calls % self.flaky_429_every == 0

    def fetch(self, params: Dict) -> Tuple[Dict, int]:
        if self._maybe_429():
            return {"error": "rate limited", "retry_after": 0.01}, 429

        if "offset" in params:
            lo, n = params["offset"], params["limit"]
            return {"items": self.records[lo:lo + n]}, 200

        if "page" in params:
            n = params["per_page"]
            lo = (params["page"] - 1) * n
            return {"items": self.records[lo:lo + n]}, 200

        if "cursor" in params:
            n = params["limit"]
            lo = int(params["cursor"] or 0)
            batch = self.records[lo:lo + n]
            nxt = str(lo + n) if lo + n < len(self.records) else None
            return {"items": batch, "next_cursor": nxt}, 200

        if "url" in params:
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(params["url"]).query)
            lo = int(q.get("after", ["0"])[0])
            n = int(q.get("limit", ["100"])[0])
            batch = self.records[lo:lo + n]
            nxt = f"/v1/records?after={lo + n}&limit={n}" if lo + n < len(self.records) else None
            return {"items": batch, "next_url": nxt}, 200

        return {"error": "bad request"}, 400
