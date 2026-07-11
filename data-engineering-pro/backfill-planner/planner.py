from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

GRANULARITIES = ("daily", "weekly", "monthly")
STATUSES = ("pending", "running", "success", "failed")


@dataclass
class Chunk:
    chunk_id: str
    start: str  # inclusive ISO date
    end: str    # exclusive ISO date
    status: str = "pending"
    attempts: int = 0
    duration_s: Optional[float] = None
    error: Optional[str] = None


def plan_chunks(start: date, end: date, granularity: str = "daily") -> List[Chunk]:
    """Split [start, end) into idempotent, non-overlapping chunks. Each chunk = one re-runnable unit."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}")
    if start >= end:
        raise ValueError("start must be before end")
    chunks: List[Chunk] = []
    cur = start
    while cur < end:
        if granularity == "daily":
            nxt = cur + timedelta(days=1)
        elif granularity == "weekly":
            nxt = cur + timedelta(days=7 - cur.weekday())  # align to Monday
        else:
            nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        nxt = min(nxt, end)
        chunks.append(Chunk(chunk_id=f"{granularity[0]}-{cur.isoformat()}", start=cur.isoformat(),
                            end=nxt.isoformat()))
        cur = nxt
    return chunks


class BackfillState:
    """JSON-persisted plan: survives crashes, resumes where it stopped, never re-runs successes."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.chunks: List[Chunk] = []
        if self.path.exists():
            self.chunks = [Chunk(**c) for c in json.loads(self.path.read_text())["chunks"]]

    def init_plan(self, chunks: List[Chunk]) -> None:
        if not self.chunks:  # never clobber an in-progress plan
            self.chunks = chunks
            self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps({"chunks": [asdict(c) for c in self.chunks]}, indent=2))

    def next_batch(self, max_parallel: int = 4, max_attempts: int = 3) -> List[Chunk]:
        """Runnable chunks: pending first, then retryable failures. Respects the parallelism cap."""
        running = sum(1 for c in self.chunks if c.status == "running")
        slots = max(0, max_parallel - running)
        candidates = [c for c in self.chunks if c.status == "pending"] + \
                     [c for c in self.chunks if c.status == "failed" and c.attempts < max_attempts]
        return candidates[:slots]

    def mark(self, chunk_id: str, status: str, duration_s: Optional[float] = None,
             error: Optional[str] = None) -> None:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                c.status = status
                if status == "running":
                    c.attempts += 1
                if duration_s is not None:
                    c.duration_s = duration_s
                c.error = error
                break
        self._save()

    def summary(self) -> Dict[str, object]:
        counts = {s: sum(1 for c in self.chunks if c.status == s) for s in STATUSES}
        done = [c for c in self.chunks if c.status == "success" and c.duration_s]
        avg = sum(c.duration_s for c in done) / len(done) if done else None
        remaining = counts["pending"] + counts["failed"]
        dead = [c for c in self.chunks if c.status == "failed" and c.attempts >= 3]
        return {
            "total": len(self.chunks), **counts,
            "avg_chunk_seconds": round(avg, 1) if avg else None,
            "eta_seconds_serial": round(avg * remaining, 1) if avg else None,
            "dead_chunks": [c.chunk_id for c in dead],
            "pct_complete": round(100 * counts["success"] / len(self.chunks), 1) if self.chunks else 0,
        }


def run_backfill(state: BackfillState, job_fn, max_parallel: int = 4, max_attempts: int = 3,
                 max_cycles: int = 1000) -> Dict[str, object]:
    """Drive the plan to completion: pull a batch, run each chunk, record outcome, repeat.
    job_fn(chunk) -> duration_s, raises on failure. Idempotent chunks make retries safe."""
    import time
    for _ in range(max_cycles):
        batch = state.next_batch(max_parallel, max_attempts)
        if not batch:
            break
        for chunk in batch:
            state.mark(chunk.chunk_id, "running")
            t0 = time.perf_counter()
            try:
                job_fn(chunk)
                state.mark(chunk.chunk_id, "success", duration_s=time.perf_counter() - t0)
            except Exception as exc:
                state.mark(chunk.chunk_id, "failed", duration_s=time.perf_counter() - t0,
                           error=str(exc))
    return state.summary()
