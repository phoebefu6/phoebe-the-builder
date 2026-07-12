from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Event:
    ts: float          # event time (epoch seconds) — when it happened, not when it arrived
    key: str
    value: float


@dataclass
class WindowResult:
    window_start: float
    window_end: float
    key: str
    count: int
    total: float
    vmin: float
    vmax: float

    @property
    def mean(self) -> float:
        return round(self.total / self.count, 3) if self.count else 0.0


@dataclass
class _Acc:
    count: int = 0
    total: float = 0.0
    vmin: float = float("inf")
    vmax: float = float("-inf")

    def add(self, v: float) -> None:
        self.count += 1
        self.total += v
        self.vmin = min(self.vmin, v)
        self.vmax = max(self.vmax, v)


class WindowAggregator:
    """Event-time windowing with watermarks — the core of every streaming engine, in one class.

    - Tumbling: slide == window (each event lands in exactly one window)
    - Sliding: slide < window (each event lands in window/slide overlapping windows)
    - Watermark: max event time seen minus allowed_lateness. A window is finalized
      (emitted, immutable) once the watermark passes its end. Events older than
      their finalized window are counted as late_dropped, never silently merged.
    """

    def __init__(self, window_seconds: float, slide_seconds: Optional[float] = None,
                 allowed_lateness_seconds: float = 0.0) -> None:
        self.window = float(window_seconds)
        self.slide = float(slide_seconds) if slide_seconds else float(window_seconds)
        if self.slide > self.window:
            raise ValueError("slide must be <= window (gaps would drop events)")
        self.lateness = float(allowed_lateness_seconds)
        self.max_event_ts = float("-inf")
        self.late_dropped = 0
        self._open: Dict[Tuple[float, str], _Acc] = {}  # (window_start, key) -> acc

    @property
    def watermark(self) -> float:
        return self.max_event_ts - self.lateness

    def _windows_for(self, ts: float) -> List[float]:
        first = (ts // self.slide) * self.slide
        starts = []
        start = first
        while start > ts - self.window:
            starts.append(start)
            start -= self.slide
        return [s for s in starts if s <= ts < s + self.window]

    def feed(self, event: Event) -> List[WindowResult]:
        """Add one event; return any windows finalized by the advancing watermark."""
        self.max_event_ts = max(self.max_event_ts, event.ts)
        placed = False
        for start in self._windows_for(event.ts):
            if start + self.window <= self.watermark:
                continue  # this window already closed — event is too late for it
            self._open.setdefault((start, event.key), _Acc()).add(event.value)
            placed = True
        if not placed:
            self.late_dropped += 1
        return self._finalize_ripe()

    def _finalize_ripe(self) -> List[WindowResult]:
        ripe = [(ws, key) for (ws, key) in self._open if ws + self.window <= self.watermark]
        out: List[WindowResult] = []
        for ws, key in sorted(ripe):
            acc = self._open.pop((ws, key))
            out.append(WindowResult(ws, ws + self.window, key, acc.count, round(acc.total, 3),
                                    acc.vmin, acc.vmax))
        return out

    def flush(self) -> List[WindowResult]:
        """End of stream: emit everything still open."""
        out = [WindowResult(ws, ws + self.window, key, acc.count, round(acc.total, 3),
                            acc.vmin, acc.vmax)
               for (ws, key), acc in sorted(self._open.items())]
        self._open.clear()
        return out


def make_event_stream(n: int = 2000, keys: Tuple[str, ...] = ("checkout", "search", "api"),
                      duration_s: float = 600.0, disorder_s: float = 8.0,
                      late_fraction: float = 0.02, late_by_s: float = 90.0,
                      seed: int = 7) -> List[Event]:
    """Out-of-order stream in *arrival* order: mild disorder plus a few very-late stragglers."""
    import random
    rng = random.Random(seed)
    events = []
    for i in range(n):
        true_ts = duration_s * i / n
        arrival_skew = rng.uniform(0, disorder_s)
        if rng.random() < late_fraction:
            arrival_skew += late_by_s
        events.append((true_ts + arrival_skew,  # arrival order key
                       Event(ts=round(true_ts, 2), key=rng.choice(keys),
                             value=round(rng.uniform(10, 200), 2))))
    events.sort(key=lambda p: p[0])
    return [e for _, e in events]
