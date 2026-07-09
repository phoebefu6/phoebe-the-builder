from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-z']+")

POSITIVE = {
    "love", "loved", "great", "excellent", "amazing", "awesome", "perfect", "good",
    "fast", "easy", "intuitive", "reliable", "helpful", "fantastic", "smooth", "best",
    "recommend", "happy", "beautiful", "responsive", "affordable", "solid", "worth",
}
NEGATIVE = {
    "hate", "hated", "bad", "terrible", "awful", "horrible", "slow", "buggy", "crash",
    "crashes", "broken", "confusing", "expensive", "useless", "disappointed", "poor",
    "frustrating", "difficult", "worst", "laggy", "unreliable", "overpriced", "clunky",
    "glitchy", "annoying",
}
NEGATORS = {"not", "no", "never", "n't", "without", "hardly"}

STOP = {
    "the", "a", "an", "is", "are", "was", "were", "it", "this", "that", "i", "we",
    "you", "to", "of", "for", "in", "on", "and", "or", "but", "my", "so", "very",
    "with", "at", "as", "be", "have", "has", "had", "they", "them", "app", "product",
    "would", "could", "really", "just", "get", "got", "im", "ive", "me", "too", "if",
    "than", "then", "there", "their", "your", "our", "all", "can", "do", "does", "did",
}


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ReviewScore:
    text: str
    label: str      # positive | neutral | negative
    score: float    # -1 .. 1


@dataclass
class FeedbackReport:
    scores: List[ReviewScore]
    distribution: Dict[str, int]
    top_praises: List[Tuple[str, int]]
    top_complaints: List[Tuple[str, int]]
    insight: str = ""

    @property
    def nps_like(self) -> float:
        """% positive minus % negative (a rough promoter-minus-detractor proxy)."""
        n = len(self.scores) or 1
        return round(100 * (self.distribution.get("positive", 0) - self.distribution.get("negative", 0)) / n, 1)


def score_sentiment(text: str) -> ReviewScore:
    """Lexicon sentiment with negation flip: a negator within the prior 2 tokens
    inverts the polarity of a sentiment word ('not good' counts negative)."""
    toks = _tokens(text)
    pos = neg = 0
    for i, tok in enumerate(toks):
        window = toks[max(0, i - 2):i]
        negated = any(w in NEGATORS for w in window)
        if tok in POSITIVE:
            neg += 1 if negated else 0
            pos += 0 if negated else 1
        elif tok in NEGATIVE:
            pos += 1 if negated else 0
            neg += 0 if negated else 1
    total = pos + neg
    score = 0.0 if total == 0 else (pos - neg) / total
    label = "neutral" if abs(score) < 0.2 else ("positive" if score > 0 else "negative")
    return ReviewScore(text=text, label=label, score=round(score, 3))


def _themes(texts: List[str], top_k: int = 8) -> List[Tuple[str, int]]:
    """Top content unigrams + bigrams across texts (stopwords + sentiment words stripped
    from unigrams so themes are nouns like 'battery' / 'customer service', not 'good')."""
    uni: Counter = Counter()
    bi: Counter = Counter()
    for t in texts:
        toks = [w for w in _tokens(t) if w not in STOP and w not in NEGATORS]
        content = [w for w in toks if w not in POSITIVE and w not in NEGATIVE and len(w) > 2]
        uni.update(content)
        for a, b in zip(toks, toks[1:]):
            if a not in STOP and b not in STOP and len(a) > 2 and len(b) > 2:
                bi.update([f"{a} {b}"])
    merged = uni + Counter({k: v + 1 for k, v in bi.items() if v >= 2})  # bigrams get a small boost
    return merged.most_common(top_k)


def _claude_insight(report_bits: Dict) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this user-feedback analysis in 2-3 sentences for a product manager. "
                    "Lead with the single most actionable takeaway.\n"
                    f"Sentiment split: {report_bits['distribution']}\n"
                    f"Top praises: {report_bits['praises']}\n"
                    f"Top complaints: {report_bits['complaints']}"
                ),
            }],
        )
        return resp.content[0].text.strip()
    except Exception:
        return None


def analyze_feedback(reviews: List[str], use_claude: bool = True) -> FeedbackReport:
    scores = [score_sentiment(r) for r in reviews if r.strip()]
    dist = Counter(s.label for s in scores)
    distribution = {k: dist.get(k, 0) for k in ("positive", "neutral", "negative")}

    praises = _themes([s.text for s in scores if s.label == "positive"])
    complaints = _themes([s.text for s in scores if s.label == "negative"])

    insight = ""
    if use_claude:
        insight = _claude_insight({
            "distribution": distribution,
            "praises": [p for p, _ in praises[:5]],
            "complaints": [c for c, _ in complaints[:5]],
        }) or ""
    if not insight:
        top_c = complaints[0][0] if complaints else "n/a"
        top_p = praises[0][0] if praises else "n/a"
        insight = (
            f"{distribution['negative']} of {len(scores)} reviews are negative. "
            f"Biggest complaint theme: '{top_c}'. Most-loved: '{top_p}'. "
            f"Fix the top complaint first."
        )

    return FeedbackReport(
        scores=scores, distribution=distribution,
        top_praises=praises, top_complaints=complaints, insight=insight,
    )


SAMPLE_REVIEWS: List[str] = [
    "Love this app, the interface is so intuitive and fast.",
    "The battery drain is terrible, my phone dies in hours.",
    "Great customer service, they resolved my issue quickly.",
    "Way too expensive for what it offers. Not worth the price.",
    "The app keeps crashing every time I upload a photo.",
    "Beautiful design and super easy to use. Highly recommend!",
    "Sync is unreliable, my data disappeared twice this week.",
    "Customer support was helpful and responsive, solid experience.",
    "Confusing navigation, I can never find the settings.",
    "Fast, reliable, and affordable. Best in its category.",
    "The latest update is buggy and slow, please fix it.",
    "Not good. The onboarding is frustrating and unclear.",
    "Amazing value, the premium features are worth every penny.",
    "It crashes on startup, completely useless right now.",
    "Smooth performance and the search is really responsive.",
    "Overpriced subscription with too many bugs.",
    "Easy setup, worked perfectly out of the box.",
    "Terrible battery life and the notifications are annoying.",
    "The customer service team is fantastic and quick to help.",
    "Laggy and clunky since the redesign, disappointed.",
]
