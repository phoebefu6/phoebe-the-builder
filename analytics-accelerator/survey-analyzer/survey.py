"""Survey Results Analyzer — core logic.

Takes a raw survey export (one row per respondent, one column per question)
and turns it into structured insight: it auto-detects each question's type
(Likert / single-choice / numeric / open-text), computes the right summary
for that type, derives an NPS score when a 0-10 recommend question exists,
and runs a lightweight lexicon sentiment + theme pass over open-text answers.

No external services or API keys — everything runs on pandas + a small
built-in word list so it works standalone in a notebook or CI.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

# --- Likert vocab: maps common ordinal text answers to a 1-5 scale ---------
LIKERT_SCALE: Dict[str, int] = {
    "strongly disagree": 1,
    "disagree": 2,
    "neutral": 3,
    "neither agree nor disagree": 3,
    "agree": 4,
    "strongly agree": 5,
    "very dissatisfied": 1,
    "dissatisfied": 2,
    "satisfied": 4,
    "very satisfied": 5,
    "very unlikely": 1,
    "unlikely": 2,
    "likely": 4,
    "very likely": 5,
    "never": 1,
    "rarely": 2,
    "sometimes": 3,
    "often": 4,
    "always": 5,
}

# --- Tiny sentiment lexicon (no model download) ----------------------------
POSITIVE_WORDS = {
    "love", "great", "excellent", "amazing", "good", "helpful", "easy",
    "fast", "intuitive", "reliable", "friendly", "fantastic", "perfect",
    "awesome", "smooth", "happy", "recommend", "best", "wonderful", "clear",
}
NEGATIVE_WORDS = {
    "hate", "bad", "terrible", "awful", "slow", "confusing", "hard",
    "buggy", "broken", "poor", "frustrating", "difficult", "expensive",
    "worst", "useless", "disappointing", "clunky", "crash", "annoying", "lacking",
}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to",
    "of", "in", "on", "for", "it", "this", "that", "i", "we", "you", "my",
    "with", "as", "at", "be", "have", "has", "not", "so", "very", "too",
    "would", "could", "more", "really", "just", "do", "did", "they", "your",
}

# Question types the analyzer recognizes.
LIKERT = "likert"
SINGLE_CHOICE = "single_choice"
NUMERIC = "numeric"
OPEN_TEXT = "open_text"
NPS = "nps"


@dataclass
class QuestionSummary:
    column: str
    qtype: str
    n: int
    summary: Dict[str, object] = field(default_factory=dict)


@dataclass
class SurveyReport:
    n_respondents: int
    questions: List[QuestionSummary]
    nps: Optional[float] = None
    nps_breakdown: Optional[Dict[str, int]] = None


def _clean(series: pd.Series) -> pd.Series:
    """Drop blanks/NA and return a trimmed copy."""
    s = series.dropna()
    if s.dtype == object:
        s = s.astype(str).str.strip()
        s = s[s.str.len() > 0]
    return s


def detect_question_type(series: pd.Series) -> str:
    """Heuristically classify one survey column.

    Order matters: NPS (0-10 numeric) > Likert > numeric > single-choice >
    open-text. The thresholds favor recall on the structured types so that
    open-text is the fallback only when nothing else fits.
    """
    s = _clean(series)
    if s.empty:
        return OPEN_TEXT

    # Numeric? (covers NPS 0-10 and rating scales)
    nums = pd.to_numeric(s, errors="coerce")
    numeric_frac = nums.notna().mean()
    if numeric_frac > 0.9:
        lo, hi = nums.min(), nums.max()
        if lo >= 0 and hi <= 10 and nums.nunique() > 2:
            return NPS if hi >= 8 else NUMERIC
        return NUMERIC

    # Text-based: is it Likert vocab?
    lowered = s.str.lower()
    likert_frac = lowered.isin(LIKERT_SCALE).mean()
    if likert_frac > 0.6:
        return LIKERT

    # Few distinct short values -> single-choice (e.g. plan tier, yes/no)
    uniq = s.nunique()
    avg_words = s.str.split().str.len().mean()
    if uniq <= max(8, int(0.2 * len(s))) and avg_words <= 4:
        return SINGLE_CHOICE

    return OPEN_TEXT


def score_sentiment(text: str) -> Tuple[str, int]:
    """Lexicon sentiment for one open-text answer.

    Returns (label, score) where score = positives - negatives. Label is
    positive / negative / neutral.
    """
    words = re.findall(r"[a-z']+", str(text).lower())
    score = sum(w in POSITIVE_WORDS for w in words) - sum(
        w in NEGATIVE_WORDS for w in words
    )
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", 0


def top_themes(texts: List[str], top_n: int = 8) -> List[Tuple[str, int]]:
    """Most frequent non-stopword tokens across open-text answers."""
    counter: Counter = Counter()
    for t in texts:
        for w in re.findall(r"[a-z']+", str(t).lower()):
            if len(w) > 2 and w not in STOPWORDS:
                counter[w] += 1
    return counter.most_common(top_n)


def _summarize_open_text(s: pd.Series) -> Dict[str, object]:
    texts = s.tolist()
    labels = [score_sentiment(t)[0] for t in texts]
    counts = Counter(labels)
    return {
        "sentiment": dict(counts),
        "sentiment_pct": {
            k: round(100 * v / len(texts), 1) for k, v in counts.items()
        },
        "themes": top_themes(texts),
        "sample": texts[:3],
    }


def _to_likert_scores(s: pd.Series) -> pd.Series:
    return s.str.lower().map(LIKERT_SCALE).dropna()


def compute_nps(series: pd.Series) -> Tuple[float, Dict[str, int]]:
    """Standard NPS: %promoters (9-10) - %detractors (0-6), on a 0-10 column."""
    nums = pd.to_numeric(_clean(series), errors="coerce").dropna()
    if nums.empty:
        return 0.0, {"promoters": 0, "passives": 0, "detractors": 0}
    promoters = int((nums >= 9).sum())
    passives = int(((nums >= 7) & (nums <= 8)).sum())
    detractors = int((nums <= 6).sum())
    total = len(nums)
    nps = 100 * (promoters - detractors) / total
    return round(nps, 1), {
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
    }


def analyze_survey(df: pd.DataFrame) -> SurveyReport:
    """Full pipeline: classify every column, summarize it, derive NPS."""
    questions: List[QuestionSummary] = []
    nps_value: Optional[float] = None
    nps_breakdown: Optional[Dict[str, int]] = None

    for col in df.columns:
        s = _clean(df[col])
        qtype = detect_question_type(df[col])
        n = int(s.shape[0])
        summary: Dict[str, object] = {}

        if qtype == NPS:
            nps_value, nps_breakdown = compute_nps(df[col])
            nums = pd.to_numeric(s, errors="coerce").dropna()
            summary = {"mean": round(float(nums.mean()), 2), "nps": nps_value}
        elif qtype == NUMERIC:
            nums = pd.to_numeric(s, errors="coerce").dropna()
            summary = {
                "mean": round(float(nums.mean()), 2),
                "median": float(nums.median()),
                "min": float(nums.min()),
                "max": float(nums.max()),
            }
        elif qtype == LIKERT:
            scores = _to_likert_scores(s)
            summary = {
                "mean_score": round(float(scores.mean()), 2) if len(scores) else None,
                "distribution": s.value_counts().to_dict(),
            }
        elif qtype == SINGLE_CHOICE:
            counts = s.value_counts()
            summary = {
                "distribution": counts.to_dict(),
                "top": counts.index[0] if len(counts) else None,
            }
        else:  # OPEN_TEXT
            summary = _summarize_open_text(s.tolist() if isinstance(s, list) else s)

        questions.append(QuestionSummary(column=col, qtype=qtype, n=n, summary=summary))

    return SurveyReport(
        n_respondents=len(df),
        questions=questions,
        nps=nps_value,
        nps_breakdown=nps_breakdown,
    )


def sample_survey() -> pd.DataFrame:
    """Deterministic mock survey for demos, tests, and the notebook."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 120
    nps = rng.integers(0, 11, size=n)
    satisfaction = rng.choice(
        ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"],
        size=n,
        p=[0.05, 0.10, 0.20, 0.40, 0.25],
    )
    plan = rng.choice(["Free", "Pro", "Enterprise"], size=n, p=[0.5, 0.35, 0.15])
    hours = rng.integers(1, 40, size=n)
    comments_pool = [
        "Love the product, very intuitive and fast.",
        "The dashboard is confusing and slow to load.",
        "Great support team, really helpful and friendly.",
        "Too expensive for what it offers, disappointing.",
        "Easy to use, best tool we have adopted this year.",
        "Buggy on mobile, crashes often, frustrating experience.",
        "Reliable and clear, would recommend to colleagues.",
        "Onboarding was hard, documentation is lacking.",
    ]
    comments = rng.choice(comments_pool, size=n)

    return pd.DataFrame(
        {
            "How likely are you to recommend us (0-10)?": nps,
            "Overall satisfaction": satisfaction,
            "Current plan": plan,
            "Hours used per week": hours,
            "What can we improve?": comments,
        }
    )
