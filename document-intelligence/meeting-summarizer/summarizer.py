from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class MeetingSummary:
    tldr: str
    decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


ACTION_VERBS = re.compile(
    r"\b(will|should|need(s)? to|going to|let's|action item|todo|follow up)\b", re.IGNORECASE
)
DECISION_WORDS = re.compile(r"\b(decided|agreed|we'll go with|final call|approved)\b", re.IGNORECASE)
QUESTION_PATTERN = re.compile(r"[^.!?]*\?")


def heuristic_summary(transcript: str) -> MeetingSummary:
    """Fallback extraction with no LLM: pattern-match sentences for action/decision/question cues."""
    sentences = re.split(r"(?<=[.!?])\s+", transcript.strip())
    decisions = [s.strip() for s in sentences if DECISION_WORDS.search(s)]
    actions = [s.strip() for s in sentences if ACTION_VERBS.search(s)]
    questions = [s.strip() for s in QUESTION_PATTERN.findall(transcript) if s.strip()]

    tldr = sentences[0].strip() if sentences else "No content to summarize."
    return MeetingSummary(
        tldr=tldr,
        decisions=decisions[:5],
        action_items=actions[:5],
        open_questions=questions[:5],
    )


def summarize(transcript: str, api_key: str | None = None) -> MeetingSummary:
    """Summarize a meeting transcript into TL;DR, decisions, action items, open questions.
    Uses Claude if an API key is available, else falls back to regex-based heuristics."""
    if not transcript.strip():
        return MeetingSummary(tldr="Empty transcript.")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return heuristic_summary(transcript)

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=800,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this meeting transcript. Respond with ONLY valid JSON, "
                    "no markdown fences, matching this schema:\n"
                    '{"tldr": "one sentence", "decisions": ["..."], '
                    '"action_items": ["..."], "open_questions": ["..."]}\n\n'
                    f"Transcript:\n{transcript}"
                ),
            }
        ],
    )
    raw = response.content[0].text.strip()
    data = json.loads(raw)
    return MeetingSummary(
        tldr=data.get("tldr", ""),
        decisions=data.get("decisions", []),
        action_items=data.get("action_items", []),
        open_questions=data.get("open_questions", []),
    )


SAMPLE_TRANSCRIPT = """\
Sarah: Let's kick off. First topic is the Q3 roadmap.
Mike: I think we should prioritize the mobile redesign over the API v2 work.
Sarah: Agreed, we decided mobile redesign is the priority for Q3.
Mike: I'll need to follow up with design to get the new mockups by Friday.
Priya: What's our budget for the contractor we discussed last week?
Sarah: Good question, I don't have that number yet, let me check with finance.
Mike: Also, we need to send the updated timeline to stakeholders by end of week.
Priya: Should we loop in the support team on this too?
Sarah: Yes, let's go with looping them in starting next sprint.
"""
