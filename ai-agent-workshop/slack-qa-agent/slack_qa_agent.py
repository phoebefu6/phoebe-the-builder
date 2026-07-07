from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "do", "does", "how", "what", "when", "where",
    "who", "i", "we", "you", "to", "of", "for", "in", "on", "and", "or", "my",
    "can", "with", "it", "this", "that", "be", "if", "at", "as", "our",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = "knowledge-base"


@dataclass
class Retrieved:
    doc: Document
    score: float


@dataclass
class Answer:
    question: str
    text: str
    confidence: float
    escalated: bool
    citations: List[Document] = field(default_factory=list)


class TfidfRetriever:
    """Tiny hand-rolled TF-IDF cosine retriever — no external embedding API, so the
    whole RAG loop runs offline in a notebook."""

    def __init__(self, docs: List[Document]) -> None:
        self.docs = docs
        self._tokens = [_tokenize(f"{d.title} {d.text}") for d in docs]
        df: Counter = Counter()
        for toks in self._tokens:
            for term in set(toks):
                df[term] += 1
        n = max(len(docs), 1)
        self._idf: Dict[str, float] = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._vecs = [self._vectorize(toks) for toks in self._tokens]

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        vec = {t: (c / total) * self._idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 3) -> List[Retrieved]:
        q = self._vectorize(_tokenize(query))
        scored: List[Retrieved] = []
        for doc, vec in zip(self.docs, self._vecs):
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            scored.append(Retrieved(doc=doc, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


def _rule_based_answer(question: str, hits: List[Retrieved]) -> str:
    best = hits[0].doc
    others = [h.doc.title for h in hits[1:] if h.score > 0]
    ans = f"{best.text}\n\n_Source: {best.title} ({best.source})_"
    if others:
        ans += f"\n\nRelated: {', '.join(others)}"
    return ans


def _claude_answer(question: str, hits: List[Retrieved]) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        context = "\n\n".join(f"[{h.doc.title}]\n{h.doc.text}" for h in hits)
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    "Answer the Slack question using ONLY the context below. Be concise and "
                    "friendly, cite the source doc title, and if the context does not answer it "
                    f"say so plainly.\n\nContext:\n{context}\n\nQuestion: {question}"
                ),
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return None


def answer_question(
    question: str,
    retriever: TfidfRetriever,
    threshold: float = 0.35,
    escalation_channel: str = "#people-ops",
    use_claude: bool = True,
) -> Answer:
    """RAG loop: retrieve -> confidence gate -> grounded answer OR escalate to a human.
    The gate is what stops the bot from confidently answering questions it has no doc for."""
    hits = retriever.search(question, k=3)
    top_score = hits[0].score if hits else 0.0

    if top_score < threshold:
        return Answer(
            question=question,
            text=(
                f"I couldn't find a confident answer in the knowledge base, so I've routed this "
                f"to {escalation_channel} for a human to follow up. (Adding a doc for this will let "
                f"me answer it next time.)"
            ),
            confidence=round(top_score, 3),
            escalated=True,
            citations=[],
        )

    text = (_claude_answer(question, hits) if use_claude else None) or _rule_based_answer(question, hits)
    return Answer(
        question=question,
        text=text,
        confidence=round(top_score, 3),
        escalated=False,
        citations=[h.doc for h in hits if h.score > 0],
    )


def post_to_slack(channel: str, text: str) -> Tuple[bool, str]:
    """Post to Slack if SLACK_BOT_TOKEN is set; otherwise dry-run (returns the payload).
    Keeps the demo runnable without a real workspace."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        return (False, f"[dry-run] would post to {channel}: {text[:80]}...")
    try:
        import urllib.request
        import json as _json

        payload = _json.dumps({"channel": channel, "text": text}).encode()
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = _json.loads(resp.read())
        return (bool(body.get("ok")), body.get("ts", body.get("error", "")))
    except Exception as exc:  # noqa: BLE001
        return (False, f"slack post failed: {exc}")


SAMPLE_KB: List[Document] = [
    Document("pto", "PTO / Vacation Policy", "Vacation and paid time off: full-time employees get 20 vacation days of PTO per year, accrued monthly. Request time off in Workday at least 3 business days in advance. Unused vacation days roll over up to 5 days into the next year.", "hr-handbook"),
    Document("vpn", "VPN Access", "To connect to the VPN, install GlobalProtect from the IT self-service portal, sign in with your SSO credentials, and use the gateway vpn.company.com. Contact IT if VPN MFA fails.", "it-wiki"),
    Document("expenses", "Expense Reimbursement", "To get reimbursed, submit your expense report in Expensify within 30 days of purchase. Attach receipts for anything over $25. Approved expense reports are reimbursed in the next payroll cycle.", "finance-wiki"),
    Document("wifi", "WiFi Password", "The office WiFi network is 'Company-Secure'. Get the rotating WiFi password from the poster in each kitchen or from the #it-support channel. 'Company-Guest' is for visitors only.", "it-wiki"),
    Document("onboarding", "New Hire Onboarding / First Day", "On your first day new hires complete IT setup, then security training in week one, and build a 30-60-90 plan with their manager. Your onboarding buddy is assigned in the welcome email. Here is what to expect when you start.", "hr-handbook"),
    Document("payday", "Payday / Payroll Schedule", "Payday: we get paid on the 15th and the last business day of each month. Payroll direct deposit details are managed in Workday under Pay > Payment Elections.", "finance-wiki"),
    Document("laptop", "Broken Laptop / Hardware Replacement", "Standard laptop refresh is every 3 years. If your laptop broke or is damaged, file a hardware ticket in the IT self-service portal and a loaner laptop will be issued within one business day.", "it-wiki"),
]
