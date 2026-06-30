from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Clause:
    """A single extracted contract clause."""

    clause_type: str
    text: str
    risk: str = "low"  # low | medium | high
    note: str = ""


@dataclass
class ContractReview:
    """Structured output of a contract review."""

    parties: list[str] = field(default_factory=list)
    effective_date: str = ""
    clauses: list[Clause] = field(default_factory=list)
    missing_clauses: list[str] = field(default_factory=list)

    def risk_counts(self) -> dict[str, int]:
        counts = {"low": 0, "medium": 0, "high": 0}
        for c in self.clauses:
            counts[c.risk] = counts.get(c.risk, 0) + 1
        return counts


# Clause types most legal reviewers scan for first, with cue patterns + a default
# risk weight when the clause is present but worth a human look.
CLAUSE_CUES: dict[str, tuple[re.Pattern, str]] = {
    "Termination": (re.compile(r"\b(terminat\w+|cancel\w+|expire|notice period)\b", re.I), "medium"),
    "Liability": (re.compile(r"\b(liabilit\w+|indemnif\w+|hold harmless|damages)\b", re.I), "high"),
    "Confidentiality": (re.compile(r"\b(confidential\w*|non-disclosure|proprietary information)\b", re.I), "low"),
    "Payment": (re.compile(r"\b(payment|fee|invoice|net \d+|compensation|\$[\d,]+)\b", re.I), "medium"),
    "Governing Law": (re.compile(r"\b(governing law|jurisdiction|governed by the laws)\b", re.I), "low"),
    "Auto-Renewal": (re.compile(r"\b(auto\w*[- ]?renew\w*|automatically renew|evergreen)\b", re.I), "high"),
    "Intellectual Property": (re.compile(r"\b(intellectual property|work product|ownership of|license to)\b", re.I), "medium"),
    "Warranty": (re.compile(r"\b(warrant\w+|as is|fitness for a particular purpose)\b", re.I), "medium"),
}

# Clauses a reviewer flags as RISKY when missing entirely from a contract.
EXPECTED_CLAUSES = ["Termination", "Liability", "Confidentiality", "Payment", "Governing Law"]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _extract_parties(text: str) -> list[str]:
    """Pull party names from a typical preamble: '... between Acme Inc. and Beta LLC'."""
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)[\.,\n]", text, re.I)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    return []


def _extract_date(text: str) -> str:
    m = re.search(
        r"\b(?:dated|effective(?:\s+as\s+of)?|entered into on)\s+([A-Z][a-z]+ \d{1,2}, \d{4}|\d{4}-\d{2}-\d{2})",
        text,
        re.I,
    )
    return m.group(1) if m else ""


def heuristic_review(contract: str) -> ContractReview:
    """Extract clauses with no LLM: scan each sentence for clause cues, flag risk + gaps.

    Fast, deterministic, zero-cost first pass. Good enough to triage a contract in seconds;
    upgrade to Claude for paraphrased / unusually worded clauses."""
    sentences = _split_sentences(contract)
    clauses: list[Clause] = []
    seen_types: set[str] = set()

    for sent in sentences:
        for ctype, (pattern, risk) in CLAUSE_CUES.items():
            if pattern.search(sent):
                clauses.append(Clause(clause_type=ctype, text=sent, risk=risk))
                seen_types.add(ctype)

    missing = [c for c in EXPECTED_CLAUSES if c not in seen_types]
    return ContractReview(
        parties=_extract_parties(contract),
        effective_date=_extract_date(contract),
        clauses=clauses,
        missing_clauses=missing,
    )


def llm_review(contract: str, api_key: str) -> ContractReview:
    """Use Claude to extract clauses, including paraphrased ones the regex misses."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    clause_list = ", ".join(CLAUSE_CUES.keys())
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a contract reviewer. Extract key clauses from the contract below. "
                    f"Look for these clause types: {clause_list}. "
                    "For each clause found, assign a risk of low, medium, or high based on how "
                    "one-sided or unusual the terms are, and a one-line note on why. "
                    "Also list any of these expected clauses that are MISSING: "
                    f"{', '.join(EXPECTED_CLAUSES)}. "
                    "Respond with ONLY valid JSON, no markdown fences, matching:\n"
                    '{"parties": ["..."], "effective_date": "...", '
                    '"clauses": [{"clause_type": "...", "text": "...", "risk": "low|medium|high", "note": "..."}], '
                    '"missing_clauses": ["..."]}\n\n'
                    f"Contract:\n{contract}"
                ),
            }
        ],
    )
    data = json.loads(response.content[0].text.strip())
    return ContractReview(
        parties=data.get("parties", []),
        effective_date=data.get("effective_date", ""),
        clauses=[
            Clause(
                clause_type=c.get("clause_type", ""),
                text=c.get("text", ""),
                risk=c.get("risk", "low"),
                note=c.get("note", ""),
            )
            for c in data.get("clauses", [])
        ],
        missing_clauses=data.get("missing_clauses", []),
    )


def review_contract(contract: str, api_key: Optional[str] = None) -> ContractReview:
    """Review a contract into structured clauses + risk flags.

    Uses Claude if an API key is available, else falls back to regex heuristics."""
    if not contract.strip():
        return ContractReview(missing_clauses=EXPECTED_CLAUSES)

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return heuristic_review(contract)
    return llm_review(contract, api_key)


def extract_text_from_pdf(path: str) -> str:
    """Pull raw text from a PDF using pypdf. Raises a clear error if pypdf is absent."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pypdf not installed. Run: pip install pypdf") from exc
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


SAMPLE_CONTRACT = """\
This Master Services Agreement ("Agreement") is entered into on March 15, 2026,
between Acme Analytics Inc. and Beta Retail LLC.

1. Services. Acme will provide data engineering services as described in each Statement of Work.

2. Payment. Beta shall pay all invoices Net 30. Fees total $48,000 per quarter.

3. Term and Termination. This Agreement begins on the effective date and remains in
effect for twelve months. Either party may terminate with 30 days written notice.

4. Auto-Renewal. This Agreement will automatically renew for successive one-year terms
unless either party gives notice of non-renewal at least 90 days before the renewal date.

5. Confidentiality. Each party agrees to keep the other's proprietary information confidential.

6. Limitation of Liability. Acme's total liability shall not exceed the fees paid in the
prior three months. Beta agrees to indemnify and hold harmless Acme against third-party claims.

7. Intellectual Property. All work product created under this Agreement is owned by Beta,
with Acme retaining a license to its pre-existing tools.
"""
