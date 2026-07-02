from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Rule:
    """A single compliance rule. A document should either CONTAIN or NOT CONTAIN a pattern.

    mode='require' -> a violation if the pattern is MISSING (e.g. must have a data-retention clause).
    mode='forbid'  -> a violation if the pattern is PRESENT (e.g. must not store plaintext passwords).
    """

    id: str
    description: str
    pattern: str
    mode: str = "require"  # require | forbid
    severity: str = "medium"  # low | medium | high | critical
    category: str = "general"

    def compiled(self) -> re.Pattern:
        return re.compile(self.pattern, re.I)


@dataclass
class Finding:
    rule_id: str
    description: str
    severity: str
    category: str
    status: str  # pass | violation
    evidence: str = ""


@dataclass
class ComplianceReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def violations(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "violation"]

    def score(self) -> int:
        """0-100 compliance score, weighted by severity of violations."""
        if not self.findings:
            return 100
        weights = {"low": 1, "medium": 3, "high": 6, "critical": 10}
        max_pen = sum(weights[f.severity] for f in self.findings)
        pen = sum(weights[f.severity] for f in self.violations)
        return round(100 * (1 - pen / max_pen)) if max_pen else 100

    def severity_counts(self) -> dict:
        counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for f in self.violations:
            counts[f.severity] += 1
        return counts


# A default GDPR/security-flavored ruleset - the "policy as code" a compliance team maintains.
DEFAULT_RULES = [
    Rule("GDPR-RETENTION", "Must state a data retention period",
         r"retention period|retain(ed)? for|delete.*after|data.*retained",
         mode="require", severity="high", category="GDPR"),
    Rule("GDPR-CONSENT", "Must describe how consent is obtained",
         r"consent|opt[- ]?in|agree to", mode="require", severity="high", category="GDPR"),
    Rule("GDPR-RIGHTS", "Must mention data subject rights (access/erasure)",
         r"right to (access|erasure|be forgotten)|data subject rights|request.*deletion",
         mode="require", severity="medium", category="GDPR"),
    Rule("SEC-NO-PLAINTEXT-PW", "Must not store or transmit plaintext passwords",
         r"plain\s*text|clear\s*text|unencrypted password",
         mode="forbid", severity="critical", category="Security"),
    Rule("SEC-ENCRYPTION", "Must state data is encrypted",
         r"encrypt(ed|ion)|aes-?256|tls", mode="require", severity="high", category="Security"),
    Rule("SEC-NO-SHARED-CREDS", "Must not permit shared credentials",
         r"shared\s+(login|credentials|account)|share\s+your\s+password",
         mode="forbid", severity="high", category="Security"),
    Rule("CONTACT-DPO", "Must give a contact for data/privacy questions",
         r"data protection officer|dpo|privacy@|contact.*privacy",
         mode="require", severity="low", category="Governance"),
]


def check_rules(text: str, rules: Optional[list[Rule]] = None) -> ComplianceReport:
    """Run a document against the ruleset with deterministic regex matching. No LLM."""
    rules = rules or DEFAULT_RULES
    findings: list[Finding] = []
    for rule in rules:
        m = rule.compiled().search(text)
        present = m is not None
        # require: violation if absent; forbid: violation if present
        violated = (not present) if rule.mode == "require" else present
        # show the matched snippet whenever there IS a match (the offending text for a
        # forbid violation, or the satisfying text for a require pass); nothing to quote
        # when a required pattern is simply absent.
        evidence = _snippet(text, m) if m else ""
        findings.append(
            Finding(
                rule_id=rule.id,
                description=rule.description,
                severity=rule.severity,
                category=rule.category,
                status="violation" if violated else "pass",
                evidence=evidence,
            )
        )
    return ComplianceReport(findings=findings)


def _snippet(text: str, m: re.Match, pad: int = 40) -> str:
    start = max(0, m.start() - pad)
    end = min(len(text), m.end() + pad)
    return ("..." if start else "") + text[start:end].strip() + ("..." if end < len(text) else "")


def llm_check(text: str, rules: Optional[list[Rule]] = None, api_key: str = "") -> ComplianceReport:
    """Use Claude to judge each rule semantically - catches paraphrased compliance the regex misses."""
    import anthropic

    rules = rules or DEFAULT_RULES
    rule_json = [
        {"id": r.id, "description": r.description, "mode": r.mode, "severity": r.severity, "category": r.category}
        for r in rules
    ]
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1800,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a compliance auditor. For each rule, decide pass or violation for the "
                    "document. mode 'require' = violation if the document does NOT satisfy it; "
                    "mode 'forbid' = violation if it DOES. Judge by meaning, not exact words. "
                    "Respond with ONLY valid JSON, no markdown fences:\n"
                    '{"findings": [{"rule_id": "...", "status": "pass|violation", "evidence": "quote or empty"}]}\n\n'
                    f"RULES:\n{json.dumps(rule_json, indent=2)}\n\nDOCUMENT:\n{text}"
                ),
            }
        ],
    )
    data = json.loads(response.content[0].text.strip())
    status_map = {f["rule_id"]: f for f in data.get("findings", [])}
    findings = []
    for r in rules:
        f = status_map.get(r.id, {"status": "pass", "evidence": ""})
        findings.append(
            Finding(
                rule_id=r.id, description=r.description, severity=r.severity,
                category=r.category, status=f.get("status", "pass"), evidence=f.get("evidence", ""),
            )
        )
    return ComplianceReport(findings=findings)


def run_compliance_check(
    text: str, rules: Optional[list[Rule]] = None, api_key: Optional[str] = None
) -> ComplianceReport:
    """Check a document against a compliance ruleset.

    Uses Claude for semantic judgment if an API key is available, else regex rules."""
    if not text.strip():
        return ComplianceReport()
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return check_rules(text, rules)
    return llm_check(text, rules, api_key=api_key)


SAMPLE_POLICY_GOOD = """\
Privacy & Security Policy

We encrypt all customer data at rest using AES-256 and in transit using TLS 1.3.
Passwords are hashed with bcrypt before storage and are never recoverable.

We collect data only after obtaining explicit consent (opt-in) at signup. Users may
exercise their right to access and right to erasure at any time by emailing privacy@acme.com.
Personal data is retained for 24 months after account closure, then permanently deleted.

For any privacy questions, contact our Data Protection Officer at dpo@acme.com.
"""

SAMPLE_POLICY_BAD = """\
Internal Systems Note

Users log in with a username and password. For convenience, team leads may use a shared
login for the ops dashboard. Passwords are stored in plaintext in the config table so we
can email them to users who forget.

We keep customer records around indefinitely in case they come back.
"""
