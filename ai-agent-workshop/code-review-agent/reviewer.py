from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Finding:
    file: str
    line: int
    severity: str  # critical | high | medium | low
    message: str


@dataclass
class ReviewResult:
    findings: List[Finding] = field(default_factory=list)
    summary: str = ""
    files_reviewed: int = 0
    lines_added: int = 0


SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id shape
]

RULES = [
    ("bare except", re.compile(r"^\s*except\s*:\s*$"), "high", "bare 'except:' swallows all errors, catch specific exceptions"),
    ("print left in", re.compile(r"^\s*print\("), "medium", "leftover print() — use logging instead"),
    ("todo marker", re.compile(r"(?i)#\s*(TODO|FIXME|HACK)"), "medium", "unresolved TODO/FIXME marker"),
    ("debugger left in", re.compile(r"^\s*(import pdb|pdb\.set_trace\(\))"), "critical", "debugger breakpoint left in code"),
]


def _parse_diff(diff_text: str):
    """Yield (file, line_number, added_line_text) for each added (+) line in a unified diff."""
    current_file = "unknown"
    current_line = 0
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ "):
            current_file = raw_line[4:].lstrip("b/").strip()
            continue
        if raw_line.startswith("@@"):
            match = re.search(r"\+(\d+)", raw_line)
            current_line = int(match.group(1)) - 1 if match else 0
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            current_line += 1
            yield current_file, current_line, raw_line[1:]
        elif not raw_line.startswith("-"):
            current_line += 1


def _rule_based_findings(diff_text: str) -> List[Finding]:
    findings: List[Finding] = []
    for file, line_no, content in _parse_diff(diff_text):
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                findings.append(Finding(file, line_no, "critical", "possible hardcoded secret/credential"))
                break
        for _name, pattern, severity, message in RULES:
            if pattern.search(content):
                findings.append(Finding(file, line_no, severity, message))
        if len(content) > 120:
            findings.append(Finding(file, line_no, "low", f"line exceeds 120 chars ({len(content)})"))
    return findings


def _call_claude_summary(diff_text: str, findings: List[Finding]) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        findings_text = "\n".join(f"- {f.severity.upper()} {f.file}:{f.line} {f.message}" for f in findings) or "none"
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"Write a 3-sentence PR review summary given these automated findings:\n{findings_text}\n\nDiff:\n{diff_text[:3000]}",
            }],
        )
        return response.content[0].text
    except Exception:
        return None


def _rule_based_summary(findings: List[Finding], files_reviewed: int, lines_added: int) -> str:
    if not findings:
        return f"Reviewed {files_reviewed} file(s), {lines_added} line(s) added. No issues flagged — looks safe to merge."
    counts = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    parts = ", ".join(f"{n} {sev}" for sev, n in sorted(counts.items(), key=lambda x: ["critical", "high", "medium", "low"].index(x[0])))
    verdict = "blocks merge until fixed" if counts.get("critical") or counts.get("high") else "safe to merge after minor cleanup"
    return f"Reviewed {files_reviewed} file(s), {lines_added} line(s) added. Found {len(findings)} issue(s): {parts}. Verdict: {verdict}."


def review_diff(diff_text: str) -> ReviewResult:
    findings = _rule_based_findings(diff_text)
    files_reviewed = len({f.file for f in findings}) or len(re.findall(r"^\+\+\+ ", diff_text, re.M))
    lines_added = len(re.findall(r"^\+[^+]", diff_text, re.M))

    summary = _call_claude_summary(diff_text, findings) or _rule_based_summary(findings, files_reviewed, lines_added)

    return ReviewResult(findings=findings, summary=summary, files_reviewed=files_reviewed, lines_added=lines_added)


def fetch_pr_diff(repo: str, pr_number: int, github_token: Optional[str] = None) -> str:
    """Fetch a PR's unified diff from the GitHub API. Requires network + optionally a token for private repos."""
    import requests

    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {"Accept": "application/vnd.github.v3.diff"}
    token = github_token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text
