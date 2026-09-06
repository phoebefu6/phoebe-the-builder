from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

VAGUE_LINK_TEXT = {"click here", "here", "read more", "more", "link", "this", "learn more"}
SEVERITY_WEIGHT = {"critical": 10, "serious": 6, "moderate": 3, "minor": 1}


@dataclass
class Finding:
    rule: str
    severity: str  # critical | serious | moderate | minor
    element: str
    message: str
    wcag: str


@dataclass
class _Element:
    tag: str
    attrs: Dict[str, Optional[str]]
    text: str = ""


class _Collector(HTMLParser):
    """Single pass over the document, collecting everything the rules need."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.images: List[_Element] = []
        self.links: List[_Element] = []
        self.buttons: List[_Element] = []
        self.inputs: List[_Element] = []
        self.headings: List[Tuple[int, str]] = []
        self.label_targets: set = set()
        self.has_html_lang = False
        self.has_title = False
        self.has_viewport = False
        self.positive_tabindex: List[str] = []
        self.inline_event_handlers: List[str] = []
        self._stack: List[_Element] = []

    def handle_starttag(self, tag: str, attrs_list) -> None:
        attrs = dict(attrs_list)
        el = _Element(tag=tag, attrs=attrs)
        if tag == "html" and attrs.get("lang", "").strip():
            self.has_html_lang = True
        elif tag == "title":
            self.has_title = True
        elif tag == "meta" and attrs.get("name", "").lower() == "viewport":
            self.has_viewport = True
        elif tag == "img":
            self.images.append(el)
        elif tag == "a":
            self._stack.append(el)
        elif tag == "button":
            self._stack.append(el)
        elif tag in ("input", "select", "textarea"):
            self.inputs.append(el)
        elif tag == "label" and attrs.get("for"):
            self.label_targets.add(attrs["for"])
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._stack.append(el)
        tabindex = attrs.get("tabindex")
        if tabindex and tabindex.lstrip("+").isdigit() and int(tabindex) > 0:
            self.positive_tabindex.append(f"<{tag} tabindex={tabindex}>")
        if any(k.startswith("on") for k in attrs) and tag not in ("button", "a", "input", "select", "textarea", "form"):
            self.inline_event_handlers.append(f"<{tag}>")

    def handle_data(self, data: str) -> None:
        if self._stack:
            self._stack[-1].text += data

    def handle_endtag(self, tag: str) -> None:
        if self._stack and self._stack[-1].tag == tag:
            el = self._stack.pop()
            if tag == "a":
                self.links.append(el)
            elif tag == "button":
                self.buttons.append(el)
            else:
                self.headings.append((int(tag[1]), el.text.strip()))


def _describe(el: _Element, limit: int = 60) -> str:
    attrs = " ".join(f'{k}="{v}"' for k, v in list(el.attrs.items())[:3] if v is not None)
    return f"<{el.tag}{' ' + attrs if attrs else ''}>"[:limit]


def check_html(html: str) -> List[Finding]:
    c = _Collector()
    c.feed(html)
    findings: List[Finding] = []

    if not c.has_html_lang:
        findings.append(Finding("html-lang", "serious", "<html>",
                                "Missing lang attribute — screen readers can't pick the right voice", "WCAG 3.1.1"))
    if not c.has_title:
        findings.append(Finding("page-title", "serious", "<head>",
                                "Missing <title> — the page has no accessible name", "WCAG 2.4.2"))
    if not c.has_viewport:
        findings.append(Finding("viewport", "moderate", "<head>",
                                "Missing viewport meta — pinch-zoom users and mobile readers suffer", "WCAG 1.4.10"))

    for img in c.images:
        if "alt" not in img.attrs:
            findings.append(Finding("img-alt", "critical", _describe(img),
                                    "Image has no alt attribute — invisible to screen readers", "WCAG 1.1.1"))
        elif img.attrs.get("alt", "").strip().lower() in ("image", "photo", "picture", "img"):
            findings.append(Finding("img-alt-vague", "moderate", _describe(img),
                                    "Alt text is a placeholder word — describe what the image shows", "WCAG 1.1.1"))

    for a in c.links:
        text = a.text.strip().lower()
        if not text and not a.attrs.get("aria-label"):
            findings.append(Finding("link-name", "critical", _describe(a),
                                    "Link has no text or aria-label — announced as just 'link'", "WCAG 2.4.4"))
        elif text in VAGUE_LINK_TEXT:
            findings.append(Finding("link-vague", "moderate", _describe(a),
                                    f"Link text '{a.text.strip()}' is meaningless out of context", "WCAG 2.4.4"))

    for b in c.buttons:
        if not b.text.strip() and not b.attrs.get("aria-label"):
            findings.append(Finding("button-name", "critical", _describe(b),
                                    "Button has no text or aria-label", "WCAG 4.1.2"))

    for inp in c.inputs:
        if inp.attrs.get("type") in ("hidden", "submit", "button"):
            continue
        has_label = (inp.attrs.get("id") in c.label_targets or inp.attrs.get("aria-label")
                     or inp.attrs.get("aria-labelledby"))
        if not has_label:
            findings.append(Finding("input-label", "critical", _describe(inp),
                                    "Form control has no label — users don't know what to enter", "WCAG 3.3.2"))

    last_level = 0
    h1_count = 0
    for level, text in c.headings:
        if level == 1:
            h1_count += 1
        if last_level and level > last_level + 1:
            findings.append(Finding("heading-skip", "moderate", f"<h{level}> '{text[:40]}'",
                                    f"Heading jumps from h{last_level} to h{level} — outline breaks", "WCAG 1.3.1"))
        last_level = level
    if h1_count == 0 and c.headings:
        findings.append(Finding("no-h1", "moderate", "<body>",
                                "No <h1> — the page has no top-level heading", "WCAG 1.3.1"))
    if h1_count > 1:
        findings.append(Finding("multi-h1", "minor", "<body>",
                                f"{h1_count} <h1> elements — keep one main heading", "WCAG 1.3.1"))

    for el in c.positive_tabindex:
        findings.append(Finding("tabindex", "serious", el,
                                "Positive tabindex hijacks natural tab order", "WCAG 2.4.3"))
    for el in c.inline_event_handlers:
        findings.append(Finding("click-div", "serious", el,
                                "Event handler on a non-interactive element — keyboard users can't reach it",
                                "WCAG 2.1.1"))
    return findings


def score(findings: List[Finding]) -> int:
    penalty = sum(SEVERITY_WEIGHT[f.severity] for f in findings)
    return max(0, 100 - penalty)


def grade(value: int) -> str:
    return "A" if value >= 90 else "B" if value >= 75 else "C" if value >= 60 else "D" if value >= 40 else "F"


def report_markdown(findings: List[Finding]) -> str:
    s = score(findings)
    lines = [f"# Accessibility Report — score {s}/100 (grade {grade(s)})\n"]
    if not findings:
        lines.append("No issues detected by automated checks. Manual testing still required.")
    for sev in ("critical", "serious", "moderate", "minor"):
        group = [f for f in findings if f.severity == sev]
        if group:
            lines.append(f"\n## {sev.capitalize()} ({len(group)})")
            for f in group:
                lines.append(f"- **{f.rule}** [{f.wcag}] `{f.element}` — {f.message}")
    lines.append("\n---\n*Automated checks catch ~30-40% of WCAG issues. Pair with keyboard and screen-reader testing.*")
    return "\n".join(lines)
