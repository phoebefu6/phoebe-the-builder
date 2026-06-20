from __future__ import annotations

"""Turn a repo profile into a README.

Two paths:
- `render_readme_template` - deterministic, offline, zero-dependency. Always works.
- `generate_readme_claude` - richer prose via the Claude API when an
  `ANTHROPIC_API_KEY` is set. The SDK is imported lazily so the template path
  (and the demo notebook / CI) never need the package or a key.
"""

import os
from typing import Dict, List, Optional

CLAUDE_MODEL = "claude-opus-4-8"


def render_readme_template(profile: Dict[str, object]) -> str:
    """Build a solid README from the profile with no LLM call."""
    name = profile.get("name", "project")
    lang = profile.get("primary_language", "Unknown")
    deps: List[str] = list(profile.get("dependencies", []))  # type: ignore[arg-type]
    entries: List[str] = list(profile.get("entry_points", []))  # type: ignore[arg-type]

    lines: List[str] = [f"# {name}", ""]
    lines.append(f"> A {lang} project with {profile.get('file_count', 0)} files.")
    lines.append("")

    lines.append("## Languages")
    lang_counts: Dict[str, int] = profile.get("language_counts", {})  # type: ignore[assignment]
    if lang_counts:
        for lg, n in lang_counts.items():
            lines.append(f"- {lg}: {n} file(s)")
    else:
        lines.append("- Not detected")
    lines.append("")

    lines.append("## Getting Started")
    if lang == "Python":
        lines.append("```bash")
        lines.append("pip install -r requirements.txt")
        if entries:
            lines.append(f"python {entries[0]}")
        lines.append("```")
    elif lang in {"JavaScript", "TypeScript"}:
        lines.append("```bash")
        lines.append("npm install")
        lines.append("npm start")
        lines.append("```")
    else:
        lines.append("_Add build/run instructions here._")
    lines.append("")

    if entries:
        lines.append("## Entry Points")
        for e in entries:
            lines.append(f"- `{e}`")
        lines.append("")

    if deps:
        lines.append("## Dependencies")
        for d in deps[:25]:
            lines.append(f"- `{d}`")
        lines.append("")

    if profile.get("has_dockerfile"):
        lines.append("## Docker")
        lines.append("```bash")
        lines.append(f"docker build -t {name} .")
        lines.append(f"docker run --rm {name}")
        lines.append("```")
        lines.append("")

    lines.append("## Project Status")
    lines.append(f"- Tests present: {'yes' if profile.get('has_tests') else 'no'}")
    lines.append(f"- Containerized: {'yes' if profile.get('has_dockerfile') else 'no'}")
    lines.append("")
    lines.append("_README scaffolded by auto-readme. Edit freely._")
    return "\n".join(lines) + "\n"


def _build_prompt(profile: Dict[str, object]) -> str:
    return (
        "Write a concise, professional README.md for this repository. "
        "Use the structured profile below. Include: a one-line description, a "
        "Features section inferred from the files/dependencies, a Getting Started "
        "section with real install/run commands for the primary language, and a "
        "short Project Structure note. Output only Markdown.\n\n"
        f"Repository profile:\n{profile}"
    )


def generate_readme_claude(profile: Dict[str, object], api_key: Optional[str] = None) -> str:
    """Generate a README via the Claude API. Falls back to the template if no key."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return render_readme_template(profile)

    # Lazy import so the template path never requires the SDK.
    from anthropic import Anthropic

    client = Anthropic(api_key=key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": _build_prompt(profile)}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")
