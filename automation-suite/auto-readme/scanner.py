from __future__ import annotations

"""Scan a repository into a structured profile.

The profile is the deterministic, offline half of the tool: it walks the tree,
detects the dominant language, parses dependency manifests, and guesses entry
points. The generator (template or Claude) turns that profile into prose.
"""

import json
from pathlib import Path
from typing import Dict, List

LANG_BY_EXT = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".go": "Go",
    ".rs": "Rust", ".java": "Java", ".rb": "Ruby", ".sh": "Shell",
    ".ipynb": "Jupyter Notebook",
}

IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".ipynb_checkpoints"}
ENTRY_HINTS = ("main.py", "app.py", "api.py", "cli.py", "index.js", "server.js", "manage.py")


def _iter_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and not any(part in IGNORE_DIRS for part in p.parts):
            out.append(p)
    return out


def _parse_python_deps(root: Path) -> List[str]:
    req = root / "requirements.txt"
    if not req.exists():
        return []
    deps = []
    for line in req.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            deps.append(line)
    return deps


def _parse_node_deps(root: Path) -> List[str]:
    pkg = root / "package.json"
    if not pkg.exists():
        return []
    try:
        data = json.loads(pkg.read_text())
    except json.JSONDecodeError:
        return []
    return sorted(list(data.get("dependencies", {}).keys()))


def scan_repo(path: str) -> Dict[str, object]:
    """Return a structured profile of the repo at `path`."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"no such path: {path}")

    files = _iter_files(root)
    ext_counts: Dict[str, int] = {}
    for f in files:
        ext_counts[f.suffix] = ext_counts.get(f.suffix, 0) + 1

    lang_counts: Dict[str, int] = {}
    for ext, n in ext_counts.items():
        lang = LANG_BY_EXT.get(ext)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + n

    primary_language = max(lang_counts, key=lang_counts.get) if lang_counts else "Unknown"

    entry_points = sorted(
        str(f.relative_to(root)) for f in files if f.name in ENTRY_HINTS
    )

    deps = _parse_python_deps(root) or _parse_node_deps(root)

    has_dockerfile = (root / "Dockerfile").exists()
    has_tests = any("test" in f.name.lower() for f in files)

    return {
        "name": root.resolve().name,
        "primary_language": primary_language,
        "language_counts": dict(sorted(lang_counts.items(), key=lambda x: -x[1])),
        "file_count": len(files),
        "entry_points": entry_points,
        "dependencies": deps,
        "has_dockerfile": has_dockerfile,
        "has_tests": has_tests,
        "top_files": sorted(str(f.relative_to(root)) for f in files)[:20],
    }
