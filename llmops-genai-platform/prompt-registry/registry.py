from __future__ import annotations

"""Prompt Registry & Versioning.

A tiny, file-backed prompt registry that solves one pain point:
"Prompts are scattered across the codebase - no history, no diff, no rollback."

- Versioned prompt store: every commit of a prompt becomes v1, v2, v3, ...
- Content hashing (sha256, first 12 chars) so an unchanged body is never
  re-versioned (idempotent commits).
- Unified diff between any two versions of the same prompt.
- Template variable extraction ({name}-style placeholders) so callers know
  what a prompt expects before rendering.
- Safe render() that fills placeholders and refuses to render with missing vars.
- Stage promotion (draft -> staging -> production) with a
  single-production-per-prompt invariant, so one and only one version is live.

No wall-clock timestamps are stamped automatically: pass created_at explicitly
if you want one, otherwise it is stored as null (keeps demos reproducible).
"""

import difflib
import hashlib
import json
import os
import re
from typing import Optional, List, Dict


VALID_STAGES = ("draft", "staging", "production")
_VAR_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def extract_variables(body: str) -> List[str]:
    """Return the ordered, de-duplicated {placeholder} names in a prompt body."""
    seen: List[str] = []
    for match in _VAR_RE.findall(body):
        if match not in seen:
            seen.append(match)
    return seen


class PromptRegistry:
    """A minimal file-backed, versioned prompt store."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.bodies_dir = os.path.join(root, "bodies")
        self.index_path = os.path.join(root, "index.json")
        os.makedirs(self.bodies_dir, exist_ok=True)
        if not os.path.exists(self.index_path):
            self._write_index([])

    # ---- index helpers -------------------------------------------------
    def _read_index(self) -> List[Dict]:
        with open(self.index_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_index(self, records: List[Dict]) -> None:
        with open(self.index_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)

    def _body_path(self, name: str, version: int) -> str:
        return os.path.join(self.bodies_dir, name, f"v{version}.txt")

    # ---- core API ------------------------------------------------------
    def commit(
        self,
        name: str,
        body: str,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict:
        """Store `body` as the next version of prompt `name`.

        Idempotent: if the latest version has identical content, no new
        version is created and the existing record is returned unchanged.
        """
        records = self._read_index()
        existing = [r for r in records if r["name"] == name]
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]

        if existing:
            latest = max(existing, key=lambda r: r["version"])
            if latest["hash"] == content_hash:
                # Nothing changed - do not bump the version.
                return latest

        version = max((r["version"] for r in existing), default=0) + 1
        body_path = self._body_path(name, version)
        os.makedirs(os.path.dirname(body_path), exist_ok=True)
        with open(body_path, "w", encoding="utf-8") as fh:
            fh.write(body)

        record: Dict = {
            "name": name,
            "version": version,
            "stage": "draft",
            "hash": content_hash,
            "variables": extract_variables(body),
            "chars": len(body),
            "tags": list(tags) if tags else [],
            "author": author,
            "created_at": created_at,
        }
        records.append(record)
        self._write_index(records)
        return record

    def list_prompts(self, name: Optional[str] = None) -> List[Dict]:
        """Return all version records, optionally filtered by prompt name."""
        records = self._read_index()
        if name is not None:
            records = [r for r in records if r["name"] == name]
        return records

    def names(self) -> List[str]:
        """Return the distinct prompt names in the registry."""
        seen: List[str] = []
        for r in self._read_index():
            if r["name"] not in seen:
                seen.append(r["name"])
        return seen

    def get(self, name: str, version: Optional[int] = None) -> Dict:
        """Return a single record (latest version if `version` is None)."""
        records = [r for r in self._read_index() if r["name"] == name]
        if not records:
            raise KeyError(f"No prompt named {name!r} in registry")
        if version is None:
            return max(records, key=lambda r: r["version"])
        for r in records:
            if r["version"] == version:
                return r
        raise KeyError(f"No version {version} for prompt {name!r}")

    def get_body(self, name: str, version: Optional[int] = None) -> str:
        """Return the raw prompt text of a version (latest if None)."""
        record = self.get(name, version)
        with open(self._body_path(name, record["version"]), "r", encoding="utf-8") as fh:
            return fh.read()

    def production(self, name: str) -> Optional[Dict]:
        """Return the current production version of a prompt, or None."""
        for r in self._read_index():
            if r["name"] == name and r["stage"] == "production":
                return r
        return None

    def diff(self, name: str, v1: int, v2: int) -> str:
        """Return a unified diff between two versions of the same prompt."""
        left = self.get_body(name, v1).splitlines(keepends=True)
        right = self.get_body(name, v2).splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                left, right, fromfile=f"{name} v{v1}", tofile=f"{name} v{v2}"
            )
        )

    def render(self, name: str, variables: Dict, version: Optional[int] = None) -> str:
        """Fill a prompt's {placeholders} with `variables`.

        Raises KeyError listing every missing variable rather than emitting a
        half-filled prompt (a silent {typo} in a prompt is a production bug).
        """
        body = self.get_body(name, version)
        required = extract_variables(body)
        missing = [v for v in required if v not in variables]
        if missing:
            raise KeyError(
                f"Cannot render {name!r}: missing variables {missing}"
            )
        rendered = body
        for key, value in variables.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        return rendered

    def promote(self, name: str, version: int, stage: str) -> Dict:
        """Set the stage of a version, enforcing one production per prompt."""
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got {stage!r}")
        records = self._read_index()

        target: Optional[Dict] = None
        for r in records:
            if r["name"] == name and r["version"] == version:
                target = r
                break
        if target is None:
            raise KeyError(f"No version {version} for prompt {name!r}")

        if stage == "production":
            # Demote any currently-production version of the same prompt.
            for r in records:
                if (
                    r["name"] == name
                    and r["stage"] == "production"
                    and r["version"] != version
                ):
                    r["stage"] = "staging"

        target["stage"] = stage
        self._write_index(records)
        return target

    def to_frame(self):
        """Return a pandas DataFrame of all records for display."""
        import pandas as pd

        rows: List[Dict] = []
        for r in self._read_index():
            rows.append(
                {
                    "name": r["name"],
                    "version": r["version"],
                    "stage": r["stage"],
                    "hash": r["hash"],
                    "chars": r["chars"],
                    "variables": ", ".join(r["variables"]),
                    "tags": ", ".join(r["tags"]),
                    "created_at": r["created_at"],
                }
            )
        return pd.DataFrame(rows)


def _demo() -> None:
    import tempfile

    root = tempfile.mkdtemp(prefix="prompt_registry_demo_")
    reg = PromptRegistry(root)
    print(f"Registry root: {root}\n")

    # Three iterations of a support-triage prompt - each a real edit.
    v1_body = (
        "You are a support agent. Answer the customer question:\n{question}"
    )
    v2_body = (
        "You are a helpful support agent for {product}.\n"
        "Answer the customer question clearly and concisely:\n{question}"
    )
    v3_body = (
        "You are a helpful support agent for {product}.\n"
        "Rules: be concise, never invent policy, escalate billing issues.\n"
        "Customer question:\n{question}\n\nAnswer:"
    )

    for i, body in enumerate([v1_body, v2_body, v3_body], start=1):
        rec = reg.commit(
            "support_triage",
            body,
            tags=["support", "triage"],
            author="phoebe",
            created_at=f"2026-07-18T09:0{i}:00",
        )
        print(
            f"committed v{rec['version']}  hash={rec['hash']}  "
            f"vars={rec['variables']}"
        )

    # Idempotent commit: same body as v3 -> no new version.
    same = reg.commit("support_triage", v3_body)
    print(f"\nre-commit identical body -> still v{same['version']} (no bump)\n")

    print("All versions:")
    print(reg.to_frame().to_string(index=False))

    print("\nDiff v1 -> v3:")
    print(reg.diff("support_triage", 1, 3))

    # Promote v3 to production, then hotfix v4 and promote it.
    reg.promote("support_triage", 3, "production")
    v4_body = v3_body.replace("escalate billing issues", "escalate billing and refund issues")
    rec4 = reg.commit("support_triage", v4_body, created_at="2026-07-18T11:00:00")
    reg.promote("support_triage", rec4["version"], "production")

    prod = [
        r for r in reg.list_prompts("support_triage") if r["stage"] == "production"
    ]
    assert len(prod) == 1, f"invariant broken: {len(prod)} production versions"
    print(
        f"Single-production invariant holds: exactly 1 live version "
        f"(v{prod[0]['version']})."
    )

    print("\nRendered production prompt:")
    print(
        reg.render(
            "support_triage",
            {"product": "Acme Cloud", "question": "How do I reset my password?"},
        )
    )

    # Missing-variable guard.
    try:
        reg.render("support_triage", {"product": "Acme Cloud"})
    except KeyError as exc:
        print(f"\nRender guard works: {exc}")


if __name__ == "__main__":
    _demo()
