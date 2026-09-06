from __future__ import annotations

# Business Glossary Manager - a single home for what business terms MEAN, so
# "active user" stops meaning five different things to five teams. Each term
# carries an owner, a definition, a status, synonyms, related-term links, and the
# data assets (table.column) it governs. validate() surfaces governance gaps -
# ownerless terms, missing definitions, orphans, synonym collisions, broken
# related refs, and deprecated terms still wired to live assets - and every
# finding says WHY it fired so a steward can act on it. Offline, no API keys.
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Term:
    """One business term. The glossary is a dict of these keyed by name."""

    name: str
    definition: str = ""
    owner: str = ""
    status: str = "draft"                       # draft | approved | deprecated
    synonyms: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)   # names of other terms
    assets: List[str] = field(default_factory=list)    # "table.column" refs


@dataclass
class Issue:
    """One governance finding. Typed + explainable so it's actionable."""

    issue_type: str      # ownerless | no-definition | orphan | synonym-collision
                         # | broken-related | deprecated-linked
    severity: str        # high | medium | low
    term: str
    message: str


VALID_STATUS = ("draft", "approved", "deprecated")


def _norm(s: str) -> str:
    """Case/space-insensitive key so 'Active User' == 'active  user'."""
    return " ".join(s.strip().lower().split())


class Glossary:
    """Manage business terms and validate the glossary for governance gaps."""

    def __init__(self) -> None:
        self._terms: Dict[str, Term] = {}   # keyed by normalized name

    def add(self, term: Term) -> None:
        """Add or replace a term (last write wins, keyed by normalized name)."""
        self._terms[_norm(term.name)] = term

    def get(self, name: str) -> Optional[Term]:
        """Exact lookup by name, or by any of a term's synonyms."""
        key = _norm(name)
        if key in self._terms:
            return self._terms[key]
        for t in self._terms.values():
            if key in (_norm(s) for s in t.synonyms):
                return t
        return None

    def all_terms(self) -> List[Term]:
        return sorted(self._terms.values(), key=lambda t: t.name.lower())

    def search(self, query: str) -> List[Term]:
        """Substring match over name, synonyms, and definition - the lookup UX."""
        q = _norm(query)
        if not q:
            return self.all_terms()
        hits: List[Term] = []
        for t in self.all_terms():
            haystack = " ".join([t.name, " ".join(t.synonyms), t.definition])
            if q in _norm(haystack):
                hits.append(t)
        return hits

    def validate(self) -> List[Issue]:
        """Return every governance gap, most severe first. Empty glossary -> []."""
        issues: List[Issue] = []
        names = set(self._terms.keys())

        # Map each synonym -> the term(s) that claim it, to catch collisions where
        # one word points at two different meanings (the exact "active user" trap).
        synonym_owners: Dict[str, List[str]] = {}

        for t in self.all_terms():
            # Ownerless: nobody accountable for keeping this definition true.
            if not t.owner.strip():
                issues.append(Issue(
                    "ownerless", "high", t.name,
                    "no owner - nobody is accountable for this definition",
                ))
            # No definition: a named term with no meaning is worse than no term.
            if not t.definition.strip():
                issues.append(Issue(
                    "no-definition", "high", t.name,
                    "no definition - the term exists but says nothing",
                ))
            # Orphan: not linked to any asset, and not related to any term. It
            # floats free of the data and the rest of the vocabulary.
            if not t.assets and not t.related:
                issues.append(Issue(
                    "orphan", "low", t.name,
                    "orphaned - no linked assets and no related terms; "
                    "may be unused or disconnected from the model",
                ))
            # Broken related ref: points at a term that isn't in the glossary.
            for r in t.related:
                if _norm(r) not in names:
                    issues.append(Issue(
                        "broken-related", "medium", t.name,
                        f"related term '{r}' does not exist in the glossary",
                    ))
            # Deprecated-but-linked: a retired term still governing live assets is
            # a landmine - consumers may still trust a definition you've dropped.
            if t.status == "deprecated" and t.assets:
                issues.append(Issue(
                    "deprecated-linked", "high", t.name,
                    f"deprecated but still linked to {len(t.assets)} asset(s): "
                    f"{', '.join(t.assets)} - migrate consumers before retiring",
                ))
            # Record synonym claims for the collision pass below.
            for syn in t.synonyms:
                synonym_owners.setdefault(_norm(syn), []).append(t.name)

        # Synonym collision: same synonym claimed by 2+ terms = ambiguous meaning.
        for syn, owners in synonym_owners.items():
            if len(owners) > 1:
                for owner in owners:
                    issues.append(Issue(
                        "synonym-collision", "high", owner,
                        f"synonym '{syn}' is also claimed by: "
                        f"{', '.join(o for o in owners if o != owner)} - "
                        "one word cannot mean two things",
                    ))

        order = {"high": 0, "medium": 1, "low": 2}
        return sorted(issues, key=lambda i: (order.get(i.severity, 3), i.issue_type))

    def to_markdown(self) -> str:
        """Export the whole glossary as a readable Markdown reference doc."""
        if not self._terms:
            return "# Business Glossary\n\n_No terms defined yet._\n"
        lines = ["# Business Glossary", ""]
        for t in self.all_terms():
            lines.append(f"## {t.name}")
            lines.append(f"- **Status:** {t.status}")
            lines.append(f"- **Owner:** {t.owner or '_unassigned_'}")
            lines.append(f"- **Definition:** {t.definition or '_missing_'}")
            if t.synonyms:
                lines.append(f"- **Synonyms:** {', '.join(t.synonyms)}")
            if t.related:
                lines.append(f"- **Related:** {', '.join(t.related)}")
            if t.assets:
                lines.append(f"- **Linked assets:** {', '.join(t.assets)}")
            lines.append("")
        return "\n".join(lines)


def make_sample_glossary() -> Glossary:
    """Realistic terms with PLANTED governance issues for the demo."""
    g = Glossary()
    g.add(Term(
        "Active User",
        "A user who logged in and took >=1 core action in the last 28 days.",
        owner="Growth Analytics", status="approved",
        synonyms=["Engaged User", "MAU"],
        related=["Churn", "Retention"],
        assets=["mart_users.is_active", "events.user_id"],
    ))
    g.add(Term(
        "Churn",
        "A customer with zero core actions for 28 consecutive days.",
        owner="Retention Team", status="approved",
        synonyms=["Attrition"],
        related=["Active User", "MRR"],
        assets=["mart_users.churned_flag"],
    ))
    g.add(Term(
        "MRR",
        "Monthly Recurring Revenue - normalized subscription revenue per month.",
        owner="Finance", status="approved",
        synonyms=["Monthly Recurring Revenue"],
        related=["ARR", "Churn"],          # ARR is a broken ref (not defined)
        assets=["fct_revenue.mrr_usd"],
    ))
    g.add(Term(
        "Retention",
        "Share of a cohort still active N periods after signup.",
        owner="Growth Analytics", status="approved",
        synonyms=["MAU"],                  # collides with Active User's MAU synonym
        related=["Active User"],
        assets=["mart_cohorts.retained_pct"],
    ))
    g.add(Term(
        "Signup",
        "",                                # PLANTED: no definition
        owner="",                          # PLANTED: ownerless
        status="draft",
        related=["Active User"],
        assets=["events.signup_ts"],
    ))
    g.add(Term(
        "Bounce Rate",
        "Share of sessions with a single pageview and no interaction.",
        owner="Marketing", status="deprecated",
        assets=["fct_sessions.bounced"],   # PLANTED: deprecated but still linked
    ))
    g.add(Term(
        "Blended CAC",
        "Total sales+marketing spend divided by all new customers in a period.",
        owner="Finance", status="draft",
        # PLANTED: orphan - no assets, no related terms
    ))
    return g


def _cli() -> None:
    g = make_sample_glossary()
    terms = g.all_terms()
    print("=== Business Glossary Manager ===\n")
    print(f"{len(terms)} terms:")
    for t in terms:
        print(f"  - {t.name:14s} [{t.status:10s}] owner={t.owner or '(none)'}")

    issues = g.validate()
    print(f"\n--- {len(issues)} governance issue(s) ---")
    if not issues:
        print("Clean glossary - no issues found.")
        return
    header = f"{'severity':8s}  {'issue_type':18s}  {'term':13s}  message"
    print(header)
    print("-" * len(header))
    for i in issues:
        print(f"{i.severity:8s}  {i.issue_type:18s}  {i.term:13s}  {i.message}")


if __name__ == "__main__":
    _cli()
