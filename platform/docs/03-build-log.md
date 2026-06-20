# 03 - Build Log (design-thinking journal)

> A running diary of what we built, what we learned, and what's next. Newest entry
> at the top. This is where Phoebe's questions and the "aha" moments get captured -
> not just the answers.

---

## 2026-06-20 - Foundation: the wiki itself

**What we did:** Set up `platform/` and the docs/wiki. Wrote the glossary, the "why"
doc, the architecture picture, and ADR-0001 (governed shell, not Databricks clone).
No platform code yet - on purpose. We're laying the map before driving.

**What Phoebe was learning:** the vocabulary. Coming from a data background, terms
like *control plane*, *gateway*, *RBAC*, *audit log*, *connector layer* were
unfamiliar. The glossary translates each into a data analogy (RBAC = database roles,
token = warehouse session, audit log = immutable query history).

**Key decisions logged:**
- [ADR-0001](decisions/adr-0001-governed-shell-not-databricks.md) - build the thin
  governance shell, mount open source for compute.

**Mentor input:** Cassie (log every decision as an ADR), Karpathy (write the smallest
readable version ourselves before importing frameworks), Ng (learn in sequence, check
understanding at each step), Brené ("I'm a data expert learning platform engineering"
is a strength).

**Open questions to revisit:**
- Which client / wedge problem do we target first? (Leaning: governed analytics for
  the analyst persona - to be decided.)
- Local file-based store for the MVP, or a real database from the start? (Leaning:
  start file-based / SQLite so it's readable, swap later.)

**Next step:** Step 1 - the **gateway with login** (authentication). We'll write the
smallest real login Phoebe can read top to bottom, with a companion explainer doc
`10-gateway-login.md`.

---

## Template for future entries

```
## YYYY-MM-DD - <component / step>

**What we did:**
**What Phoebe was learning:**
**Key decisions logged:** (link ADRs)
**Mentor input:**
**Open questions to revisit:**
**Next step:**
```
