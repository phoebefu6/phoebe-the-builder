"""Tests for the four guarantees. Run: python3 test_dsar.py"""

from __future__ import annotations

from datetime import date

from dsar import (
    MENTION,
    RETAIN,
    SPEC_BY_NAME,
    SUBJECT_EMAIL,
    WITHHELD,
    build_corpus,
    coverage,
    disclosure_pack,
    erasure_plan,
    extract,
    naive_extract,
    normalize_email,
    redact,
    resolve_identity,
    weak_link_cost,
)

CORPUS = build_corpus()
IDENT = resolve_identity(CORPUS, SUBJECT_EMAIL)
HITS = extract(CORPUS, IDENT)

checks = 0
failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if condition:
        print(f"  PASS  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


print("\n1. Normalization: same mailbox, different spellings")
check(
    "case-folded",
    normalize_email("Amara.Osei@Example.com") == "amara.osei@example.com",
)
check(
    "plus-tag stripped",
    normalize_email("amara.osei+support@example.com") == "amara.osei@example.com",
)
check(
    "dots NOT stripped (would merge two corporate mailboxes)",
    normalize_email("a.osei@corp.example") != normalize_email("aosei@corp.example"),
)

print("\n2. Completeness: the resolved extract is a strict superset of the naive one")
naive = naive_extract(CORPUS, SUBJECT_EMAIL)
full = {(h.table, h.pk) for h in HITS}
check("no row lost", not (naive - full), f"naive-only rows: {sorted(naive - full)[:5]}")
check("rows recovered", len(full - naive) > 0, "resolution found nothing extra")
check("free-text rows found", any(h.how == MENTION for h in HITS))

print("\n3. Abstention: weak links are never joined on")
weak_ids = {k.value for k in IDENT.weak if k.key_type == "customer_id"}
check("a weak link exists to abstain on", bool(weak_ids))
leaked = [
    h for h in HITS if str(h.row.get("customer_id") or "") in weak_ids
]
check("no weak-linked row in the extract", not leaked, f"leaked: {[h.pk for h in leaked]}")
cost = weak_link_cost(CORPUS, IDENT)
check("abstention has a measured cost", cost.get("orders", 0) > 0, str(cost))

print("\n4. Minimization: no reference-table row is disclosed")
ref_tables = {t for t, s in SPEC_BY_NAME.items() if s.category == "reference"}
check(
    "no products in the pack",
    not (ref_tables & {h.table for h in HITS}),
    f"reference rows present: {ref_tables & {h.table for h in HITS}}",
)

print("\n5. Third-party withholding: no other person's identifier survives redaction")
pack = disclosure_pack(CORPUS, IDENT, HITS)
subject_emails = IDENT.values("email")
subject_customers = IDENT.values("customer_id")
shared = [h for h in HITS if h.shared]
check("shared rows exist to redact", bool(shared))

residual = []
for hit in shared:
    spec = SPEC_BY_NAME[hit.table]
    out = redact(hit, IDENT, CORPUS)
    for key in spec.keys:
        raw = out.get(key.column)
        if raw in (None, "", WITHHELD):
            continue
        if key.key_type == "email" and normalize_email(str(raw)) not in subject_emails:
            residual.append(f"{hit.table}.{hit.pk}.{key.column}={raw}")
        if key.key_type == "customer_id" and str(raw) not in subject_customers:
            residual.append(f"{hit.table}.{hit.pk}.{key.column}={raw}")
    for col in spec.text_cols:
        for other in (
            m
            for m in __import__("re").findall(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", str(out.get(col) or "")
            )
            if normalize_email(m) not in subject_emails
        ):
            if normalize_email(other) in {
                normalize_email(str(c["email"])) for c in CORPUS["customers"]
            }:
                residual.append(f"{hit.table}.{hit.pk}.{col}~{other}")
check("nothing third-party left in the pack", not residual, f"residual: {residual[:5]}")

print("\n6. The subject's own data is never redacted away")
own = [h for h in HITS if h.table == "customers"][0]
check("subject email intact", redact(own, IDENT, CORPUS)["email"] == own.row["email"])

print("\n7. Access is not erasure: some disclosed rows cannot be deleted")
plan = erasure_plan(HITS, CORPUS, today=date(2026, 8, 3))
retained = [a for a in plan if a.action == RETAIN]
check("plan covers every disclosed row", len(plan) == len(HITS), f"{len(plan)} vs {len(HITS)}")
check("some rows are retention-blocked", bool(retained), "nothing blocked")
check("every blocked row cites a basis", all(a.basis for a in retained))
old_orders_deleted = [
    a
    for a in plan
    if a.table == "orders" and a.action != RETAIN
]
check("orders past the 7-year window are deletable", bool(old_orders_deleted))

print("\n8. Determinism")
check("same corpus twice", build_corpus()["orders"][0] == build_corpus()["orders"][0])
c1 = coverage(CORPUS, HITS)
c2 = coverage(build_corpus(), extract(build_corpus(), resolve_identity(build_corpus(), SUBJECT_EMAIL)))
check("same coverage twice", c1 == c2, f"{c1} vs {c2}")

print(f"\n{checks - len(failures)}/{checks} checks passed")
if failures:
    raise SystemExit("FAILED:\n" + "\n".join(failures))
print("Coverage:", coverage(CORPUS, HITS))
