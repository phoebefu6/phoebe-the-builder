"""Generate demo.ipynb for dsar-extractor. Run once, then nbconvert --execute."""

from __future__ import annotations

import json
import pathlib

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def md(src: str) -> None:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src: str) -> None:
    nb["cells"].append(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
    )


BASE = "data-quality-governance/dsar-extractor"

ENGINE = pathlib.Path("dsar.py").read_text()
ENGINE = ENGINE.split('if __name__ == "__main__":')[0].rstrip() + "\n"

md(f"""# DSAR Extractor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/{BASE}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath={BASE}/demo.ipynb)

> A subject access request arrives with one email address. Finding the rows is easy. Finding
> *all* of them, and *only* them, is the actual job - and getting it wrong fails in two
> opposite directions, both of which are breaches.

**What this covers**

1. The request, and the four things wrong with `WHERE email = ?`
2. One person, many keys - identity resolution with an evidence trail
3. **Abstention**: the weak link the tool refuses to join on, and what including it would have disclosed
4. Scope: why following every foreign key hands over the product catalog
5. Third-party withholding - Art. 15(4), the clause that makes a disclosure a breach
6. The rows no key join reaches at all
7. Access is not erasure: what you must disclose but may not delete
8. Measured: under-collection, over-collection, and the retention wall
9. Try your own subject map

*Fully offline - the 11-table warehouse is generated, no database and no real personal data.*""")

md("""## 1. The request

> *"Please provide a copy of all personal data you hold about me. My email is
> `amara.osei@example.com`."*

Eleven tables. The obvious query is one line:

```sql
SELECT * FROM customers WHERE email = 'amara.osei@example.com';
```

Four separate things are wrong with it, and each one has a name in the regulation.
Start with the smallest: the address is not stored the way it was typed.""")

code('''import re

def normalize_email(value):
    """Case-fold and strip plus-addressing - both are the same mailbox everywhere."""
    value = value.strip().lower()
    if "@" not in value:
        return value
    local, _, domain = value.partition("@")
    return f"{local.split('+', 1)[0]}@{domain}"

requested = "amara.osei@example.com"
stored = [
    "Amara.Osei@Example.com",          # customers.email, display-cased at signup
    "amara.osei+support@example.com",  # support_tickets, she tags her own mail
    "amara.osei+shop@example.com",     # support_tickets, a different tag
    "amara.osei@example.com",          # marketing_contacts, lowercased on import
]

print(f"{'stored spelling':<34} {'== requested?':<14} same mailbox?")
print("-" * 66)
for value in stored:
    exact = value == requested
    same = normalize_email(value) == normalize_email(requested)
    print(f"{value:<34} {str(exact):<14} {same}")''')

md("""Three of the four are the same mailbox and would not match. A case-insensitive
comparison rescues one of them. Nothing about `=` rescues the plus-addressed pair.

One caution, because the fix has a failure mode of its own: this normalizer does **not**
strip dots from the local part. Dot-insensitivity is a Gmail convention. Apply it to a
corporate domain and you have just merged two different employees into one data subject -
and the DSAR response now mails one person's records to the other.""")

code('''print("gmail convention applied to a corporate domain:")
print("  a.osei@corp.example  vs  aosei@corp.example")
print("  -> two different people. Do not merge them.")
print()
print("what this normalizer does:", normalize_email("A.Osei+billing@Corp.Example"))''')

md("""## 2. One person, many keys

The subject exists in the warehouse under four key shapes: an email (in three spellings),
a `customer_id`, a phone number, and two `anon_id` values from before she ever logged in.

Resolution walks these outward from the seed and records **why** each key is believed to
be the same person. That evidence column is not decoration - a DSAR response has to be
defensible, and "the join found it" is not a defence.

Here is the whole engine. It is plain Python: no database, no network, no dependencies
beyond the standard library.""")

code(ENGINE)

md("""### Build the warehouse and resolve the subject

Eleven tables, deterministic, no real personal data.""")

code('''corpus = build_corpus()

print(f"{'table':<20} {'rows':>6}   scope")
print("-" * 44)
for spec in SUBJECT_MAP:
    print(f"{spec.name:<20} {len(corpus[spec.name]):>6}   {spec.category}")
print(f"\\n{sum(len(v) for v in corpus.values()):>26} rows total")''')

code('''ident = resolve_identity(corpus, SUBJECT_EMAIL)

print("RESOLVED IDENTITY\\n")
print(f"{'use':<22} {'strength':<9} {'key_type':<12} {'value':<26} evidence")
print("-" * 122)
for key in ident.keys:
    use = "join" if key.usable else "HOLD - human decision"
    print(f"{use:<22} {key.strength:<9} {key.key_type:<12} {key.value:<26} {key.evidence}")

print("\\nSTORED SPELLINGS OF THE SAME MAILBOX")
for note in ident.variants:
    print(f"   {note}")''')

md("""Two things worth pausing on.

**The `anon_id` keys are tagged `linked`, not `exact`.** They come from a device-stitching
table - a claim the controller made, not an observed fact. They are deterministic enough to
join on, but the strength label is what lets a reviewer challenge them later.

**One key is on hold.** Which brings us to the mechanism most extractors skip.

## 3. Abstention: the link the tool refuses to use

There is a second Amara Osei in the customer table. Same name, different mailbox, different
person. Every fuzzy identity-resolution library on the shelf will happily match them.""")

code('''same_name = [c for c in corpus["customers"] if c["full_name"] == "Amara Osei"]
for row in same_name:
    print(f"{row['customer_id']}  {row['full_name']:<14} {row['email']:<30} joined {row['created_at']}")

print()
for key in ident.weak:
    print(f"HELD BACK: {key.key_type}={key.value}")
    print(f"           {key.evidence}")''')

md("""The tool records the candidate and refuses to join on it. Here is the cost of the
alternative - what an automatic name match would have posted to the requester:""")

code('''cost = weak_link_cost(corpus, ident)
print("If the weak link had been auto-included, this DSAR response would have contained:\\n")
for k, v in cost.items():
    if k != "other_people":
        print(f"   {v:>3} {k}")
print(f"\\n...belonging to {cost['other_people']} completely different person.")
print("\\nUnder-collection annoys the requester. Over-resolution is a personal data breach.")''')

md("""## 4. Scope: relatedness is not aboutness

Now the traversal. The subject's orders are hers, so the line items on those orders are
hers too - that is a **subject-scoped** edge, declared in the map:

```python
TableSpec("order_items", "order_item_id",
          scoped_via=ScopedVia("order_id", "orders", "order_id"))
```

But `order_items.product_id` points at the product catalog, and a product row is not
personal data about anybody. It is *reachable* from her data, which is a different claim.
That edge is declared `reference` and the walk stops there.

Watch what happens when it does not.""")

code('''hits = extract(corpus, ident)
sweep = naive_fk_sweep(corpus, hits)

print("Following every foreign key outward from her rows also collects:\\n")
print(f"   {sweep['products']:>4} product rows   (not personal data about anyone)")
print()
print("And one step further - joining those products back to order_items:\\n")
print(f"   {sweep['_reverse_order_items']:>4} line items")
print(f"   {sweep['_reverse_customers']:>4} OTHER CUSTOMERS whose purchases are now in the export")
print("\\nThis is the mechanism behind most 'we sent the wrong data' DSAR incidents.")''')

md("""## 5. Third-party withholding

GDPR Art. 15(4): the right to obtain a copy *"shall not adversely affect the rights and
freedoms of others."* Singapore's PDPA s.21(3) and Schedule 5 carve out the same thing.

The awkward part is that some rows genuinely belong to two people at once:

- a support thread another customer replied inside
- a gift order carrying the recipient's address
- a referral, where one side is the subject and the other side is not

You cannot drop these - they are her data too. You cannot ship them raw either.""")

code('''shared = [h for h in hits if h.shared]
print(f"{len(shared)} of {len(hits)} disclosed rows contain another living person.\\n")
for hit in shared:
    print(f"{hit.table:<17} {hit.pk:<8} third party in: {', '.join(hit.third_parties)}")''')

md("""Redaction replaces the other person's identifiers - including ones sitting inside free
text - and leaves the subject's own values untouched:""")

code('''example = [h for h in shared if h.table == "ticket_messages"][0]
print("BEFORE (raw row):")
for k, v in example.row.items():
    print(f"   {k:<14} {v}")

print("\\nAFTER (safe to disclose):")
for k, v in redact(example, ident, corpus).items():
    print(f"   {k:<14} {v}")''')

code('''# The referral where she is the referee: the other party is withheld, she is not.
ref = [h for h in shared if h.table == "referrals"]
for hit in ref:
    print(f"{hit.pk}:")
    for k, v in redact(hit, ident, corpus).items():
        print(f"   {k:<24} {v}")
    print()''')

md("""## 6. The rows no join reaches

The last gap is not a key problem. Somebody wrote her email address into the body of
*someone else's* support ticket. It is personal data about her, held by the controller,
and squarely within Art. 15 - and no join on any key will ever find it.""")

code('''mentions = [h for h in hits if h.how == MENTION]
print(f"{len(mentions)} rows reachable only by sweeping free text:\\n")
for hit in mentions:
    print(f"{hit.table} {hit.pk}  (ticket {hit.row['ticket_id']}, author {hit.row['author_email']})")
    print(f"   why:  {hit.reason}")
    print(f"   body: {redact(hit, ident, corpus)['body']}")
    print()''')

md("""Note the second one: *"Refund was actually issued to amara.osei@example.com in error."*
That is the kind of row a requester most wants and a key-join extract never returns.

The text sweep is the crudest mechanism here and the one most likely to need tuning - it
matches on resolved identifiers, not on names, precisely because name matching is what
Section 3 abstained from.

## 7. Access is not erasure

If the request is an *erasure* request rather than an access one, the answer is a different
set of rows. Art. 17(3)(b) yields to a legal obligation, and seven years of accounting
records outrank the erasure right.

So the plan gives every disclosed row an action and a basis.""")

code('''from collections import Counter

plan = erasure_plan(hits, corpus)
counts = Counter(a.action for a in plan)

print(f"{'action':<34} rows")
print("-" * 42)
for action, n in counts.most_common():
    print(f"{action:<34} {n:>4}")

blocked = [a for a in plan if a.action == RETAIN]
print(f"\\n{len(blocked)} of {len(hits)} disclosed rows CANNOT be deleted.\\n")
for action in blocked[:6]:
    print(f"   {action.table:<13} {action.pk:<9} {action.basis}")''')

md("""Two design notes in that output.

`anonymize` rather than `delete` for behavioural events: nulling the identifier satisfies
erasure while leaving the aggregate intact. Deleting the rows outright silently restates
your historical traffic numbers.

`redact subject fields, keep row` for shared records: deleting a thread the subject shares
with another customer would erase *their* data to satisfy *her* request.

And the trap: telling a requester "your data has been deleted" when 19 rows are sitting
under a statutory hold is its own compliance failure. The plan names the release date.

## 8. Measured

Both failure directions, on one 77-row request.""")

code('''cov = coverage(corpus, hits)
naive = naive_extract(corpus, SUBJECT_EMAIL)

print("COVERAGE")
print(f"   ordinary query (case-insensitive email + customer_id join)  {cov['naive']:>4} rows")
print(f"   resolved extract                                            {cov['resolved']:>4} rows")
print(f"   missed by the ordinary query                                {cov['missed_by_naive']:>4} rows"
      f"  ({cov['missed_by_naive'] / cov['resolved']:.0%})")
print()
print("The baseline is not a strawman - it is what most controllers actually run.")
print("It still misses:")
missed = [h for h in hits if (h.table, h.pk) not in naive]
by_table = Counter(h.table for h in missed)
for table, n in by_table.most_common():
    print(f"   {n:>3} {table}")''')

code('''%matplotlib inline
import matplotlib.pyplot as plt

INK, UNDER, OVER, OK, MUTED = "#1d2433", "#c2410c", "#1d4ed8", "#0f766e", "#94a3b8"
cost = weak_link_cost(corpus, ident)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
fig.suptitle("One subject access request: both failure directions are quantifiable",
             fontsize=14, fontweight="bold", color=INK, y=0.99)

by_mech = {"plus-addressed\\nmailbox": 0, "pre-login\\nanon_id": 0, "free-text\\nmention": 0}
for hit in missed:
    if hit.how == MENTION:
        by_mech["free-text\\nmention"] += 1
    elif hit.table == "web_events":
        by_mech["pre-login\\nanon_id"] += 1
    else:
        by_mech["plus-addressed\\nmailbox"] += 1

ax = axes[0]
bars = ax.bar(list(by_mech), list(by_mech.values()), color=UNDER, width=0.6)
ax.bar_label(bars, fmt="%d", padding=3, fontsize=11, fontweight="bold", color=INK)
ax.set_title(f"UNDER-COLLECTION\\n{cov['naive']} rows found, {cov['resolved']} actually held",
             fontsize=11, color=UNDER, fontweight="bold", pad=10)
ax.set_ylabel("rows the requester would never have seen")
ax.set_ylim(0, max(by_mech.values()) * 1.28)

ax = axes[1]
over = {
    "reference rows\\n(products)": sweep["products"],
    "one reverse join\\n(other people's items)": sweep["_reverse_order_items"],
    "weak name link\\n(a stranger's rows)": cost["orders"] + cost["payments"] + cost["order_items"],
}
bars = ax.barh(list(over), list(over.values()), color=[MUTED, OVER, OVER], height=0.55)
ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold", color=INK)
ax.set_title(f"OVER-COLLECTION\\nrows a naive traversal adds - {sweep['_reverse_customers']} "
             "other customers exposed", fontsize=11, color=OVER, fontweight="bold", pad=10)
ax.set_xlabel("rows wrongly included")
ax.set_xlim(0, max(over.values()) * 1.3)
ax.invert_yaxis()

ax = axes[2]
order = [a for a, _ in counts.most_common()]
bars = ax.barh([a.replace(" - ", "\\n") for a in order], [counts[a] for a in order],
               color=[UNDER if a == RETAIN else OK for a in order], height=0.55)
ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold", color=INK)
ax.set_title(f"ACCESS IS NOT ERASURE\\n{counts[RETAIN]} of {len(hits)} disclosed rows cannot be deleted",
             fontsize=11, color=INK, fontweight="bold", pad=10)
ax.set_xlabel("rows")
ax.set_xlim(0, max(counts.values()) * 1.3)
ax.invert_yaxis()

for i, ax in enumerate(axes):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y" if i == 0 else "x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)

fig.text(0.5, 0.015,
         f"{cov['shared']} disclosed rows contain another living person and are redacted before the pack ships.",
         ha="center", fontsize=9.5, color=INK, style="italic")
fig.tight_layout(rect=(0, 0.045, 1, 0.94))
plt.show()''')

md("""### The per-table pack

What actually ships, and how each row was reached.""")

code('''print(summarize(corpus, ident, hits))''')

md("""## 9. Try your own subject map

The engine is generic; the map is the part you replace. For a real warehouse, declare each
table once:

- **which columns identify the subject** (`keys`) and which name the *other* party (`role="counterparty"`)
- **how the table inherits scope** (`scoped_via`) - the only edges the walk follows
- **whether it is reference data** (`category="reference"`) - the walk stops
- **which free-text columns to sweep** (`text_cols`)
- **what erasure means for it** (`delete` / `anonymize` / `redact`) and any statutory `retention`

Everything else - resolution, abstention, redaction, the erasure plan - follows from that
declaration.""")

code('''# ---------------------------------------------------------------------------------
# Sketch of a real map. Uncomment, point it at your own tables, and the four
# mechanisms come along unchanged.
# ---------------------------------------------------------------------------------
#
# MY_MAP = (
#     TableSpec("dim_customer", "customer_sk",
#               keys=(KeyCol("email_address", "email"),
#                     KeyCol("customer_sk", "customer_id"),
#                     KeyCol("mobile", "phone"))),
#
#     TableSpec("fct_invoice", "invoice_id",
#               keys=(KeyCol("customer_sk", "customer_id"),),
#               fks=(("customer_sk", "dim_customer", "customer_sk"),),
#               retention=Retention("Tax: 7 years from invoice date", 7, "invoice_date")),
#
#     TableSpec("dim_product", "product_sk", category="reference"),   # walk stops here
#
#     TableSpec("crm_note", "note_id",
#               keys=(KeyCol("author_email", "email"),),
#               text_cols=("note_body",),                            # swept for mentions
#               erasure="redact"),
# )
#
# Then, on your own data:
#
# my_corpus = {"dim_customer": rows_from_your_query, ...}
# ident = resolve_identity(my_corpus, "requester@example.com")
#
# # Review the abstained links BEFORE extracting - this is a human decision.
# for key in ident.weak:
#     print("needs a decision:", key.key_type, key.value, "-", key.evidence)
#
# hits = extract(my_corpus, ident)
# pack = disclosure_pack(my_corpus, ident, hits)       # redacted, safe to send
# plan = erasure_plan(hits, my_corpus)                 # a different set of rows
#
# # Ship nothing that still carries a third party.
# assert not any(h.shared and "@" in str(redact(h, ident, my_corpus)) for h in [])''')

md(f"""---

**Streamlit version** - change the identifier, move the retention date, and watch the
disclosure pack, the withheld rows and the erasure plan change:

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Tests** - the four guarantees, asserted:

```bash
python3 test_dsar.py
```

Nothing here is legal advice. Article and section numbers are cited so a privacy counsel
can check the mapping against your own obligations.

Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder) · [`{BASE}`](https://github.com/phoebefu6/phoebe-the-builder/tree/main/{BASE})""")

pathlib.Path("demo.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote demo.ipynb ({len(nb['cells'])} cells)")
