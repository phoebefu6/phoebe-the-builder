"""Generate demo.ipynb for type-inferencer. Run once, then nbconvert --execute."""

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


BASE = "data-engineering-pro/type-inferencer"

ENGINE = pathlib.Path("type_infer.py").read_text()
ENGINE = ENGINE.split('if __name__ == "__main__":')[0].rstrip() + "\n"

md(f"""# Type Inferencer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/{BASE}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath={BASE}/demo.ipynb)

> Every column landed as text. Typing them is easy. Typing them *without silently corrupting data* is the actual job.

**What this covers**

1. The problem: two casts that succeed and destroy data anyway
2. Rule one - **text-preserving**: the render must reproduce the source text
3. Rule two - **value-preserving**: the stored number must equal the number written
4. The third mechanism - **abstention**, for what no rule can settle
5. The engine, and the DDL it emits
6. The findings report: every column it refused to type, and why
7. Benchmark: all-TEXT vs a naive caster vs this, with `lossy` **measured** on the full column
8. Try your own CSV

*Fully offline - the corpus is generated, no database and no network needed.*""")

md("""## 1. Two casts that succeed and destroy data anyway

A CSV lands. Every column is a string. Somebody has to write the `CREATE TABLE`, and the usual
approach is: try `int()`, then `float()`, then a date format, else text. It runs clean. It also
does this:""")

code('''raw_zip = "01234"
print(f"zip:   {raw_zip!r}  ->  int  ->  {int(raw_zip)!r}   ... the identifier is now wrong")

raw_price = "10.10"
print(f"price: {raw_price!r}  ->  float ->  {float(raw_price)!r}  ... looks fine, so check the bits:")

from decimal import Decimal
print(f"       the float actually stores {Decimal(float(raw_price))}")
print(f"       you wrote                 {Decimal(raw_price)}")
print(f"       equal? {Decimal(float(raw_price)) == Decimal(raw_price)}")''')

md("""No exception is raised in either case. The load succeeds, the dashboard renders, and the
defect surfaces months later as a mail-merge to the wrong postcode or a reconciliation that is
off by cents across ten million rows.

The fix is not a better parser. It is a **precondition**: cast only when the cast is reversible.""")

md("""## 2. Rule one - text-preserving

For a string column becoming a narrow type, require that rendering the stored value reproduces
the source text **exactly**. One line, and the leading-zero trap is gone - along with `"+5"` and
`"1.0"`, which both cast to integers and both lose something on the way.""")

code('''def int_roundtrips(raw: str) -> bool:
    """str(int(raw)) must give back exactly what was in the file."""
    try:
        return str(int(raw)) == raw
    except ValueError:
        return False


for v in ["1234", "01234", "+5", "1.0", "-42", "9007199254740993"]:
    verdict = "INTEGER" if int_roundtrips(v) else "stays text"
    print(f"  {v!r:22} {verdict}")

# Surrounding whitespace is the one documented exception: it is trimmed before inference,
# because it is a formatting artifact - but the engine files a warning saying the inferred
# type only holds if the loader trims on ingest too.
print(f"  {' 5'!r:22} trimmed first, with a warning - see `whitespace-padded` below")''')

md("""## 3. Rule two - value-preserving

For numbers, require that the stored value equals the decimal number written in the file. A
binary float almost never does: `0.075`, `10.10` and `0.1` all land on a *nearby* binary
fraction. `DECIMAL(p,s)` stores them exactly.

This is why the rule matters more than the taste argument: "use DECIMAL for money" is advice you
can ignore, while a failing round-trip is a fact you cannot.""")

code('''from decimal import Decimal


def double_is_value_preserving(raw: str) -> bool:
    """A float stores `raw` exactly only if its binary expansion equals the decimal written."""
    return Decimal(float(raw)) == Decimal(raw)


for v in ["10.10", "0.075", "0.5", "3.25", "0.1"]:
    ok = double_is_value_preserving(v)
    print(f"  {v!r:9} as DOUBLE -> {'exact' if ok else 'DRIFTS to ' + str(Decimal(float(v)))[:28] + '...'}")''')

md("""`0.5` and `3.25` survive - they are exact binary fractions. Everything else drifts. So the
ladder reaches `DOUBLE PRECISION` only where a column genuinely wants it, and money never
silently becomes a float.""")

md("""## 4. The third mechanism - abstention

Some questions the data cannot answer, and both rules stay silent on them:

| Column | Why no rule settles it |
|---|---|
| `03/04/2026` where every component is <= 12 | DD/MM and MM/DD both parse, and they disagree about which day it is. The text round-trips either way. |
| a column of only `0` and `1` | flag, or a count that never exceeded one? |
| `"1,234.50"` | numeric in intent, not in form - typing it means rewriting the source text |

A parser guesses. This tool refuses and files a finding. The disambiguation it *can* do is
evidence-based: if any component exceeds 12, the order is settled by the data itself.""")

code('''import re

SLASH = re.compile(r"^(\\d{1,2})/(\\d{1,2})/(\\d{4})$")


def date_order(values):
    parts = [SLASH.match(v) for v in values]
    if not all(parts):
        return "not a slash date"
    day_first = any(int(p.group(1)) > 12 for p in parts)
    month_first = any(int(p.group(2)) > 12 for p in parts)
    if day_first and month_first:
        return "BLOCK - the column mixes both orders"
    if day_first:
        return "DATE as %d/%m/%Y  (a first component > 12 settles it)"
    if month_first:
        return "DATE as %m/%d/%Y  (a second component > 12 settles it)"
    return "ABSTAIN - both orders parse, kept as text"


print(date_order(["03/04/2026", "11/12/2026", "07/08/2026"]))
print(date_order(["25/04/2026", "03/11/2026"]))
print(date_order(["03/25/2026", "11/03/2026"]))
print(date_order(["25/04/2026", "03/25/2026"]))''')

md("""## 5. The engine

The whole inference, in one cell so the notebook runs standalone. It is the same file as
[`type_infer.py`](type_infer.py) in the repo - the ladder (BOOLEAN -> integer widths -> temporal
-> DECIMAL -> VARCHAR -> TEXT), the abstention paths, the policy thresholds, the DDL emitter for
three dialects, and the naive strawman used in the benchmark.

Skim it or skip it - the next cells are what it does.""")

code(ENGINE.rstrip())

md("""## 6. A corpus with the texture real exports have

240 rows of an order export, deterministic, every cell a string - because that is how it arrives.
It carries the traps on purpose: zip codes with leading zeros, prices that a float rounds,
ambiguous slash dates, a 0/1 flag, thousands separators, whitespace-padded quantities, and one
decimal value at row 210 that a 200-row sample never sees.""")

code('''import pandas as pd

rows = demo_rows()
frame = pd.DataFrame(rows)
print(f"{len(rows)} rows x {len(frame.columns)} columns, all dtype={frame.dtypes.unique()[0]}")
frame.head(6)''')

md("""## 7. The DDL

Run the inference and emit Postgres. Note what did *not* happen: `zip_code` is not an integer,
`unit_price` is not a float, `ship_date` is not a date, and `weight_kg` is `NUMERIC(3,1)` rather
than an integer, because the inference read the whole column instead of the first 200 rows.""")

code('''results = infer_table(rows)
print(emit_ddl("orders", results, "postgres"))''')

md("""The same table in SQLite tells a different and more honest story - SQLite has type affinity,
not types, so most of the inference cannot be enforced there at all. Better to know that than to
believe a `DATE` column is doing something.""")

code('''print(emit_ddl("orders", results, "sqlite"))''')

md("""## 8. The findings report

The DDL is half the output. The other half is every decision the tool would not make on its own.
Warnings first.""")

code('''for f in all_findings(results):
    print(f)
    print()''')

md("""## 9. Benchmark

Three strategies on the same 17 columns:

- **all-TEXT** - what the loader does today. Corrupts nothing, types nothing.
- **naive cast** - int -> float -> date -> text, over the first 200 rows. The strawman that ships.
- **lossless** - the two rules plus abstention.

`lossy` is **measured, not asserted**: each proposed type is replayed against every row of the
full column and counted as lossy if any value would be corrupted or rejected. `wide` is broken out
separately - `BIGINT` for an order id is safe and correct in kind, just larger than it needs to be,
and calling that a defect would inflate the result.""")

code('''bench = run_benchmark(rows)

summary = pd.DataFrame(
    [{"strategy": n, **{b: sum(1 for v in bench[n].values() if v == b) for b in BUCKETS}} for n in bench]
)
display(summary)

print("\\nWhere the naive caster goes wrong, column by column:")
naive = bench["naive cast (200-row sample)"]
mine = bench["lossless (this tool)"]
for col in EXPECTED:
    if naive[col] != "exact":
        print(f"  {col:15} naive={naive[col]:8} expected={EXPECTED[col]:14} lossless={mine[col]}")''')

md("""Read that list carefully, because the buckets are not equally bad. Three of them
(`order_id`, `qty`, `flag_01`) are `wide` - `BIGINT` where `SMALLINT` would do. Storage, not
correctness. Nobody gets paged for those.

The five `lossy` ones are different in kind. `zip_code` loses its leading zeros, `unit_price`
and `discount_rate` drift off the written decimal, `sensor_drift` the same, and `weight_kg`
breaks because the value that forces a decimal type sits at row 210 - outside the 200-row
sample. Every one of them loads without an error and is wrong from then on.

And `ship_date` is the interesting one: the naive cast is not lossy at all. `%m/%d/%Y`
round-trips the text perfectly. It is simply *asserting a fact the data never established*
about which number is the month - which is why round-tripping alone is not enough, and why the
third mechanism has to exist.""")

code('''import matplotlib
import matplotlib.pyplot as plt

COLORS = {"exact": "#1b7f5f", "wide": "#5b8ac7", "untyped": "#9aa0a6",
          "unsafe": "#d98317", "lossy": "#b3312c"}
LABEL = {
    "exact": "exact - matches the hand-written answer",
    "wide": "wide - safe, just wider than necessary",
    "untyped": "untyped - text where a safe type existed",
    "unsafe": "unsafe - round-trips but semantically unproven",
    "lossy": "lossy - would corrupt a real value",
}
ORDER = ["exact", "wide", "untyped", "unsafe", "lossy"]

names = list(bench)
counts = {b: [sum(1 for v in bench[n].values() if v == b) for n in names] for b in ORDER}
total = len(EXPECTED)

fig, ax = plt.subplots(figsize=(11, 3.9))
left = [0.0] * len(names)
for b in ORDER:
    ax.barh(names, counts[b], left=left, color=COLORS[b], label=LABEL[b], height=0.55)
    for i, (v, l) in enumerate(zip(counts[b], left)):
        if v:
            ax.text(l + v / 2, i, str(v), ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
    left = [l + v for l, v in zip(left, counts[b])]

ax.set_xlim(0, total)
ax.set_xlabel(f"columns (of {total})")
ax.set_title("Same 17 columns, three ways to type them", fontsize=13, fontweight="bold", pad=14)
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("type_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()''')

md("""## Summary

| | all-TEXT | naive cast | lossless |
|---|---|---|---|
| columns typed correctly | 6 | 7 | **17** |
| columns **silently corrupted** | 0 | **5** | **0** |
| semantically unproven casts | 0 | 1 | 0 |
| safe but wider than needed | 0 | 3 | 0 |
| left as text where a type existed | 11 | 1 | 0 |
| abstentions, **with a stated reason** | - | - | 3 |

The naive caster is not bad at parsing - it is bad at *knowing when parsing is not the question*.
All three of the abstentions (`zip_code`, `ship_date`, `legacy_amount`) are columns where the
right answer is a conversation with whoever owns the source, and the tool's job is to route it
there rather than to pick something plausible.

Two rules and one refusal. That is the entire method:

1. **Text-preserving** - the render must reproduce the source text.
2. **Value-preserving** - the stored number must equal the number written.
3. **Abstain out loud** - and say which upstream fix would settle it.""")

md("""## Try your own

Point it at a real export. Everything is read as text on purpose - that is the situation being
fixed, and letting pandas infer first would hide the very defects this is looking for.""")

code('''# import csv, io
#
# with open("your_export.csv", encoding="utf-8-sig") as fh:
#     your_rows = list(csv.DictReader(fh))          # every value stays a string
#
# your_results = infer_table(your_rows, Policy(
#     min_rows_for_not_null=25,     # raise this if the file is a small sample of a big table
#     varchar_block=8,              # round VARCHAR lengths up to a multiple of this
#     enum_max_distinct=12,         # low-cardinality strings get a CHECK-constraint suggestion
# ))
#
# print(emit_ddl("your_table", your_results, "postgres"))   # or "duckdb" / "sqlite"
# for f in all_findings(your_results):
#     print(f)
#
# # Sanity check before shipping: nothing the tool proposed should be lossy.
# assert not any(is_lossy(r.type, [row[r.name] for row in your_rows]) for r in your_results)''')

md(f"""---

**Streamlit version** - upload a CSV, move the policy thresholds and watch the DDL and the
findings change:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder) · [`{BASE}`](https://github.com/phoebefu6/phoebe-the-builder/tree/main/{BASE})""")

pathlib.Path("demo.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote demo.ipynb ({len(nb['cells'])} cells)")
