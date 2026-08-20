"""Build demo.ipynb.

The notebook is self-contained: it does not import collate.py. The keyer cell
is *extracted verbatim* from collate.py rather than paraphrased, so the
notebook and the engine cannot silently drift apart, and the analysis cells
are written fresh in notebook style. A verification cell at the end recomputes
the headline counts and asserts them.
"""

from __future__ import annotations

import io
import json
import re
from typing import List


def extract(names: List[str]) -> str:
    """Pull top-level blocks out of collate.py by name, in file order."""
    src = io.open("collate.py", encoding="utf-8").read()
    lines = src.splitlines()
    starts = {}
    for i, line in enumerate(lines):
        for n in names:
            if re.match(rf"^(def {n}\(|class {n}\b|{n}(: [^=]+)? = )", line):
                starts.setdefault(n, i)
    out = []
    for n in names:
        if n not in starts:
            raise KeyError(n)
        i = starts[n]
        while i > 0 and lines[i - 1].startswith("@"):  # keep decorators
            i -= 1
        j = starts[n] + 1
        while j < len(lines) and (
            not lines[j] or lines[j][0] in " )}]#" or lines[j].startswith(("    ", "\t"))
        ):
            j += 1
        block = "\n".join(lines[i:j]).rstrip()
        out.append(block)
    return "\n\n\n".join(out)


ENGINE = extract(
    [
        "ACCENT",
        "CLS_VARIABLE",
        "CLS_SYMBOL",
        "CLS_DIGIT",
        "CLS_LETTER",
        "CLS_OTHER",
        "FOLD_SECONDARY",
        "BASE_FOLD",
        "SENT",
        "Collation",
        "DE_PHONEBOOK",
        "SV_TAILOR",
        "TR_TAILOR",
        "COLLATIONS",
        "locale_lower",
        "_is_variable",
        "sort_key",
    ]
)

CORPUS_CELL = (
    "from __future__ import annotations\n\n"
    "import unicodedata\n"
    "from dataclasses import dataclass\n"
    "from typing import Dict, Tuple\n\n\n" + extract(["Row", "CORPUS"])
)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


CELLS = [
    md(
        """# Sort-Order Drift: `ORDER BY name` is a collation, not an order

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/sort-order-drift/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/sort-order-drift/demo.ipynb)

**Day 150 of the FDE portfolio.** A text column, ten collations, and the
question nobody asks of `ORDER BY`: is the answer *determined*?

A collation is three decisions the SQL never states:

1. **which sequence the characters are in** - locale tailoring,
2. **how many levels of difference count as a difference** - strength,
3. **whether two strings that compare equal are the same value** for `=`,
   `DISTINCT`, `GROUP BY` and `UNIQUE` - determinism.

Decision 1 makes two servers return the same rows in different orders.
Decision 2 creates **ties**, and the order inside a tie is whatever the plan
produced - so paginating it drops rows with no error. Decision 3 changes how
many rows a report returns.

What this notebook builds, in order:

1. the column - 28 ordinary names, two of which are the same string
2. a sort key, three levels deep, tailored per locale
3. ten orders, and how many of the 45 collation pairs agree
4. the ties, and the two very different things a tie can cost
5. what OFFSET and keyset pagination do to a tied sort
6. a range predicate that returns a different number of rows per collation
7. the one-clause fix, verified
"""
    ),
    md(
        """## 1. The column

Nothing here is exotic. Nordic and German surnames, a Turkish name, a Czech
name, a company with a fullwidth `A`, three inventory codes with digit runs,
and one name written twice - once as NFC (`é` is one code point) and once as
NFD (`e` + U+0301). Those two rows are **the same string** by Unicode's own
definition of equivalence.
"""
    ),
    code(
        CORPUS_CELL
        + """


print(f"{len(CORPUS)} rows\\n")
for r in CORPUS:
    cps = " ".join(f"U+{ord(c):04X}" for c in r.name)
    flag = "  <-- non-ASCII" if any(ord(c) > 0x7F for c in r.name) else ""
    print(f"{r.id:2d}  {r.display:<16s}{flag}")
    if flag:
        print(f"    {cps}")"""
    ),
    md(
        """## 2. The sort key

A collation compares *weights*, not characters, and it compares them in
levels: base letter first, then accents, then case. Punctuation is
"variable" - ICU shifts it to a fourth level, glibc drops it outright.

The cell below is lifted verbatim from `collate.py` so the notebook cannot
drift from the engine. Ten collations are defined:

| key | models |
|---|---|
| `C` | PostgreSQL `COLLATE "C"`, SQLite `BINARY`, `LC_ALL=C sort` |
| `UTF16_BIN` | Java `String.compareTo`, JS `Array#sort`, SQL Server `*_BIN2` |
| `en_US_icu` | ICU root/en, PostgreSQL `en-US-x-icu` |
| `de_DIN` | DIN 5007-1 |
| `de_phonebook` | DIN 5007-2, ICU `de-u-co-phonebk` - same language, different answer |
| `sv_SE` | ICU `sv` - å, ä, ö are three letters after Z |
| `tr_TR` | ICU `tr` - dotless ı is a letter before i |
| `ai_ci` | MySQL 8's **default** `utf8mb4_0900_ai_ci` - and nondeterministic |
| `glibc_en_US` | glibc `en_US.UTF-8`, the PostgreSQL default on most Linux hosts |
| `icu_numeric` | ICU `kn-true` - a digit run is a number |
"""
    ),
    code(
        """from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

BASE_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


"""
        + ENGINE
        + """


COLLATION_BY_NAME = {c.key_name: c for c in COLLATIONS}
print(f"{len(COLLATIONS)} collations\\n")
for c in COLLATIONS:
    print(f"  {c.key_name:13s} strength {c.strength}  "
          f"{'nondeterministic' if not c.deterministic else 'deterministic  '}  "
          f"variable={c.variable}")"""
    ),
    md(
        """### What a key looks like

Three levels for `Müller`, and the same three for `Mueller`, under two German
standards. Under the phonebook rule the two keys are **identical**.
"""
    ),
    code(
        '''for name in ("Müller", "Mueller", "Muller"):
    for key_name in ("de_DIN", "de_phonebook"):
        k = sort_key(name, COLLATION_BY_NAME[key_name])
        print(f"{name:<9s} {key_name:<13s} primary={k[0]}\\n{'':24s}secondary={k[1]} tertiary={k[2]}")
    print()

pb = COLLATION_BY_NAME["de_phonebook"]
print("phonebook: Müller == Mueller ?", sort_key("Müller", pb) == sort_key("Mueller", pb))
din = COLLATION_BY_NAME["de_DIN"]
print("DIN 5007-1: Müller == Mueller ?", sort_key("Müller", din) == sort_key("Mueller", din))'''
    ),
    md(
        """## 3. Ten orders, and how many agree

`order()` is a plain stable sort on the key - which is what every engine does
once it has committed to a plan.
"""
    ),
    code(
        '''import pandas as pd


def order(rows, coll):
    return sorted(rows, key=lambda r: sort_key(r.name, coll))


def positions(coll, rows=CORPUS):
    return {r.id: i for i, r in enumerate(order(rows, coll))}


table = pd.DataFrame({c.key_name: [r.display for r in order(CORPUS, c)] for c in COLLATIONS})
table.index = [f"#{i}" for i in range(len(CORPUS))]
identical = [
    (a.key_name, b.key_name)
    for a, b in combinations(COLLATIONS, 2)
    if [r.id for r in order(CORPUS, a)] == [r.id for r in order(CORPUS, b)]
]
print(f"collation pairs returning the identical sequence: {len(identical)} of "
      f"{len(list(combinations(COLLATIONS, 2)))} -> {identical}\\n")
print("ORDER BY name LIMIT 1, which is also MIN(name):")
for c in COLLATIONS:
    print(f"  {c.key_name:13s} {order(CORPUS, c)[0].display!r}")
table.head(12)'''
    ),
    md(
        """### How far one row moves

`Åberg` is a vowel with a ring in English and German, and a **letter after Z**
in Swedish. `Item 9` is after `Item 10` unless the collation reads digit runs
as numbers.
"""
    ),
    code(
        '''pos = {c.key_name: positions(c) for c in COLLATIONS}
spread = []
for r in CORPUS:
    ps = [pos[c.key_name][r.id] for c in COLLATIONS]
    spread.append({"name": r.display, "first": min(ps), "last": max(ps), "spread": max(ps) - min(ps)})
pd.DataFrame(sorted(spread, key=lambda d: -d["spread"])[:8])'''
    ),
    md(
        """## 4. The ties

Two rows are tied when their keys are **equal**. A stable sort then preserves
whatever physical order it was handed, so the result order inside a tie group
is a property of the plan, not of the query.

Two very different costs live here:

- a **deterministic** collation's ties are ordering only. PostgreSQL still
  compares bytes for `=`, so the rows stay distinct and only the order is
  undefined. This is the one that shows up as *a report whose rows moved*.
- a **nondeterministic** collation makes the tie into equality, so
  `DISTINCT`, `GROUP BY` and `UNIQUE` change meaning, and the row *count* moves.
"""
    ),
    code(
        '''class Verdict(str, Enum):
    STABLE_TOTAL = "stable-total"
    TOTAL = "total"
    TIED = "tied"
    MERGING = "merging"


def tie_groups(coll, rows=CORPUS):
    groups = {}
    for r in rows:
        groups.setdefault(sort_key(r.name, coll), []).append(r)
    return [g for g in groups.values() if len(g) > 1]


def verdict(coll, rows=CORPUS):
    if not tie_groups(coll, rows):
        return Verdict.STABLE_TOTAL if coll.kind in ("bytes", "utf16") else Verdict.TOTAL
    return Verdict.TIED if coll.deterministic else Verdict.MERGING


def distinct_count(coll, rows=CORPUS):
    if coll.deterministic:
        return len({r.name for r in rows})
    return len({sort_key(r.name, coll) for r in rows})


for c in COLLATIONS:
    groups = tie_groups(c)
    print(f"{c.key_name:13s} {verdict(c).value:13s} "
          f"{len(groups):2d} tie groups  DISTINCT(name)={distinct_count(c):2d}")
    for g in groups:
        print(f"{'':15s}{' = '.join(r.display for r in g)}")'''
    ),
    md(
        """Note the line that has nothing to do with any locale: **every**
Unicode-aware collation ties `José` with `José`, because they are the same
string. Only the byte orders escape it, by not knowing what a string is.
That is also why no collation here earns the `total` verdict.

And note what `ai_ci` - MySQL 8's *default* - does to a uniqueness constraint.
"""
    ),
    code(
        '''ai = COLLATION_BY_NAME["ai_ci"]
violations = [
    (a, b)
    for a, b in combinations(CORPUS, 2)
    if a.name != b.name and sort_key(a.name, ai) == sort_key(b.name, ai)
]
print(f"UNIQUE(name) under {ai.key_name} rejects {len(violations)} pairs of different strings:")
for a, b in violations:
    print(f"  {a.display!r:<16s} = {b.display!r}")
print(f"\\nCOUNT(DISTINCT name): {distinct_count(ai)} under ai_ci, "
      f"{len({r.name for r in CORPUS})} under every deterministic collation")'''
    ),
    md(
        """## 5. Pagination

Each page is a separate execution, so each may be handed a different physical
row order: insertion order, a backward index scan, a table rewritten by
`VACUUM FULL`. Three plans below, and four pagination schemes:

- `OFFSET` - `ORDER BY name LIMIT n OFFSET k`
- `OFFSET` + a unique tiebreak - `ORDER BY name, id`
- keyset with `>` - `WHERE name > $last`
- keyset with `>=` - `WHERE name >= $last`
"""
    ),
    code(
        '''def plan_orders(rows=CORPUS):
    base = list(rows)
    return [base, list(reversed(base)), sorted(base, key=lambda r: (len(r.name), r.id))]


def offset_pagination(coll, page_size, rows=CORPUS, tiebreak=False):
    plans = plan_orders(rows)
    seen = []
    for page_no in range((len(rows) + page_size - 1) // page_size):
        physical = plans[page_no % len(plans)]
        key = (lambda r: (sort_key(r.name, coll), r.id)) if tiebreak else (lambda r: sort_key(r.name, coll))
        ordered = sorted(physical, key=key)
        seen.extend(r.id for r in ordered[page_no * page_size:(page_no + 1) * page_size])
    lost = sorted({r.id for r in rows} - set(seen))
    dup = sorted(i for i in set(seen) if seen.count(i) > 1)
    return lost, dup


def keyset_pagination(coll, page_size, rows=CORPUS, strict=True):
    plans = plan_orders(rows)
    seen, last_key, prev, stalled = [], None, None, False
    for page_no in range(len(rows) + 1):
        ordered = sorted(plans[page_no % len(plans)], key=lambda r: sort_key(r.name, coll))
        if last_key is not None:
            ordered = [r for r in ordered
                       if (sort_key(r.name, coll) > last_key if strict
                           else sort_key(r.name, coll) >= last_key)]
        page = ordered[:page_size]
        if not page:
            break
        ids = tuple(r.id for r in page)
        if prev is not None and set(ids) == set(prev):
            stalled = True
            break
        seen.extend(ids)
        prev = ids
        last_key = sort_key(page[-1].name, coll)
    lost = sorted({r.id for r in rows} - set(seen))
    dup = sorted(i for i in set(seen) if seen.count(i) > 1)
    return lost, dup, stalled


PAGE_SIZES = (2, 3, 4, 5, 6, 7, 8, 10)
rows_out = []
for c in COLLATIONS:
    for n in PAGE_SIZES:
        lost, dup = offset_pagination(c, n)
        tlost, tdup = offset_pagination(c, n, tiebreak=True)
        klost, _, _ = keyset_pagination(c, n, strict=True)
        _, kdup, stall = keyset_pagination(c, n, strict=False)
        rows_out.append({"collation": c.key_name, "page": n, "offset_lost": len(lost),
                         "offset_dup": len(dup), "with_id": len(tlost) + len(tdup),
                         "keyset_gt_lost": len(klost), "keyset_ge_dup": len(kdup),
                         "keyset_ge_stall": stall})
pag = pd.DataFrame(rows_out)
print(f"{len(pag)} runs\\n")
print("OFFSET is wrong in these runs:")
display(pag[pag.offset_lost > 0][["collation", "page", "offset_lost", "offset_dup"]])
print(f"\\ntotals: OFFSET lost {pag.offset_lost.sum()}, repeated {pag.offset_dup.sum()}")
print(f"        with `, id`: {pag.with_id.sum()} wrong rows")
print(f"        keyset `>` lost {pag.keyset_gt_lost.sum()}")
print(f"        keyset `>=` repeated {pag.keyset_ge_dup.sum()}, "
      f"stalled in {int(pag.keyset_ge_stall.sum())} of {len(pag)} runs")'''
    ),
    md(
        """Three things in that table are worth stopping on.

1. `ai_ci` is **exact at page size 7** and wrong at 6, 8 and 10. Whether a tie
   group straddles a page boundary depends on the page size, so a test suite
   that picked a clean one passes.
2. `sv_SE` at page size 3 loses a row, and `sv_SE` is *deterministic*. This is
   not a MySQL problem.
3. The `>=` keyset column stalls in **every** run, including `C`, which has no
   ties at all: the last row of a page always satisfies `name >= $last`, so it
   opens the next page forever. A cursor needs a comparison it can strictly
   advance - `WHERE (name, id) > ($n, $i)`.
"""
    ),
    md(
        """## 6. A range predicate is collation-dependent too

`WHERE name >= 'A' AND name < 'N'` is the shape of every shard boundary,
archive sweep and alphabetical tab.
"""
    ),
    code(
        '''def in_range(name, coll, lo="A", hi="N"):
    k = sort_key(name, coll)
    return sort_key(lo, coll) <= k < sort_key(hi, coll)


counts = {c.key_name: sum(1 for r in CORPUS if in_range(r.name, c)) for c in COLLATIONS}
for k, v in counts.items():
    print(f"  {k:13s} {v:2d} rows")
print(f"\\n{min(counts.values())} to {max(counts.values())} rows, same table, same predicate\\n")
for r in CORPUS:
    yes = [c.key_name for c in COLLATIONS if in_range(r.name, c)]
    if yes and len(yes) < len(COLLATIONS):
        print(f"  {r.display:<16s} in: {', '.join(yes)}")'''
    ),
    md("""## 7. The picture"""),
    code(
        '''import matplotlib
import matplotlib.pyplot as plt
import numpy as np

INK, WARM, COOL, GREEN, SAND, GRID = "#1d2733", "#c2571a", "#2d5a68", "#2f6b39", "#e8d9c0", "#dfe4ea"
plt.rcParams.update({"font.size": 8, "figure.facecolor": "white", "axes.edgecolor": GRID})
NAMES = [c.key_name for c in COLLATIONS]


def chart_label(name):
    return "".join(ch if ord(ch) < 0x2000 else f"U+{ord(ch):04X}" for ch in name)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

for rid, col in zip([2, 5, 21, 27], [WARM, COOL, GREEN, "#5d4a7e"]):
    ax1.plot(range(len(NAMES)), [pos[n][rid] for n in NAMES], "-o", ms=3.6, lw=1.2,
             color=col, label=chart_label(CORPUS[rid - 1].name))
ax1.set_xticks(range(len(NAMES)))
ax1.set_xticklabels(NAMES, rotation=55, ha="right", fontsize=6)
ax1.invert_yaxis()
ax1.set_ylabel("position in the result (0 = first)")
ax1.grid(axis="y", color=GRID, lw=0.6)
ax1.set_axisbelow(True)
ax1.legend(fontsize=6, frameon=False, loc="center left")
ax1.set_title("The same four rows, ten collations", fontsize=9, loc="left")

grid = np.zeros((len(NAMES), len(PAGE_SIZES)))
for i, c in enumerate(COLLATIONS):
    for j, n in enumerate(PAGE_SIZES):
        grid[i, j] = len(offset_pagination(c, n)[0])
im = ax2.imshow(grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "h", ["#f6f3ee", SAND, WARM, "#7d2f0c"]), vmin=0, vmax=max(1, grid.max()))
ax2.set_xticks(range(len(PAGE_SIZES)))
ax2.set_xticklabels([str(n) for n in PAGE_SIZES], fontsize=6)
ax2.set_yticks(range(len(NAMES)))
ax2.set_yticklabels(NAMES, fontsize=6)
ax2.set_xlabel("page size")
for i in range(len(NAMES)):
    for j in range(len(PAGE_SIZES)):
        if grid[i, j]:
            ax2.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=6,
                     color="white" if grid[i, j] > grid.max() * 0.55 else INK)
ax2.set_title("Rows OFFSET paging never returns (0 everywhere with `, id`)", fontsize=9, loc="left")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)
fig.tight_layout()
fig.savefig("sort_order_notebook.png", dpi=150, bbox_inches="tight")
plt.show()'''
    ),
    md(
        """## 8. Summary, and the fix

| mechanism | number |
|---|---|
| collation pairs returning the identical sequence | 1 of 45 |
| verdicts | 2 stable-total, 0 total, 7 tied, 1 merging |
| `COUNT(DISTINCT name)` under `ai_ci` | 19, not 28 |
| `UNIQUE(name)` pairs rejected under `ai_ci` | 10 |
| OFFSET runs that are wrong | 8 of 80 |
| rows OFFSET paging never returns | 15 |
| rows keyset `>` never returns | 24 |
| keyset `>=` runs that never terminate | 80 of 80 |
| rows matched by `name >= 'A' AND name < 'N'` | 16 to 21 |
| **all of the above, with `ORDER BY name, id`** | **0** |

The fix, cheapest first:

1. **`ORDER BY name, id`** - one clause, and every pagination number above goes to zero.
2. **Keyset on a composite** - `WHERE (name, id) > ($n, $i) ORDER BY name, id`. Strictly advances, so it terminates.
3. **Name the collation in the DDL** rather than inheriting the host's `lc_collate`.
4. **Normalise on write** (NFC) - no collation can undo two spellings of one string in a byte-ordered index.
5. **Decide determinism deliberately** - nondeterministic gives you case-insensitive uniqueness *and* changes `DISTINCT`/`GROUP BY`. Both are defensible; the accident is not.
6. **Treat the collation version as a migration input** - glibc 2.28 changed `en_US.UTF-8` and invalidated existing PostgreSQL text indexes.
"""
    ),
    code(
        '''# Verification: the counts this notebook computed, asserted against the
# numbers published in the README (which evidence.py prints from collate.py).
checks = {
    "identical collation pairs": (len(identical), 1),
    "tied collations": (sum(1 for c in COLLATIONS if verdict(c) is Verdict.TIED), 7),
    "merging collations": (sum(1 for c in COLLATIONS if verdict(c) is Verdict.MERGING), 1),
    "total collations": (sum(1 for c in COLLATIONS if verdict(c) is Verdict.TOTAL), 0),
    "DISTINCT under ai_ci": (distinct_count(ai), 19),
    "UNIQUE violations under ai_ci": (len(violations), 10),
    "dirty OFFSET runs": (int((pag.offset_lost > 0).sum()), 8),
    "rows lost by OFFSET": (int(pag.offset_lost.sum()), 15),
    "rows lost by keyset `>`": (int(pag.keyset_gt_lost.sum()), 24),
    "keyset `>=` stalls": (int(pag.keyset_ge_stall.sum()), 80),
    "wrong rows with `, id`": (int(pag.with_id.sum()), 0),
    "range predicate min..max": ((min(counts.values()), max(counts.values())), (16, 21)),
}
for label, (got, want) in checks.items():
    status = "ok " if got == want else "MISMATCH"
    print(f"  [{status}] {label:32s} {got} (expected {want})")
assert all(got == want for got, want in checks.values())
print("\\nnotebook agrees with the engine on every headline count")'''
    ),
    md(
        """## Try your own column

Paste your own names below - a real customer surname column, a product list, a
tag vocabulary - and see which collations tie it, and which page sizes lose a
row.
"""
    ),
    code(
        '''# MY_NAMES = """
# Ostergaard
# Østergaard
# Oestergaard
# o'brien
# O'Brien
# """.strip().splitlines()
#
# mine = tuple(Row(i, n.strip(), "") for i, n in enumerate(MY_NAMES, start=1))
# for c in COLLATIONS:
#     print(f"{c.key_name:13s} {verdict(c, mine).value:13s} "
#           f"{len(tie_groups(c, mine))} tie groups   "
#           f"first: {order(mine, c)[0].name!r}")
# for n in (2, 3, 4, 5):
#     lost, dup = offset_pagination(COLLATION_BY_NAME["ai_ci"], n, mine)
#     print(f"page size {n}: OFFSET lost {len(lost)}, repeated {len(dup)}")'''
    ),
    md(
        """---

**Day 150 of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder)** -
[project README](README.md) with every mechanism and the findings table,
`python evidence.py` for every number, `python -m pytest -q` for the 53 tests,
and `streamlit run app.py` to paste a column into the audit interactively.

Previous days on the same theme - an operation that looks total and is not:
[`currency-rounder`](../currency-rounder/) (Day 143),
[`filename-sanitiser`](../../automation-suite/filename-sanitiser/) (146),
[`duration-parser`](../../automation-suite/duration-parser/) (147),
[`percent-recomputer`](../../analytics-engineering-bi/percent-recomputer/) (148),
[`header-casing`](../../automation-suite/header-casing/) (149).
"""
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    with io.open("demo.ipynb", "w", encoding="utf-8") as fh:
        json.dump(NB, fh, ensure_ascii=False, indent=1)
    print(f"wrote demo.ipynb ({len(CELLS)} cells)")
