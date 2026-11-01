"""Are these two tables the same? Five whole-table fingerprints against what really happens to a table in transit.

NOT a row differ (`data-diff`, Day 100) and NOT a reconciliation report (`reconciliation-checker`, Day 99): this
scores the ONE number you compare when you cannot ship both tables to one place. A fictional 1,000-row keyless
order-event export (it contains exact duplicate rows, as event logs do), seeded Random(207). Each fingerprint is
asked about benign transforms (same data, should MATCH) and real mutations (different data, should DIFFER).

Fingerprints:
  csv_md5     md5 of the CSV dump as written (row order, column order, float repr all count)
  count_sums  row count + SUM of every money column, to the cent (the reconciliation-report habit)
  xor_rows    XOR of 64-bit row hashes, row = "|".join(str(v)) in the given column order
  sum_rows    the same row hashes summed mod 2**64 (a multiset, so duplicates count)
  canonical   sum_rows over a TYPED row: columns by name, money quantised to the cent, NULL a distinct token,
              strings length-prefixed, -0 folded to 0, string values parsed by the declared type
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
from collections import Counter
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Callable, Dict, List, Optional, Tuple

SEED = 207
N_ROWS = 1000
N_DUP_PAIRS = 20
TRIALS = 100
TAX = 1.07
COLS = ["event_id", "customer", "note", "amount", "discount"]
SCHEMA = {"event_id": "int", "customer": "str", "note": "str", "amount": "money", "discount": "money"}
NULL = "\\N"  # how the CSV round trip writes a NULL (Postgres COPY style)
CUSTOMERS = ["Ada Lim", "Ben Okafor", "Chloe Tan", "Dev Rao", "Eli Novak", "Fay Wong", "Gus Berg", "Hana Sato"]
NOTES = ["gift", "rush", "fragile", "repeat", "VIP"]

Row = Dict[str, object]
Table = Tuple[List[str], List[Row]]  # (column order, rows)


def amount_of(lines: List[Tuple[int, int]], reverse: bool = False) -> float:
    """Sum of price_cents/100 * qty * TAX in float, the way an order service adds it up. Order of addition matters."""
    seq = list(reversed(lines)) if reverse else lines
    total = 0.0
    for cents, qty in seq:
        total += cents / 100 * qty * TAX
    return total


def make_table(seed: int = SEED) -> Tuple[Table, Dict[int, List[Tuple[int, int]]]]:
    rng = random.Random(seed)
    rows: List[Row] = []
    items: Dict[int, List[Tuple[int, int]]] = {}
    for i in range(N_ROWS - N_DUP_PAIRS):
        lines = [(rng.randint(99, 9999), rng.randint(1, 5)) for _ in range(rng.randint(1, 6))]
        u = rng.random()
        note: Optional[str] = None if u < 0.5 else ("" if u < 0.65 else rng.choice(NOTES))
        disc: Optional[float] = None if rng.random() < 0.6 else float(rng.choice([0, 0, 5, 10, 12.5]))
        eid = 10_000 + i
        items[eid] = lines
        rows.append({"event_id": eid, "customer": rng.choice(CUSTOMERS), "note": note,
                     "amount": amount_of(lines), "discount": disc})
    for r in rng.sample(rows, N_DUP_PAIRS):  # a keyless event log: a retry wrote some events twice
        rows.append(dict(r))
    rng.shuffle(rows)
    return (list(COLS), rows), items


# ---------------------------------------------------------------- fingerprints

def h64(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big")


def csv_md5(t: Table) -> str:
    cols, rows = t
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow(["" if r[c] is None else r[c] for c in cols])
    return hashlib.md5(buf.getvalue().encode()).hexdigest()


def money(v: object) -> Decimal:
    d = Decimal(repr(v)) if isinstance(v, float) else Decimal(str(v))
    d = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    return Decimal("0.00") if d == 0 else d  # -0.00 and 0.00 are the same amount


def sum_col(rows: List[Row], c: str) -> Decimal:
    return sum((money(r[c]) for r in rows if r[c] not in (None, NULL)), Decimal("0.00"))


def count_sums(t: Table) -> str:
    cols, rows = t
    return "|".join([str(len(rows))] + [str(sum_col(rows, c)) for c in sorted(cols) if SCHEMA.get(c) == "money"])


def naive_row(cols: List[str], r: Row) -> str:
    return "|".join(str(r[c]) for c in cols)


def xor_rows(t: Table) -> str:
    acc = 0
    for r in t[1]:
        acc ^= h64(naive_row(t[0], r))
    return f"{acc:016x}"


def sum_rows(t: Table) -> str:
    acc = sum(h64(naive_row(t[0], r)) for r in t[1]) % 2**64
    return f"{acc:016x}"


def canon_value(kind: str, v: object) -> str:
    if v is None or v == NULL:
        return "\x00N"
    if kind == "int":
        return f"i{int(v)}"
    if kind == "money":
        return f"m{money(v)}"
    s = str(v)
    return f"s{len(s)}:{s}"


def canonical_row(r: Row) -> str:
    return "".join(canon_value(SCHEMA[c], r[c]) for c in sorted(SCHEMA))


def canonical(t: Table) -> str:
    acc = sum(h64(canonical_row(r)) for r in t[1]) % 2**64
    return f"{acc:016x}"


SCHEMES: Dict[str, Callable[[Table], str]] = {
    "csv_md5": csv_md5, "count_sums": count_sums, "xor_rows": xor_rows, "sum_rows": sum_rows, "canonical": canonical,
}


# ---------------------------------------------------------------- what happens in transit

def _copy(t: Table) -> Table:
    return list(t[0]), [dict(r) for r in t[1]]


def b_shuffle(t, items, rng):
    c = _copy(t)
    rng.shuffle(c[1])
    return c


def b_reorder_columns(t, items, rng):
    cols = list(t[0])
    while cols == t[0]:
        rng.shuffle(cols)
    return cols, [dict(r) for r in t[1]]


def b_csv_roundtrip(t, items, rng):
    """Every value comes back a string, NULL as \\N, floats as Python writes them."""
    return list(t[0]), [{c: NULL if v is None else str(v) for c, v in r.items()} for r in t[1]]


def b_export_15g(t, items, rng):
    """The target exports floats with 15 significant digits, as many databases and spreadsheets do."""
    return list(t[0]), [{c: (float(f"{v:.15g}") if isinstance(v, float) else v) for c, v in r.items()} for r in t[1]]


def b_recompute_amount(t, items, rng):
    """The target re-adds the same line items in the opposite order."""
    c = _copy(t)
    for r in c[1]:
        r["amount"] = amount_of(items[r["event_id"]], reverse=True)
    return c


def b_negative_zero(t, items, rng):
    c = _copy(t)
    for r in c[1]:
        if r["discount"] == 0:
            r["discount"] = -0.0
    return c


def b_int_discount(t, items, rng):
    """The target stores whole-number discounts in an integer column: 5.0 comes back as 5."""
    c = _copy(t)
    for r in c[1]:
        if isinstance(r["discount"], float) and r["discount"].is_integer():
            r["discount"] = int(r["discount"])
    return c


def m_one_cent(t, items, rng):
    c = _copy(t)
    r = rng.choice(c[1])
    r["amount"] = r["amount"] + 0.01
    return c


def m_offsetting_cents(t, items, rng):
    c = _copy(t)
    a, b = rng.sample(c[1], 2)
    a["amount"], b["amount"] = a["amount"] + 0.01, b["amount"] - 0.01
    return c


def m_swap_amounts(t, items, rng):
    c = _copy(t)
    a, b = rng.sample([r for r in c[1]], 2)
    a["amount"], b["amount"] = b["amount"], a["amount"]
    return c


def m_null_to_empty(t, items, rng):
    c = _copy(t)
    rng.choice([r for r in c[1] if r["note"] is None])["note"] = ""
    return c


def m_dup_pair_swapped(t, items, rng):
    """The dedup step dropped both copies of a retried event, and a retry wrote another event twice more."""
    c = _copy(t)
    keys = Counter(naive_row(COLS, r) for r in c[1])
    dup = rng.choice([r for r in c[1] if keys[naive_row(COLS, r)] == 2])
    gone = naive_row(COLS, dup)
    rows = [r for r in c[1] if naive_row(COLS, r) != gone]
    extra = rng.choice([r for r in rows if keys[naive_row(COLS, r)] == 1])
    return list(c[0]), rows + [dict(extra), dict(extra)]


BENIGN = {"shuffle rows": b_shuffle, "reorder columns": b_reorder_columns, "CSV round trip": b_csv_roundtrip,
          "export at 15 digits": b_export_15g, "re-add line items": b_recompute_amount,
          "-0.0 discount": b_negative_zero, "5.0 stored as 5": b_int_discount}
MUTATIONS = {"one amount +1 cent": m_one_cent, "two offsetting cents": m_offsetting_cents,
             "two amounts swapped": m_swap_amounts, "NULL note became ''": m_null_to_empty,
             "dup pair dropped, another doubled": m_dup_pair_swapped}


def delimiter_case(t: Table, rng: random.Random) -> Tuple[Table, Table]:
    """Two tables identical except one row's customer/note boundary: 'Ada Lim|' + 'VIP' vs 'Ada Lim' + '|VIP'."""
    a, b = _copy(t), _copy(t)
    i = rng.choice([j for j, r in enumerate(t[1]) if r["note"]])
    a[1][i]["customer"] = t[1][i]["customer"] + "|"
    b[1][i]["note"] = "|" + t[1][i]["note"]
    return a, b


def score(t: Table, items: Dict[int, List[Tuple[int, int]]], trials: int = TRIALS) -> dict:
    """Per transform, per scheme: share of seeded trials where the fingerprints MATCH."""
    base = {k: f(t) for k, f in SCHEMES.items()}
    out: dict = {"benign": {}, "mutations": {}}
    for group, table in (("benign", BENIGN), ("mutations", MUTATIONS)):
        for name, fn in table.items():
            rng = random.Random(f"{SEED}-{name}")
            n = 1 if name in ("CSV round trip", "export at 15 digits", "re-add line items", "-0.0 discount",
                              "5.0 stored as 5") else trials
            match = {k: 0 for k in SCHEMES}
            for _ in range(n):
                u = fn(t, items, rng)
                for k, f in SCHEMES.items():
                    match[k] += f(u) == base[k]
            out[group][name] = {"trials": n, "match": {k: v / n for k, v in match.items()}}
    rng = random.Random(f"{SEED}-delim")
    match = {k: 0 for k in SCHEMES}
    for _ in range(trials):
        a, b = delimiter_case(t, rng)
        for k, f in SCHEMES.items():
            match[k] += f(a) == f(b)
    out["mutations"]["'|' moved across a column"] = {"trials": trials, "match": {k: v / trials for k, v in match.items()}}
    return out


def float_detail(t: Table, items: Dict[int, List[Tuple[int, int]]]) -> dict:
    """Where the re-added amounts disagree: in the bits, and after rounding to the cent."""
    bits = cents = export15 = 0
    straddle: List[Tuple[int, str, str]] = []
    for r in t[1]:
        a, b = r["amount"], amount_of(items[r["event_id"]], reverse=True)
        bits += a != b
        if money(a) != money(b):
            cents += 1
            straddle.append((r["event_id"], repr(a), repr(b)))
        export15 += money(a) != money(float(f"{a:.15g}"))
    exact = sum(1 for r in t[1] if (sum(c * q for c, q in items[r["event_id"]]) * 107) % 100 == 50)
    return {"rows": len(t[1]), "bits_differ": bits, "cents_differ": cents, "export15_cents_differ": export15,
            "half_cent_exact": exact, "straddle": straddle[:5]}


def study() -> dict:
    t, items = make_table()
    keys = [naive_row(COLS, r) for r in t[1]]
    return {"rows": len(t[1]), "distinct": len(set(keys)), "dup_pairs": N_DUP_PAIRS, "trials": TRIALS,
            "nulls": sum(r["note"] is None for r in t[1]), "empties": sum(r["note"] == "" for r in t[1]),
            "scores": score(t, items), "float": float_detail(t, items)}
