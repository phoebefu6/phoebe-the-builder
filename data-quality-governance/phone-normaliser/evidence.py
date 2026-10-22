"""Run the study, write results.json (read by the chart and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def _n(x) -> str:
    return "-" if x is None else str(x)


def lines(r: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    c = r["ceiling"]
    w(
        f"phone-normaliser evidence  ({r['n']} CRM rows, {r['true_customers']} real customers; truth = hand-labelled "
        f"E.164 per row; default region {r['hq']}; seeded sweep, every number exact)"
    )
    w("")
    w(
        "1. Each normaliser against the truth (pairs = row pairs; 'customers' = records left after matching on the key)"
    )
    w(
        f"{'normaliser':<36} {'E.164 right':>11} {'wrong':>6} {'refused':>8} {'false split':>12} "
        f"{'false merge':>12} {'customers':>10}"
    )
    for n, p in r["per"].items():
        w(
            f"{n:<36} {_n(p['right']):>11} {_n(p['wrong']):>6} {p['refused']:>8} "
            f"{len(p['false_splits']):>7}/{p['true_pairs']:<4} {len(p['false_merges']):>12} "
            f"{p['customers']:>5}/{r['true_customers']}"
        )
    w("")
    w("2. Two customers made one (false merges)")
    for n, p in r["per"].items():
        seen = sorted({f"{a!r} = {b!r}" for a, b in p["false_merges"]})
        if seen:
            w(f"{n:<36} " + "; ".join(seen[:3]) + (f"; +{len(seen) - 3} more" if len(seen) > 3 else ""))
    w("")
    w("3. Wrong numbers produced by the library (a valid number, not the customer's)")
    for n in r["per"]:
        if n in E.KEY_ONLY or n.startswith("US script"):
            continue
        for (s, cf, t, _), k in zip(E.CORPUS, r["per"][n]["keys"]):
            if k is not None and k != t:
                w(f"{n:<36} {s!r} [{cf or 'blank'}] -> {k}, truth {t}")
    w("")
    w("4. How many valid numbers each typed string can be (distinct E.164 across 245 regions)")
    top = sorted(r["ambiguity"].items(), key=lambda kv: (-kv[1], kv[0]))
    for s, k in top[:8]:
        w(f"   {s!r:<22} {k:>3}")
    w(f"   strings with exactly one reading: {sum(1 for _, k in top if k == 1)} of {len(top)} distinct")
    w("")
    w("5. Seeded sweep: random strings typed without a country code")
    for lab, sw in r["sweep"].items():
        w(
            f"   {lab:<34} n={sw['n']}  valid somewhere {sw['valid_somewhere']:.0%}  valid as 3+ numbers "
            f"{sw['valid_3plus']:.0%}  median {sw['median_numbers']}  max {sw['max_numbers']}"
        )
    w("")
    w("6. The ceiling")
    w(f"string only           : {c['by_string']}/{r['n']} rows recoverable (one valid reading in the world)")
    w(
        f"string + country field: {c['by_string_and_field']}/{r['n']}  (same input, two truths: "
        f"{', '.join(c['two_truths'])})"
    )
    best = max((n for n in r["per"] if r["per"][n]["right"] is not None), key=lambda n: r["per"][n]["right"])
    w(f"best normaliser here  : {best}, {r['per'][best]['right']}/{r['n']}")
    return out


def main() -> None:
    r = E.study()
    json.dump({"main": r}, open("results.json", "w"), indent=1, ensure_ascii=False)
    open("evidence.txt", "w").write("\n".join(lines(r)) + "\n")
    print("\n".join(lines(r)))


if __name__ == "__main__":
    main()
