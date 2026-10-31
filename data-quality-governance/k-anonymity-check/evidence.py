"""Run the study, write results.json (read by the chart and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E

SHOW = [
    ("6 digits", "full date", "kept"),
    ("6 digits", "month", "kept"),
    ("6 digits", "year", "kept"),
    ("6 digits", "decade", "kept"),
    ("6 digits", "dropped", "kept"),
    ("4 digits", "year", "kept"),
    ("4 digits", "decade", "kept"),
    ("4 digits", "dropped", "dropped"),
    ("sector (2 digits)", "full date", "kept"),
    ("sector (2 digits)", "month", "kept"),
    ("sector (2 digits)", "year", "kept"),
    ("sector (2 digits)", "decade", "kept"),
    ("sector (2 digits)", "decade", "dropped"),
    ("sector (2 digits)", "dropped", "kept"),
    ("dropped", "year", "kept"),
    ("dropped", "decade", "kept"),
]


def lv(levels) -> str:
    p, d, s = levels
    return f"postcode {p} / dob {d} / sex {s}"


def pct(a: int, n: int) -> str:
    return f"{100 * a / n:.1f}%"


def lines(r: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    n = r["export_n"]
    pc = r["postcodes"]
    w(
        f"k-anonymity-check evidence  (fictional town of {r['pop_n']:,} residents in {pc['postcodes']} postcodes "
        f"holding {pc['min']} to {pc['max']:,} people, median {pc['median']}; a {n:,}-row clinic export = "
        f"{100 * n / r['pop_n']:g}% of the town; three quasi-identifiers: postcode, date of birth, sex; "
        f"k = {E.K}; seeded Random({E.SEED}), every person fictional)"
    )
    w("")
    w("1. How few columns make a row unique (full granularity; 'in the town' = no other resident shares the key)")
    w(f"{'columns':<36} {'unique in export':>17} {'unique in town':>15} {'classes':>8}")
    for p in r["subsets"]:
        w(
            f"{' + '.join(p['columns']):<36} {p['unique_export']:>7} ({pct(p['unique_export'], n):>6}) "
            f"{p['unique_pop']:>6} ({pct(p['unique_pop'], n):>6}) {p['classes']:>8}"
        )
    w("")
    w(f"2. The generalisation ladder ({len(SHOW)} of {len(r['ladder'])} combinations; all 40 in results.json)")
    w(f"{'generalisation':<58} {'classes':>7} {'k_min':>5} {'uniq export':>11} {'uniq town':>9} {'<5 export':>9} {'<5 town':>7}")
    by = {tuple(p["levels"]): p for p in r["ladder"]}
    for levels in SHOW:
        p = by[levels]
        w(
            f"{lv(levels):<58} {p['classes']:>7} {p['k_min']:>5} {p['unique_export']:>11} {p['unique_pop']:>9} "
            f"{p['below_k']:>9} {p['below_k_pop']:>7}"
        )
    ok = r["k_anonymous"]
    b = r["best"]
    rel = r["release"]
    w(
        f"k >= {E.K} in the export with nothing suppressed: {len(ok)} of {len(r['ladder'])} combinations, the finest "
        f"= {lv(b['levels'])} ({b['classes']} classes)"
    )
    w(
        f"k >= {E.K} after suppressing at most 5% of rows: finest = {lv(rel['levels'])} ({rel['classes']} classes, "
        f"{rel['below_k']} rows suppressed, {rel['released']:,} released)"
    )
    w("")
    w("3. Unique in the export vs unique in the town, as the export grows (two keys)")
    w(f"{'export':>8} {'rows':>7} {'full key: export':>17} {'town':>7} {'sector+year+sex: export':>24} {'town':>7}")
    sw = r["sweep"]
    fr = sorted({s["frac"] for s in sw})
    for f in fr:
        a = next(s for s in sw if s["frac"] == f and tuple(s["levels"]) == E.FULL)
        c = next(s for s in sw if s["frac"] == f and tuple(s["levels"]) != E.FULL)
        w(
            f"{100 * f:>7g}% {a['n']:>7,} {a['unique_export_pct']:>16.1f}% {a['unique_pop_pct']:>6.1f}% "
            f"{c['unique_export_pct']:>23.1f}% {c['unique_pop_pct']:>6.2f}%"
        )
    sy = by[("sector (2 digits)", "year", "kept")]
    w(
        f"two attackers at sector+year+sex on the {n:,}-row export: one who knows the person IS a patient needs only "
        f"export-uniqueness and re-identifies {sy['unique_export']:,}; a stranger needs town-uniqueness and "
        f"re-identifies {sy['unique_pop']}"
    )
    w("")
    w("4. Linkage with a public register (name + the three columns), complete and missing 20% of the town")
    w(f"{'generalisation':<58} {'register':>8} {'right':>6} {'wrong':>6} {'ambiguous':>9} {'none':>5}")
    for k in r["linkage"]:
        w(
            f"{lv(k['levels']):<58} {100 * k['coverage']:>7g}% {k['right']:>6} {k['wrong']:>6} "
            f"{k['ambiguous']:>9} {k['none']:>5}"
        )
    w("")
    w(f"5. k does not protect the value (baseline: {pct(r['sensitive_base'], n)} of export rows carry a sensitive diagnosis)")
    w(
        f"the release ({lv(rel['levels'])}): {rel['homog_classes']} classes where everyone shares one diagnosis "
        f"({rel['homog_rows']} rows); {rel['near_homog_rows']} rows in classes where one diagnosis is >= 80%; "
        f"{rel['sensitive_confident_rows']} of {rel['sensitive_rows_released']} sensitive rows sit in a class that "
        f"makes an attacker >= 50% sure of it"
    )
    hot = [p for p in r["ladder"] if p["sensitive_confident_rows"] > 0]
    w(f"generalisations whose k >= {E.K} release leaves an attacker >= 50% sure of a sensitive diagnosis: {len(hot)} of {len(r['ladder'])}")
    for p in sorted(hot, key=lambda p: -p["sensitive_confident_rows"])[:4]:
        w(
            f"   {lv(p['levels']):<56} {p['sensitive_confident_rows']:>3} rows ({p['homog_classes']} homogeneous classes, "
            f"{p['below_k']} suppressed)"
        )
    w("")
    w("6. The ceiling")
    w(
        "smallest postcodes            : "
        + ", ".join(f"{c} ({k} residents)" for c, k in r["smallest_postcodes"])
        + " - a six-digit postcode can never be hidden among many"
    )
    w(
        f"export k is not town k        : at sector+year+sex the export has k_min {sy['k_min']} and {sy['below_k']:,} rows "
        f"below {E.K}; the town has {sy['below_k_pop']} below {E.K}. The right-hand column needs the town, which is "
        "not in the file"
    )
    full = by[E.FULL]
    w(
        f"two columns are already enough: postcode + date of birth make {r['subsets'][3]['unique_pop']:,} "
        f"of {n:,} rows unique in the town; adding sex makes {full['unique_pop']:,}"
    )
    return out


def main() -> None:
    r = E.study()
    json.dump({"main": r}, open("results.json", "w"), indent=1, ensure_ascii=False, default=str)
    open("evidence.txt", "w").write("\n".join(lines(r)) + "\n")
    print("\n".join(lines(r)))


if __name__ == "__main__":
    main()
