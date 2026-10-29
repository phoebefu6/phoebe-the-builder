"""Run the study, write results.json (read by the chart and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def lines(r: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    c = r["ceiling"]
    tp = c["true_pairs"]
    w(
        f"dedupe-key-choice evidence  ({r['n']} CRM rows, {r['true_customers']} real customers, {tp} true duplicate "
        f"pairs; truth = hand-labelled customer id per row; every name fictional; no randomness)"
    )
    w("")
    w("1. Each key against the truth ('customers' = records left after matching on the key)")
    w(f"{'key':<26} {'false split':>12} {'false merge':>12} {'customers':>10} {'chained':>8}")
    for n, p in r["per"].items():
        w(
            f"{n:<26} {len(p['false_splits']):>8}/{tp:<3} {len(p['false_merges']):>12} "
            f"{p['customers']:>5}/{r['true_customers']} {p['chained_pairs']:>8}"
        )
    w("")
    w("2. Pairs the ROW key merges that the COLUMN key keeps apart")
    short = {n: n.split(" >=")[0] for n in E.NAMES}
    w(f"{'':<26} " + " ".join(f"{short[b]:>12}" for b in E.NAMES))
    for a in E.NAMES:
        w(f"{a:<26} " + " ".join(f"{r['disagreement'][a][b]:>12}" for b in E.NAMES))
    w("")
    w("3. Two customers made one (false merges, first three per key)")
    for n, p in r["per"].items():
        seen = sorted({f"{a!r} = {b!r}" for a, b in p["false_merges"]})
        if seen:
            w(f"{n:<26} " + "; ".join(seen[:3]) + (f"; +{len(seen) - 3} more" if len(seen) > 3 else ""))
    w("")
    w("4. Chained: in one cluster although the pair itself is below the threshold")
    for n in E.PAIR_KEYS:
        sim, t = E.PAIR_KEYS[n]
        lab = r["per"][n]["labels"]
        ch = [
            (E.CORPUS[i][0], E.CORPUS[j][0], sim(E.CORPUS[i][0], E.CORPUS[j][0]))
            for i, j in E.combinations(range(r["n"]), 2)
            if lab[i] == lab[j] and sim(E.CORPUS[i][0], E.CORPUS[j][0]) < t
        ]
        d = r["direct"][n]
        w(
            f"{n:<26} direct {d['false_splits']} splits / {d['false_merges']} merges -> clustered "
            f"{len(r['per'][n]['false_splits'])} / {len(r['per'][n]['false_merges'])}; {len(ch)} chained pairs"
        )
        for a, b, s in ch[:3]:
            w(f"   {a!r} ~ {b!r} at {s:.3g}")
    w("")
    w("5. The frontier: the best any single threshold reaches, judged pair by pair")
    for n, b in c["best_frontier"].items():
        w(
            f"{n:<26} best threshold {b['threshold']:g}: {b['false_splits']} splits / {b['false_merges']} merges; "
            f"a zero-error threshold exists: {'yes' if b['zero_exists'] else 'no'}"
        )
    w("")
    w("6. The ceiling")
    w(f"identical string, two people  : {', '.join(repr(s) for s in c['two_truths'])} (every key merges them)")
    w(f"true pairs no key here merges : {len(r['never_merged'])} of {tp}")
    for a, b in r["never_merged"]:
        w(f"   {a!r} = {b!r}")
    return out


def main() -> None:
    r = E.study()
    json.dump({"main": r}, open("results.json", "w"), indent=1, ensure_ascii=False)
    open("evidence.txt", "w").write("\n".join(lines(r)) + "\n")
    print("\n".join(lines(r)))


if __name__ == "__main__":
    main()
