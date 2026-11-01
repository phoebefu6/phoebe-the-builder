"""Run the study, write results.json (read by the chart and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def tally(r: dict) -> dict:
    """Per scheme: benign transforms it false-alarms on (matched in <100% of trials) and mutations it misses (>0%)."""
    out = {}
    for k in E.SCHEMES:
        fa = [n for n, v in r["scores"]["benign"].items() if v["match"][k] < 1]
        miss = [n for n, v in r["scores"]["mutations"].items() if v["match"][k] > 0]
        out[k] = {"false_alarms": fa, "misses": miss}
    return out


def lines(r: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    s, f = r["scores"], r["float"]
    w(
        f"table-checksum evidence  (fictional keyless order-event export, {r['rows']:,} rows, {r['distinct']} distinct, "
        f"{r['dup_pairs']} exact duplicate pairs; note column {r['nulls']} NULL and {r['empties']} empty string; "
        f"amount = float sum of line items x {E.TAX}; seeded Random({E.SEED}); {r['trials']} seeded trials per random "
        f"transform; every row fictional)"
    )
    w("")
    head = f"{'':<40}" + "".join(f"{k:>12}" for k in E.SCHEMES)
    w("1. Benign transforms: same data. Cell = share of trials where the fingerprint MATCHES (want 100%)")
    w(head)
    for n, v in s["benign"].items():
        w(f"{n + ' (' + str(v['trials']) + ')':<40}" + "".join(f"{100 * v['match'][k]:>11.0f}%" for k in E.SCHEMES))
    w("")
    w("2. Mutations: different data. Cell = share of trials where the fingerprint still MATCHES (want 0%)")
    w(head)
    for n, v in s["mutations"].items():
        w(f"{n + ' (' + str(v['trials']) + ')':<40}" + "".join(f"{100 * v['match'][k]:>11.0f}%" for k in E.SCHEMES))
    w("")
    w("3. Score per fingerprint")
    for k, v in tally(r).items():
        w(
            f"{k:<11} false alarms {len(v['false_alarms'])}/{len(s['benign'])}  misses {len(v['misses'])}/"
            f"{len(s['mutations'])}  missed: {', '.join(v['misses']) or 'none'}"
        )
    w("")
    w("4. The ceiling: re-adding the same line items in the opposite order")
    w(f"rows whose float amount differs in the bits           {f['bits_differ']} of {f['rows']}")
    w(f"rows whose true amount sits exactly on a half cent    {f['half_cent_exact']}")
    w(f"rows that round to a DIFFERENT cent (half-even)       {f['cents_differ']}")
    for eid, a, b in f["straddle"]:
        w(f"  event {eid}: {a} -> {E.money(float(a))}   vs   {b} -> {E.money(float(b))}")
    w(f"rows that change cent when exported at 15 digits      {f['export15_cents_differ']}")
    w(
        "Any fingerprint that rounds has a boundary; a value one ulp either side of it splits. Only storing the "
        "amount at its declared scale (NUMERIC(12,2)) before hashing removes it."
    )
    return out


def main() -> None:
    r = E.study()
    r["tally"] = tally(r)
    json.dump(r, open("results.json", "w"), indent=1)
    text = "\n".join(lines(r)) + "\n"
    open("evidence.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
