"""Run the study, write results.json (read by the chart, the app and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def lines(r: dict, s: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    wr = r["writers"]
    w(f"float-roundtrip evidence  ({E.N:,} fictional values per group, seeded Generator({E.SEED}); "
      f"{s['n']} more worlds for the computed column; bits compared, NaN read as NaN counts as exact)")
    w("")
    w("1. Exact / changed / lost per writer and value group")
    w(f"{'writer':<18}" + "".join(f"{g:>22}" for g in E.GROUPS) + "   max ULP off")
    for k in E.WRITERS:
        cells = "".join(f"{d['exact']:>8}/{d['changed']:>6}/{d['lost']:>4}   " for d in wr[k].values())
        w(f"{k:<18}{cells}{max(d['max_ulp'] for d in wr[k].values()):>12,}")
    clean = [k for k in E.WRITERS if all(d["changed"] + d["lost"] == 0 for d in wr[k].values())]
    w("max ULP off counts adjacent doubles between sent and received; across a flush to 0.0 (pandas %.6f, float32) "
      "it is astronomical and means the value was destroyed, not nudged")
    w(f"bit-exact on all four groups: {len(clean)} of {len(E.WRITERS)} ({', '.join(clean)})")
    w("")
    w("2. A fixture of typed amounts passes every writer except float32")
    passes = [k for k in E.WRITERS if wr[k]["entered"]["exact"] == E.N]
    w(f"entered (two decimals) exact for {len(passes)} of {len(E.WRITERS)} writers; "
      f"the same writers on the computed column: "
      + ", ".join(f"{k} {100 * (1 - wr[k]['computed']['exact'] / E.N):.1f}% changed"
                  for k in passes if wr[k]["computed"]["exact"] < E.N))
    w("")
    w("3. The total does not move; the keys do (computed column, after the round trip)")
    w(f"{'writer':<18}{'total drift':>14}{'join keys lost':>16}   30-world range")
    for k in E.WRITERS:
        lg = r["ledger"][k]
        lo, hi = s["range"][k]
        w(f"{k:<18}{lg['drift']:>14.2e}{100 * lg['join_miss']:>15.1f}%   {100 * lo:.1f}-{100 * hi:.1f}%")
    w(f"total before = {r['ledger']['repr']['total_before']:,.2f}; 'drift' is math.fsum(after) - math.fsum(before)")
    w("")
    w("4. Edge values: what came back")
    w("in:  " + " | ".join(repr(x) for x in E.EDGE))
    for k in E.WRITERS:
        got = r["edge"][k]
        diff = [f"{E.EDGE[i]!r} -> {g}" for i, g in enumerate(got) if g != repr(E.EDGE[i])]
        w(f"{k:<18}" + ("all exact" if not diff else "; ".join(diff)))
    w("")
    w("5. pandas: the file is exact, the default reader is not")
    for k in ("pandas round_trip", "pandas csv", "pandas legacy"):
        d = wr[k]["measured"]
        w(f"{k:<18} measured column: {d['changed']} of {E.N} changed, worst {d['max_ulp']:,} ULP")
    return out


def main() -> None:
    r = E.study()
    s = E.seeds()
    json.dump({"study": r, "seeds": s}, open("results.json", "w"), indent=1)
    text = "\n".join(lines(r, s)) + "\n"
    open("evidence.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
