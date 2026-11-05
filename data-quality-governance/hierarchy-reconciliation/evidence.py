"""Run the study, write results.json (read by the chart and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def lines(r: dict, s: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    d = r["defects"]
    w(
        f"hierarchy-reconciliation evidence  (fictional cost-centre hierarchy: 5 regions, {r['areas']} areas, "
        f"{r['leaves']} cost centres, {r['rows']} extract rows; {r['posts']} postings, ledger {r['ledger']:,.0f}; "
        f"seeded Random({E.SEED}); every cost centre fictional)"
    )
    w("")
    w("1. What a validator finds in the extract")
    w(f"cost centres with two parent rows (moved, old row kept)   {d['multi_parent']}   "
      f"{pct(r['share']['moved'])} of postings")
    w(f"orphans (parent area retired, row kept)                 {d['orphan']}   {pct(r['share']['orphan'])}")
    w(f"nodes cut off by one cycle (area parent = own child)    {d['cycle']}   {pct(r['share']['cycle'])}")
    w(f"codes posted to but missing from the extract            {d['unknown_code']}   {pct(r['share']['unknown'])}")
    w(f"posted at area level, above the leaves                  -    {pct(r['share']['internal'])}")
    w("")
    w("2. Five roll-ups, scored per posted dollar")
    w(f"{'method':<15}{'regions=total':>14}{'total=ledger':>13}{'total':>8}{'right':>8}{'misfiled':>10}"
      f"{'unplaced':>10}{'twice':>8}")
    for k in E.METHODS:
        a = r["methods"][k]
        yn = lambda b: "yes" if b else "NO"  # noqa: E731
        w(f"{k:<15}{yn(a['sums_to_total']):>14}{yn(a['total_is_ledger']):>13}{pct(a['total']):>8}"
          f"{pct(a['right']):>8}{pct(a['misfiled']):>10}{pct(a['dropped']):>10}{pct(a['double']):>8}")
    w("naive_walk = recursive descent on every edge, leaf postings only; plug = naive_walk + 'Other' = ledger minus "
      "regions; first_parent / latest_parent = one row per child by file order / by latest load; validated = "
      "latest_parent + unplaced dollars named in an Unassigned bucket")
    w("")
    w("3. The plug")
    p = r["methods"]["plug"]
    w(f"net plug that makes the regions tie to the ledger      {pct(r['plug_net'])}")
    w(f"dollars actually wrong underneath it                    {pct(p['wrong'])}   "
      f"({p['wrong'] / r['plug_net']:.1f}x the plug)")
    w(f"across {s['n']} seeds, wrong / plug: median {s['gross_over_net_median']:.1f}x, "
      f"smallest {s['gross_over_net_min']:.1f}x")
    w("")
    w("4. Which parent row to keep")
    w(f"moved cost centres filed under their OLD area, first row in file   {r['first_wrong_moves']} of "
      f"{d['multi_parent']}")
    w(f"moved cost centres filed under their OLD area, latest load        {r['latest_wrong_moves']} of "
      f"{d['multi_parent']}")
    w("")
    w("5. What no roll-up rule can place")
    w(f"unplaced after validation (orphans + cycle + unknown codes)   {pct(r['quarantined'])} of the ledger")
    w(f"regions within 0.5% of the truth, validated                   {r['methods']['validated']['regions_right']} of 5")
    w(f"across {s['n']} seeds, validated still leaves {pct(s['validated_err_min'])} to {pct(s['validated_err_max'])} "
      "unplaced")
    w("The extract does not say where an orphan or a cycle belongs. Only a fix in the source hierarchy places it.")
    return out


def main() -> None:
    r = E.study()
    s = E.seeds()
    json.dump({**r, "seeds": s}, open("results.json", "w"), indent=1)
    text = "\n".join(lines(r, s)) + "\n"
    open("evidence.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
