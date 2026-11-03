"""Run the study, write results.json (read by the chart and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def lines(r: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    ev = ", ".join(f"{v} {k}" for k, v in r["events"].items())
    w(
        f"reference-data-versioning evidence  (fictional territory list: 40 territories, 5 regions, {E.MONTHS} months; "
        f"changes: {ev}; {r['backdated']} moves recorded 2-4 months late; {r['facts']} monthly sales facts; "
        f"seeded Random({E.SEED}); every territory fictional)"
    )
    w("")
    w("1. Rebuild last year's published reports (as-was): 180 region-month cells, 36 months")
    w(f"{'strategy':<12}{'cells same':>12}{'months same':>13}{'revenue reported':>18}{'dropped by join':>17}")
    for k in E.STRATEGIES:
        a = r["asw"][k]
        w(f"{k:<12}{a['cells']:>12}{a['months']:>13}{pct(a['reported']):>18}{pct(a['lost']):>17}")
    w("overwrite = today's table only; code_only = the full history joined on code alone; scd2_asof = history joined "
      "on code and sale month; bitemporal = sale month AND what the table said on the run day")
    w("")
    w("2. The reused code")
    u = r["reuse"]
    w(f"code {u['code']} retired in month {u['retired']} (territory absorbed), reused in month {u['reused']} for a new "
      f"territory in {u['new_region']}")
    w(f"revenue sold under the old {u['code']} ({u['old_region']}), which overwrite files under {u['new_region']}   "
      f"{u['rev']:,}")
    w("")
    w("3. Backdated corrections: what valid time alone cannot reproduce")
    a = r["asw"]["scd2_asof"]
    w(f"months scd2_asof rebuilds differently from what was published   {36 - a['months']} of 36 "
      f"({180 - a['cells']} cells)")
    w(f"rows kept: bitemporal table {r['rows_bitemporal']}   vs   a monthly snapshot of the table {r['rows_snapshots']}")
    w("")
    w("4. Restate to today's regions (as-is), within 1% of the true restatement")
    w(f"{'method':<12}{'cells':>8}{'months':>9}")
    for k, v in r["asis"].items():
        w(f"{k:<12}{v['cells']:>8}{v['months']:>9}")
    w("scd2_asof is the as-was answer: correct for what was reported, wrong for a trend on today's regions")
    w("")
    w("5. The split nobody can undo")
    w(f"revenue sold under a code that later split          {pct(r['split_pre_rev'])} of all revenue")
    w(f"of it put in the wrong region, 50/50 allocation      {pct(r['misalloc_even'])}")
    w(f"of it put in the wrong region, post-split shares     {pct(r['misalloc_post_share'])}")
    w("The sale never recorded which half it came from. No rule over the reference data recovers it; only capturing "
      "the finer location on the sale does.")
    return out


def main() -> None:
    r = E.study()
    json.dump(r, open("results.json", "w"), indent=1)
    text = "\n".join(lines(r)) + "\n"
    open("evidence.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
