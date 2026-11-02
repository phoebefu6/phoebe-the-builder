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
    n = r["clusters"]
    w(
        f"golden-record evidence  (fictional customers: {r['customers']} people, {n} captured by 2+ systems = "
        f"{r['records']} records from crm / billing / web; true details change over time; seeded Random({E.SEED}); "
        f"every person fictional)"
    )
    w("")
    w("1. Field accuracy: share of clusters where the kept value equals the customer's value today")
    w(f"{'rule-set':<17}" + "".join(f"{f:>10}" for f in E.FIELDS) + f"{'all five':>11}")
    for k in E.RULES:
        w(f"{k:<17}" + "".join(f"{pct(r['acc'][k][f]):>10}" for f in E.FIELDS) + f"{pct(r['whole'][k]):>11}")
    w(f"{'ceiling':<17}" + "".join(f"{pct(r['ceiling'][f]):>10}" for f in E.FIELDS) + f"{pct(r['oracle_whole']):>11}")
    w("ceiling = the true value is held by at least one record in the cluster; no rule can beat it")
    w("")
    w("2. Disagreement: the five generic rule-sets on the same cluster")
    w(f"clusters where all five keep the same golden record   {r['distinct'].get(1, 0)} of {n} ({pct(r['all_agree'])})")
    for k, v in r["distinct"].items():
        w(f"  {k} distinct golden record(s)                        {v}")
    w("share of clusters where the five disagree, per field   " +
      "  ".join(f"{f} {pct(v)}" for f, v in r["disagree_field"].items()))
    w("")
    w("3. Which timestamp does 'most recent' read? (address)")
    t = r["ts_fix"]
    w(f"record updated_at (an opt-in touch refreshes it)      {pct(t['record_ts'])}")
    w(f"the field's own capture day                           {pct(t['field_ts'])}")
    w(f"the field's capture day with lineage through the sync {pct(t['lineage_ts'])}")
    w(f"ceiling                                               {pct(r['ceiling']['address'])}")
    s = r["stale_sync"]
    w(f"clusters where CRM and billing share a stale address (synced) and web has the true one: {s['clusters']}; "
      f"majority keeps the stale one in {s['majority_stale']}")
    w("")
    w("4. The ceiling")
    w(f"clusters where at least one true value is held by NO system  {r['missing_any']} of {n} "
      f"({pct(r['missing_any'] / n)})")
    best = max(E.RULES, key=lambda k: r["whole"][k])
    w(f"best rule-set, all five fields right                         {best} {pct(r['whole'][best])}")
    w(f"a perfect chooser (picks the true value when one exists)     {pct(r['oracle_whole'])}")
    w("Survivorship chooses among the values the systems hold. When none holds today's value, only a fresh capture "
      "(ask the customer, verify against a payment) can fix the field.")
    return out


def main() -> None:
    r = E.study()
    json.dump(r, open("results.json", "w"), indent=1)
    text = "\n".join(lines(r)) + "\n"
    open("evidence.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
