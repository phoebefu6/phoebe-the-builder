"""Run the study, write results.json (read by the chart, the app and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def lines(r: dict, s: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    sc = r["scenarios"]
    w(f"fanout-join-audit evidence  (fictional shop: {E.N_CUST:,} customers, {E.N_PROD} products, {E.N_ORD:,} orders; "
      f"report = revenue by tier and by category; seeded Generator({E.SEED}), {s['n']} more worlds; all fictional)")
    w("")
    w("1. What each broken join does to the report")
    w(f"{'join':<16}{'rows':>8}{'revenue':>9}   worst report cell")
    for k in E.SCENARIOS:
        v = sc[k]
        w(f"{k:<16}{pct(v['rows']):>8}{pct(v['revenue']):>9}   {v['worst_cell']} {pct(v['worst'])}")
    w("scd2_no_filter = customer history joined without WHERE is_current; dup_load = six product rows loaded twice; "
      "promo_bridge = promo codes joined in for display; late_inner = 3% of orders have a customer not loaded yet, "
      "inner join; rows_offset / revenue_offset = scd2_no_filter + late_inner, tuned so rows / revenue cancel; "
      "orphan_dups = duplicate keys only on products no order references")
    w("")
    w("2. Which pre-aggregate check fires (X = alarm), seeded world")
    w(f"{'join':<16}" + "".join(f"{c:>22}" for c in E.CHECKS))
    for k in E.SCENARIOS:
        f = sc[k]["fires"]
        w(f"{k:<16}" + "".join(f"{'X' if f[c] else '.':>22}" for c in E.CHECKS))
    broken = [k for k in E.SCENARIOS if k not in ("clean", "orphan_dups")]
    for c in E.CHECKS:
        caught = sum(sc[k]["fires"][c] for k in broken)
        false = int(sc["orphan_dups"]["fires"][c]) + int(sc["clean"]["fires"][c])
        w(f"{c:<22} catches {caught} of {len(broken)} broken joins, false alarms on {false} of 2 correct reports")
    same = all(s["fires"][k][c] == s["n"] * int(sc[k]["fires"][c]) for k in E.SCENARIOS for c in E.CHECKS)
    w(f"Across {s['n']} more worlds every check gives the same verdict on every join in every world: {same}")
    w("")
    w("3. The declared test passes on the biggest defect")
    v = sc["scd2_no_filter"]
    w(f"customer history: {r['scd2_rows']:,} rows for {E.N_CUST:,} customers; {r['n_upgraded']} upgraded customers "
      f"keep an old row")
    w(f"declared test unique(customer_id, valid_from) passes; the join on customer_id alone adds {pct(v['rows'])} "
      f"rows and {pct(v['revenue'])} revenue")
    w("")
    w("4. The row count understates the revenue error when the duplicates are not random")
    w(f"upgraded customers place {100 * r['upgraded_order_share']:.1f}% of orders and {100 * r['upgraded_revenue_share']:.1f}%"
      f" of revenue")
    for k in ("scd2_no_filter", "dup_load", "promo_bridge"):
        w(f"{k:<16} revenue inflation / row inflation   median {s['ratio_median'][k]:.2f}x   lowest "
          f"{s['ratio_min'][k]:.2f}x   ({s['n']} worlds)")
    w("A 1% row-tolerance check is a revenue tolerance only when the duplicated rows are average rows.")
    w("")
    w("5. Two defects that cancel")
    a, b = sc["rows_offset"], sc["revenue_offset"]
    w(f"rows_offset     row count matches exactly; revenue {pct(a['revenue'])}, {a['worst_cell']} {pct(a['worst'])}")
    w(f"revenue_offset  revenue total matches; rows {pct(b['rows'])}, {b['worst_cell']} {pct(b['worst'])}")
    w(f"rows_offset across {s['n']} worlds: revenue overstated by median {pct(s['offset_revenue_median'])}, "
      f"least {pct(s['offset_revenue_min'])}")
    w("Only pk_preserved (every source order exactly once in the joined rows) catches all six broken joins with no "
      "false alarm. It cannot say WHICH join broke; referenced_key_unique names the dimension.")
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
