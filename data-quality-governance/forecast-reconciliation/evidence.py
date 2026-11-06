"""Run the study, write results.json (read by the chart and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def lines(r: dict, s: dict, sh: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    w(f"forecast-reconciliation evidence  (fictional retailer: 1 total, {len(E.REGIONS)} regions, "
      f"{len(E.REGIONS) * E.STORES_PER} stores; {E.T} months of history, {E.H} months scored; "
      f"seeded Generator({E.SEED}), {s['n']} more worlds for the spread; every store fictional)")
    w("")
    w("1. Why the numbers do not add up")
    w(f"same linear model on every series, raw scale: largest gap      {100 * r['raw_scale_gap']:.2f}% of the total")
    w(f"log-scale model on every series: regions vs national forecast  {100 * r['base_regions_vs_total']:.2f}%")
    w(f"log-scale model on every series: stores vs national forecast   {100 * r['base_stores_vs_total']:.2f}%")
    w(f"store mix moved between history and the scored year           {100 * r['share_drift']:.1f}% of sales changed hands")
    w("")
    w("2. Seven ways to make them add up, RMSE change vs the unreconciled forecasts (minus = better)")
    w(f"{'method':<15}{'adds up':>8}{'total':>9}{'region':>9}{'store':>9}   {'30 worlds: total':>17}{'region':>9}"
      f"{'store':>9}{'better at all 3':>17}")
    for k in E.METHODS:
        a, m = r["methods"][k], s["mean"][k]
        adds = "yes" if a["gap"] < 1e-9 else "NO"
        w(f"{k:<15}{adds:>8}{pct(a['vs_base']['total']):>9}{pct(a['vs_base']['region']):>9}"
          f"{pct(a['vs_base']['store']):>9}   {pct(m['total']):>17}{pct(m['region']):>9}{pct(m['store']):>9}"
          f"{s['all_levels_better'][k]:>14} / {s['n']}")
    w("base = each series forecast on its own; bottom_up = sum the store forecasts; top_down_hist = split the national "
      "forecast by each store's average historical share; prorate = scale each level to the number above it; ols / "
      "wls / mint_shrink = least-squares reconciliation with identity, structural and shrunk-residual weights")
    w("")
    w("3. Which method wins where, across the worlds")
    for lv in E.LEVELS:
        bc = s["best_count"][lv]
        top = sorted(bc, key=lambda m: -bc[m])[:3]
        w(f"lowest {lv} RMSE   " + "   ".join(f"{m} {bc[m]}" for m in top) + f"   (of {s['n']})")
    tb = s["total_bias"]
    w(f"median bias of the national number   own model {pct(tb['base'])}   sum of store forecasts {pct(tb['bottom_up'])}")
    w("Stores grow at different rates; their sum is not one exponential, so the national log model under-forecasts.")
    w("")
    w("4. The check that passes for the worst method")
    td = r["methods"]["top_down_hist"]
    w(f"top_down_hist adds up exactly, and its region RMSE is {pct(td['vs_base']['region'])}, store "
      f"{pct(td['vs_base']['store'])} vs base")
    w("'Do the regions sum to the forecast?' passes for all six reconciled methods; it tests arithmetic, not accuracy.")
    w("")
    w("5. What no reconciliation can fix")
    w(f"one region's sales cut {100 * sh['shock']:.0f}% in the scored year, a cut no history contains ({sh['n']} worlds)")
    w("median over-forecast of the cut region   " + "   ".join(
        f"{m} {pct(sh['median_bias'][m])}" for m in ["base", "bottom_up", "wls", "mint_shrink"]))
    w("Every method redistributes the same forecasts; none of them contains the cut. Coherent and wrong.")
    return out


def main() -> None:
    r = E.study()
    s = E.seeds()
    sh = E.shock_study()
    json.dump({**r, "seeds": s, "shock": sh}, open("results.json", "w"), indent=1)
    text = "\n".join(lines(r, s, sh)) + "\n"
    open("evidence.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
