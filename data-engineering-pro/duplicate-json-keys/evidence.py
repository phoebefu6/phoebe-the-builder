"""Write evidence.txt and results.json. Every number in the README comes from here. Seeded, fictional."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Dict, List

import explore as E


def study() -> Dict[str, Any]:
    feed = E.make_feed()
    return {"matrix": E.matrix(), "feed": E.feed_study(feed), "detect": E.detect(feed["lines"]),
            "n_lines": len(feed["lines"])}


def seeds(n: int = 30) -> Dict[str, Any]:
    """The detection gap and the refund count over 30 feeds. Reader behaviour itself is deterministic."""
    miss, refunds, hook_missed = [], [], []
    for s in range(1, n + 1):
        f = E.make_feed(seed=s)
        d = E.detect(f["lines"])
        miss.append(d["strict hook"] - d["regex"])
        refunds.append(sum(f["refunded"]))
        hook_missed.append(sum(f["refunded"]) - d["strict hook"])
    return {"regex_missed": miss, "refunds": refunds, "hook_missed": hook_missed}


def pairs(m: Dict[str, Dict[str, str]], fixture: str = "top level") -> Dict[str, int]:
    names = list(m)
    total = len(list(combinations(names, 2)))
    agree = sum(m[a][fixture] == m[b][fixture] for a, b in combinations(names, 2))
    return {"pairs": total, "disagree": total - agree}


def lines(r: Dict[str, Any], s: Dict[str, Any]) -> List[str]:
    m, fd, d = r["matrix"], r["feed"], r["detect"]
    fx = list(E.FIXTURES)
    out = [f"duplicate-json-keys evidence: {len(m)} readers, {len(fx)} fixtures, a {r['n_lines']:,}-line order "
           f"feed seeded Generator(215). Every value fictional.", ""]
    out.append("1. Which value each reader returns (fixture -> first / last / refuse)")
    out.append(f"   {'reader':<22}" + "".join(f"{f[:13]:<15}" for f in fx))
    for k, row in m.items():
        out.append(f"   {k:<22}" + "".join(f"{row[f][:13]:<15}" for f in fx))
    cls: Dict[str, List[str]] = {}
    for k, row in m.items():
        cls.setdefault(row["top level"], []).append(k)
    out.append("   top level: " + "; ".join(f"{c} {len(v)}" for c, v in cls.items()))
    p = pairs(m)
    out.append(f"   reader pairs that disagree on {{\"a\":1,\"a\":2}}: {p['disagree']} of {p['pairs']}")
    out.append("")
    t = fd["_truth"]
    out.append(f"2. The order feed: {t['refunds']} of {t['rows']:,} orders carry a refund patch "
               f"(status twice, refund last), ${t['amount']:,.2f} refunded")
    out.append(f"   {'reader':<22}{'rows read':>10}{'refunds':>9}{'refunded $':>13}  verdict")
    for k, v in fd.items():
        if k == "_truth":
            continue
        if v["raised"]:
            verdict = f"refused the whole file ({v['raised']})"
        elif v["refunds"] == t["refunds"]:
            verdict = "matches intent"
        else:
            verdict = f"missed {t['refunds'] - v['refunds']} refunds, ${t['amount'] - v['amount']:,.2f}, no error"
        out.append(f"   {k:<22}{v['rows']:>10,}{v['refunds']:>9}{v['amount']:>13,.2f}  {verdict}")
    out.append("")
    out.append(f"3. Pre-flight detection on the same feed: {d['strict hook']} duplicated lines")
    out.append(f"   regex for a repeated quoted key    {d['regex']:>3} found  (misses every \"st\\u0061tus\")")
    out.append(f"   json + object_pairs_hook (strict)  {d['strict hook']:>3} found")
    mi = s["regex_missed"]
    out.append(f"   over 30 seeds the regex missed {min(mi)}-{max(mi)} duplicated lines per feed "
               f"(feeds held {min(s['refunds'])}-{max(s['refunds'])}); the strict hook missed {min(s['hook_missed'])}-"
               f"{max(s['hook_missed'])}")
    out.append("")
    out.append("4. One engine, three entry points, two behaviours (duckdb, same document {\"a\":1,\"a\":2})")
    out.append(f"   read_json -> {m['duckdb read_json']['top level']}; read_json(ignore_errors=true) -> "
               f"{m['duckdb ignore_errors']['top level']}; json_extract -> {m['duckdb json_extract']['top level']}")
    out.append(f"   and {{\"A\":1,\"a\":2}} - not a duplicate - selecting column a returns "
               f"{m['duckdb read_json']['case only (not a dup)']}: identifiers are case-insensitive, 'a' became a_1")
    return out


def main() -> None:
    r, s = study(), seeds()
    out = lines(r, s)
    open("evidence.txt", "w").write("\n".join(out) + "\n")
    json.dump({"study": r, "seeds": s}, open("results.json", "w"), indent=1, default=str)
    print("\n".join(out))


if __name__ == "__main__":
    main()
