"""Run the study, write results.json (read by the chart and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def lines(r: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    c = r["ceiling"]
    w(
        f"name-splitter evidence  ({r['n']} fictional full names in {len(r['groups'])} naming conventions; truth = "
        f"hand-labelled given + family from each person's own convention; no randomness, every number exact)"
    )
    w("")
    w("1. Each splitter against the truth (both right = given AND family exactly right)")
    w(f"{'splitter':<28} {'both right':>10} {'given wrong':>12} {'family wrong':>13} {'greeted wrong':>14}")
    for n, p in r["per"].items():
        w(
            f"{n:<28} {p['both_right']:>7}/{r['n']} {len(p['given_wrong']):>12} {len(p['family_wrong']):>13} "
            f"{p['greeted_wrong']:>14.2%}"
        )
    w("")
    w("2. Share of each convention split exactly right")
    w(f"{'convention':<28} " + " ".join(f"{i + 1:>4}" for i in range(len(r["per"]))))
    for g in r["groups"]:
        w(f"{g:<28} " + " ".join(f"{r['by_group'][n][g]:>4.0%}" for n in r["per"]))
    w("   columns: " + "; ".join(f"{i + 1} = {n}" for i, n in enumerate(r["per"])))
    w("")
    w("3. Greetings that came out as a title or a particle ('Dear ___')")
    for n, p in r["per"].items():
        if p["greeted_as_affix"]:
            w(f"{n:<28} " + ", ".join(f"{s!r} -> {g!r}" for s, g in p["greeted_as_affix"]))
    w("")
    w(f"4. Names every splitter gets wrong ({len(r['wrong_by_all'])})")
    for s in r["wrong_by_all"]:
        w(f"   {s}")
    w("")
    w("5. The ceiling: the best any splitter that sees only the string can do")
    w(f"by string: {c['by_string']}/{r['n']}  (same string, two truths: {', '.join(c['ambiguous_strings'])})")
    w(
        f"by shape : {c['by_shape']}/{r['n']}  (three plain words W W W: middle name {c['WWW'].get('G-F', 0)}, "
        f"two surnames or double-barrelled {c['WWW'].get('GFF', 0)})"
    )
    w(f"best splitter here: {r['best']}, {r['per'][r['best']]['both_right']}/{r['n']}")
    return out


def main() -> None:
    r = E.study()
    json.dump({"main": r, "share": E.SHARE}, open("results.json", "w"), indent=1, ensure_ascii=False)
    open("evidence.txt", "w").write("\n".join(lines(r)) + "\n")
    print("\n".join(lines(r)))


if __name__ == "__main__":
    main()
