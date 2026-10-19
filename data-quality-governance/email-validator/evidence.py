"""Run the study, write results.json (read by the chart, the app and the tests) and evidence.txt."""

from __future__ import annotations

import json
from typing import List

import explore as E


def show(a: str) -> str:
    return repr(a) if len(a) <= 40 else repr(a[:18] + "..." + a[-14:]) + f" ({len(a.encode())} octets)"


def lines(r: dict) -> List[str]:
    out: List[str] = []
    w = out.append
    t = r["tiers"]
    w(
        f"email-validator evidence  ({len(E.CORPUS)} addresses: {t['ok']} deliverable, {t['grey']} valid-but-local, "
        f"{t['bad']} invalid; truth = RFC 5321 + RFC 6531 reference parser, agreeing with a hand label on every row; "
        f"no randomness, every number exact)"
    )
    w("")
    w("1. Each validator against the reference (FR = deliverable rejected, FA = invalid accepted)")
    w(f"{'validator':<22} {'FR':>3} {'FA':>3} {'grey ok':>8} {'sign-ups lost':>14}")
    for n, p in r["per"].items():
        w(
            f"{n:<22} {len(p['false_reject']):>3} {len(p['false_accept']):>3} {len(p['grey_accept']):>8} "
            f"{p['signups_lost']:>14.2%}"
        )
    w("")
    w("2. Which deliverable groups each validator rejects")
    for n, p in r["per"].items():
        w(f"{n:<22} {', '.join(p['groups_rejected']) or '-'}")
    w("")
    w("3. Deliverable addresses rejected by the most validators (of 10)")
    for a, k in sorted(r["ok_rejected_by_k"].items(), key=lambda x: -x[1])[:8]:
        w(f"{k:>2}  {show(a)}")
    w("")
    w("4. Invalid addresses accepted by the most validators (of 10)")
    for a, k in sorted(r["bad_accepted_by_k"].items(), key=lambda x: -x[1])[:8]:
        w(f"{k:>2}  {show(a)}  ({E.reference(a)[1]})")
    w("")
    w("5. The overlap")
    w(f"validators with zero errors: {len(r['perfect'])}")
    w(f"accept 'john@example.com\\n' (trailing newline): {len(r['newline_accepted_by'])} of 10")
    w(
        f"stacked {' + '.join(r['stack'])}: rejects {len(r['stack_false_reject'])} deliverable addresses, "
        f"{r['stack_signups_lost']:.2%} of sign-ups"
    )
    return out


def main() -> None:
    r = E.study()
    json.dump({"main": r, "share": E.SHARE}, open("results.json", "w"), indent=1, ensure_ascii=False)
    open("evidence.txt", "w").write("\n".join(lines(r)) + "\n")
    print("\n".join(lines(r)))


if __name__ == "__main__":
    main()
