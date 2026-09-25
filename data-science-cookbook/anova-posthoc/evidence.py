"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Nothing downstream recomputes a verdict.
Every rate is a Monte Carlo estimate at REPS replicates; every verdict is a 99% Wilson interval at
that count, and seeds are indices into the library's design lists, so results.json is identical
across processes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import posthoc as P

OUT: List[str] = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def rule(title: str) -> None:
    say()
    say("=" * 96)
    say(title)
    say("=" * 96)


def fwer_line(r: Dict[str, Any]) -> str:
    return f"  k={r['k']:>2} ({r['null_pairs']:>2} null pairs)" + "".join(
        f"{r[p + '_fwer']:>9.4f} {r[p + '_fwer_verdict'][:5]:<6}" for p in P.PROCS)


results: Dict[str, Any] = {"config": {
    "alpha": P.ALPHA, "band": [P.BAND_LO, P.BAND_HI], "reps": P.REPS, "n_per_group": P.N_PER_GROUP,
    "numpy": np.__version__, "scipy": __import__("scipy").__version__,
}}
HEAD = " " * 24 + "".join(f"{P.LABEL[p]:>16}" for p in P.PROCS)

# ------------------------------------------------------------------------ 1. calibration
rule("1.  CALIBRATION  -  checked against scipy, and three logical identities counted, not assumed")
cal = P.calibrate_against_scipy()
ids = P.identity_checks()
results["calibration"] = {**cal, **ids}
say(f"  Tukey decisions vs scipy.stats.tukey_hsd     {cal['tukey_decision_mismatches']} mismatches "
    f"in {cal['pairs_compared']} pairs")
say(f"  F p-value vs scipy.stats.f_oneway            max gap {cal['max_F_p_gap']:.1e}")
say(f"  over {ids['replicates']:,} complete-null replicates (k = {', '.join(map(str, P.KS))}):")
say(f"    pairs Bonferroni rejects that Holm does not       {ids['holm_misses_bonferroni_pair']}")
say(f"    replicates where Holm and Bonferroni differ on    {ids['holm_vs_bonferroni_any_differs_under_null']}"
    "   <- so their FWER is IDENTICAL under the complete null")
say("       'any rejection'")
say(f"    LSD rejections without a significant F            {ids['lsd_without_F']}")

# ------------------------------------------------------------------------ 2. complete null
rule(f"2.  COMPLETE NULL  -  all k means equal, n={P.N_PER_GROUP}/group, {P.REPS:,} reps; FWER = P(any pair called)")
null = P.null_table()
results["null"] = null
say(HEAD)
for r in null:
    say(fwer_line(r))
say()
say("  Tukey is EXACT here, so its column is the calibration cell: inside the band at every k.")

# ------------------------------------------------------------------------ 3. partial null
rule("3.  PARTIAL NULL  -  one group 3 SD away (F always significant), the other k-1 equal")
partial = P.partial_table()
results["partial"] = partial
say(HEAD)
for r in partial:
    say(fwer_line(r))
say()
say(f"  F significant in at least {min(r['F_sig'] for r in partial):.4f} of replicates on every row, so the")
say("  'protection' has already been spent: protected LSD IS the unadjusted column, to the replicate.")

# ------------------------------------------------------------------------ 4. power
rule("4.  POWER  -  per true pair; a procedure is compared only where its partial-null FWER is not INFLATED")
power = P.power_table(partial)
results["power"] = power
say("  design           F sig" + "".join(f"{P.LABEL[p]:>16}" for p in P.PROCS) + "   Tukey-Holm (99% +-)  winner")
for r in power:
    cells = "".join(f"{r[p + '_per_pair']:>15.4f}{'' if r[p + '_comparable'] else '*'}"
                    + (" " if r[p + "_comparable"] else "") for p in P.PROCS)
    say(f"  k={r['k']:>2} {r['shape']:<8} {r['F_sig']:>7.4f}{cells}   {r['tukey_minus_holm']:+.4f} "
        f"(+-{r['tukey_minus_holm_half']:.4f})  {r['tukey_vs_holm']}")
say("  * = not comparable: that procedure's FWER is INFLATED at this k, so its lead is not power")
tally = {w: sum(r["tukey_vs_holm"] == w for r in power) for w in ("tukey", "holm", "tie")}
results["tukey_vs_holm_tally"] = tally
say(f"\n  Tukey beats Holm materially on {tally['tukey']}, Holm beats Tukey on {tally['holm']}, "
    f"tie on {tally['tie']} of {len(power)}")

# ------------------------------------------------------------------------ 5. F vs post-hoc
rule("5.  THE F TEST AND THE PAIR HUNT DISAGREE, BOTH WAYS")
say("  design          P(F sig)  P(no Tukey pair | F sig)  P(no Holm pair | F sig)  P(Tukey pair, F not sig)")
for r in power:
    say(f"  k={r['k']:>2} {r['shape']:<8} {r['F_sig']:>9.4f} {r['tukey_no_pair_given_F']:>24.4f} "
        f"{r['holm_no_pair_given_F']:>24.4f} {r['tukey_pair_no_F']:>25.4f}")
worst = max(power, key=lambda r: r["tukey_no_pair_given_F"])
rev = max(power, key=lambda r: r["tukey_pair_no_F"])
results["disagreement"] = {"worst_no_pair": [worst["k"], worst["shape"], worst["tukey_no_pair_given_F"]],
                           "worst_pair_no_F": [rev["k"], rev["shape"], rev["tukey_pair_no_F"]]}

# ------------------------------------------------------------------------ 6. exemplar
rule("6.  ONE DATASET  -  first seed of a 5-group ladder where F is significant and Tukey names nothing")
ex = P.exemplar()
results["exemplar"] = {k: (v if k != "pairs" else [list(p) for p in v]) for k, v in ex.items()
                       if k != "significant"}
results["exemplar"]["significant"] = {p: [list(x) for x in v] for p, v in ex["significant"].items()}
say(f"  seed {ex['seed']}, group means " + ", ".join(f"{m:.2f}" for m in ex["means"]))
say(f"  ANOVA F p = {ex['F_p']:.4f}")
for p in P.PROCS:
    names = ", ".join(f"{chr(65 + i)}-{chr(65 + j)}" for i, j in ex["significant"][p]) or "none"
    say(f"    {P.LABEL[p]:<16} names: {names}")

with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
