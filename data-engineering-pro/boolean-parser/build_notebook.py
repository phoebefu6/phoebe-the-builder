"""Generate demo.ipynb.

The notebook is self-contained - it does not import `boolparse`.  It
carries a *recorded* grid (this machine's 720 readings, baked in at build
time) so every cell renders on GitHub, and it re-derives as much of that
grid as the reader's machine can actually run, then diffs the two.  A
notebook that checks itself against the environment it landed in is worth
more than one that only replays a transcript.

    python build_notebook.py   ->  demo.ipynb (unexecuted)
"""

from __future__ import annotations

import json

import nbformat as nbf

import boolparse as B

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-engineering-pro/boolean-parser"

CODE_TO_VERDICT = {"T": B.TRUE, "F": B.FALSE, "!": B.REFUSED, "?": B.NOTBOOL}
VERDICT_TO_CODE = {v: k for k, v in CODE_TO_VERDICT.items()}


def recorded() -> dict:
    """The build machine's grid, compacted to one character per reading."""
    return {
        name: "".join(VERDICT_TO_CODE[r.verdict] for r in readings)
        for name, readings in B.grid().items()
    }


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    # ---------------------------------------------------------------- 1
    cells.append(md(f"""
# A string does not contain a boolean — a reader assigns one

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

**Day 154 of the FDE portfolio.** `"false"` comes back **true** in six of the
sixteen readers below, and that is not a bug in any of them.

A boolean reader is three separate decisions:

1. an **accept table** — which spellings mean true, which mean false;
2. a **normalisation** — case, whitespace, Unicode, applied *before* the lookup;
3. a **failure policy** — refuse, or quietly fall back to one of the two answers.

The word `true` in a config file carries none of them. Sixteen layers each pick
their own three, and the value changes as it crosses them.

**What this notebook does**

| § | |
|---|---|
| 1 | The corpus — 45 strings somebody actually typed |
| 2 | The 16 readers, and re-running as many as *your* machine can |
| 3 | The headline: no string is read the same way by all sixteen |
| 4 | `"false"` → true, six times |
| 5 | Ten of sixteen readers cannot fail |
| 6 | Succeeding is not deciding: the *not-a-boolean* verdict |
| 7 | The Norway problem — YAML 1.1 vs YAML 1.2 |
| 8 | In SQLite the string `'true'` is false (live, on your machine) |
| 9 | `Boolean(s)` and `s == true` are different readers |
| 10 | The file the value travelled in is part of the value |
| 11 | Normalisation: `lower()` and `casefold()` are different functions |
| 12 | The picture |
| 13 | Written for one reader, read by another — and it runs one way |
| 14 | Try your own |
"""))

    # ---------------------------------------------------------------- 2
    cells.append(md("""
## 1. The corpus

Forty-five strings. Every one of them is something a person has typed into a
`.env` file, a YAML key, a spreadsheet cell or a CSV column meaning *yes* or
*no* — plus the residue that files leave behind, which the author never typed
at all.

`intent` is what the author meant, where that is unambiguous. It is the
yardstick for **silently wrong**: a confident verdict *opposite* the intent.
"""))

    corpus_literal = json.dumps(
        [[s.text, s.family, s.intent, s.note] for s in B.CORPUS],
        ensure_ascii=False, indent=0,
    ).replace("\n", "")
    cells.append(code(f'''
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

CORPUS: List[Tuple[str, str, Optional[bool], str]] = json.loads(r"""{corpus_literal}""")
TEXTS = [row[0] for row in CORPUS]


def show(text: str) -> str:
    """Render a string so its invisible parts survive printing."""
    if text == "":
        return "''"
    out = []
    for ch in text:
        cp = ord(ch)
        if ch == "\\ufeff":
            out.append("<BOM>")
        elif ch == "\\r":
            out.append("<CR>")
        elif ch == " ":
            out.append("\\u2423")
        elif cp < 0x20 or cp == 0x7F:
            out.append("<%02X>" % cp)
        else:
            out.append(ch)
    return "".join(out)


LABELS = [show(t) for t in TEXTS]

print(f"{{len(CORPUS)}} strings\\n")
for family in dict.fromkeys(row[1] for row in CORPUS):
    members = [show(r[0]) for r in CORPUS if r[1] == family]
    print(f"  {{family:<11}} {{'  '.join(members)}}")
'''))

    # ---------------------------------------------------------------- 3
    cells.append(md("""
## 2. Sixteen readers — and how many of them your machine can run

Fourteen of the sixteen are real interpreters invoked at run time: node, git,
SQLite, awk, perl, ruby, jq, bash, PyYAML, ruamel. Two — Go's
`strconv.ParseBool` and Java's `Boolean.parseBoolean` — are transcribed from
their published contracts, because neither toolchain was on the build machine.

The grid below is **recorded** from the build machine so this notebook renders
in full on GitHub. The next cell re-derives whatever your machine can actually
run and diffs it against the recording, so you can see for yourself that none
of it is invented.
"""))

    rec = recorded()
    readers_literal = json.dumps(
        [[r.name, r.stack, r.source, r.can_refuse, r.seen_in] for r in B.READERS],
        ensure_ascii=False,
    )
    grid_literal = json.dumps(rec, ensure_ascii=False, indent=1)
    cells.append(code(f'''
TRUE, FALSE, REFUSED, NOTBOOL = "true", "false", "refused", "not-a-boolean"
CODE = {{"T": TRUE, "F": FALSE, "!": REFUSED, "?": NOTBOOL}}
CONFIDENT = (TRUE, FALSE)

READERS = json.loads(r"""{readers_literal}""")
RECORDED = json.loads(r"""{grid_literal}""")

GRID: Dict[str, List[str]] = {{name: [CODE[c] for c in row] for name, row in RECORDED.items()}}
NAMES = [r[0] for r in READERS]

print(f"{{'reader':<16}} {{'stack':<15}} {{'src':<5}} {{'refuses?':<9}} where you meet it")
print("-" * 88)
for name, stack, source, can_refuse, seen_in in READERS:
    print(f"{{name:<16}} {{stack:<15}} {{source:<5}} "
          f"{{('yes' if can_refuse else 'never'):<9}} {{seen_in}}")
print(f"\\n{{len(NAMES)}} readers x {{len(CORPUS)}} strings = {{len(NAMES) * len(CORPUS)}} readings")
'''))

    cells.append(md("""
### Re-deriving the grid here

Six of the readers need nothing but the Python standard library plus PyYAML, so
they run anywhere this notebook runs — including Colab and Binder. The rest need
an interpreter that may or may not be installed; where it is missing the cell
says so rather than pretending.
"""))

    cells.append(code('''
import shutil
import sqlite3
import subprocess


def live_py_truthy(texts): return [TRUE if bool(t) else FALSE for t in texts]


def live_json_strict(texts):
    out = []
    for t in texts:
        try:
            v = json.loads(t)
        except Exception:
            out.append(REFUSED)
            continue
        out.append((TRUE if v else FALSE) if isinstance(v, bool) else NOTBOOL)
    return out


def live_sqlite_where(texts):
    con = sqlite3.connect(":memory:")
    try:
        return [TRUE if con.execute("SELECT CASE WHEN ? THEN 1 ELSE 0 END", (t,)
                                    ).fetchone()[0] else FALSE for t in texts]
    finally:
        con.close()


def live_yaml11(texts):
    import yaml
    out = []
    for t in texts:
        try:
            v = yaml.safe_load(t)
        except Exception:
            out.append(REFUSED)
            continue
        out.append((TRUE if v else FALSE) if isinstance(v, bool) else NOTBOOL)
    return out


GO_TRUE = {"1", "t", "T", "TRUE", "true", "True"}
GO_FALSE = {"0", "f", "F", "FALSE", "false", "False"}


def live_go(texts):
    return [TRUE if t in GO_TRUE else FALSE if t in GO_FALSE else REFUSED for t in texts]


def live_java(texts):
    return [TRUE if (t.lower() == "true" and t.isascii()) else FALSE for t in texts]


def live_ruby(texts):
    if not shutil.which("ruby"):
        return None
    out = subprocess.run(["ruby", "-e", 'ARGV.each { |a| puts(a ? "T" : "F") }', *texts],
                         capture_output=True, text=True)
    return [CODE[c] for c in out.stdout.split()]


def live_awk(texts):
    if not shutil.which("awk"):
        return None
    out = subprocess.run(["awk", '{ print ($0 ? "T" : "F") }'],
                         input="\\n".join(texts) + "\\n", capture_output=True, text=True)
    return [CODE[c] for c in out.stdout.split()]


CHECKS = {
    "py_truthy": live_py_truthy, "json_strict": live_json_strict,
    "sqlite_where": live_sqlite_where, "yaml11": live_yaml11,
    "go_parsebool": live_go, "java_parsebool": live_java,
    "ruby_truthy": live_ruby, "awk_field": live_awk,
}

matched, skipped, drifted = [], [], []
for name, fn in CHECKS.items():
    got = fn(TEXTS)
    if got is None:
        skipped.append(name)
    elif got == GRID[name]:
        matched.append(name)
    else:
        drifted.append((name, [(show(TEXTS[i]), a, b)
                               for i, (a, b) in enumerate(zip(got, GRID[name])) if a != b]))

print(f"re-derived live on this machine and MATCHED the recording: {len(matched)}")
for n in matched:
    print(f"    {n}")
print(f"\\ninterpreter not installed here, recording used: {len(skipped)} {skipped}")
if drifted:
    print("\\nDRIFT - your environment disagrees with the build machine:")
    for name, rows in drifted:
        print(f"  {name}: {rows}")
else:
    print("\\nno drift.")
'''))

    # ---------------------------------------------------------------- 4
    cells.append(md("""
## 3. The headline: there is no string that means true

Not one of the 45 strings is read the same way by all sixteen readers. Not
`yes`, not `1`, not `true`.
"""))
    cells.append(code('''
import collections


def verdicts_for(i: int) -> Dict[str, str]:
    return {name: GRID[name][i] for name in NAMES}


def distinct(i: int) -> int:
    return len(set(verdicts_for(i).values()))


unanimous = [LABELS[i] for i in range(len(CORPUS)) if distinct(i) == 1]
print(f"strings read the same way by all {len(NAMES)} readers: {len(unanimous)} {unanimous}\\n")

for text in ("true", "false", "1", "0", "yes"):
    i = TEXTS.index(text)
    c = collections.Counter(verdicts_for(i).values())
    print(f"  {show(text):<7} -> " + ",  ".join(f"{k}={c[k]}" for k in
                                                (TRUE, FALSE, REFUSED, NOTBOOL) if c[k]))

i = TEXTS.index("true")
dissent = sorted(n for n, v in verdicts_for(i).items() if v != TRUE)
print(f"\\nEven 'true' is not unanimous. Not reading it as true: {dissent}")
print("js_loose_eq is `s == true`, which numifies BOTH sides; Number('true') is NaN.")
'''))

    # ---------------------------------------------------------------- 5
    cells.append(md("""
## 4. `"false"` came back true

Six readers. Every one of them is a **truthiness** reader: it never consulted a
boolean table at all. It asked *is this string non-empty*, and for `"false"` the
answer is yes.
"""))
    cells.append(code('''
i = TEXTS.index("false")
v = verdicts_for(i)
print(f"{'reader':<16} verdict")
print("-" * 44)
for name in NAMES:
    flag = "   <-- the bug" if v[name] == TRUE else ""
    print(f"{name:<16} {v[name]}{flag}")

trues = sorted(n for n, x in v.items() if x == TRUE)
print(f"\\n{len(trues)} of {len(NAMES)} readers read the string 'false' as true:")
print("   " + ", ".join(trues))
'''))

    # ---------------------------------------------------------------- 6
    cells.append(md("""
## 5. Ten of sixteen readers cannot fail

A reader with no failure policy returns a confident boolean for every string it
is ever handed — including `undefined`, a bare UTF-8 BOM and the empty string.
It cannot tell you that you spelled it wrong. It can only be quietly wrong.
"""))
    cells.append(code('''
INTENT = [row[2] for row in CORPUS]

refusals = {n: GRID[n].count(REFUSED) for n in NAMES}
deferrals = {n: GRID[n].count(NOTBOOL) for n in NAMES}
wrong = {
    n: sum(1 for i, v in enumerate(GRID[n])
           if INTENT[i] is not None and v in CONFIDENT and (v == TRUE) != INTENT[i])
    for n in NAMES
}

never = [n for n in NAMES if refusals[n] == 0 and deferrals[n] == 0]
print(f"{'reader':<16} {'refused':>8} {'deferred':>9} {'silently wrong':>15}")
print("-" * 52)
for n in NAMES:
    print(f"{n:<16} {refusals[n]:>8} {deferrals[n]:>9} {wrong[n]:>15}")

print(f"\\n{len(never)} of {len(NAMES)} readers answer every string confidently:")
print("   " + ", ".join(never))
print(f"\\nOnly {sum(1 for n in NAMES if refusals[n])} readers ever refuse anything.")
print("The two columns are mutually exclusive: no reader is both permissive and safe.")
assert all(not (wrong[n] and refusals[n]) for n in NAMES)
'''))

    # ---------------------------------------------------------------- 7
    cells.append(md("""
## 6. Succeeding is not deciding

A YAML or JSON reader that hands back `1`, `'yes'` or `None` has not failed and
has not decided. The decision has been *deferred* to whichever `if value:` runs
next — in a different file, written by somebody else, with no idea what the
original string was.

`yaml11` and `yaml12` score zero for both refusals and silent errors. That is
not a clean bill of health; it is what deferring 25 and 35 of the 45 strings
looks like.
"""))
    cells.append(code('''
for name in ("yaml11", "yaml12", "json_strict"):
    idx = [i for i, v in enumerate(GRID[name]) if v == NOTBOOL]
    print(f"{name}: {len(idx)}/{len(CORPUS)} strings come back as something that is not a bool")
    print("   " + "  ".join(LABELS[i] for i in idx[:12]) + ("  ..." if len(idx) > 12 else ""))
    print()
'''))

    # ---------------------------------------------------------------- 8
    cells.append(md("""
## 7. The Norway problem

YAML 1.1's implicit resolver makes `yes`, `no`, `on` and `off` booleans. YAML
1.2 deleted them from the core schema for exactly this reason. PyYAML still
implements 1.1; Go's `yaml.v3` and the JS `yaml` package implement 1.2. An
unquoted `NO` in a country column is the boolean **false** in one and the string
`"NO"` in the other.
"""))
    cells.append(code('''
differ = [i for i in range(len(CORPUS)) if GRID["yaml11"][i] != GRID["yaml12"][i]]
print(f"{'string':<8} {'YAML 1.1':<16} {'YAML 1.2':<16}")
print("-" * 44)
for i in differ:
    print(f"{LABELS[i]:<8} {GRID['yaml11'][i]:<16} {GRID['yaml12'][i]:<16}")
print(f"\\n{len(differ)} of {len(CORPUS)} strings change meaning between the two YAML versions.")
print("\\nWorth knowing separately: YAML 1.1's type repository lists `y` and `n` as")
print("booleans, and PyYAML does not implement them - so 'y' is the string 'y'.")
for t in ("y", "n"):
    print(f"   {t!r}: yaml11 -> {GRID['yaml11'][TEXTS.index(t)]}")
'''))

    # ---------------------------------------------------------------- 9
    cells.append(md("""
## 8. In SQLite, the string `'true'` is false

This one runs live on your machine — `sqlite3` ships with Python.

SQLite has no boolean type. A TEXT value in boolean position is cast to a
number, and a string that does not begin with digits casts to `0`. A feature
flag stored as the string `'true'` is **off**, in every row, forever, and
nothing ever raises.
"""))
    cells.append(code('''
con = sqlite3.connect(":memory:")
con.execute("CREATE TABLE feature (name TEXT, flag TEXT)")
con.executemany("INSERT INTO feature VALUES (?, ?)",
                [("beta", "true"), ("dark_mode", "yes"), ("audit", "on"),
                 ("legacy", "1"), ("off_by_default", "0")])

print("stored as TEXT, then queried the way everybody queries a flag:\\n")
print("  SELECT name FROM feature WHERE flag")
print("  ->", [r[0] for r in con.execute("SELECT name FROM feature WHERE flag")])
print("\\n  SELECT name FROM feature WHERE flag = TRUE")
print("  ->", [r[0] for r in con.execute("SELECT name FROM feature WHERE flag = TRUE")])

truthy = [LABELS[i] for i, v in enumerate(GRID["sqlite_where"]) if v == TRUE]
print(f"\\nOf the {len(CORPUS)} corpus strings, {len(truthy)} are truthy in a WHERE clause: {truthy}")
print("Every word-shaped spelling of true selects zero rows.")
con.close()
'''))

    # ---------------------------------------------------------------- 10
    cells.append(md("""
## 9. `Boolean(s)` and `s == true` are different readers

The `== true` comparison is the fix people reach for after the first bug. Loose
equality against a boolean converts **both** sides to numbers, so `"1" == true`
is `true` and `"true" == true` is `false`. The fix inverts the original bug
instead of removing it.
"""))
    cells.append(code('''
disagree = [i for i in range(len(CORPUS)) if GRID["js_truthy"][i] != GRID["js_loose_eq"][i]]
print(f"Boolean(s) and (s == true) disagree on {len(disagree)} of {len(CORPUS)} strings.\\n")
print(f"{'string':<12} {'Boolean(s)':<12} {'s == true':<12}")
print("-" * 40)
for text in ("true", "false", "1", "0", "yes", "TRUE", ""):
    i = TEXTS.index(text)
    print(f"{LABELS[i]:<12} {GRID['js_truthy'][i]:<12} {GRID['js_loose_eq'][i]:<12}")
'''))

    # ---------------------------------------------------------------- 11
    cells.append(md("""
## 10. The file the value travelled in is part of the value

The author wrote `true` in all four of these. What arrived was `true` plus
whatever the file format left behind: a `\\r` from CRLF line endings, a UTF-8
BOM on line 1, a space after the `=`.

Day 151 showed that a file has no lines in it until a splitter makes them. Here
the splitter's leftovers change a boolean.
"""))
    cells.append(code('''
for text, why in (("true\\r", "a .env file saved with CRLF line endings"),
                  ("\\ufefftrue", "a UTF-8 BOM on the first line"),
                  (" true", "a space after the = sign"),
                  ("true ", "a trailing space nobody sees in a diff")):
    i = TEXTS.index(text)
    v = verdicts_for(i)
    t = sorted(n for n, x in v.items() if x == TRUE)
    f = sorted(n for n, x in v.items() if x == FALSE)
    e = sorted(n for n, x in v.items() if x == REFUSED)
    print(f"{show(text):<12} {why}")
    print(f"    true    ({len(t):>2}): {', '.join(t)}")
    print(f"    false   ({len(f):>2}): {', '.join(f)}")
    print(f"    refused ({len(e):>2}): {', '.join(e) or '-'}\\n")
'''))

    # ---------------------------------------------------------------- 12
    cells.append(md("""
## 11. Normalisation: `lower()` and `casefold()` are different functions

Not a strictness ordering — different functions. `casefold()` maps LATIN SMALL
LETTER LONG S (`ſ`) to `s`, so `FALſE` casefolds to `false` and is **accepted**
by a casefolding reader that a lowercasing reader refuses. Fullwidth `ＴＲＵＥ`
needs NFKC before any casing helps it at all.
"""))
    cells.append(code('''
import unicodedata

BASE_TABLE = {"true", "t", "yes", "y", "on", "1", "false", "f", "no", "n", "off", "0"}
EXTENDED_TABLE = BASE_TABLE | {"enabled", "disabled"}


def normalisations(text: str) -> Dict[str, str]:
    return {
        "as-written": text,
        "strip": text.strip(),
        "lower": text.lower(),
        "casefold": text.casefold(),
        "NFKC+casefold": unicodedata.normalize("NFKC", text).casefold(),
    }


keys = list(normalisations("x"))
print(f"{'string':<12} " + " ".join(f"{k:<14}" for k in keys))
print("-" * 84)
for text in ("TRUE", "tRuE", " true", "true\\r", "ye\\u017f", "FAL\\u017fE", "\\uff34\\uff32\\uff35\\uff25"):
    row = {k: (v in BASE_TABLE) for k, v in normalisations(text).items()}
    print(f"{show(text):<12} " + " ".join(f"{str(row[k]):<14}" for k in keys))

print("\\n'FAL\\u017fE'.casefold() ->", repr("FAL\\u017fE".casefold()))
print("'FAL\\u017fE'.lower()    ->", repr("FAL\\u017fE".lower()))
print("\\nAnd the Turkish-I hazard reaches exactly one literal in this vocabulary:")
print("  'DISABLED' lowercased in a tr locale is 'd\\u0131sabled', which matches nothing.")
print("  The twelve core literals are immune only because not one contains the letter I:")
print("  ", sorted(BASE_TABLE))
assert all("i" not in w for w in BASE_TABLE)
'''))

    # ---------------------------------------------------------------- 13
    cells.append(md("""
## 12. The picture

Left: every one of the 720 readings, one cell each. Not a single row is one
colour — that is the whole finding in one image.

Right: refusals against silent errors. The two sides never both light up for the
same reader.
"""))
    cells.append(code('''
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRIDC, PAPER = "#1d1a17", "#8a8178", "#e3ddd5", "#faf7f2"
ACCENT, COOL, WARM, GREEN = "#c8553d", "#2f6f8f", "#e0a458", "#4f7942"
COLOUR = {TRUE: COOL, FALSE: WARM, REFUSED: GREEN, NOTBOOL: "#cfc7bd"}
ORDER = [TRUE, FALSE, REFUSED, NOTBOOL]
plt.rcParams.update({"figure.facecolor": PAPER, "axes.facecolor": PAPER,
                     "text.color": INK, "font.size": 8})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1.15, 1]})

idx = {v: i for i, v in enumerate(ORDER)}
data = np.array([[idx[GRID[n][j]] for n in NAMES] for j in range(len(CORPUS))])
ax1.imshow(data, cmap=matplotlib.colors.ListedColormap([COLOUR[v] for v in ORDER]),
           aspect="auto", vmin=-0.5, vmax=3.5, interpolation="nearest")
ax1.set_xticks(range(len(NAMES)))
ax1.set_xticklabels(NAMES, rotation=90, fontsize=7)
ax1.set_yticks(range(len(CORPUS)))
ax1.set_yticklabels([lab.replace("\\uff34\\uff32\\uff35\\uff25", "[fullwidth]TRUE")
                     for lab in LABELS],
                    fontsize=6)
for x in range(1, len(NAMES)):
    ax1.axvline(x - 0.5, color=PAPER, lw=0.6)
for y in range(1, len(CORPUS)):
    ax1.axhline(y - 0.5, color=PAPER, lw=0.4)
ax1.tick_params(length=0)
for s in ax1.spines.values():
    s.set_visible(False)
ax1.set_title(f"{len(CORPUS)} strings x {len(NAMES)} readers = {len(CORPUS)*len(NAMES)} readings\\n"
              f"{len(unanimous)} rows are a single colour",
              loc="left", fontsize=10, fontweight="bold")
ax1.legend(handles=[mpatches.Patch(color=COLOUR[v], label=v) for v in ORDER],
           loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7)

order = sorted(NAMES, key=lambda n: refusals[n] + deferrals[n])
y = np.arange(len(order))
span = max(max(wrong.values()), max(refusals[n] + deferrals[n] for n in order))
ax2.barh(y, [-wrong[n] for n in order], color=ACCENT, height=0.66, label="silently wrong")
ax2.barh(y, [refusals[n] for n in order], color=GREEN, height=0.66, label="refused")
ax2.barh(y, [deferrals[n] for n in order], left=[refusals[n] for n in order],
         color=COLOUR[NOTBOOL], height=0.66, label="deferred")
ax2.axvline(0, color=INK, lw=0.9)
ax2.set_yticks([])
for i, n in enumerate(order):
    left = wrong[n] > 0
    ax2.text(-span * 0.02 if left else span * 0.02, i, n,
             ha="right" if left else "left", va="center", fontsize=7.5)
ax2.set_xlim(-span * 1.15, span * 1.15)
ax2.set_xlabel("<- wrong quietly            told you ->")
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(length=0)
ax2.legend(frameon=False, fontsize=7, loc="lower right")
ax2.set_title("no reader is both permissive and safe", loc="left",
              fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig("notebook_boolean.png", dpi=140, facecolor=PAPER)
plt.show()
'''))

    # ---------------------------------------------------------------- 14
    cells.append(md("""
## 13. Written for one reader, read by another — and it runs one way

Take the strings a given reader reads *confidently* — the spellings somebody
would plausibly write into a config authored against it — and count how many a
second reader fails to reproduce.

The matrix is **not symmetric**, and the asymmetry is the actionable part:
migrating a config from a permissive reader to a strict one breaks almost
everything, while the reverse direction is nearly free.
"""))
    cells.append(code('''
def lost(writer: str, reader: str) -> int:
    return sum(1 for a, b in zip(GRID[writer], GRID[reader]) if a in CONFIDENT and b != a)


print(f"{'A':<16} {'B':<16} {'A->B':>6} {'B->A':>6}")
print("-" * 48)
for a, b in (("py_truthy", "json_strict"), ("ruby_truthy", "go_parsebool"),
             ("js_truthy", "git_bool"), ("yaml11", "yaml12"),
             ("bash_eq_true", "py_truthy")):
    print(f"{a:<16} {b:<16} {lost(a, b):>6} {lost(b, a):>6}")

pairs = [(a, b) for a in NAMES for b in NAMES if a != b]
clean = [p for p in pairs if lost(*p) == 0]
print(f"\\n{len(pairs)} ordered pairs; {len(clean)} lose nothing.")
print("Migrations have a safe direction, and it points at strictness.")
'''))

    # ---------------------------------------------------------------- 15
    cells.append(md("""
## 14. Try your own

Drop your own strings in and see which of your stacks disagree about them. If
you have a config file to hand, paste its actual boolean values — the interesting
ones are never `true` and `false`.
"""))
    cells.append(code('''
# MY_STRINGS = ["TRUE", "Y", "on ", "vrai", "\\u662f"]
#
# import shutil, subprocess, json as _json
#
# def audit(strings):
#     rows = {}
#     rows["py_truthy"]    = [TRUE if bool(s) else FALSE for s in strings]
#     rows["json_strict"]  = live_json_strict(strings)
#     rows["sqlite_where"] = live_sqlite_where(strings)
#     rows["yaml11"]       = live_yaml11(strings)
#     rows["go_parsebool"] = live_go(strings)
#     rows["java_parsebool"] = live_java(strings)
#     if shutil.which("node"):
#         js = _json.loads(subprocess.run(
#             ["node", "-e",
#              "const xs=JSON.parse(require('fs').readFileSync(0,'utf8'));"
#              "process.stdout.write(JSON.stringify(xs.map(s=>[Boolean(s),s==true])))"],
#             input=_json.dumps(strings), capture_output=True, text=True).stdout)
#         rows["js_truthy"]   = [TRUE if a else FALSE for a, _ in js]
#         rows["js_loose_eq"] = [TRUE if b else FALSE for _, b in js]
#     return rows
#
# rows = audit(MY_STRINGS)
# print(f"{'string':<14} " + " ".join(f"{n[:11]:<12}" for n in rows))
# for i, s in enumerate(MY_STRINGS):
#     print(f"{show(s):<14} " + " ".join(f"{rows[n][i]:<12}" for n in rows))
# for i, s in enumerate(MY_STRINGS):
#     vs = {rows[n][i] for n in rows}
#     if TRUE in vs and FALSE in vs:
#         print(f"\\n{show(s)!r} flips sign across your stack.")
'''))

    # ---------------------------------------------------------------- 16
    cells.append(md(f"""
## What to take away

A boolean reader is an **accept table**, a **normalisation** and a **failure
policy**. If you do not choose them, sixteen layers choose sixteen different
combinations on your behalf.

1. **Name the accept table in the schema**, not in the code that reads it.
   `true|false` is a defensible table. So is git's. `bool(s)` is not a table.
2. **Choose the normalisation deliberately.** Strip first — all four residue
   strings in §10 are a stripping bug — then `casefold()`, not `lower()`, and
   NFKC only if fullwidth input is possible.
3. **Refuse.** A reader that returns `false` for an unrecognised string has
   thrown away the only fact it had: that nobody knows what the value means.
   Only four of these sixteen readers can do that.

And the rule §8 earns, which is not "pick a better parser": **do not store a
boolean as text**, so that nothing downstream has to read one. A `BOOLEAN`
column, or an `INTEGER` holding 0 and 1, is never parsed. That is the whole fix.
Everything above is what it costs not to do it.

---

**Full audit:** `python evidence.py` in
[`{PATH}`](https://github.com/{REPO}/tree/main/{PATH}) prints all sixteen
sections, and `pytest` asserts every number on this page.

**Interactive version:** `streamlit run app.py` — paste a string, see all sixteen
readers disagree about it live.

Part of [phoebe-the-builder](https://github.com/{REPO}) — one small, real tool a day.
Days 147, 149-153 are the rest of this arc: a duration, a header name, a sort
order, a line ending, a number and a truncation, each of which turns out to be a
reader's decision rather than a property of the bytes.
"""))

    nb["cells"] = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


if __name__ == "__main__":
    nbf.write(build(), "demo.ipynb")
    print(f"wrote demo.ipynb ({len(build()['cells'])} cells)")
