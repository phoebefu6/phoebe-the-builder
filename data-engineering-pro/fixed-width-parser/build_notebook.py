"""Generates demo.ipynb with fwf.py and evidence.py embedded.

The notebook writes both modules to disk from embedded source, then imports them.
That keeps it self-contained on Colab and Binder without a clone step, and means
there is no second copy of the logic that can drift from the tested one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-engineering-pro/fixed-width-parser"

HERE = Path(__file__).parent


def md(text: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def embed(name: str) -> str:
    """Embed a module as a JSON-encoded literal.

    Not a raw triple-quoted string: both modules contain docstrings, so any
    triple-quote wrapper terminates early. json.dumps produces escapes that are
    valid Python string-literal syntax, which side-steps the whole problem.
    """
    src = (HERE / name).read_text()
    return (
        f"_src = {json.dumps(src)}\n"
        f"from pathlib import Path\n"
        f'Path("{name}").write_text(_src)\n'
        f'print("wrote {name}:", len(_src.splitlines()), "lines")\n'
    )


CELLS: List[Dict[str, Any]] = [
    md(
        f"""# Fixed-width files are a byte format

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

**Day 138 - Data Engineering Pro.**

`pandas.read_fwf` decodes the record to a `str`, then slices by character offset. On
pure-ASCII data byte offsets and character offsets are the same number, so it works, and
it keeps working right up until a customer with an umlaut in their name signs up.

This notebook takes six things that decide whether a flat file loads correctly and turns
each into a measurement:

| | mechanism | effect on the sample file |
|---|---|---|
| A | byte offsets vs character offsets | 4 of 12 records scrambled |
| B | 1-indexed spec read 0-indexed | totals 9.6x too large |
| C | signed zoned decimal (overpunch) | revenue +9.2%, or -89%, depending which repair you reach for |
| D | implied decimal | exactly 100x |
| E | RECFM=F framing, and COMP-3 bytes that *are* newlines | 1 record recovered instead of 6 |
| F | packed decimal decoded as latin-1 | 12 amounts silently to mojibake |

Five of the six raise no exception. Four produce a *stable* wrong answer, so a
month-on-month reconciliation agrees with itself.
"""
    ),
    md(
        """## Setup

Both modules are embedded in this notebook and written to disk, so it runs standalone on
Colab or Binder. `fwf.py` is standard library only - no pandas, no numpy."""
    ),
    code(embed("fwf.py")),
    code(embed("evidence.py")),
    md(
        """## The two sample files

`build_customer_file()` is a newline-framed UTF-8 customer master. Two pairs of records are
deliberately the same customer twice - once transliterated to ASCII, once spelled
properly - which gives every encoding claim below a control.

`build_balance_file()` is RECFM=F: fixed-length records, no separator byte at all, with two
COMP-3 packed decimal fields."""
    ),
    code(
        """from fwf import CUSTOMER_SPEC, BALANCE_SPEC, build_customer_file, build_balance_file

data = build_customer_file()
print(CUSTOMER_SPEC.describe())
print()
print("first record, as bytes:")
print(repr(data.split(b"\\n")[0]))
print()
print("record 5 (CJK name) - note it is 62 bytes but fewer than 62 characters:")
rec = data.split(b"\\n")[5]
print(f"  bytes: {len(rec)}   characters: {len(rec.decode('utf-8'))}")"""
    ),
    md(
        """## Parsing it

`parse()` slices bytes and decodes each field independently. Money comes back as
`decimal.Decimal`, never `float`: the file stores cents as integers, which is exact, and
converting that to binary floating point on the way in throws the exactness away for
nothing."""
    ),
    code(
        """from fwf import parse

result = parse(data, CUSTOMER_SPEC)
print(f"{len(result.rows)} records, framing={result.framing}, {len(result.errors)} field error(s)")
print()
for row in result.rows[:6]:
    print({k: str(v) for k, v in row.items()})"""
    ),
    md("""## A. The accent that moves every field after it

Experiment A compares the byte-accurate parse against the character-sliced one. The two
pairs of twin records are what make it an argument rather than an anecdote: the reader
gets the transliterated spelling right and the real spelling wrong."""),
    code("""from evidence import exp_byte_vs_char
_ = exp_byte_vs_char()"""),
    md(
        """## B. The one-byte shift

Copybooks, data dictionaries and hand-written column specs are 1-indexed and inclusive.
`pandas` `colspecs` are 0-indexed and half-open. Pasting one into the other shifts every
field by one byte - each field drops its last character and borrows the previous field's.

It still parses."""
    ),
    code("""from evidence import exp_index_base
_ = exp_index_base()"""),
    md(
        """## C. The minus sign is a letter

`PIC S9(7)V99 DISPLAY` punches the sign into the final digit. `+24500.00` ends in `{`,
`-1425.30` ends in `}`. The magnitudes are identical and exactly one byte differs.

`int()` refuses the column, so it arrives as text, so somebody cleans it. Both obvious
cleanings are wrong, in opposite directions."""
    ),
    code("""from evidence import exp_overpunch
_ = exp_overpunch()"""),
    md("""## D. Implied decimal

`PIC 9(7)V99` stores `26000.00` as `002600000`. There is no decimal point in the file and
nothing to detect - the scale is metadata, and if it is not in the layout it does not
exist anywhere."""),
    code("""from evidence import exp_implied_decimal
_ = exp_implied_decimal()"""),
    md(
        """## E. A record separator that is also a number

COMP-3 stores two digits per byte with the sign in the final nibble: `0xD` negative, `0xC`
positive. So any negative amount whose last digit is `0` ends in the byte `0x0D` - a
carriage return.

The file below has no newlines in it by design. Splitting on line breaks anyway does not
throw."""
    ),
    code("""from evidence import exp_framing
_ = exp_framing()"""),
    md("""## F. Packed decimal read as text

latin-1 maps all 256 byte values to a character, so it never raises. The column loads,
has the declared width, contains no nulls, and passes a not-null check."""),
    code("""from evidence import exp_packed_is_not_text
_ = exp_packed_is_not_text()"""),
    md("""## The audit runs before the load

`audit()` works on bytes and reports a verdict first. Every finding it emits is a failure
that produces a *plausible* answer rather than an exception - the only class of defect
worth a pre-flight check, because the ones that raise were never going to ship."""),
    code(
        """from fwf import audit

print(audit(build_customer_file(), CUSTOMER_SPEC).text())
print()
print(audit(build_balance_file(), BALANCE_SPEC).text())"""
    ),
    md("""## The damage ledger"""),
    code("""from evidence import damage_ledger
_ = damage_ledger()"""),
    md("""## The figure

Six panels, one per mechanism. Panel A is a per-cell agreement grid; panel E is a byte map
of the packed file showing exactly where the stray `0x0D` bytes sit."""),
    code(
        """import os
from IPython.display import Image, display

# Pre-rendered in the repo. On Colab the file is not present, so fetch it.
if not os.path.exists("fwf_audit_nb.png"):
    import urllib.request
    url = (
        "https://raw.githubusercontent.com/phoebefu6/phoebe-the-builder/main/"
        "data-engineering-pro/fixed-width-parser/fwf_audit_nb.png"
    )
    try:
        urllib.request.urlretrieve(url, "fwf_audit_nb.png")
    except Exception as exc:
        print("could not fetch the figure:", exc)

if os.path.exists("fwf_audit_nb.png"):
    display(Image("fwf_audit_nb.png"))
else:
    print("Run `python3 make_chart.py` in the repo to regenerate it.")"""
    ),
    md(
        """## Try your own layout

Describe the fields, state the two conventions, and parse. The conventions are required
arguments rather than defaults because neither one is inferable from the file."""
    ),
    code(
        '''from fwf import RecordSpec, Field, Text, Int, Implied, Overpunch, Packed, Date, parse, audit

# my_spec = RecordSpec(
#     [
#         Field("id",      1,  6, Text()),
#         Field("name",    7, 24, Text()),
#         Field("amount", 31, 11, Overpunch(scale=2)),   # PIC S9(9)V99 DISPLAY
#         Field("qty",    42,  5, Int()),
#         Field("posted", 47,  8, Date("%Y%m%d")),
#     ],
#     index_base=1,        # 1 if your spec is a copybook; 0 if it came from pandas colspecs
#     encoding="utf-8",    # "cp037" for real EBCDIC
#     length=54,           # declare it - it is the closest thing a flat file has to a checksum
# )
#
# raw = open("my_file.dat", "rb").read()     # note "rb" - never open a flat file in text mode
# print(audit(raw, my_spec).text())          # verdict first
# result = parse(raw, my_spec)
# import pandas as pd
# pd.DataFrame(result.rows).head()

print("Uncomment and edit the block above.")'''
    ),
    md(
        f"""---

**Streamlit version:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Reproduce every number here:

```bash
python3 test_fwf.py    # 46 tests over the core
python3 evidence.py    # every table above
python3 make_chart.py  # the six-panel figure
```

Part of [phoebe-the-builder](https://github.com/{REPO}) - Day 138, Data Engineering Pro.
"""
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / "demo.ipynb").write_text(json.dumps(nb, indent=1))
    print(f"wrote demo.ipynb with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
