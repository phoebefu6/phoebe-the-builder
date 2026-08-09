"""Generates demo.ipynb with tznorm.py and evidence.py embedded.

Modules are embedded as JSON-encoded string literals rather than triple-quoted
blocks: both contain docstrings, so any triple-quote wrapper terminates early.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-engineering-pro/timezone-normalizer"

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
    src = (HERE / name).read_text()
    return (
        f"_src = {json.dumps(src)}\n"
        f"from pathlib import Path\n"
        f'Path("{name}").write_text(_src)\n'
        f'print("wrote {name}:", len(_src.splitlines()), "lines")\n'
    )


CELLS: List[Dict[str, Any]] = [
    md(
        f"""# A local timestamp is not a point in time

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

**Day 139 - Data Engineering Pro.**

`2024-11-03 01:30` in New York happened twice, an hour apart. `2024-03-10 02:30`
in New York never happened at all. `zoneinfo` returns a UTC value for both without
raising - PEP 495 made both readings representable via the `fold` attribute, which
is correct and complete, and which also means the guess is made for you silently.

Six mechanisms, each measured against instants that actually happened:

| | mechanism | effect on the sample |
|---|---|---|
| A | the wall clock reads 01:00-01:59 twice | a session lasts **-40 minutes** |
| B | the clock skips 02:00-02:59 (or 02:00-02:29) | accepted silently, round trip fails |
| C | an offset in the payload | recovers both contested sessions exactly |
| D | `Asia/Calcutta` == `Asia/Kolkata`; `Etc/GMT+5` is UTC**-**5 | one office split in two |
| E | UTC day vs local day | 6 of 24 events change day, totals still reconcile |
| F | +05:45, +13:45, and a 30-minute DST shift | 3 of 7 zones miss the hour grid |

None of the eight entries in the damage ledger raise. Six are *stable*, so
re-running the pipeline reproduces the same wrong number.

**The sample is generated from true UTC instants** and the local wall-clock strings
are rendered from them with the offset dropped - which is exactly what a system does
when it writes local `now()` into a `TIMESTAMP` column. That makes the ground truth
intrinsic, so everything below is a real recovery error rather than two guesses
compared to each other.
"""
    ),
    md(
        """## Setup

Both modules are embedded and written to disk, so this runs standalone on Colab or
Binder. `tznorm.py` is standard library only - `zoneinfo`, no pytz, no pandas."""
    ),
    code(embed("tznorm.py")),
    code(embed("evidence.py")),
    md(
        """## Which rules are we using?

Time zone rules are political and change several times a year. Two runs of the same
code on the same input give different answers if the tz database moved underneath
them, so the version belongs next to the results."""
    ),
    code(
        """from tznorm import tzdata_version, build_session_log, audit

print(tzdata_version())
print()
rows = build_session_log()
for r in rows[:8]:
    print(f"{r['session_id']:8}{r['office']:12}{r['zone']:22}{r['local_ts']:20}{r['event']}")"""
    ),
    md(
        """## The three-way classification

`classify()` decides between `ok`, `ambiguous` and `nonexistent` with a round trip -
convert the wall time to UTC and back. An ordinary time returns itself. A skipped
time cannot, and comes back as some *other* wall time. An ambiguous time returns
itself under both folds while the two folds carry different offsets.

That definition needs no transition table, which is why it works on every zone
including the ones nobody writes tests for."""
    ),
    code(
        """import datetime as dt
from tznorm import classify

cases = [
    ("America/New_York",    dt.datetime(2024, 6, 15, 12, 0)),
    ("America/New_York",    dt.datetime(2024, 11, 3, 1, 30)),
    ("America/New_York",    dt.datetime(2024, 3, 10, 2, 30)),
    ("Australia/Lord_Howe", dt.datetime(2024, 10, 6, 2, 15)),
    ("Australia/Lord_Howe", dt.datetime(2024, 10, 6, 2, 30)),
    ("Asia/Singapore",      dt.datetime(2024, 11, 3, 1, 30)),
]
for zone, naive in cases:
    print(f"{zone:22}{naive}  ->  {classify(naive, zone)}")"""
    ),
    md("""## A. The hour that happens twice"""),
    code("""from evidence import exp_ambiguous
_ = exp_ambiguous()"""),
    md(
        """## B. The hour that never happens

Lord Howe Island is the one worth remembering: its DST shift is thirty minutes, so
its gap is half an hour wide. Any guard written as "is the hour 02:xx suspicious"
misses it, and any test suite built only on `America/New_York` never sees it."""
    ),
    code("""from evidence import exp_nonexistent
_ = exp_nonexistent()"""),
    md("""## C. Six characters of offset fix it - and what they still do not fix"""),
    code("""from evidence import exp_offset_vs_zone
_ = exp_offset_vs_zone()"""),
    md("""## D. The zone identifier is not canonical"""),
    code("""from evidence import exp_identifiers
_ = exp_identifiers()"""),
    md("""## E. Which day did it happen on"""),
    code("""from evidence import exp_day_bucketing
_ = exp_day_bucketing()"""),
    md("""## F. Offsets are not whole hours"""),
    code("""from evidence import exp_sub_hour
_ = exp_sub_hour()"""),
    md(
        """## The audit runs before the conversion

Every finding is a failure that produces a *plausible* answer rather than an
exception. A timestamp that will not parse announces itself and is not interesting
here."""
    ),
    code("""print(audit(build_session_log()).text())"""),
    md("""## The damage ledger"""),
    code("""from evidence import damage_ledger
_ = damage_ledger()"""),
    md("""## The figure"""),
    code(
        """import os
from IPython.display import Image, display

if not os.path.exists("tz_audit_nb.png"):
    import urllib.request
    url = (
        "https://raw.githubusercontent.com/phoebefu6/phoebe-the-builder/main/"
        "data-engineering-pro/timezone-normalizer/tz_audit_nb.png"
    )
    try:
        urllib.request.urlretrieve(url, "tz_audit_nb.png")
    except Exception as exc:
        print("could not fetch the figure:", exc)

if os.path.exists("tz_audit_nb.png"):
    display(Image("tz_audit_nb.png"))
else:
    print("Run `python3 make_chart.py` in the repo to regenerate it.")"""
    ),
    md(
        """## Try your own column

The policies default to `raise` on a single value and `flag` on a column, because in
both hard cases the input does not determine the answer and a default that quietly
picks one is how a duration ends up negative in a table nobody re-reads."""
    ),
    code(
        '''from tznorm import resolve, normalize, audit, local_day, utc_day

# my_rows = [
#     {"local_ts": "2024-11-03 01:30", "zone": "America/New_York"},
#     {"local_ts": "2024-03-10 02:30", "zone": "America/New_York"},
#     {"local_ts": "2024-11-04 00:25", "zone": "Asia/Kathmandu"},
# ]
#
# print(audit(my_rows).text())                       # verdict first
# for r in normalize(my_rows):                       # flags rather than guesses
#     print(r.status, r.raw, r.utc, r.note)
#
# # when you genuinely have to pick, say so out loud:
# normalize(my_rows, ambiguous="earlier", nonexistent="shift_forward")

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
python3 test_tznorm.py   # 36 tests over the core
python3 evidence.py      # every table above
python3 make_chart.py    # the six-panel figure
```

Part of [phoebe-the-builder](https://github.com/{REPO}) - Day 139, Data Engineering Pro.
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
