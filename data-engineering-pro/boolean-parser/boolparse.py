"""Reading a boolean out of a string.

`"false"` is a true value in six of the sixteen readers below.  That is
not a bug in any of them.  A string does not *contain* a boolean - a
reader *assigns* one, and a reader is three separate decisions:

  * an **accept table** - which spellings mean true, which mean false;
  * a **normalisation** - case, whitespace, Unicode, before the lookup;
  * a **failure policy** - refuse, or fall back to one of the two answers.

The integer-free word `true` in a config file carries none of them.  Ten
of the sixteen readers here have no failure policy at all: they return a
confident boolean for every string ever handed to them, including
`undefined`, a UTF-8 BOM, and the empty string.  A reader that cannot
fail cannot tell you that you spelled it wrong.

Almost nothing here is modelled.  `js_truthy` and `js_loose_eq` are a
real `node` subprocess, `sqlite_where` is a real SQLite query, `git_bool`
shells out to `git config --type=bool` in a scratch repository, and
`awk_field`, `perl_truthy`, `ruby_truthy`, `jq_truthy` and `bash_eq_true`
are the real interpreters.  The two exceptions are `go_parsebool` and
`java_parsebool`: no Go toolchain or JRE is present on the build machine,
so their documented value tables are transcribed verbatim and marked
`spec` rather than `live`.  `test_boolparse.py` asserts the transcription
against the published tables.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------
#
# Four outcomes, not two.  The distinction that matters most is the one
# between REFUSED and NOTBOOL:
#
#   REFUSED  the reader raised.  You find out at the boundary, with the
#            offending string in the message.
#   NOTBOOL  the reader succeeded and handed back something that is not a
#            boolean - an int, a string, None.  Nothing has failed and
#            nothing has been decided; the decision has been deferred to
#            whichever `if value:` runs next, in a different file, written
#            by somebody else, under a different set of rules.

TRUE = "true"
FALSE = "false"
REFUSED = "refused"
NOTBOOL = "not-a-boolean"

VERDICTS = (TRUE, FALSE, REFUSED, NOTBOOL)

#: The two verdicts that hand the caller a boolean without hesitating.
CONFIDENT = (TRUE, FALSE)


@dataclass(frozen=True)
class Reading:
    """One reader's answer for one string."""

    verdict: str
    raw: str = ""

    @property
    def confident(self) -> bool:
        return self.verdict in CONFIDENT

    def as_bool(self) -> Optional[bool]:
        if self.verdict == TRUE:
            return True
        if self.verdict == FALSE:
            return False
        return None


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """A string somebody actually wrote into a config file or a CSV cell.

    `intent` is what the author meant, where that is unambiguous - it is
    the yardstick for "silently wrong", which is a *confident* verdict
    opposite to the intent.  `None` means the string has no boolean
    intent at all and any confident verdict is an invention.
    """

    text: str
    family: str
    intent: Optional[bool]
    note: str = ""

    @property
    def label(self) -> str:
        """A printable form - every invisible character made visible."""
        return show(self.text)


def show(text: str) -> str:
    """Render a string so its invisible parts survive a terminal."""
    if text == "":
        return "''"
    out = []
    for ch in text:
        cp = ord(ch)
        if ch == "﻿":
            out.append("<BOM>")
        elif ch == "\r":
            out.append("<CR>")
        elif ch == "\n":
            out.append("<LF>")
        elif ch == "\t":
            out.append("<TAB>")
        elif ch == " ":
            out.append("␣")  # OPEN BOX, the classic visible-space mark
        elif cp < 0x20 or cp == 0x7F:
            out.append(f"<{cp:02X}>")
        else:
            out.append(ch)
    return "".join(out)


CORPUS: Tuple[Sample, ...] = (
    # -- the two spellings every reader is supposed to agree on ----------
    Sample("true", "canonical", True),
    Sample("false", "canonical", False, "the title bug: truthy in six readers"),
    # -- case ------------------------------------------------------------
    Sample("True", "case", True, "Python's repr of the value"),
    Sample("TRUE", "case", True, "a spreadsheet export"),
    Sample("tRuE", "case", True, "accepted by four readers, refused by two"),
    Sample("False", "case", False),
    Sample("FALSE", "case", False, "a spreadsheet export"),
    # -- single letters --------------------------------------------------
    Sample("t", "letter", True, "Postgres and Go say true; git refuses"),
    Sample("f", "letter", False),
    Sample("T", "letter", True),
    Sample("F", "letter", False),
    Sample("y", "letter", True, "in the YAML 1.1 spec, not in PyYAML"),
    Sample("n", "letter", False, "in the YAML 1.1 spec, not in PyYAML"),
    Sample("N", "letter", False),
    # -- words -----------------------------------------------------------
    Sample("yes", "word", True),
    Sample("Yes", "word", True),
    Sample("YES", "word", True),
    Sample("no", "word", False),
    Sample("No", "word", False),
    Sample("NO", "word", False, "also the ISO 3166 code for Norway"),
    Sample("on", "word", True),
    Sample("ON", "word", True),
    Sample("off", "word", False),
    Sample("OFF", "word", False),
    Sample("enabled", "word", True, "no reader here accepts it"),
    Sample("disabled", "word", False, "no reader here accepts it"),
    # -- numeric ---------------------------------------------------------
    Sample("1", "numeric", True),
    Sample("0", "numeric", False),
    Sample("2", "numeric", None, "git reads it as true"),
    Sample("-1", "numeric", None, "git reads it as true"),
    Sample("0.0", "numeric", None, "false to awk, true to Perl"),
    Sample("00", "numeric", None),
    # -- empty and whitespace --------------------------------------------
    Sample("", "empty", None, "an unset variable that still exists"),
    Sample(" ", "empty", None, "a trailing space in the .env file"),
    Sample("true ", "whitespace", True, "trailing space"),
    Sample(" true", "whitespace", True, "leading space"),
    # -- residue from the file it travelled in ---------------------------
    Sample("true\r", "residue", True, "a .env file with CRLF line endings"),
    Sample("\ufeff" + "true", "residue", True, "a UTF-8 BOM on line 1"),
    # -- null-shaped -----------------------------------------------------
    Sample("null", "nullish", None),
    Sample("None", "nullish", None, "Python's repr of None, str()'d into a cell"),
    Sample("nil", "nullish", None),
    Sample("undefined", "nullish", None, "JavaScript's, stringified"),
    # -- Unicode ---------------------------------------------------------
    Sample("yeſ", "unicode", True, "LATIN SMALL LETTER LONG S; casefolds to 'yes'"),
    Sample("FALſE", "unicode", False, "casefolds to 'false', lowercases to 'falſe'"),
    Sample("ＴＲＵＥ", "unicode", True, "fullwidth; NFKC-normalises to 'TRUE'"),
)


# --------------------------------------------------------------------------
# Subprocess plumbing
# --------------------------------------------------------------------------


class ReaderUnavailable(RuntimeError):
    """The interpreter this reader needs is not on the machine."""


@lru_cache(maxsize=None)
def have(binary: str) -> bool:
    return shutil.which(binary) is not None


def _run(argv: Sequence[str], stdin: Optional[str] = None) -> str:
    proc = subprocess.run(
        list(argv),
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def _require(binary: str) -> None:
    if not have(binary):
        raise ReaderUnavailable(f"`{binary}` is not installed; this reader is real, not modelled")


def _texts(samples: Sequence[Sample]) -> List[str]:
    return [s.text for s in samples]


# --------------------------------------------------------------------------
# Python-side readers
# --------------------------------------------------------------------------


def read_py_truthy(samples: Sequence[Sample]) -> List[Reading]:
    """`bool(s)` - what `if os.environ.get("DEBUG"):` actually asks.

    Every non-empty string is true.  This is the reader in the title.
    """
    out = []
    for text in _texts(samples):
        out.append(Reading(TRUE if bool(text) else FALSE, repr(bool(text))))
    return out


@lru_cache(maxsize=1)
def _strtobool() -> Callable[[str], int]:
    """The real `distutils.util.strtobool`, wherever it survived to.

    Removed from the stdlib in 3.12; setuptools vendors it as
    `setuptools._distutils.util.strtobool`.  We import rather than
    reimplement so the accept table is the shipped one.
    """
    try:
        from setuptools._distutils.util import strtobool  # type: ignore

        return strtobool
    except Exception:  # pragma: no cover - only on a setuptools-free box
        from distutils.util import strtobool  # type: ignore

        return strtobool


def read_py_strtobool(samples: Sequence[Sample]) -> List[Reading]:
    """`strtobool` - y/yes/t/true/on/1 and n/no/f/false/off/0, else ValueError.

    Note it returns `1`/`0`, not `True`/`False`; a caller that writes
    `if strtobool(v) is True` gets false for every input.
    """
    fn = _strtobool()
    out = []
    for text in _texts(samples):
        try:
            value = fn(text)
        except (ValueError, AttributeError) as exc:
            out.append(Reading(REFUSED, type(exc).__name__))
            continue
        out.append(Reading(TRUE if value else FALSE, repr(value)))
    return out


def read_json_strict(samples: Sequence[Sample]) -> List[Reading]:
    """`json.loads`, insisting the result is a boolean.

    Only the lowercase literals parse.  Anything else that *does* parse
    (a number, `null`) is NOTBOOL - it succeeded without deciding.
    """
    out = []
    for text in _texts(samples):
        try:
            value = json.loads(text)
        except Exception:
            out.append(Reading(REFUSED, "JSONDecodeError"))
            continue
        if isinstance(value, bool):
            out.append(Reading(TRUE if value else FALSE, repr(value)))
        else:
            out.append(Reading(NOTBOOL, f"{type(value).__name__}={value!r}"))
    return out


def read_yaml11(samples: Sequence[Sample]) -> List[Reading]:
    """PyYAML `safe_load` - YAML 1.1's implicit `bool` resolver.

    This is the Norway problem: an unquoted `NO` in a country column
    resolves to the boolean false.  Note also what PyYAML does *not*
    implement: YAML 1.1's type repository lists `y` and `n`, and PyYAML
    leaves both as strings.
    """
    import yaml

    out = []
    for text in _texts(samples):
        try:
            value = yaml.safe_load(text)
        except Exception:
            out.append(Reading(REFUSED, "YAMLError"))
            continue
        if isinstance(value, bool):
            out.append(Reading(TRUE if value else FALSE, repr(value)))
        else:
            out.append(Reading(NOTBOOL, f"{type(value).__name__}={value!r}"))
    return out


@lru_cache(maxsize=1)
def _ruamel12():
    from ruamel.yaml import YAML

    loader = YAML(typ="safe", pure=True)
    loader.version = (1, 2)
    return loader


def read_yaml12(samples: Sequence[Sample]) -> List[Reading]:
    """ruamel.yaml pinned to YAML 1.2 - `true`/`false` and nothing else.

    The 1.2 core schema deleted `yes`/`no`/`on`/`off` precisely because
    of the resolver above.  Two YAML parsers, one document, two values.
    """
    import io

    loader = _ruamel12()
    out = []
    for text in _texts(samples):
        try:
            value = loader.load(io.StringIO(text))
        except Exception:
            out.append(Reading(REFUSED, "YAMLError"))
            continue
        if isinstance(value, bool):
            out.append(Reading(TRUE if value else FALSE, repr(value)))
        else:
            out.append(Reading(NOTBOOL, f"{type(value).__name__}={value!r}"))
    return out


def read_sqlite_where(samples: Sequence[Sample]) -> List[Reading]:
    """`WHERE flag` against a TEXT column - a real SQLite query.

    SQLite has no boolean type.  A TEXT value in boolean position is cast
    to a number, and a string that does not *begin* with digits casts to
    0.  So a feature flag stored as the string `'true'` is off, in every
    row, forever, and no error is ever raised.
    """
    con = sqlite3.connect(":memory:")
    try:
        out = []
        for text in _texts(samples):
            row = con.execute("SELECT CASE WHEN ? THEN 1 ELSE 0 END", (text,)).fetchone()
            cast = con.execute("SELECT CAST(? AS NUMERIC)", (text,)).fetchone()[0]
            out.append(Reading(TRUE if row[0] else FALSE, f"CAST->{cast!r}"))
        return out
    finally:
        con.close()


# --------------------------------------------------------------------------
# Subprocess readers - real interpreters, one call each
# --------------------------------------------------------------------------

_NODE_SCRIPT = r"""
const xs = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = xs.map(s => ({ truthy: Boolean(s), loose: (s == true), looseFalse: (s == false) }));
process.stdout.write(JSON.stringify(out));
"""


@lru_cache(maxsize=1)
def _node_batch(payload: str) -> List[Dict[str, bool]]:
    _require("node")
    return json.loads(_run(["node", "-e", _NODE_SCRIPT], stdin=payload))


def _node_readings(samples: Sequence[Sample]) -> List[Dict[str, bool]]:
    return _node_batch(json.dumps(_texts(samples)))


def read_js_truthy(samples: Sequence[Sample]) -> List[Reading]:
    """`Boolean(s)` / `if (s)` in a real node 22 - every non-empty string is true."""
    return [
        Reading(TRUE if r["truthy"] else FALSE, repr(r["truthy"]))
        for r in _node_readings(samples)
    ]


def read_js_loose_eq(samples: Sequence[Sample]) -> List[Reading]:
    """`s == true` in a real node 22 - the fix people reach for, which is worse.

    Loose equality against a boolean converts *both* sides to numbers, so
    `"1" == true` is true and `"true" == true` is false.  This reader
    disagrees with `Boolean(s)` on almost everything, which is how a
    codebase ends up with two flags that are never both on.
    """
    return [
        Reading(TRUE if r["loose"] else FALSE, repr(r["loose"]))
        for r in _node_readings(samples)
    ]


def js_loose_false(samples: Sequence[Sample]) -> List[bool]:
    """`s == false`, kept separate: it is not the negation of `s == true`."""
    return [r["looseFalse"] for r in _node_readings(samples)]


_AWK_PROGRAM = r'{ print ($0 ? "T" : "F") }'


def read_awk_field(samples: Sequence[Sample]) -> List[Reading]:
    """`awk '$0 { ... }'` on a value that arrived as input.

    awk's *strnum* rule: a field that looks numeric is compared as a
    number, so the input line `0` is false - and so are `00`, `0.0` and
    `0e0`.  The same characters written as a literal inside the program
    are a string constant and therefore true.  One implementation, one
    string, two answers, chosen by how the string got in.
    """
    _require("awk")
    payload = "\n".join(_texts(samples)) + "\n"
    lines = _run(["awk", _AWK_PROGRAM], stdin=payload).splitlines()
    if len(lines) != len(samples):
        raise RuntimeError(f"awk returned {len(lines)} lines for {len(samples)} samples")
    return [Reading(TRUE if v == "T" else FALSE, v) for v in lines]


def awk_program_literal(text: str) -> bool:
    """The same string as a *string constant* compiled into the program."""
    _require("awk")
    prog = "BEGIN { print (" + json.dumps(text) + ' ? "T" : "F") }'
    return _run(["awk", prog]).strip() == "T"


def awk_input_field(text: str) -> bool:
    """The same string arriving as a line of input, for any text at all."""
    _require("awk")
    return _run(["awk", _AWK_PROGRAM], stdin=text + "\n").strip() == "T"


def awk_assigned_var(text: str) -> bool:
    """The same string arriving through `-v name=value`."""
    _require("awk")
    out = _run(["awk", "-v", f"s={text}", 'BEGIN { print (s ? "T" : "F") }'])
    return out.strip() == "T"


def read_perl_truthy(samples: Sequence[Sample]) -> List[Reading]:
    """Perl truthiness - false is `""`, `"0"`, and nothing else.

    `"0.0"` and `"00"` are true; `"0"` is false.  Perl is the only reader
    here whose falsy set is exactly two strings.
    """
    _require("perl")
    script = r'print(($_ ? "T" : "F"), "\n") for @ARGV;'
    lines = _run(["perl", "-e", script, *_texts(samples)]).splitlines()
    return [Reading(TRUE if v == "T" else FALSE, v) for v in lines]


def read_ruby_truthy(samples: Sequence[Sample]) -> List[Reading]:
    """Ruby truthiness - only `nil` and `false` are falsy, so *every* string is true.

    Maximally permissive and therefore maximally uninformative: this
    reader returns the same answer for all 45 strings in the corpus.
    """
    _require("ruby")
    script = 'ARGV.each { |a| puts(a ? "T" : "F") }'
    lines = _run(["ruby", "-e", script, *_texts(samples)]).splitlines()
    return [Reading(TRUE if v == "T" else FALSE, v) for v in lines]


def read_jq_truthy(samples: Sequence[Sample]) -> List[Reading]:
    """`jq 'if . then'` on a string - falsy is `false` and `null` only.

    Same shape as Ruby, reached from a different direction: jq's booleans
    are JSON booleans, and a JSON string is never one of them.
    """
    _require("jq")
    lines = _run(
        ["jq", "-nr", "--args", '$ARGS.positional[] | if . then "T" else "F" end', "--", *_texts(samples)]
    ).splitlines()
    return [Reading(TRUE if v == "T" else FALSE, v) for v in lines]


def read_bash_eq_true(samples: Sequence[Sample]) -> List[Reading]:
    """`[ "$x" = true ]` - the entrypoint-script idiom.

    An exact byte comparison against one spelling.  It never errors and
    never accepts `True`, `1`, `yes` or a value with a stray `\\r` from a
    CRLF file, so a Windows-edited `.env` turns every flag off in silence.
    """
    _require("bash")
    script = 'for x in "$@"; do if [ "$x" = true ]; then echo T; else echo F; fi; done'
    lines = _run(["bash", "-c", script, "bash", *_texts(samples)]).splitlines()
    return [Reading(TRUE if v == "T" else FALSE, v) for v in lines]


def read_git_bool(samples: Sequence[Sample]) -> List[Reading]:
    """`git config --type=bool` in a real scratch repository.

    git's own table: true/yes/on/1 and false/no/off/0/empty, plus *any*
    integer - `2` and `-1` both read as true.  It is one of only four
    readers here that will tell you when it does not understand.
    """
    _require("git")
    out: List[Reading] = []
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
        subprocess.run(["git", "init", "-q", tmp], check=True, capture_output=True, env=env)
        cfg = os.path.join(tmp, ".git", "config")
        for text in _texts(samples):
            subprocess.run(
                ["git", "config", "--file", cfg, "flag.value", text],
                check=True, capture_output=True, env=env,
            )
            proc = subprocess.run(
                ["git", "config", "--file", cfg, "--type=bool", "flag.value"],
                capture_output=True, text=True, env=env,
            )
            if proc.returncode != 0:
                out.append(Reading(REFUSED, "bad numeric config value"))
            else:
                value = proc.stdout.strip()
                out.append(Reading(TRUE if value == "true" else FALSE, value))
    return out


# --------------------------------------------------------------------------
# Spec readers - transcribed tables, no toolchain on this machine
# --------------------------------------------------------------------------

#: Go, `strconv.ParseBool`: "It accepts 1, t, T, TRUE, true, True, 0, f,
#: F, FALSE, false, False. Any other value returns an error."
GO_TRUE = frozenset({"1", "t", "T", "TRUE", "true", "True"})
GO_FALSE = frozenset({"0", "f", "F", "FALSE", "false", "False"})


def read_go_parsebool(samples: Sequence[Sample]) -> List[Reading]:
    """`strconv.ParseBool` - twelve exact spellings, everything else an error.

    Marked `spec`: there is no Go toolchain on this machine, so the
    twelve-entry table is transcribed from the documented behaviour and
    asserted in the test suite.  Note what is missing: `yes`, `no`, `on`,
    `off`, `y`, `n`, and every mixed case that is not `True`/`TRUE`.
    """
    out = []
    for text in _texts(samples):
        if text in GO_TRUE:
            out.append(Reading(TRUE, "true"))
        elif text in GO_FALSE:
            out.append(Reading(FALSE, "false"))
        else:
            out.append(Reading(REFUSED, "strconv.ErrSyntax"))
    return out


def read_java_parsebool(samples: Sequence[Sample]) -> List[Reading]:
    """`Boolean.parseBoolean` - true iff the string equalsIgnoreCase "true".

    Marked `spec`: no JRE on this machine.  The rule is one line and it is
    the most dangerous one in the roster, because *everything else is
    false and nothing raises*.  `parseBoolean("yes")`, `("1")` and
    `("ture")` are all false, indistinguishably.
    """
    out = []
    for text in _texts(samples):
        # Java's equalsIgnoreCase compares char-by-char via toUpperCase then
        # toLowerCase on each char; for ASCII "true" this is plain ASCII casing.
        out.append(Reading(TRUE if text.lower() == "true" and text.isascii() else FALSE, "boolean"))
    return out


# --------------------------------------------------------------------------
# The roster
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reader:
    name: str
    stack: str
    source: str  # "live" | "spec"
    can_refuse: bool
    seen_in: str
    fn: Callable[[Sequence[Sample]], List[Reading]]


READERS: Tuple[Reader, ...] = (
    Reader("py_truthy", "Python 3.11", "live", False,
           'if os.environ.get("DEBUG"):', read_py_truthy),
    Reader("py_strtobool", "Python 3.11", "live", True,
           "distutils.util.strtobool, argparse recipes", read_py_strtobool),
    Reader("json_strict", "Python 3.11", "live", True,
           "a JSON request body, a JSONB column", read_json_strict),
    Reader("yaml11", "PyYAML 6", "live", True,
           "docker-compose, GitHub Actions, Ansible", read_yaml11),
    Reader("yaml12", "ruamel 1.2", "live", True,
           "a YAML 1.2 parser: Go yaml.v3, JS yaml", read_yaml12),
    Reader("js_truthy", "node 22", "live", False,
           "if (process.env.FLAG) in any Node service", read_js_truthy),
    Reader("js_loose_eq", "node 22", "live", False,
           "the == true 'fix' applied after the first bug", read_js_loose_eq),
    Reader("sqlite_where", "SQLite 3.51", "live", False,
           "WHERE flag on a TEXT column", read_sqlite_where),
    Reader("git_bool", "git 2.50", "live", True,
           "git config --type=bool, core.* settings", read_git_bool),
    Reader("awk_field", "awk 20200816", "live", False,
           "awk '$3 { ... }' over a TSV export", read_awk_field),
    Reader("perl_truthy", "perl 5", "live", False,
           "a legacy ETL script", read_perl_truthy),
    Reader("ruby_truthy", "ruby 2.6", "live", False,
           "if flag in Rails, Chef, Puppet", read_ruby_truthy),
    Reader("jq_truthy", "jq 1.6", "live", False,
           "jq 'select(.flag)' in a pipeline", read_jq_truthy),
    Reader("bash_eq_true", "bash 3.2", "live", False,
           '[ "$X" = true ] in an entrypoint', read_bash_eq_true),
    Reader("go_parsebool", "Go (spec)", "spec", True,
           "strconv.ParseBool, most Go flag parsing", read_go_parsebool),
    Reader("java_parsebool", "Java (spec)", "spec", False,
           "Boolean.parseBoolean, Spring @Value", read_java_parsebool),
)

READERS_BY_NAME: Dict[str, Reader] = {r.name: r for r in READERS}


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def grid() -> Dict[str, List[Reading]]:
    """Every reader's answer for every corpus string. Reader name -> readings."""
    return {r.name: r.fn(CORPUS) for r in READERS}


def verdict(reader: str, sample: Sample) -> Reading:
    return grid()[reader][CORPUS.index(sample)]


def verdicts_for(sample: Sample) -> Dict[str, Reading]:
    i = CORPUS.index(sample)
    return {name: readings[i] for name, readings in grid().items()}


def distinct_verdicts(sample: Sample) -> List[str]:
    seen = []
    for reading in verdicts_for(sample).values():
        if reading.verdict not in seen:
            seen.append(reading.verdict)
    return seen


def unanimous() -> List[Sample]:
    """Strings every reader reads the same way. Expect very few."""
    return [s for s in CORPUS if len(distinct_verdicts(s)) == 1]


def sign_flips() -> List[Tuple[Sample, List[str], List[str]]]:
    """Strings that at least one reader calls true and another calls false.

    Both sides confident, neither raising: nothing anywhere in the stack
    can detect that these two services disagree.
    """
    out = []
    for sample in CORPUS:
        readings = verdicts_for(sample)
        trues = sorted(n for n, r in readings.items() if r.verdict == TRUE)
        falses = sorted(n for n, r in readings.items() if r.verdict == FALSE)
        if trues and falses:
            out.append((sample, trues, falses))
    return out


def silently_wrong() -> List[Tuple[Sample, str, str]]:
    """Confident verdicts that contradict the author's intent.

    Only counted where the intent is unambiguous.  A REFUSED reading is
    never in here - being told is the good outcome.
    """
    out = []
    for sample in CORPUS:
        if sample.intent is None:
            continue
        for name, reading in verdicts_for(sample).items():
            if reading.confident and reading.as_bool() is not sample.intent:
                out.append((sample, name, reading.verdict))
    return out


def refusal_counts() -> Dict[str, int]:
    return {
        name: sum(1 for r in readings if r.verdict == REFUSED)
        for name, readings in grid().items()
    }


def notbool_counts() -> Dict[str, int]:
    return {
        name: sum(1 for r in readings if r.verdict == NOTBOOL)
        for name, readings in grid().items()
    }


def never_refuse() -> List[str]:
    """Readers that return a confident boolean for every string in the corpus."""
    return [
        name
        for name, readings in grid().items()
        if all(r.confident for r in readings)
    ]


def agreement(a: str, b: str) -> Tuple[int, int, int]:
    """(identical, disagree-confidently, one-side-declined) over the corpus."""
    same = flip = decline = 0
    for ra, rb in zip(grid()[a], grid()[b]):
        if ra.verdict == rb.verdict:
            same += 1
        elif ra.confident and rb.confident:
            flip += 1
        else:
            decline += 1
    return same, flip, decline


def round_trip() -> Dict[Tuple[str, str], int]:
    """Writer -> reader survival, and it is genuinely asymmetric.

    Filter on the strings the *writer* reads confidently - those are the
    spellings somebody would plausibly have written into a config file
    authored against that reader's accept table.  Then count how many of
    them the *reader* fails to reproduce: a flipped verdict, a refusal,
    or a non-boolean.  The filter depends on the writer, so
    `(a, b) != (b, a)`.

    This is the "the compose file was written for Ansible and is read by
    the Go service" number.
    """
    out: Dict[Tuple[str, str], int] = {}
    for writer in READERS:
        for reader in READERS:
            broken = 0
            for rw, rr in zip(grid()[writer.name], grid()[reader.name]):
                if rw.confident and rr.verdict != rw.verdict:
                    broken += 1
            out[(writer.name, reader.name)] = broken
    return out


def round_trip_flips(writer: str, reader: str) -> Dict[str, int]:
    """The same pair, split by how the reader failed to reproduce it."""
    flipped = refused = deferred = 0
    for rw, rr in zip(grid()[writer], grid()[reader]):
        if not rw.confident or rr.verdict == rw.verdict:
            continue
        if rr.verdict in CONFIDENT:
            flipped += 1
        elif rr.verdict == REFUSED:
            refused += 1
        else:
            deferred += 1
    return {"flipped": flipped, "refused": refused, "deferred": deferred}


def most_asymmetric(n: int = 6) -> List[Tuple[str, str, int, int]]:
    """Pairs where the damage runs mostly one way."""
    rt = round_trip()
    names = [r.name for r in READERS]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = abs(rt[(a, b)] - rt[(b, a)])
            out.append((a, b, rt[(a, b)], rt[(b, a)], gap))
    out.sort(key=lambda row: -row[4])
    return [(a, b, ab, ba) for a, b, ab, ba, _ in out[:n]]


# --------------------------------------------------------------------------
# Normalisation - the second of the three decisions
# --------------------------------------------------------------------------


def normalisations(text: str) -> Dict[str, str]:
    """The four normalisations a reader might apply before its lookup.

    They are not interchangeable.  `.casefold()` maps LATIN SMALL LETTER
    LONG S to `s`, so `FALſE` becomes `false` and is accepted by a
    casefolding reader while a `.lower()` reader refuses it.  NFKC maps
    fullwidth Latin to ASCII, so `ＴＲＵＥ` becomes `TRUE`.
    """
    return {
        "as-written": text,
        "strip": text.strip(),
        "lower": text.lower(),
        "casefold": text.casefold(),
        "NFKC+casefold": unicodedata.normalize("NFKC", text).casefold(),
    }


BASE_TABLE = frozenset({"true", "t", "yes", "y", "on", "1", "false", "f", "no", "n", "off", "0"})

#: A vocabulary that grew past the twelve core literals. `disabled` is the
#: first entry with an `I` in it, which is where locale-sensitive casing
#: starts to matter (see `locale_lower`).
EXTENDED_TABLE = BASE_TABLE | {"enabled", "disabled"}


def locale_lower(words: Sequence[str], locale: str) -> Dict[str, str]:
    """Real locale-aware lowercasing, via node's `toLocaleLowerCase`.

    ECMA-402 in node 22, not a transcribed table. In `tr` and `az`,
    U+0049 LATIN CAPITAL LETTER I lowercases to U+0131 DOTLESS I, so
    `DISABLED` becomes `dısabled` and matches nothing.
    """
    _require("node")
    script = (
        "const [loc, ws] = JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "process.stdout.write(JSON.stringify("
        "Object.fromEntries(ws.map(w => [w, w.toLocaleLowerCase(loc)]))));"
    )
    return json.loads(_run(["node", "-e", script], stdin=json.dumps([locale, list(words)])))


def accepted_after(text: str) -> Dict[str, bool]:
    """Whether one fixed accept table matches, under each normalisation."""
    return {k: v in BASE_TABLE for k, v in normalisations(text).items()}
