"""Filename sanitising, and the collisions a sanitiser creates on the way.

`sanitise(name) -> str` is the shape every library ships. It is the wrong shape,
for a reason that has nothing to do with which characters are on the deny-list.

A sanitiser is a *projection*: it maps a large set of source names onto the
smaller set of names a target filesystem will accept. Projections onto smaller
sets collide. That is not a defect in any particular implementation - it is what
the word means. The only real question is whether the function tells you, and a
function whose return type is `str` structurally cannot: the collision is a fact
about a *pair* of names, and it only has one name in scope.

So the three failures that actually lose data are all invisible per-name:

1. Two distinct sources land on one target. `a:b.txt`, `a/b.txt`, `a*b.txt` and
   `a?b.txt` all become `a_b.txt`. Four files, one file written, three gone, and
   every call returned successfully.
2. Two distinct targets collide *on the volume* rather than in the sanitiser.
   `Report.csv` / `report.csv` are two names on ext4 and one on NTFS and on a
   default APFS volume. `café` in NFC and NFD are two on ext4 and one on APFS.
   Nothing is wrong with either name.
3. The name is fine and the *path* is not. A 200-character name is legal
   everywhere and unwritable at a destination 100 characters deep on Win32.
   Validity is a property of `(name, target, destination)`, and a pure function
   of the name has two thirds of that missing.

And one that is worse than invisible - the sanitiser is *more* destructive than
the filesystem it protects you from. Case-insensitive volumes fold with a 1:1
case table (NTFS's `$UpCase`; APFS's equivalent). Python's `str.casefold()`
implements full Unicode case folding, which expands `ß` to `ss`. So a sanitiser
that lowercases via `casefold()` merges `Straße.txt` into `STRASSE.txt` - two
files NTFS would have kept apart. `str.lower()` gets that pair right and gets
Greek final sigma wrong in the other direction. Neither models a filesystem.

Core ideas
----------
1. Validity and injectivity trade off directly. The more a sanitiser rewrites,
   the more source names it merges. Measured over one corpus, the ranking by
   validity is the reverse of the ranking by collisions.
2. The length limit is in *bytes* and sanitisers count *characters*. 100 emoji
   is 100 characters and 400 bytes against a 255-byte `NAME_MAX`.
3. Reserved device names survive extensions. `CON.txt` is reserved. So is
   `con.tar.gz`, and `CON.` and `CON ` - because Win32 strips trailing dots and
   spaces *before* it looks the name up.
4. The verdict is three-valued and narrow:
   `portable` - every name is writable on every target, and the mapping from
                source names to targets is injective on every target.
   `lossy`    - every name is writable, but two sources land on one file
                somewhere. The write succeeds and a file is gone.
   `rejected` - at least one name cannot be written at all on some target.

Standard library only: `re`, `unicodedata`, `dataclasses`, `enum`, `collections`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Target profiles
# ---------------------------------------------------------------------------
#
# A profile is what a sanitiser is never given. Each field below changes the
# answer for the same input string, which is the point.
#
# `max_component_bytes` is NAME_MAX: 255 *bytes* on ext4/XFS/APFS/NTFS-as-used.
#   NTFS's own limit is 255 UTF-16 code units, which is a different unit again -
#   modelled as `component_unit`.
# `max_path_chars` is the Win32 MAX_PATH ceiling of 260 *including* the
#   terminating NUL, so 259 usable characters. Absent on POSIX and on Win32
#   when long paths are enabled.
# `case_table` is how the volume compares two names that differ only in case:
#   None      - byte-exact, `Report` and `report` are two files (ext4, XFS)
#   "simple"  - 1:1 Unicode case mapping (NTFS `$UpCase`, APFS case-insensitive)
# `normalisation` is how it compares NFC against NFD:
#   None      - byte-exact, two files (ext4, XFS, and NTFS)
#   "insensitive" - one file (APFS on macOS 10.13+)


@dataclass(frozen=True)
class Profile:
    """One target filesystem, as far as naming is concerned."""

    name: str
    forbidden_chars: str
    forbids_control_chars: bool
    strips_trailing_dot_space: bool
    reserved_stems: FrozenSet[str]
    max_component_bytes: int
    component_unit: str  # "utf-8 bytes" | "utf-16 code units"
    max_path_chars: Optional[int]
    max_path_bytes: Optional[int]
    case_table: Optional[str]
    normalisation: Optional[str]
    note: str = ""


# The Microsoft-documented set. The reservation is on the stem *before the first
# dot*, so `CON.txt` is reserved too - the single most-missed rule in this file.
# COM0/LPT0 appear in current Win32 naming documentation alongside 1-9.
WINDOWS_RESERVED: FrozenSet[str] = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{d}" for d in range(0, 10)]
    + [f"LPT{d}" for d in range(0, 10)]
)

# Win32 API layer, not NTFS. NTFS itself stores most of these happily; it is
# `CreateFileW` and everything above it that refuses.
WINDOWS_FORBIDDEN = '<>:"/\\|?*'

WINDOWS = Profile(
    name="windows-ntfs",
    forbidden_chars=WINDOWS_FORBIDDEN,
    forbids_control_chars=True,
    strips_trailing_dot_space=True,
    reserved_stems=WINDOWS_RESERVED,
    max_component_bytes=255,
    component_unit="utf-16 code units",
    max_path_chars=259,  # MAX_PATH 260 includes the terminating NUL
    max_path_bytes=None,
    case_table="simple",
    normalisation=None,
    note="Win32 API layer. MAX_PATH applies unless long paths are opted into.",
)

WINDOWS_LONG = Profile(
    name="windows-longpath",
    forbidden_chars=WINDOWS_FORBIDDEN,
    forbids_control_chars=True,
    strips_trailing_dot_space=True,
    reserved_stems=WINDOWS_RESERVED,
    max_component_bytes=255,
    component_unit="utf-16 code units",
    max_path_chars=None,
    max_path_bytes=None,
    case_table="simple",
    normalisation=None,
    note="Long paths enabled. Everything else about Win32 is unchanged.",
)

MACOS_APFS = Profile(
    name="macos-apfs",
    forbidden_chars=":",  # the Finder shows a typed "/" and stores ":"
    forbids_control_chars=False,
    strips_trailing_dot_space=False,
    reserved_stems=frozenset(),
    max_component_bytes=255,
    component_unit="utf-8 bytes",
    max_path_chars=None,
    max_path_bytes=1024,
    case_table="simple",
    normalisation="insensitive",
    note="Case-insensitive and normalisation-insensitive by default (10.13+).",
)

LINUX_EXT4 = Profile(
    name="linux-ext4",
    forbidden_chars="/",  # POSIX forbids exactly this and NUL
    forbids_control_chars=False,
    strips_trailing_dot_space=False,
    reserved_stems=frozenset(),
    max_component_bytes=255,
    component_unit="utf-8 bytes",
    max_path_chars=None,
    max_path_bytes=4096,
    case_table=None,
    normalisation=None,
    note="Byte-exact. Any byte sequence except / and NUL is a legal name.",
)

OBJECT_STORE = Profile(
    name="object-store",
    forbidden_chars="",
    forbids_control_chars=False,
    strips_trailing_dot_space=False,
    reserved_stems=frozenset(),
    max_component_bytes=1024,
    component_unit="utf-8 bytes",
    max_path_chars=None,
    max_path_bytes=1024,
    case_table=None,
    normalisation=None,
    note="Keys are opaque UTF-8. Permissive enough to make the target the variable.",
)

PROFILES: Dict[str, Profile] = {
    p.name: p for p in [WINDOWS, WINDOWS_LONG, MACOS_APFS, LINUX_EXT4, OBJECT_STORE]
}

# The interchange set: a file that has to survive all three desktop platforms
# is bound by the intersection of their rules, which is Windows' rules plus
# ext4's byte limit plus APFS's folding.
PORTABLE_TARGETS: Tuple[Profile, ...] = (WINDOWS, MACOS_APFS, LINUX_EXT4)


# ---------------------------------------------------------------------------
# Case folding: three models, none of which agree
# ---------------------------------------------------------------------------


def fold_py_lower(s: str) -> str:
    """`str.lower()`. What almost every sanitiser calls.

    Applies Unicode SpecialCasing, so it gets Greek final sigma *wrong* for
    this purpose: `ΣΙΣΥΦΟΣ`.lower() ends in `ς` while `σισυφοσ`.lower() ends in
    `σ`, so the two look distinct - and a case-insensitive volume, which folds
    both to `Σ`, treats them as one file.
    """
    return s.lower()


def fold_py_casefold(s: str) -> str:
    """`str.casefold()`. Full Unicode case folding.

    Expands `ß` to `ss`, so `Straße` and `STRASSE` compare equal - two names
    that NTFS and APFS both keep apart, because their case tables are 1:1.
    Using this to deduplicate merges files the filesystem never would have.
    """
    return s.casefold()


def fold_simple_upper(s: str) -> str:
    """1:1 uppercase folding: the model that matches a real volume.

    NTFS compares through `$UpCase`, a per-volume table of single code-unit
    uppercase mappings; APFS's case-insensitive comparison is likewise 1:1.
    Python exposes only full case mappings, so this approximates the simple
    mapping by taking the first code point of each character's expansion:
    `ß` -> `ß` (unchanged, correct), `ς` and `σ` -> `Σ` (merged, correct),
    `İ` -> `İ`.
    """
    out: List[str] = []
    for ch in s:
        up = ch.upper()
        out.append(up[0] if up else ch)
    return "".join(out)


FOLDS: Dict[str, Callable[[str], str]] = {
    "py_lower": fold_py_lower,
    "py_casefold": fold_py_casefold,
    "simple_upper": fold_simple_upper,
}


# ---------------------------------------------------------------------------
# Win32 name resolution
# ---------------------------------------------------------------------------


def win32_effective(name: str) -> str:
    """What Win32 actually opens when handed `name`.

    Trailing dots and spaces are stripped from each path component by the API
    before anything else happens. This is why `report.` and `report` are the
    same file, and why `CON.` is still the console device.
    """
    return name.rstrip(" .")


def reserved_stem(name: str) -> str:
    """The part of a name the reserved-device lookup sees.

    Everything before the first dot, after trailing dots and spaces are gone,
    uppercased. `con.tar.gz` -> `CON`.
    """
    return win32_effective(name).split(".")[0].strip().upper()


def is_reserved(name: str, profile: Profile) -> bool:
    if not profile.reserved_stems:
        return False
    return reserved_stem(name) in profile.reserved_stems


def component_length(name: str, profile: Profile) -> int:
    """Length in the unit the *target* counts in, which is never characters."""
    if profile.component_unit == "utf-16 code units":
        return len(name.encode("utf-16-le")) // 2
    return len(name.encode("utf-8"))


def collision_key(name: str, profile: Profile, fold: str = "simple_upper") -> str:
    """The key under which the *volume* stores this name.

    Two names with the same key are one file on that volume, whatever the
    sanitiser thinks. Applies the profile's normalisation, then its case table.
    """
    key = win32_effective(name) if profile.strips_trailing_dot_space else name
    if profile.normalisation == "insensitive":
        key = unicodedata.normalize("NFD", key)
    if profile.case_table == "simple":
        key = FOLDS[fold](key)
    return key


def truncate_to_bytes(name: str, limit: int, encoding: str = "utf-8") -> str:
    """Truncate to `limit` bytes without splitting a code point.

    The naive version - `name.encode()[:limit].decode()` - raises or produces
    mojibake whenever the cut lands mid-sequence, which for CJK is two times in
    three. Extension is preserved where it fits, because a truncated name that
    lost its `.csv` is a different kind of broken.
    """
    raw = name.encode(encoding)
    if len(raw) <= limit:
        return name
    stem, dot, ext = name.rpartition(".")
    if dot and len(ext.encode(encoding)) + 1 < limit:
        keep = limit - len(ext.encode(encoding)) - 1
        return _cut(stem, keep, encoding) + "." + ext
    return _cut(name, limit, encoding)


def _cut(s: str, limit: int, encoding: str) -> str:
    raw = s.encode(encoding)[:limit]
    while raw:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            raw = raw[:-1]
    return ""


def naive_truncate(name: str, limit: int, encoding: str = "utf-8") -> str:
    """The version people write. Kept so the difference can be measured."""
    return name.encode(encoding)[:limit].decode(encoding, errors="replace")


# ---------------------------------------------------------------------------
# Six sanitisers, modelled from published implementations
# ---------------------------------------------------------------------------
#
# Each is `str -> str`, which is the interface under examination. They are
# written from the documented behaviour of widely-used implementations rather
# than vendored, because the point is that they disagree with each other on
# ordinary input and none of them returns the collision it just created.


def s_passthrough(name: str) -> str:
    """The control. Rewrites nothing, so it merges nothing.

    Zero collisions and zero validity. Every other row in the comparison is a
    trade against this one.
    """
    return name


def s_strip_bad_chars(name: str) -> str:
    """The regex everyone writes: `re.sub(r'[<>:"/\\|?*]', '_', name)`.

    Found in a thousand snippets. Maps every forbidden character to the same
    replacement, which is exactly the operation that merges `a:b`, `a/b`,
    `a*b` and `a?b` into one file. Leaves control characters, reserved device
    names, trailing dots and every length limit untouched.
    """
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def s_django_valid(name: str) -> str:
    """Django's `get_valid_filename`: strip, spaces to underscores, then
    `re.sub(r'(?u)[^-\\w.]', '', s)`.

    `\\w` is Unicode-aware, so accents and CJK survive - and `:` `/` `*` `?`
    are *deleted* rather than replaced, which merges more aggressively than
    replacing does. Rejects `.`/`..`/empty by raising; modelled here as
    returning `""` so the audit can report it instead of crashing.
    """
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        return ""
    return s


def s_werkzeug_secure(name: str) -> str:
    """Werkzeug's `secure_filename`: NFKD to ASCII, keep `[A-Za-z0-9_.-]`,
    collapse runs to `_`, strip leading/trailing `_.`, then prefix Windows
    device names with `_`.

    The only one of the six that knows reserved names exist. It is also the
    most destructive: the ASCII fold sends every CJK name to the empty string,
    so a corpus of Chinese filenames collapses to one target.
    """
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii")
    for sep in ("/", "\\", " "):
        s = s.replace(sep, "_")
    s = "_".join(re.sub(r"[^A-Za-z0-9_.-]", "", part) for part in s.split("_"))
    s = re.sub(r"_+", "_", s).strip("._")
    if s and s.split(".")[0].upper() in WINDOWS_RESERVED:
        s = "_" + s
    return s


def s_pathvalidate(name: str) -> str:
    """`pathvalidate.sanitize_filename`-shaped: drop forbidden and control
    characters, strip trailing dots and spaces, suffix reserved device names.

    The most correct of the five real ones on *validity*. It still returns a
    string, so the collisions it creates by dropping characters are unreported.
    """
    s = "".join(c for c in name if c not in WINDOWS_FORBIDDEN and ord(c) >= 32)
    s = s.rstrip(" .")
    if s and s.split(".")[0].upper() in WINDOWS_RESERVED:
        stem, dot, ext = s.partition(".")
        s = stem + "_" + (dot + ext if dot else "")
    return s


def s_slugify(name: str) -> str:
    """Aggressive slug: ASCII fold, lowercase, non-alphanumerics to `-`.

    What a CMS does to an upload. Maximum validity, maximum merging - and the
    lowercase step means it collides `Report.csv` with `report.csv` on ext4,
    which is a collision the *sanitiser* invented on a byte-exact volume.

    When the ASCII fold empties the stem, this returns bare `.csv` - so a CJK
    filename becomes a dot-file, invisible in `ls` and in every file picker.
    Preserved rather than patched, because it is what the real thing does and
    `LEADING_DASH_OR_DOT` is the finding that catches it.
    """
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    stem, dot, ext = s.rpartition(".")
    if not dot:
        stem, ext = s, ""
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    ext = re.sub(r"[^a-z0-9]+", "", ext)
    return f"{stem}.{ext}" if ext else stem


SANITISERS: Dict[str, Callable[[str], str]] = {
    "passthrough": s_passthrough,
    "strip_bad_chars": s_strip_bad_chars,
    "django_valid": s_django_valid,
    "werkzeug_secure": s_werkzeug_secure,
    "pathvalidate": s_pathvalidate,
    "slugify": s_slugify,
}


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Severity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Verdict(Enum):
    PORTABLE = "portable"
    LOSSY = "lossy"
    REJECTED = "rejected"


@dataclass
class Finding:
    code: str
    severity: Severity
    message: str
    names: List[str] = field(default_factory=list)
    detail: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        shown = ", ".join(repr(n) for n in self.names[:3])
        more = f" (+{len(self.names) - 3} more)" if len(self.names) > 3 else ""
        return f"[{self.severity.value:>8}] {self.code}: {self.message}\n{' ' * 11}{shown}{more}"


@dataclass
class Report:
    profile: str
    sanitiser: str
    dest: str
    verdict: Verdict
    findings: List[Finding]
    mapping: Dict[str, str]          # source name -> sanitised name
    written: Dict[str, List[str]]    # volume key -> source names landing there

    @property
    def critical(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.CRITICAL]

    # One partition, computed once. Every source name lands in exactly one of
    # three buckets, so `delivered + overwritten + rejected == len(names)` with
    # no remainder. Deriving any of the three as a residual elsewhere is how the
    # first version of this file reported 6 and 10 for the same quantity.

    FATAL_CODES = frozenset(
        {"PATH_TRAVERSAL", "CONTROL_CHARACTER", "RESERVED_DEVICE_NAME",
         "BYTE_LENGTH_EXCEEDED", "PATH_LENGTH_EXCEEDED", "SANITISER_EMPTIED_NAME"}
    )

    def partition(self) -> Tuple[List[str], List[str], List[str]]:
        """(delivered, overwritten, rejected) source names.

        `rejected` is a write that *fails*, which at least raises somewhere.
        `overwritten` is a write that *succeeds* onto a file another source
        already claimed - no error, no log line, one fewer file. That is why
        the second list is the dangerous one.
        """
        rejected = sorted(
            {n for f in self.findings if f.code in self.FATAL_CODES for n in f.names}
        )
        bad = set(rejected)
        keys: Dict[str, List[str]] = defaultdict(list)
        for src, out in self.mapping.items():
            if src in bad or not out:
                continue
            keys[collision_key(out, PROFILES[self.profile])].append(src)
        delivered = sorted(v[0] for v in keys.values() if len(v) == 1)
        overwritten = sorted(n for v in keys.values() if len(v) > 1 for n in v)
        return delivered, overwritten, rejected

    @property
    def rejected_names(self) -> List[str]:
        return self.partition()[2]

    @property
    def delivered(self) -> int:
        """Sources that arrive as their own distinct file. The only number that
        matters, and the one a `str` return value cannot express."""
        return len(self.partition()[0])

    @property
    def lost(self) -> int:
        """Sources that share a target with another source. All of them are at
        risk, not just the losers - which one survives depends on write order."""
        return len(self.partition()[1])


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

_CONFUSABLE_FOLD = str.maketrans(
    {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
        "\u2014": "-", "\u2212": "-", "\u00a0": " ", "\u2018": "'",
        "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u0430": "a",
        "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    }
)


def audit(
    names: Sequence[str],
    profile: Profile = WINDOWS,
    dest: str = r"C:\data",
    sanitiser: str = "strip_bad_chars",
    fold: str = "simple_upper",
) -> Report:
    """Audit a whole corpus against one target, at one destination.

    Takes the set because every failure worth catching is a fact about a pair.
    """
    fn = SANITISERS[sanitiser]
    findings: List[Finding] = []
    mapping: Dict[str, str] = {}

    traversal, control, forbidden, reserved_hits = [], [], [], []
    trailing, too_long_component, too_long_path = [], [], []
    truncated, empty_out, hidden_ext, leading_dash = [], [], [], []
    path_submitted = []

    sep = "\\" if profile.name.startswith("windows") else "/"

    for src in names:
        # Every fatal check below runs on the sanitiser's *output*, because the
        # output is the string handed to the filesystem. Checking the source
        # instead - the obvious way round, and how this was first written -
        # denies a sanitiser credit for the problems it does fix and blames it
        # for ones it removed. The source name is still what gets reported, so
        # a finding remains traceable back to the row it came from.
        out = fn(src)

        # A separator anywhere means the caller handed over a path, not a name.
        # Worth reporting even when the sanitiser flattens it, because
        # flattening is itself a merge: `a/b` and `a\b` and `a_b` converge.
        if len(re.split(r"[/\\]", src)) > 1 or src in {".", ".."} or re.match(
            r"^[A-Za-z]:", src
        ):
            path_submitted.append(src)

        if out == "":
            empty_out.append(src)
            mapping[src] = ""
            continue

        # ...and traversal is only still *fatal* if it survived sanitising.
        out_parts = re.split(r"[/\\]", out)
        if len(out_parts) > 1 or out in {".", ".."} or ".." in out_parts or re.match(
            r"^[A-Za-z]:", out
        ):
            traversal.append(src)

        if any(ord(c) < 32 for c in out):
            control.append(src)
        if profile.forbidden_chars and any(c in profile.forbidden_chars for c in out):
            forbidden.append(src)
        if is_reserved(out, profile):
            reserved_hits.append(src)
        if profile.strips_trailing_dot_space and out != win32_effective(out):
            trailing.append(src)
        # A name made only of dots and spaces survives the deny-list, has legal
        # length, and is not a device - and Win32 reduces it to nothing, so
        # there is no name left to open. Without this it looked like an ordinary
        # collision onto an empty key, which reported two names merging into a
        # file that cannot exist.
        if profile.strips_trailing_dot_space and win32_effective(out) == "":
            empty_out.append(src)
            mapping[src] = ""
            continue
        if re.match(r"^[.-]", out) and out not in {".", ".."}:
            leading_dash.append(src)
        if out.count(".") >= 2 and out.rsplit(".", 1)[-1].lower() in {
            "exe", "scr", "bat", "cmd", "com", "js", "vbs"
        }:
            hidden_ext.append(src)

        # Length is measured in the target's unit, and truncating to fit is a
        # second, separate collision generator - so both are recorded.
        if component_length(out, profile) > profile.max_component_bytes:
            too_long_component.append(src)
            truncated.append(src)
            out = truncate_to_bytes(out, profile.max_component_bytes)

        full = f"{dest.rstrip(sep)}{sep}{out}"
        if profile.max_path_chars is not None and len(full) > profile.max_path_chars:
            too_long_path.append(src)
        if profile.max_path_bytes is not None and len(
            full.encode("utf-8")
        ) > profile.max_path_bytes:
            too_long_path.append(src)

        mapping[src] = out

    # --- where the files actually land on the volume -----------------------
    written: Dict[str, List[str]] = defaultdict(list)
    for src, out in mapping.items():
        if out == "":
            continue
        written[collision_key(out, profile, fold)].append(src)
    written = {k: v for k, v in written.items()}

    merged = {k: v for k, v in written.items() if len(v) > 1}

    # --- findings ---------------------------------------------------------
    def add(code: str, sev: Severity, msg: str, ns: Iterable[str], **detail: object) -> None:
        ns = list(ns)
        if ns:
            findings.append(Finding(code, sev, msg, ns, dict(detail)))

    add("PATH_TRAVERSAL", Severity.CRITICAL,
        "sanitised name is still a path, not a name: it contains a separator, a "
        "`..`, or a leading drive letter. `a:b.txt` is not a filename with a "
        "colon in it - it is drive-relative, meaning `b.txt` in drive A:'s "
        "current directory, which is why `:` is forbidden in the first place",
        sorted(set(traversal)))
    add("PATH_SUBMITTED_AS_NAME", Severity.WARNING,
        "source was a path, not a filename; flattening it to one component is "
        "itself a merge, because every separator becomes the same character",
        sorted(set(path_submitted)))
    add("CONTROL_CHARACTER", Severity.CRITICAL,
        "contains a control character; NUL cannot be encoded in a POSIX name at all",
        control)
    add("RESERVED_DEVICE_NAME", Severity.CRITICAL,
        "stem before the first dot is a reserved Win32 device; the extension does not help",
        reserved_hits,
        stems=sorted({reserved_stem(n) for n in reserved_hits}))
    add("BYTE_LENGTH_EXCEEDED", Severity.CRITICAL,
        f"component exceeds {profile.max_component_bytes} {profile.component_unit} "
        f"(the limit is not in characters)",
        too_long_component)
    add("PATH_LENGTH_EXCEEDED", Severity.CRITICAL,
        f"full path at {dest!r} exceeds the target's limit; the name alone is legal",
        sorted(set(too_long_path)))
    add("SANITISER_EMPTIED_NAME", Severity.CRITICAL,
        "nothing is left to write: the sanitiser returned the empty string, or "
        "the name was only dots and spaces and Win32 strips all of them",
        empty_out)

    if merged:
        losers = sorted({n for v in merged.values() for n in v[1:]})
        add("COLLISION_AFTER_SANITISE", Severity.CRITICAL,
            f"{len(merged)} target name(s) receive more than one source; "
            f"{len(losers)} file(s) are overwritten and every call returns success",
            losers,
            groups={k: v for k, v in sorted(merged.items())[:8]})

    # These two are volume-level, not sanitiser-level: the names are already
    # legal and already distinct, and the *filesystem* merges them. Report the
    # actual pairs, not every name that happens to share a key for some other
    # reason.
    if profile.case_table:
        pairs = _pairs_differing_by(merged, "case")
        add("CASE_FOLD_COLLISION", Severity.CRITICAL,
            f"names differ only by case and {profile.name} folds them into one file; "
            f"no sanitiser was involved",
            sorted({n for p in pairs for n in p}), pairs=pairs)
    if profile.normalisation == "insensitive":
        pairs = _pairs_differing_by(merged, "nfc")
        add("NORMALISATION_COLLISION", Severity.CRITICAL,
            f"names differ only by Unicode normalisation; one file on {profile.name}, "
            f"two on a byte-exact volume",
            sorted({n for p in pairs for n in p}), pairs=pairs)
    pairs = _pairs_differing_by(merged, "trailing")
    add("TRAILING_STRIP_COLLISION", Severity.CRITICAL,
        "two names that differ only by a trailing dot or space; Win32 strips both "
        "and opens the same file",
        sorted({n for p in pairs for n in p}), pairs=pairs)

    add("TRUNCATION_COLLISION", Severity.CRITICAL,
        "cutting the name to fit produced a target another source already claimed",
        [n for n in truncated if len(written.get(collision_key(mapping[n], profile, fold), [])) > 1])

    add("TRAILING_DOT_OR_SPACE", Severity.WARNING,
        "Win32 strips trailing dots and spaces before opening, so this is not "
        "the name you asked for",
        trailing)
    add("RESERVED_CHARACTER", Severity.WARNING,
        f"contains a character {profile.name} rejects: {profile.forbidden_chars!r}",
        forbidden)
    add("EXTENSION_MASQUERADE", Severity.WARNING,
        "double extension ending in an executable type; a hidden-extensions "
        "desktop shows the harmless one",
        hidden_ext)

    disagree = case_table_disagreements(names)
    if disagree:
        add("CASE_TABLE_DISAGREEMENT", Severity.WARNING,
            "py_lower, py_casefold and simple_upper give different answers for these; "
            "only simple_upper models a real volume's case table",
            sorted({n for pair in disagree for n in pair}),
            pairs=disagree[:8])

    conf = confusable_pairs(names)
    if conf:
        add("CONFUSABLE_PAIR", Severity.INFO,
            "visually identical, distinct on every filesystem; no index catches this "
            "and no human distinguishes the two rows",
            sorted({n for pair in conf for n in pair}),
            pairs=conf[:8])
    add("LEADING_DASH_OR_DOT", Severity.INFO,
        "leading dash reads as a CLI flag; leading dot hides the file on POSIX",
        [n for n in leading_dash if n not in set(traversal)])

    # --- verdict ----------------------------------------------------------
    codes = {f.code for f in findings}
    if codes & Report.FATAL_CODES:
        verdict = Verdict.REJECTED
    elif merged:
        verdict = Verdict.LOSSY
    else:
        verdict = Verdict.PORTABLE

    # Criticals first. The checks above are emitted in the order the target
    # applies them, which is not the order a reader needs them in.
    rank = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    findings.sort(key=lambda f: rank[f.severity])

    return Report(profile.name, sanitiser, dest, verdict, findings, mapping, written)


def collision_reason(a: str, b: str) -> str:
    """Which rule finally merges these two names. Exactly one answer.

    The rules are applied *cumulatively*, in the order the target applies them:
    Win32 strips trailing dots and spaces first, then the volume normalises,
    then it case-folds. The answer is the rule at which the two names become
    equal - earlier rules may also have been needed to get there.

    Checking each rule in isolation instead, which is how this was first
    written, misclassifies every pair that needs two of them: `CAFÉ.txt.` and
    `café.txt` need the trailing strip *and* the case fold, matched neither
    check on its own, and were reported as the sanitiser's fault.
    """
    if a == b:
        return "identical"
    a, b = a.rstrip(" ."), b.rstrip(" .")
    if a == b:
        return "trailing"
    a, b = unicodedata.normalize("NFD", a), unicodedata.normalize("NFD", b)
    if a == b:
        return "nfc"
    if fold_simple_upper(a) == fold_simple_upper(b):
        return "case"
    return "sanitiser"


def _pairs_differing_by(merged: Dict[str, List[str]], kind: str) -> List[Tuple[str, str]]:
    """Within each merged group, the pairs whose reason for merging is `kind`.

    That they collide is already known - they share a key. What this answers is
    *why*, which decides whether there was anything a sanitiser could have done
    about it. For `trailing`, `nfc` and `case` the answer is no: the names were
    already legal and already distinct, and the volume merged them.
    """
    out: List[Tuple[str, str]] = []
    for group in merged.values():
        ordered = sorted(group)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                if collision_reason(a, b) == kind:
                    out.append((a, b))
    return out


def case_table_disagreements(names: Sequence[str]) -> List[Tuple[str, str]]:
    """Pairs where the three fold models do not agree on whether they collide.

    This is the finding that says the tool cannot answer the question for you:
    the answer depends on the volume's case table, and `str.casefold()` is not
    any volume's case table.
    """
    out: List[Tuple[str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            votes = {k: FOLDS[k](a) == FOLDS[k](b) for k in FOLDS}
            if len(set(votes.values())) > 1:
                out.append((a, b))
    return out


def confusable_pairs(names: Sequence[str]) -> List[Tuple[str, str]]:
    """Pairs that render identically and are distinct bytes everywhere."""
    out: List[Tuple[str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if a == b:
                continue
            if a.translate(_CONFUSABLE_FOLD) == b.translate(_CONFUSABLE_FOLD):
                out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Cross-sanitiser comparison: the validity / injectivity frontier
# ---------------------------------------------------------------------------


@dataclass
class Row:
    """One sanitiser's result over one corpus and one target.

    `source` = `delivered` + `overwritten` + `rejected`, exactly, with no
    remainder - which is the accounting a `str` return value cannot give you.
    """

    sanitiser: str
    source: int
    delivered: int
    overwritten: int
    rejected: int
    distinct_out: int
    critical: int
    verdict: Verdict

    @property
    def delivery_rate(self) -> float:
        return self.delivered / self.source if self.source else 0.0


def compare(
    names: Sequence[str],
    profile: Profile = WINDOWS,
    dest: str = r"C:\data",
    fold: str = "simple_upper",
) -> List[Row]:
    """Run every sanitiser over the same corpus and target.

    Validity and injectivity trade off directly: a sanitiser buys legal names by
    throwing information away, and the information it throws away is what kept
    two source names apart. `passthrough` merges the least and is legal least
    often; `slugify` is legal nearly always and merges most. Neither extreme
    delivers the most files, which is the actual objective.
    """
    rows: List[Row] = []
    for key in SANITISERS:
        r = audit(names, profile, dest, key, fold)
        delivered, overwritten, rejected = r.partition()
        assert len(delivered) + len(overwritten) + len(rejected) == len(names), (
            "the three buckets must partition the corpus exactly"
        )
        rows.append(
            Row(
                sanitiser=key,
                source=len(names),
                delivered=len(delivered),
                overwritten=len(overwritten),
                rejected=len(rejected),
                distinct_out=len({v for v in r.mapping.values() if v}),
                critical=len(r.critical),
                verdict=r.verdict,
            )
        )
    return rows


def round_trip(names: Sequence[str], built_on: Profile, opened_on: Profile) -> Dict[str, object]:
    """Build an archive on one filesystem, extract it on another.

    Names that are distinct on `built_on` and identical on `opened_on` are the
    files that silently do not arrive - the zip is valid, the extraction reports
    no error, and the count of files on disk is lower than the count in the
    archive.
    """
    packed: Dict[str, List[str]] = defaultdict(list)
    for n in names:
        packed[collision_key(n, built_on)].append(n)
    entries = [v[0] for v in packed.values()]

    landed: Dict[str, List[str]] = defaultdict(list)
    for n in entries:
        landed[collision_key(n, opened_on)].append(n)

    return {
        "built_on": built_on.name,
        "opened_on": opened_on.name,
        "entries": len(entries),
        "files_on_disk": len(landed),
        "lost": len(entries) - len(landed),
        "casualties": sorted(
            {n for v in landed.values() if len(v) > 1 for n in v[1:]}
        ),
    }


# ---------------------------------------------------------------------------
# Sample corpus: one export dump, every mechanism represented once
# ---------------------------------------------------------------------------

SAMPLE_NAMES: List[str] = [
    # ordinary, and the case pair nobody notices
    "Q3 Report.csv",
    "Q3 report.csv",
    # reserved devices, including the "with an extension" cases
    "CON",
    "CON.txt",
    "nul.log",
    "COM1.csv",
    "aux.tar.gz",
    "CON.",
    # trailing dot and space: two names, one file, on Win32
    "monthly report.",
    "monthly report",
    "budget ",
    # four distinct names, one replacement character
    "a:b.txt",
    "a*b.txt",
    "a?b.txt",
    "a|b.txt",
    # normalisation: one file on APFS, two on ext4
    "caf\u00e9.txt",
    "cafe\u0301.txt",
    # case tables that disagree with each other
    "Stra\u00dfe.txt",
    "STRASSE.txt",
    "\u03a3\u0399\u03a3\u03a5\u03a6\u039f\u03a3.txt",
    "\u03c3\u03b9\u03c3\u03c5\u03c6\u03bf\u03c3.txt",
    "\u0130stanbul.txt",
    "istanbul.txt",
    # confusables: distinct everywhere, identical to a human
    "report\u20102024.pdf",
    "report-2024.pdf",
    # not filenames at all
    "../../etc/passwd",
    "/absolute/path.txt",
    "C:\\Windows\\system32.dll",
    # unencodable
    "data\x00.csv",
    "log\nline.txt",
    # length: characters are not the unit. 90 CJK characters is 90 UTF-16 code
    # units (legal on NTFS) and 270 UTF-8 bytes (over ext4's NAME_MAX). The
    # emoji name is 70 characters and 280 bytes. Same two names, opposite
    # verdicts, decided by which unit the target counts in.
    "\u5b63\u5ea6\u9500\u552e\u62a5\u544a" * 15 + ".csv",
    "\U0001f4c8" * 70 + ".png",
    "a" * 300 + ".csv",
    # legal components everywhere, and unwritable at a deep destination. These
    # are the names that make validity a property of the *path*, not the name -
    # and they are what real export filenames actually look like.
    "Regional Sales Performance and Margin Analysis - EMEA and APAC Combined - "
    "Q3 2026 - Final Draft Reviewed by Group Finance.xlsx",
    "Customer Churn Cohort Analysis with Retention Curves and Segment Breakdown "
    "- Enterprise and Mid-Market - FY2026 Q1 through Q3 - Prepared by the Data "
    "Team - Reviewed by Revenue Operations - Final v3.xlsx",
    # shapes that break tools rather than filesystems
    "-rf.txt",
    ".hidden",
    "invoice.pdf.exe",
    "...",
    "   ",
    "\u00d6l\u00fcm.txt",
    "\u00f6l\u00fcm.txt",
]
