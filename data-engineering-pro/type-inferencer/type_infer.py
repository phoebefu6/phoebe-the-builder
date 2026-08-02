"""Infer SQL column types from raw text values - casting only when the cast is reversible.

Two guarantees, and the whole tool falls out of them:

1. TEXT-PRESERVING (string -> narrow type): rendering the stored value must reproduce
   the source text exactly. This is what kills the leading-zero trap ("01234" is not
   the integer 1234), along with "+5" and "1.0". Surrounding whitespace is the one
   documented exception: it is trimmed before inference, with a finding stating that
   the inferred type only holds if the loader trims too.

2. VALUE-PRESERVING (numeric types): the stored value must equal the decimal number
   written in the file, exactly. This is what refuses DOUBLE PRECISION for money -
   the binary float nearest "10.10" is 10.09999999999999964..., which is a different
   number, not a rounding detail.

Where the evidence cannot support a decision (an ambiguous DD/MM vs MM/DD column, a
0/1 column that could be a flag or a count, a thousands-separated amount), the tool
leaves the column as text and emits a finding saying why. Abstaining out loud beats
guessing silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------

DEFAULT_NULL_TOKENS = frozenset(
    {"", "null", "none", "n/a", "na", "nil", "-", "--", "unknown", "?", "\\n"}
)

TRUE_TOKENS = frozenset({"true", "t", "yes", "y"})
FALSE_TOKENS = frozenset({"false", "f", "no", "n"})

INT16_MAX = 32767
INT32_MAX = 2147483647
INT64_MAX = 9223372036854775807


@dataclass
class Policy:
    """Every threshold the inference leans on, in one place and visible in the UI."""

    null_tokens: frozenset = DEFAULT_NULL_TOKENS
    min_rows_for_not_null: int = 25       # below this, absence of NULL is not evidence
    min_rows_for_narrowing: int = 25      # below this, do not pick SMALLINT/enums
    varchar_block: int = 8                # round VARCHAR length up to a multiple of this
    varchar_max: int = 255                # above this, use TEXT
    enum_max_distinct: int = 12
    enum_max_ratio: float = 0.15          # distinct / non-null
    headroom_warn_ratio: float = 0.80     # observed max within 80% of type ceiling -> warn
    double_scale_hint: int = 5            # scale >= this looks like measurement, not money


# --------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------

SEVERITY_ORDER = {"block": 0, "warn": 1, "info": 2}


@dataclass
class Finding:
    column: str
    code: str
    severity: str  # block | warn | info
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper():5}] {self.column} - {self.code}\n        {self.message}"


@dataclass
class ColumnType:
    kind: str  # BOOLEAN INTEGER SMALLINT BIGINT DECIMAL DOUBLE DATE TIMESTAMP VARCHAR TEXT
    precision: Optional[int] = None
    scale: Optional[int] = None
    length: Optional[int] = None
    nullable: bool = True
    enum_values: Optional[List[str]] = None  # only ever a *suggestion*, never applied

    def sql(self, dialect: str = "postgres") -> str:
        k = self.kind
        if dialect == "sqlite":
            # SQLite has type affinity, not types. Collapse honestly.
            affinity = {
                "BOOLEAN": "INTEGER", "SMALLINT": "INTEGER", "INTEGER": "INTEGER",
                "BIGINT": "INTEGER", "DECIMAL": "NUMERIC", "DOUBLE": "REAL",
                "DATE": "TEXT", "TIMESTAMP": "TEXT", "VARCHAR": "TEXT", "TEXT": "TEXT",
            }
            return affinity[k]
        if k == "DECIMAL":
            name = "NUMERIC" if dialect == "postgres" else "DECIMAL"
            return f"{name}({self.precision},{self.scale})"
        if k == "DOUBLE":
            return "DOUBLE PRECISION" if dialect == "postgres" else "DOUBLE"
        if k == "VARCHAR":
            return f"VARCHAR({self.length})"
        return k

    def __str__(self) -> str:
        return self.sql("postgres")


@dataclass
class ColumnProfile:
    name: str
    n_rows: int
    n_null: int
    n_value: int
    distinct: int
    min_len: int
    max_len: int


@dataclass
class ColumnResult:
    name: str
    profile: ColumnProfile
    type: ColumnType
    findings: List[Finding] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        """True when the column was left as text *because* evidence was insufficient."""
        return self.type.kind in ("TEXT", "VARCHAR") and any(
            f.code in ABSTAIN_CODES for f in self.findings
        )


ABSTAIN_CODES = {
    "ambiguous-date-order", "date-order-conflict", "invalid-calendar-date",
    "thousands-separator", "sign-prefix", "padded-numeric", "single-valued-boolean",
    "no-observations",
}


# --------------------------------------------------------------------------------------
# Value-level helpers
# --------------------------------------------------------------------------------------


def is_null(raw: str, policy: Policy) -> bool:
    return raw.strip().lower() in policy.null_tokens


def int_roundtrips(raw: str) -> bool:
    """Text-preserving integer check: str(int(raw)) must reproduce raw exactly."""
    try:
        return str(int(raw)) == raw
    except (ValueError, TypeError):
        return False


def decimal_of(raw: str) -> Optional[Decimal]:
    try:
        d = Decimal(raw)
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not d.is_finite():
        return None
    return d


def double_is_value_preserving(raw: str) -> bool:
    """A float stores raw exactly only if its binary expansion equals the decimal written."""
    d = decimal_of(raw)
    if d is None:
        return False
    try:
        return Decimal(float(raw)) == d
    except (OverflowError, ValueError):
        return False


def decimal_shape(values: Sequence[str]) -> Tuple[int, int]:
    """(precision, scale) that stores every value exactly."""
    max_scale = 0
    max_int_digits = 1
    for raw in values:
        d = decimal_of(raw)
        if d is None:
            continue
        exp = d.as_tuple().exponent
        digits = len(d.as_tuple().digits)
        scale = max(0, -int(exp))
        int_digits = max(1, digits + int(exp)) if exp < 0 else digits + int(exp)
        max_scale = max(max_scale, scale)
        max_int_digits = max(max_int_digits, int_digits)
    return max_int_digits + max_scale, max_scale


SLASH_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$")
THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
SIGNED = re.compile(r"^\+\d+$")
PADDED = re.compile(r"^0\d+$")
DIGITS = re.compile(r"^\d+$")


def parses_with(raw: str, fmt: str) -> bool:
    """Text-preserving temporal check: the format must also *reproduce* the source text."""
    try:
        return datetime.strptime(raw, fmt).strftime(fmt) == raw
    except ValueError:
        return False


# --------------------------------------------------------------------------------------
# Per-kind candidates
# --------------------------------------------------------------------------------------


def try_boolean(values: Sequence[str]) -> Optional[ColumnType]:
    low = {v.strip().lower() for v in values}
    if not low or not low <= (TRUE_TOKENS | FALSE_TOKENS):
        return None
    if not (low & TRUE_TOKENS) or not (low & FALSE_TOKENS):
        return None  # a single-valued column is not evidence of a boolean
    return ColumnType("BOOLEAN")


def try_integer(values: Sequence[str], policy: Policy, n_rows: int) -> Optional[ColumnType]:
    if not all(int_roundtrips(v) for v in values):
        return None
    biggest = max(abs(int(v)) for v in values)
    if biggest <= INT16_MAX and n_rows >= policy.min_rows_for_narrowing:
        return ColumnType("SMALLINT")
    if biggest <= INT32_MAX:
        return ColumnType("INTEGER")
    if biggest <= INT64_MAX:
        return ColumnType("BIGINT")
    return None  # beyond int64 - fall through to DECIMAL


def try_decimal(values: Sequence[str]) -> Optional[ColumnType]:
    if any(decimal_of(v) is None for v in values):
        return None
    precision, scale = decimal_shape(values)
    return ColumnType("DECIMAL", precision=precision, scale=scale)


def try_temporal(values: Sequence[str], name: str) -> Tuple[Optional[ColumnType], List[Finding]]:
    findings: List[Finding] = []

    if all(parses_with(v, "%Y-%m-%d") for v in values) and all(ISO_DATE.match(v) for v in values):
        return ColumnType("DATE"), findings

    if all(ISO_TS.match(v) for v in values):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            if all(parses_with(v, fmt) for v in values):
                return ColumnType("TIMESTAMP"), findings

    if all(SLASH_DATE.match(v) for v in values):
        firsts = [int(SLASH_DATE.match(v).group(1)) for v in values]
        seconds = [int(SLASH_DATE.match(v).group(2)) for v in values]
        day_first = any(a > 12 for a in firsts)
        month_first = any(b > 12 for b in seconds)
        if day_first and month_first:
            findings.append(Finding(
                name, "date-order-conflict", "block",
                "Some rows require DD/MM and others require MM/DD in the same column. "
                "The source is mixing two formats - no single date type is safe. Kept as text.",
            ))
            return None, findings
        if day_first or month_first:
            fmt = "%d/%m/%Y" if day_first else "%m/%d/%Y"
            if all(parses_with(v, fmt) for v in values):
                findings.append(Finding(
                    name, "date-order-resolved", "info",
                    f"Slash dates disambiguated as {fmt} by a component > 12 in the data.",
                ))
                return ColumnType("DATE"), findings
            findings.append(Finding(
                name, "invalid-calendar-date", "block",
                f"Order resolves to {fmt} but at least one value is not a real calendar date.",
            ))
            return None, findings
        findings.append(Finding(
            name, "ambiguous-date-order", "warn",
            "Every day and month component is <= 12, so DD/MM and MM/DD both parse and "
            "disagree on the actual dates. Kept as text - supply the source format instead "
            "of letting a loader pick one.",
        ))
        return None, findings

    return None, findings


def size_string(
    values: Sequence[str], policy: Policy, name: str, quiet: bool = False
) -> Tuple[ColumnType, List[Finding]]:
    """Size a string column. `quiet=True` for columns we are abstaining on - a column
    parked as text pending an upstream fix should not also collect enum advice."""
    findings: List[Finding] = []
    lengths = [len(v) for v in values]
    max_len, min_len = max(lengths), min(lengths)
    distinct = len(set(values))

    if max_len > policy.varchar_max:
        return ColumnType("TEXT"), findings

    if min_len == max_len:
        length = max_len
        if not quiet:
            findings.append(Finding(
                name, "fixed-width", "info",
                f"Every value is exactly {max_len} characters - a code format, not free text. "
                f"VARCHAR({max_len}) with no headroom; widen it if the format can change.",
            ))
    else:
        length = max(policy.varchar_block, -(-max_len // policy.varchar_block) * policy.varchar_block)
        if not quiet:
            findings.append(Finding(
                name, "length-is-a-sample-max", "info",
                f"Longest observed value is {max_len} chars; sized VARCHAR({length}). "
                "An observed maximum is not a specified maximum.",
            ))

    t = ColumnType("VARCHAR", length=length)
    if (
        not quiet
        and distinct <= policy.enum_max_distinct
        and distinct / max(1, len(values)) <= policy.enum_max_ratio
    ):
        t.enum_values = sorted(set(values))
        findings.append(Finding(
            name, "enum-candidate", "info",
            f"Only {distinct} distinct values across {len(values)} rows "
            f"({', '.join(repr(v) for v in t.enum_values)}). "
            "A CHECK constraint would catch new values at write time.",
        ))
    return t, findings


# --------------------------------------------------------------------------------------
# Column inference
# --------------------------------------------------------------------------------------


def infer_column(name: str, raw: Sequence[str], policy: Optional[Policy] = None) -> ColumnResult:
    policy = policy or Policy()
    n_rows = len(raw)
    values = [v for v in raw if not is_null(v, policy)]
    n_null = n_rows - len(values)

    pre_findings: List[Finding] = []
    n_padded_ws = sum(1 for v in values if v != v.strip())
    if n_padded_ws:
        # Surrounding whitespace is a formatting artifact, not data - but stripping it IS a
        # rewrite of the source, so the type below only holds if the loader also strips.
        pre_findings.append(Finding(
            name, "whitespace-padded", "warn",
            f"{n_padded_ws}/{len(values)} values carry leading or trailing whitespace. Inference "
            "ran on the trimmed values, so the inferred type only holds if the loader trims too "
            "(COPY ... , pandas `skipinitialspace`, or an explicit TRIM in the staging query).",
        ))
        values = [v.strip() for v in values]

    profile = ColumnProfile(
        name=name,
        n_rows=n_rows,
        n_null=n_null,
        n_value=len(values),
        distinct=len(set(values)),
        min_len=min((len(v) for v in values), default=0),
        max_len=max((len(v) for v in values), default=0),
    )
    findings: List[Finding] = list(pre_findings)

    if not values:
        findings.append(Finding(
            name, "no-observations", "warn",
            f"All {n_rows} values are null or a null sentinel. No type can be inferred; kept as text.",
        ))
        return ColumnResult(name, profile, ColumnType("TEXT", nullable=True), findings)

    nullable = n_null > 0
    if not nullable and n_rows < policy.min_rows_for_not_null:
        nullable = True
        findings.append(Finding(
            name, "thin-null-evidence", "warn",
            f"No nulls seen, but only {n_rows} rows (< {policy.min_rows_for_not_null}). "
            "Left nullable - absence of evidence is not evidence of NOT NULL.",
        ))

    # --- text shapes that a naive caster would silently swallow -----------------------
    n_padded = sum(1 for v in values if PADDED.match(v))
    if n_padded and all(DIGITS.match(v) for v in values):
        findings.append(Finding(
            name, "padded-numeric", "warn",
            f"{n_padded}/{len(values)} values are digits with a leading zero (zip code, account "
            "number, product code). The whole column casts to an integer cleanly and loses the "
            "padding forever - '01234' and '1234' are not the same identifier. Kept as text.",
        ))
        t, more = size_string(values, policy, name, quiet=True)
        t.nullable = nullable
        return ColumnResult(name, profile, t, findings + more)

    if any(THOUSANDS.match(v) for v in values):
        findings.append(Finding(
            name, "thousands-separator", "warn",
            "Values carry thousands separators (e.g. '1,234.50'). They are numeric in intent "
            "but not in form. Normalise upstream, then re-infer - this tool will not silently "
            "rewrite source text.",
        ))
        t, more = size_string(values, policy, name, quiet=True)
        t.nullable = nullable
        return ColumnResult(name, profile, t, findings + more)

    if any(SIGNED.match(v) for v in values):
        findings.append(Finding(
            name, "sign-prefix", "warn",
            "Some values carry an explicit '+' prefix. It casts to a number cleanly and the "
            "prefix is gone forever - harmless for a quantity, not harmless if the '+' is "
            "carrying meaning (a signed adjustment, a phone country code). Strip it upstream "
            "and re-infer. Kept as text.",
        ))
        t, more = size_string(values, policy, name, quiet=True)
        t.nullable = nullable
        return ColumnResult(name, profile, t, findings + more)

    # --- the ladder, narrow to wide ---------------------------------------------------
    boolean = try_boolean(values)
    if boolean is not None:
        boolean.nullable = nullable
        return ColumnResult(name, profile, boolean, findings)

    low = {v.strip().lower() for v in values}
    if low <= (TRUE_TOKENS | FALSE_TOKENS) and len(low) == 1:
        findings.append(Finding(
            name, "single-valued-boolean", "warn",
            f"Every value is {sorted(low)[0]!r}. That is consistent with a boolean but does not "
            "demonstrate one - a column that has only ever been true is not evidence the false "
            "case exists. Left as text; type it by hand if the source says it is a flag.",
        ))

    if set(values) == {"0", "1"}:
        findings.append(Finding(
            name, "zero-one-ambiguous", "warn",
            "Column holds only 0 and 1. That is a flag or a count of at-most-one - the data "
            "cannot tell which. Typed SMALLINT rather than BOOLEAN; confirm with the source owner.",
        ))

    integer = try_integer(values, policy, n_rows)
    if integer is not None:
        integer.nullable = nullable
        biggest = max(abs(int(v)) for v in values)
        ceiling = {"SMALLINT": INT16_MAX, "INTEGER": INT32_MAX, "BIGINT": INT64_MAX}[integer.kind]
        if biggest > ceiling * policy.headroom_warn_ratio:
            findings.append(Finding(
                name, "near-type-ceiling", "warn",
                f"Largest observed value ({biggest}) is within "
                f"{100 * (1 - policy.headroom_warn_ratio):.0f}% of the {integer.kind} ceiling. "
                "Widen one step before this column grows into an overflow at 3am.",
            ))
        if n_rows < policy.min_rows_for_narrowing and integer.kind == "INTEGER":
            findings.append(Finding(
                name, "thin-range-evidence", "info",
                f"Only {n_rows} rows; did not narrow to SMALLINT on this little evidence.",
            ))
        return ColumnResult(name, profile, integer, findings)

    temporal, tf = try_temporal(values, name)
    findings.extend(tf)
    if temporal is not None:
        temporal.nullable = nullable
        return ColumnResult(name, profile, temporal, findings)
    if any(f.code in ("ambiguous-date-order", "date-order-conflict", "invalid-calendar-date") for f in tf):
        t, more = size_string(values, policy, name, quiet=True)
        t.nullable = nullable
        return ColumnResult(name, profile, t, findings + more)

    dec = try_decimal(values)
    if dec is not None:
        dec.nullable = nullable
        lossy_as_double = [v for v in values if not double_is_value_preserving(v)]
        if lossy_as_double:
            example = lossy_as_double[0]
            findings.append(Finding(
                name, "decimal-over-double", "info",
                f"Chose DECIMAL({dec.precision},{dec.scale}) over DOUBLE PRECISION: the nearest "
                f"binary float to '{example}' is {Decimal(float(example))}, a different number. "
                "Fine for a sensor reading, wrong for money.",
            ))
        if (dec.scale or 0) >= policy.double_scale_hint and profile.distinct > 0.5 * profile.n_value:
            findings.append(Finding(
                name, "measurement-not-money", "info",
                f"Scale {dec.scale} with high cardinality reads as a measurement rather than a "
                "currency amount. DOUBLE PRECISION is the idiomatic choice there and is faster; "
                "switch if ~1e-16 relative error is acceptable for this column.",
            ))
        return ColumnResult(name, profile, dec, findings)

    t, more = size_string(values, policy, name)
    t.nullable = nullable
    return ColumnResult(name, profile, t, findings + more)


def infer_table(
    rows: Sequence[Dict[str, str]], policy: Optional[Policy] = None
) -> List[ColumnResult]:
    if not rows:
        return []
    columns: List[str] = []
    for r in rows:                      # union of keys, first-seen order - a ragged export
        for k in r:                     # must not lose the columns that appear late
            if k not in columns:
                columns.append(k)
    return [infer_column(c, [r.get(c, "") or "" for r in rows], policy) for c in columns]


# --------------------------------------------------------------------------------------
# DDL
# --------------------------------------------------------------------------------------


def emit_ddl(table: str, results: Sequence[ColumnResult], dialect: str = "postgres") -> str:
    width = max(len(r.name) for r in results) + 2
    lines = [f"CREATE TABLE {table} ("]
    body = []
    for r in results:
        null_sql = "" if r.type.nullable else " NOT NULL"
        body.append(f"    {r.name:<{width}}{r.type.sql(dialect)}{null_sql}")
    lines.append(",\n".join(body))
    lines.append(");")

    checks = []
    for r in results:
        if r.type.enum_values:
            vals = ", ".join(f"'{v}'" for v in r.type.enum_values)
            checks.append(f"-- ALTER TABLE {table} ADD CONSTRAINT {r.name}_allowed "
                          f"CHECK ({r.name} IN ({vals}));")
    if checks:
        lines.append("")
        lines.append("-- Suggested, not applied - confirm the value set is closed first:")
        lines.extend(checks)
    if dialect == "sqlite":
        lines.append("")
        lines.append("-- NOTE: SQLite has type affinity, not types. DATE/TIMESTAMP/VARCHAR all")
        lines.append("-- collapse to TEXT and nothing enforces the inferred shape at write time.")
    return "\n".join(lines)


def all_findings(results: Sequence[ColumnResult]) -> List[Finding]:
    out = [f for r in results for f in r.findings]
    return sorted(out, key=lambda f: (SEVERITY_ORDER[f.severity], f.column))


# --------------------------------------------------------------------------------------
# The strawman every loader ships: sample N rows, try int -> float -> date -> text
# --------------------------------------------------------------------------------------


def naive_infer(name: str, raw: Sequence[str], sample: int = 200) -> ColumnType:
    values = [v for v in raw[:sample] if v.strip() not in ("", "NULL", "null", "N/A")]
    if not values:
        return ColumnType("TEXT")
    try:
        [int(v) for v in values]
        return ColumnType("BIGINT")
    except ValueError:
        pass
    try:
        [float(v) for v in values]
        return ColumnType("DOUBLE")
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            [datetime.strptime(v, fmt) for v in values]
            return ColumnType("TIMESTAMP" if "%H" in fmt else "DATE")
        except ValueError:
            continue
    return ColumnType("TEXT")


# --------------------------------------------------------------------------------------
# Objective grading: does a proposed type actually survive the whole column?
# --------------------------------------------------------------------------------------


def is_lossy(t: ColumnType, raw: Sequence[str], policy: Optional[Policy] = None) -> bool:
    """True if storing the FULL column under type `t` loses information or fails to load.

    Applies the same documented pre-step as the engine: null sentinels dropped, surrounding
    whitespace trimmed. Anything beyond that would be scoring one strategy against a
    different input than the other.
    """
    policy = policy or Policy()
    values = [v.strip() for v in raw if not is_null(v, policy)]
    if not values:
        return False
    k = t.kind
    if k == "TEXT":
        return False
    if k == "VARCHAR":
        return any(len(v) > (t.length or 0) for v in values)
    if k == "BOOLEAN":
        return not {v.strip().lower() for v in values} <= (TRUE_TOKENS | FALSE_TOKENS)
    if k in ("SMALLINT", "INTEGER", "BIGINT"):
        ceiling = {"SMALLINT": INT16_MAX, "INTEGER": INT32_MAX, "BIGINT": INT64_MAX}[k]
        return any(not int_roundtrips(v) or abs(int(v)) > ceiling for v in values)
    if k == "DOUBLE":
        return any(not double_is_value_preserving(v) for v in values)
    if k == "DECIMAL":
        p, s = decimal_shape(values)
        return any(decimal_of(v) is None for v in values) or s > (t.scale or 0) or p > (t.precision or 0)
    if k == "DATE":
        return not (
            all(parses_with(v, "%Y-%m-%d") for v in values)
            or all(parses_with(v, "%m/%d/%Y") for v in values)
            or all(parses_with(v, "%d/%m/%Y") for v in values)
        )
    if k == "TIMESTAMP":
        return not any(
            all(parses_with(v, fmt) for v in values)
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
        )
    return False


# --------------------------------------------------------------------------------------
# Demo corpus - deterministic, no RNG, shaped like a real ops export
# --------------------------------------------------------------------------------------

STATUSES = ["shipped", "pending", "cancelled", "delivered"]
ZIPS = ["01234", "02138", "10001", "94103", "07030"]
LONG_NOTE = ("Customer called about a delayed shipment; agent confirmed the carrier scan "
             "and reissued the tracking link after verifying the delivery window. " * 2).strip()


def demo_rows(n: int = 240) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for i in range(n):
        day = 1 + (i % 12)
        month = 1 + (i % 11)
        weight = "12.5" if i == 210 else str([3, 7, 12, 5, 21][i % 5])
        rows.append({
            "order_id": str(1001 + i),
            "zip_code": ZIPS[i % 5],
            "qty": f" {1 + i % 5}" if i % 6 == 0 else str(1 + i % 5),
            "unit_price": ["10.10", "3.99", "249.00", "18.75"][i % 4],
            "discount_rate": ["0.075", "0.100", "0.000", "0.250"][i % 4],
            "sensor_drift": f"0.{(i * 4177 + 13) % 1000000:06d}",
            "is_gift": "Y" if i % 3 == 0 else "N",
            "flag_01": str(i % 2),
            "order_date": f"2026-{month:02d}-{day:02d}",
            "ship_date": f"{1 + i % 12:02d}/{1 + (i * 5) % 12:02d}/2026",
            "delivered_at": f"2026-{month:02d}-{day:02d} {8 + i % 10:02d}:22:11",
            "customer_ref": f"CUS-{1 + i % 90:04d}",
            "status": STATUSES[i % 4],
            "big_id": str(9007199254740993 + i),
            "weight_kg": "" if i % 40 == 7 else weight,
            "legacy_amount": f"{1 + i % 9},{100 + i % 800:03d}.50",
            "notes": LONG_NOTE if i == 3 else ("" if i % 5 == 0 else f"case {i} reviewed"),
        })
    return rows


# What a careful data engineer would write by hand after reading the file.
EXPECTED: Dict[str, str] = {
    "order_id": "SMALLINT",
    "zip_code": "VARCHAR(5)",
    "qty": "SMALLINT",
    "unit_price": "NUMERIC(5,2)",
    "discount_rate": "NUMERIC(4,3)",
    "sensor_drift": "NUMERIC(7,6)",
    "is_gift": "BOOLEAN",
    "flag_01": "SMALLINT",
    "order_date": "DATE",
    "ship_date": "VARCHAR(10)",       # ambiguous - must not become a DATE
    "delivered_at": "TIMESTAMP",
    "customer_ref": "VARCHAR(8)",
    "status": "VARCHAR(16)",
    "big_id": "BIGINT",
    "weight_kg": "NUMERIC(3,1)",      # the tail row that a 200-row sample never sees
    "legacy_amount": "VARCHAR(8)",    # thousands separators - normalise, do not guess
    "notes": "TEXT",
}


BUCKETS = ("exact", "lossy", "unsafe", "untyped", "wide")


def grade(
    rows: Sequence[Dict[str, str]], proposed: Dict[str, ColumnType], policy: Optional[Policy] = None
) -> Dict[str, str]:
    """Bucket each proposed type against the hand-written answer key.

    - `lossy`   - *measured* on the full column: the type would corrupt or reject a real value
    - `exact`   - matches the answer key
    - `unsafe`  - a narrow type where the answer key says text: it round-trips but asserts
                  something the data never proved (an ambiguous date direction)
    - `untyped` - text where a safe narrower type existed
    - `wide`    - safe and correct in kind, just wider than necessary (BIGINT for an order id).
                  Not a defect. Counted separately so it is not confused with one.
    """

    def is_texty(kind: str) -> bool:
        return kind in ("TEXT", "VARCHAR")

    verdicts: Dict[str, str] = {}
    for col, t in proposed.items():
        raw = [r[col] for r in rows]
        expected = EXPECTED[col]
        expected_texty = expected == "TEXT" or expected.startswith("VARCHAR")
        if is_lossy(t, raw, policy):
            verdicts[col] = "lossy"
        elif t.sql("postgres") == expected:
            verdicts[col] = "exact"
        elif expected_texty and is_texty(t.kind):
            verdicts[col] = "exact"          # text is text; the length is a sizing detail
        elif expected_texty:
            verdicts[col] = "unsafe"
        elif is_texty(t.kind):
            verdicts[col] = "untyped"
        else:
            verdicts[col] = "wide"
    return verdicts


def run_benchmark(rows: Optional[Sequence[Dict[str, str]]] = None) -> Dict[str, Dict[str, str]]:
    rows = rows or demo_rows()
    cols = list(rows[0].keys())
    strategies = {
        "all-TEXT (today's loader)": {c: ColumnType("TEXT") for c in cols},
        "naive cast (200-row sample)": {c: naive_infer(c, [r[c] for r in rows]) for c in cols},
        "lossless (this tool)": {r.name: r.type for r in infer_table(rows)},
    }
    return {name: grade(rows, proposed) for name, proposed in strategies.items()}


if __name__ == "__main__":
    rows = demo_rows()
    results = infer_table(rows)
    print(emit_ddl("orders", results))
    print()
    for f in all_findings(results):
        print(f)
    print()
    for name, verdicts in run_benchmark(rows).items():
        counts = {b: sum(1 for v in verdicts.values() if v == b) for b in BUCKETS}
        print(f"{name:32} {counts}")
