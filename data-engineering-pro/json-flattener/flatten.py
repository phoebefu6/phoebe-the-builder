from __future__ import annotations

# JSON Flattener - core logic.
#
# Every API response and event payload is nested. Every warehouse table is flat.
# The gap between them gets closed by hand-written, one-off unnesting code that
# breaks the first time a field is missing or an array changes length.
#
# This module turns nested JSON into tabular rows with dot-path columns, and
# takes an explicit position on the two decisions that actually matter:
#
#   1. Arrays: EXPLODE into rows (fact-table shape) or INDEX into columns
#      (arr.0.x, arr.1.x). Wrong choice here is the usual cause of either
#      duplicated measures or an ever-widening table.
#   2. Missing keys: records are ragged, so the column set is the UNION across
#      all records, with nulls where a record didn't have the path.
#
# It also emits a type-inferred schema (Spark/SQL DDL) so the flattened shape
# can be landed rather than just looked at. Fully offline, stdlib only + pandas.
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SEP = "."

# Values that are containers we recurse into vs leaves we record.
_SCALARS = (str, int, float, bool, type(None))


@dataclass
class FlattenStats:
    """What the flattener did - the audit trail for a pipeline step."""

    input_records: int = 0
    output_rows: int = 0
    columns: int = 0
    max_depth: int = 0
    exploded_paths: List[str] = field(default_factory=list)
    ragged_paths: Dict[str, int] = field(default_factory=dict)  # path -> records missing it
    type_conflicts: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def row_multiplier(self) -> float:
        return round(self.output_rows / self.input_records, 2) if self.input_records else 0.0

    @property
    def fanout_warning(self) -> Optional[str]:
        """Set when more than one array level exploded - measures are duplicated.

        Two exploded paths where one is a prefix of the other means a cross
        join happened: the parent's scalar columns repeat once per child element,
        so SUM() over them overcounts. This is the single most common silent bug
        in hand-rolled unnesting code.
        """
        nested = [
            (a, b) for a in self.exploded_paths for b in self.exploded_paths
            if a != b and b.startswith(a + SEP)
        ]
        if not nested:
            return None
        a, b = nested[0]
        return (
            f"'{b}' exploded inside '{a}' - every scalar on '{a}' now repeats once "
            f"per '{b}' element, so SUM() over it double-counts. "
            f"Pass explode_paths=['{a}'] to pin the grain."
        )


def _is_scalar(v: Any) -> bool:
    return isinstance(v, _SCALARS)


def _depth(obj: Any, current: int = 0) -> int:
    if isinstance(obj, dict):
        return max([_depth(v, current + 1) for v in obj.values()] or [current])
    if isinstance(obj, list):
        return max([_depth(v, current + 1) for v in obj] or [current])
    return current


def flatten_record(
    record: Dict[str, Any],
    array_mode: str = "explode",
    sep: str = SEP,
    max_array_cols: int = 20,
    explode_paths: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Flatten one nested record into one or more flat rows.

    array_mode:
      "explode" - each array element becomes its own row (fact-table shape).
        Nested arrays multiply, which is exactly what a cross join does; the
        multiplier is reported in stats so it can't happen silently.
      "index"   - each element becomes its own column (items.0.sku, items.1.sku).
        Keeps one row per record but widens with the longest array, so it is
        capped by max_array_cols.
      "json"    - keep the array as a JSON string in one column. The right call
                  when the array is a payload, not a set of facts.

    explode_paths:
      Only meaningful with array_mode="explode". Explode ONLY these paths and
      JSON-encode every other array. This is the fix for the trap the tool
      exposes: exploding a nested array (items.tags) multiplies the parent's
      rows, so summing items.price afterwards double-counts revenue. Naming the
      grain explicitly - explode_paths=["items"] - keeps one row per item.
      None means explode every array.
    """
    if array_mode not in ("explode", "index", "json"):
        raise ValueError("array_mode must be 'explode', 'index' or 'json'")
    allowed = None if explode_paths is None else set(explode_paths)

    rows: List[Dict[str, Any]] = [{}]

    def assign(rows_in: List[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
        for r in rows_in:
            r[key] = value
        return rows_in

    def walk(obj: Any, prefix: str, rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if isinstance(obj, dict):
            # An empty dict is a leaf, not a no-op: recording it keeps the column
            # present so downstream schemas stay stable across batches.
            if not obj:
                return assign(rows_in, prefix or "_root", None)
            out = rows_in
            for k, v in obj.items():
                path = f"{prefix}{sep}{k}" if prefix else str(k)
                out = walk(v, path, out)
            return out

        if isinstance(obj, list):
            if not obj:
                # Empty array: keep the record as a row with nulls in the array's
                # columns (explode_outer semantics) rather than dropping it. No
                # scalar column is created for `prefix` - that would collide with
                # the `prefix.child` columns other records produce and leave a
                # half-typed wart in the DDL.
                return assign(rows_in, prefix, None) if array_mode == "json" else rows_in
            # An array not named in explode_paths is kept whole, so it cannot
            # silently multiply the rows of the grain the caller asked for.
            keep_whole = array_mode == "json" or (
                array_mode == "explode" and allowed is not None and prefix not in allowed
            )
            if keep_whole:
                return assign(rows_in, prefix, json.dumps(obj))
            if array_mode == "index":
                out = rows_in
                for i, item in enumerate(obj[:max_array_cols]):
                    out = walk(item, f"{prefix}{sep}{i}", out)
                if len(obj) > max_array_cols:
                    out = assign(out, f"{prefix}{sep}_truncated_count", len(obj) - max_array_cols)
                return out
            # explode: cross the current rows with the array elements
            exploded: List[Dict[str, Any]] = []
            for item in obj:
                branch = walk(item, prefix, [dict(r) for r in rows_in])
                exploded.extend(branch)
            return exploded

        return assign(rows_in, prefix, obj)

    return walk(record, "", rows)


def flatten(
    records: Sequence[Dict[str, Any]],
    array_mode: str = "explode",
    sep: str = SEP,
    max_array_cols: int = 20,
    explode_paths: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], FlattenStats]:
    """Flatten many records into a rectangle, unioning columns across records."""
    stats = FlattenStats(input_records=len(records))
    all_rows: List[Dict[str, Any]] = []
    per_record_paths: List[set] = []

    for rec in records:
        stats.max_depth = max(stats.max_depth, _depth(rec))
        rows = flatten_record(rec, array_mode, sep, max_array_cols, explode_paths)
        if array_mode == "explode":
            for path, length in _array_paths(rec, sep):
                if explode_paths is not None and path not in set(explode_paths):
                    continue
                if length > 1 and path not in stats.exploded_paths:
                    stats.exploded_paths.append(path)
        all_rows.extend(rows)
        paths = set()
        for r in rows:
            paths |= set(r.keys())
        per_record_paths.append(paths)

    # Column set is the UNION across records - ragged input is the normal case,
    # not an error. Records missing a path get null, and the count of how many
    # were missing is reported so a silently-optional field is visible.
    union: List[str] = []
    seen = set()
    for r in all_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                union.append(k)

    for path in union:
        missing = sum(1 for paths in per_record_paths if path not in paths)
        if missing:
            stats.ragged_paths[path] = missing

    normalised = [{k: r.get(k) for k in union} for r in all_rows]
    stats.output_rows = len(normalised)
    stats.columns = len(union)
    stats.type_conflicts = _type_conflicts(normalised)
    return normalised, stats


def _array_paths(obj: Any, sep: str = SEP, prefix: str = "") -> Iterable[Tuple[str, int]]:
    """Yield (path, length) for every array in a record."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}{sep}{k}" if prefix else str(k)
            yield from _array_paths(v, sep, path)
    elif isinstance(obj, list):
        yield (prefix, len(obj))
        for item in obj:
            yield from _array_paths(item, sep, prefix)


def _type_conflicts(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Paths where different records disagree on the scalar type.

    This is the failure that turns up at load time, not flatten time - a column
    that is int in 9,000 records and str in 3 will reject the whole batch.
    """
    kinds: Dict[str, set] = {}
    for r in rows:
        for k, v in r.items():
            if v is None:
                continue
            kinds.setdefault(k, set()).add(type(v).__name__)
    return {k: sorted(v) for k, v in kinds.items() if len(v) > 1}


# --------------------------------------------------------------------------
# Schema inference - so the flat shape can be landed, not just viewed
# --------------------------------------------------------------------------

_SQL_TYPE = {
    "bool": "BOOLEAN",
    "int": "BIGINT",
    "float": "DOUBLE",
    "str": "VARCHAR",
    "NoneType": "VARCHAR",
}


def infer_schema(rows: List[Dict[str, Any]]) -> List[Dict[str, object]]:
    """Per-column type, nullability and fill rate."""
    if not rows:
        return []
    cols = list(rows[0].keys())
    out = []
    for c in cols:
        vals = [r.get(c) for r in rows]
        present = [v for v in vals if v is not None]
        types = Counter(type(v).__name__ for v in present)
        # Mixed int/float is not a conflict, it is a widening - a column with
        # both is a DOUBLE, and calling that an error would fail valid data.
        names = set(types)
        if names <= {"int", "float"} and names:
            py = "float" if "float" in names else "int"
        elif len(names) > 1:
            py = "str"  # genuine conflict: widen to text rather than reject the batch
        else:
            py = next(iter(names), "NoneType")
        out.append({
            "column": c,
            "python_type": py,
            "sql_type": _SQL_TYPE.get(py, "VARCHAR"),
            "nullable": len(present) < len(vals),
            "fill_rate": round(len(present) / len(vals), 3),
            "mixed_types": ",".join(sorted(types)) if len(types) > 1 else "",
        })
    return out


def to_ddl(rows: List[Dict[str, Any]], table: str = "flattened") -> str:
    """CREATE TABLE for the flattened shape, with dot paths quoted."""
    schema = infer_schema(rows)
    if not schema:
        return f"-- no rows to infer {table} from"
    lines = [f"CREATE TABLE {table} ("]
    for i, s in enumerate(schema):
        comma = "," if i < len(schema) - 1 else ""
        null = "" if s["nullable"] else " NOT NULL"
        note = f"  -- mixed: {s['mixed_types']}" if s["mixed_types"] else ""
        lines.append(f'  "{s["column"]}" {s["sql_type"]}{null}{comma}{note}')
    lines.append(");")
    return "\n".join(lines)


def to_dataframe(rows: List[Dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sample payload - deliberately ragged, with a type conflict and nested arrays
# --------------------------------------------------------------------------

SAMPLE_ORDERS: List[Dict[str, Any]] = [
    {
        "order_id": "A-1001",
        "placed_at": "2026-07-28T09:12:00Z",
        "customer": {"id": 552, "name": "Nadia", "address": {"city": "Singapore", "zip": "018956"}},
        "items": [
            {"sku": "KB-01", "qty": 1, "price": 89.0, "tags": ["input", "wireless"]},
            {"sku": "MS-07", "qty": 2, "price": 25.5, "tags": ["input"]},
        ],
        "total": 140.0,
        "coupon": None,
    },
    {
        # ragged: no address.zip, no coupon key at all, extra channel field
        "order_id": "A-1002",
        "placed_at": "2026-07-28T10:03:00Z",
        "customer": {"id": 771, "name": "Wei", "address": {"city": "Kuala Lumpur"}},
        "items": [{"sku": "MN-27", "qty": 1, "price": 310.0, "tags": []}],
        "total": 310.0,
        "channel": "mobile",
    },
    {
        # type conflict: total arrives as a string from a different producer
        "order_id": "A-1003",
        "placed_at": "2026-07-29T14:40:00Z",
        "customer": {"id": 552, "name": "Nadia", "address": {"city": "Singapore", "zip": "018956"}},
        "items": [],
        "total": "0.00",
        "coupon": "WELCOME10",
    },
]


def main() -> None:
    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("=" * 84)
    print("INPUT: 3 nested order records (ragged, one type conflict, nested arrays)")
    print("=" * 84)
    print(json.dumps(SAMPLE_ORDERS[0], indent=2)[:420] + "\n  ...")

    for mode in ("explode", "index", "json"):
        rows, stats = flatten(SAMPLE_ORDERS, array_mode=mode)
        print("\n" + "-" * 84)
        print(f"array_mode = {mode!r}")
        print(f"  {stats.input_records} records -> {stats.output_rows} rows "
              f"({stats.row_multiplier}x), {stats.columns} columns, depth {stats.max_depth}")
        if stats.exploded_paths:
            print(f"  exploded: {', '.join(stats.exploded_paths)}")
        df = to_dataframe(rows)
        cols = [c for c in df.columns if c.startswith(("order_id", "items", "customer.address"))]
        print(df[cols[:6]].to_string(index=False))

    print("\n" + "-" * 84)
    print("THE FAN-OUT TRAP - does SUM(items.price) survive the flatten?")
    truth = sum(
        it["price"] * it["qty"] for o in SAMPLE_ORDERS for it in o.get("items", [])
    )
    for label, kwargs in (
        ("explode everything", {}),
        ("explode_paths=['items']", {"explode_paths": ["items"]}),
    ):
        rows_x, st = flatten(SAMPLE_ORDERS, array_mode="explode", **kwargs)
        got = sum(
            (r["items.price"] or 0) * (r["items.qty"] or 0) for r in rows_x
        )
        flag = "OK" if abs(got - truth) < 1e-9 else f"WRONG (+{got - truth:.2f})"
        print(f"  {label:<26} {st.output_rows} rows  revenue {got:>8.2f}  vs truth "
              f"{truth:.2f}  -> {flag}")
        if st.fanout_warning:
            print(f"       warning: {st.fanout_warning}")

    rows, stats = flatten(SAMPLE_ORDERS, array_mode="explode", explode_paths=["items"])
    print("\n" + "-" * 84)
    print("RAGGED PATHS (records missing the path, filled with null)")
    for p, n in sorted(stats.ragged_paths.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {p:<34} missing in {n} of {stats.output_rows} rows")

    print("\nTYPE CONFLICTS (would reject the batch at load time)")
    for p, kinds in stats.type_conflicts.items():
        print(f"  {p:<34} {' + '.join(kinds)}")
    if not stats.type_conflicts:
        print("  none")

    print("\n" + "-" * 84)
    print("INFERRED DDL")
    print(to_ddl(rows, "orders_flat"))


if __name__ == "__main__":
    main()
