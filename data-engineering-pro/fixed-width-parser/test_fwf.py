"""Tests for fwf.py. Plain asserts, no pytest needed: ``python3 test_fwf.py``."""

from __future__ import annotations

import datetime as dt
import sys
import traceback
from decimal import Decimal
from typing import Callable, List, Tuple

from fwf import (
    BALANCE_SPEC,
    CUSTOMER_SPEC,
    Date,
    Field,
    FieldError,
    Implied,
    Int,
    Overpunch,
    Packed,
    RecordSpec,
    SpecError,
    Text,
    _overpunch,
    _packed,
    audit,
    build_balance_file,
    build_customer_file,
    frame_records,
    parse,
    parse_naive,
)

_TESTS: List[Tuple[str, Callable[[], None]]] = []


def test(fn: Callable[[], None]) -> Callable[[], None]:
    _TESTS.append((fn.__name__, fn))
    return fn


def raises(exc, fn, *a, **kw) -> Exception:
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


# --------------------------------------------------------------------------
# kinds
# --------------------------------------------------------------------------


@test
def text_strips_both_pad_bytes() -> None:
    assert Text().decode(b"abc   ", "utf-8") == "abc"
    assert Text().decode(b"abc\x00\x00", "utf-8") == "abc"
    assert Text().decode(b"      ", "utf-8") is None
    assert Text(strip=False).decode(b"ab ", "utf-8") == "ab "


@test
def text_reports_the_encoding_it_was_given() -> None:
    err = raises(FieldError, Text().decode, b"\xff\xfe", "utf-8")
    assert "utf-8" in str(err)
    # latin-1 accepts every byte, which is the point of experiment F
    assert Text().decode(b"\xff\xfe", "latin-1") == "ÿþ"


@test
def int_rejects_and_explains_overpunch() -> None:
    assert Int().decode(b"00042", "ascii") == 42
    assert Int().decode(b"     ", "ascii") is None
    err = raises(FieldError, Int().decode, b"0004J", "ascii")
    assert "Overpunch" in str(err)


@test
def implied_decimal_scales() -> None:
    assert Implied(2).decode(b"000012345", "ascii") == Decimal("123.45")
    assert Implied(0).decode(b"000012345", "ascii") == Decimal(12345)
    assert Implied(3).decode(b"1", "ascii") == Decimal("0.001")
    raises(SpecError, Implied, -1)


@test
def overpunch_round_trips_every_digit() -> None:
    for d in range(10):
        for sign in (1, -1):
            v = Decimal(sign * (1230 + d)) / 100
            enc = _overpunch(v, 9, 2)
            assert Overpunch(2).decode(enc, "ascii") == v, (v, enc)


@test
def overpunch_sign_is_one_byte_from_its_opposite() -> None:
    pos = _overpunch(Decimal("1425.30"), 9, 2)
    neg = _overpunch(Decimal("-1425.30"), 9, 2)
    assert pos[:-1] == neg[:-1]
    assert pos != neg
    assert sum(a != b for a, b in zip(pos, neg)) == 1


@test
def overpunch_accepts_an_unpunched_positive() -> None:
    assert Overpunch(2).decode(b"000142530", "ascii") == Decimal("1425.30")


@test
def overpunch_leading_sign_variant() -> None:
    # the sign rides the FIRST digit, so -1425.30 in nine bytes is }00142530
    assert Overpunch(2, leading=True).decode(b"}00142530", "ascii") == Decimal("-1425.30")
    assert Overpunch(2, leading=True).decode(b"A00142530", "ascii") == Decimal("1001425.30")


@test
def overpunch_rejects_a_non_sign() -> None:
    raises(FieldError, Overpunch(2).decode, b"0001425*", "ascii")


@test
def packed_round_trips() -> None:
    for v in ("0.00", "18420.55", "-45.99", "1250000.00", "-1234.50", "9.90"):
        enc = _packed(Decimal(v), 5, 2)
        assert len(enc) == 5
        assert Packed(2).decode(enc, "ascii") == Decimal(v), v


@test
def packed_negative_ending_in_zero_emits_carriage_return() -> None:
    """The whole of experiment E rests on this byte existing."""
    enc = _packed(Decimal("-1234.50"), 5, 2)
    assert enc[-1] == 0x0D
    assert 0x0D in enc
    assert Packed(2).decode(enc, "ascii") == Decimal("-1234.50")


@test
def packed_rejects_a_non_digit_nibble() -> None:
    raises(FieldError, Packed(2).decode, bytes([0xAB, 0x00, 0x00, 0x00, 0x0C]), "ascii")


@test
def packed_rejects_a_missing_sign_nibble() -> None:
    raises(FieldError, Packed(2).decode, bytes([0x00, 0x00, 0x00, 0x00, 0x01]), "ascii")


@test
def packed_treats_all_zero_bytes_as_null() -> None:
    assert Packed(2).decode(b"\x00" * 5, "ascii") is None


@test
def date_handles_mainframe_nulls() -> None:
    assert Date().decode(b"20240131", "ascii") == dt.date(2024, 1, 31)
    assert Date().decode(b"00000000", "ascii") is None
    assert Date().decode(b"        ", "ascii") is None
    raises(FieldError, Date().decode, b"20240230", "ascii")


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


@test
def index_base_shifts_every_slice_by_exactly_one() -> None:
    a = CUSTOMER_SPEC
    b = a.rebase(0)
    for f in a.fields:
        s0, e0 = a.slice_of(f.name)
        s1, e1 = b.slice_of(f.name)
        assert (s1 - s0, e1 - e0) == (1, 1)


@test
def declared_length_catches_the_rebase() -> None:
    err = raises(SpecError, CUSTOMER_SPEC.rebase(0, keep_length=True).validate)
    assert "63" in str(err) and "62" in str(err)


@test
def rebase_without_length_is_silent() -> None:
    CUSTOMER_SPEC.rebase(0).validate()  # no raise - that is the failure mode


@test
def overlap_is_an_error_and_gap_is_a_note() -> None:
    overlapping = RecordSpec(
        [Field("a", 1, 5, Text()), Field("b", 4, 5, Text())], index_base=1
    )
    issues = overlapping.structural_issues()
    assert any(i.startswith("ERROR overlap") for i in issues)
    raises(SpecError, overlapping.validate)

    gapped = RecordSpec([Field("a", 1, 4, Text()), Field("b", 9, 4, Text())], index_base=1)
    issues = gapped.structural_issues()
    assert any("gap: 4 undeclared" in i for i in issues)
    gapped.validate()  # a gap is filler, not an error


@test
def negative_offset_under_the_wrong_base_is_rejected() -> None:
    raises(SpecError, RecordSpec, [Field("a", 0, 4, Text())], index_base=1)


@test
def zero_length_field_is_rejected() -> None:
    raises(SpecError, Field, "a", 1, 0, Text())


@test
def span_and_record_length_differ_when_length_is_declared() -> None:
    spec = RecordSpec([Field("a", 1, 4, Text())], index_base=1, length=10)
    assert spec.span == 4 and spec.record_length == 10


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


@test
def auto_framing_picks_block_for_packed_layouts() -> None:
    data = build_balance_file()
    recs, used, notes = frame_records(data, BALANCE_SPEC, "auto")
    assert used == "block"
    assert len(recs) == 6
    assert all(len(r) == 31 for r in recs)
    assert any("packed" in n for n in notes)


@test
def auto_framing_picks_lines_for_the_customer_file() -> None:
    recs, used, _ = frame_records(build_customer_file(), CUSTOMER_SPEC, "auto")
    assert used == "lines"
    assert len(recs) == 12


@test
def line_framing_on_a_packed_file_loses_records() -> None:
    data = build_balance_file()
    lines, _, notes = frame_records(data, BALANCE_SPEC, "lines")
    assert len(lines) < 6
    assert any(n.startswith("WARNING") for n in notes)


@test
def block_framing_flags_a_partial_trailing_record() -> None:
    data = build_balance_file() + b"XY"
    recs, _, notes = frame_records(data, BALANCE_SPEC, "block")
    assert len(recs) == 7 and len(recs[-1]) == 2
    assert any("partial record" in n for n in notes)


@test
def crlf_is_stripped_but_a_bare_cr_is_not_a_separator() -> None:
    spec = RecordSpec([Field("a", 1, 3, Text())], index_base=1, length=3)
    recs, _, _ = frame_records(b"abc\r\ndef\r\n", spec, "lines")
    assert recs == [b"abc", b"def"]


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


@test
def customer_file_parses_to_expected_values() -> None:
    r = parse(build_customer_file(), CUSTOMER_SPEC)
    assert len(r.rows) == 12
    first = r.rows[0]
    assert first["cust_id"] == "C0001001"
    assert first["name"] == "Acme Industrial"
    assert first["net_amount"] == Decimal("24500.00")
    assert first["list_price"] == Decimal("26000.00")
    assert first["open_date"] == dt.date(2021, 3, 4)


@test
def wide_characters_do_not_shift_later_fields() -> None:
    rows = parse(build_customer_file(), CUSTOMER_SPEC).rows
    cjk = rows[5]
    assert cjk["name"] == "陈晓贸易"
    assert cjk["country"] == "CN"
    assert cjk["qty"] == 900
    assert cjk["net_amount"] == Decimal("182300.00")


@test
def accented_and_ascii_twins_agree_on_every_non_name_field() -> None:
    rows = parse(build_customer_file(), CUSTOMER_SPEC).rows
    for a, b in ((1, 2), (3, 4)):
        for col in ("country", "qty", "list_price", "open_date"):
            assert rows[a][col] == rows[b][col], (a, b, col)


@test
def character_slicing_disagrees_on_exactly_the_wide_rows() -> None:
    data = build_customer_file()
    right = parse(data, CUSTOMER_SPEC).rows
    naive = parse_naive(data, CUSTOMER_SPEC)
    wide = {i for i, r in enumerate(right) if r["name"] and not r["name"].isascii()}
    disagree = {i for i in range(12) if str(right[i]["country"]) != str(naive[i]["country"])}
    assert disagree == wide, (disagree, wide)


@test
def short_record_is_reported_not_silently_nulled() -> None:
    r = parse(build_customer_file(), CUSTOMER_SPEC)
    assert r.rows[11]["open_date"] is None
    assert any(f == "<record>" and "short record" in m for _, f, m in r.errors)


@test
def a_file_without_short_records_parses_clean() -> None:
    r = parse(build_customer_file(short_record_at=None), CUSTOMER_SPEC)
    assert r.ok, r.errors


@test
def on_error_raise_names_the_record_and_field() -> None:
    data = build_customer_file()
    spec = RecordSpec(
        [f if f.name != "net_amount" else Field("net_amount", 37, 9, Int()) for f in CUSTOMER_SPEC.fields],
        index_base=1,
        length=62,
    )
    err = raises(FieldError, parse, data, spec, on_error="raise")
    assert "record 0" in str(err) and "net_amount" in str(err)


@test
def balance_file_parses_to_expected_values() -> None:
    r = parse(build_balance_file(), BALANCE_SPEC, framing="block")
    assert r.ok, r.errors
    assert r.rows[1]["adjustment"] == Decimal("-1234.50")
    assert r.rows[3]["balance"] == Decimal("1250000.00")
    assert r.total("balance") == Decimal("1346587.06")


@test
def totals_ignore_nulls() -> None:
    r = parse(build_customer_file(), CUSTOMER_SPEC)
    assert r.total("net_amount") == Decimal("390215.20")


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


@test
def audit_flags_multibyte_records_with_a_count() -> None:
    rep = audit(build_customer_file(), CUSTOMER_SPEC)
    assert rep.stats["divergent_records"] == 4
    assert any("multi-byte" in f for f in rep.findings)


@test
def audit_flags_the_short_record() -> None:
    rep = audit(build_customer_file(), CUSTOMER_SPEC)
    assert any("shorter than" in f for f in rep.findings)


@test
def audit_is_clean_on_a_well_formed_ascii_file() -> None:
    spec = RecordSpec([Field("a", 1, 3, Text()), Field("b", 4, 2, Int())], index_base=1, length=5)
    rep = audit(b"abc12\ndef34\n", spec)
    assert rep.verdict.startswith("CLEAN"), rep.text()


@test
def audit_flags_overpunch_in_an_int_field() -> None:
    spec = RecordSpec(
        [f if f.name != "net_amount" else Field("net_amount", 37, 9, Int()) for f in CUSTOMER_SPEC.fields],
        index_base=1,
        length=62,
    )
    rep = audit(build_customer_file(), spec)
    hit = [f for f in rep.findings if "overpunch sign" in f]
    assert hit and "3 of them negative" in hit[0], rep.text()


@test
def audit_flags_separator_bytes_inside_packed_fields() -> None:
    rep = audit(build_balance_file(), BALANCE_SPEC)
    hit = [f for f in rep.findings if "0x0A or 0x0D" in f]
    assert hit and "3 value(s)" in hit[0]


@test
def audit_verdict_is_fatal_on_a_broken_layout() -> None:
    spec = RecordSpec([Field("a", 1, 5, Text()), Field("b", 4, 5, Text())], index_base=1)
    rep = audit(b"abcdefghi\n", spec)
    assert rep.verdict.startswith("NOT SAFE TO LOAD")


@test
def audit_counts_undecodable_records_without_calling_them_an_error() -> None:
    spec = RecordSpec(list(BALANCE_SPEC.fields), index_base=1, encoding="utf-8", length=31)
    rep = audit(build_balance_file(), spec)
    assert rep.stats["undecodable_records"] > 0
    assert any("not decodable" in f for f in rep.findings)


# --------------------------------------------------------------------------
# claims the README makes
# --------------------------------------------------------------------------


@test
def claim_index_base_shift_is_roughly_ten_x() -> None:
    data = build_customer_file()
    right = parse(data, CUSTOMER_SPEC)
    shifted = parse(data, CUSTOMER_SPEC.rebase(0))
    ratio = sum(v for v in shifted.column("qty") if v) / sum(
        v for v in right.column("qty") if v
    )
    assert ratio == 10.0


@test
def claim_sign_flip_is_exactly_twice_the_negative_balance() -> None:
    r = parse(build_customer_file(), CUSTOMER_SPEC)
    vals = [v for v in r.column("net_amount") if v is not None]
    negatives = sum(v for v in vals if v < 0)
    assert sum(abs(v) for v in vals) - sum(vals) == -2 * negatives


@test
def claim_implied_decimal_is_exactly_one_hundred_x() -> None:
    data = build_customer_file()
    r = parse(data, CUSTOMER_SPEC)
    s, e = CUSTOMER_SPEC.slice_of("list_price")
    raw = [rec[s:e] for rec in data.split(b"\n") if len(rec) >= e]
    assert sum(int(x) for x in raw) == r.total("list_price") * 100


def main() -> int:
    failures = 0
    for name, fn in _TESTS:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
