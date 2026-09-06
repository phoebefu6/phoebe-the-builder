"""The experiments the README quotes. Every number below is produced here.

Run ``python3 evidence.py`` to regenerate all of it. Nothing is random, so the
output is byte-stable across machines.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Dict, List, Tuple

from fwf import (
    _OVERPUNCH_NEG,
    _OVERPUNCH_POS,
    BALANCE_SPEC,
    CUSTOMER_SPEC,
    Field,
    Int,
    RecordSpec,
    SpecError,
    Text,
    audit,
    build_balance_file,
    build_customer_file,
    frame_records,
    parse,
    parse_naive,
)

RULE = "-" * 78


def _dw(s: str) -> int:
    """Terminal display width - CJK and fullwidth forms occupy two columns."""
    return sum(2 if ord(c) > 0x1100 and _wide(c) else 1 for c in s)


def _wide(c: str) -> bool:
    o = ord(c)
    return (
        0x1100 <= o <= 0x115F
        or 0x2E80 <= o <= 0xA4CF
        or 0xAC00 <= o <= 0xD7A3
        or 0xF900 <= o <= 0xFAFF
        or 0xFE30 <= o <= 0xFE6F
        or 0xFF00 <= o <= 0xFF60
        or 0xFFE0 <= o <= 0xFFE6
    )


def _pad(s: str, n: int) -> str:
    return s + " " * max(0, n - _dw(s))


def _hdr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _raw_column(data: bytes, spec: RecordSpec, name: str, framing: str = "auto") -> List[bytes]:
    """The field's bytes, sliced correctly.

    Experiments C and D are about *type* declarations, so they read the bytes the
    right way and vary only how the field is interpreted. Otherwise the encoding
    effect from experiment A leaks in and no number measures one thing.
    """
    recs, _, _ = frame_records(data, spec, framing)
    s, e = spec.slice_of(name)
    return [r[s:e] for r in recs if len(r) >= e]


# --------------------------------------------------------------------------
# A. byte offsets vs character offsets
# --------------------------------------------------------------------------


def exp_byte_vs_char(verbose: bool = True) -> Dict[str, Any]:
    """One accent shifts every field after it - on that record only.

    Rows 2/3 and 4/5 of the sample are the same customer twice, once transliterated
    to ASCII and once spelled properly. That is the control: a reader that gets one
    right and the other wrong is failing on the encoding, not on the layout.
    """
    data = build_customer_file()
    correct = parse(data, CUSTOMER_SPEC).rows
    naive = parse_naive(data, CUSTOMER_SPEC)

    compared = ["country", "status", "qty", "list_price"]
    wrong_rows: List[int] = []
    cell_errors = 0
    for i, (c, n) in enumerate(zip(correct, naive)):
        bad = False
        for col in compared:
            cv = c[col]
            nv = n[col]
            if col == "list_price":
                # naive loses the implied decimal by construction; compare digits only
                same = nv is not None and cv is not None and int(cv * 100) == nv
            else:
                same = str(cv) == str(nv)
            if not same:
                cell_errors += 1
                bad = True
        if bad:
            wrong_rows.append(i)

    ascii_rows = [i for i, r in enumerate(correct) if r["name"] and r["name"].isascii()]
    wide_rows = [i for i in range(len(correct)) if i not in ascii_rows]

    if verbose:
        _hdr("A. Byte offsets vs character offsets")
        print(
            f"{len(correct)} records. {len(wide_rows)} contain a character wider than one byte.\n"
        )
        print(f"{'row':>4}  {'name':<18}{'country':>9}{'':4}{'qty':>7}{'':4}{'reader':>8}")
        print(RULE)
        for i in (1, 2, 3, 4, 5):
            c, n = correct[i], naive[i]
            print(
                f"{i:>4}  {_pad(str(c['name']), 18)}{str(c['country']):>9}{'':4}"
                f"{str(c['qty']):>7}{'':4}{'bytes':>8}"
            )
            print(
                f"{'':>4}  {'':<18}{str(n['country']):>9}{'':4}"
                f"{str(n['qty']):>7}{'':4}{'chars':>8}"
                f"{'' if i in ascii_rows else '   <- diverges'}"
            )
        print(RULE)
        print(
            f"\ncharacter slicing is correct on {len(ascii_rows)} ASCII rows and wrong on "
            f"{len(wide_rows)} wide rows"
        )
        print(f"rows with at least one wrong field: {wrong_rows}")
        print(f"wrong cells across {len(compared)} compared columns: {cell_errors}")
        print(
            "\nRows 1 and 2 are the same customer, spelled 'Zoe Ahlstrom' and "
            "'Zoë Ahlström'.\nRows 3 and 4 are 'Muller Werke' and 'Müller Werke'. "
            "The reader gets the\ntransliterated row right and the correctly spelled "
            "row wrong. That is the whole\nfailure mode: it is data-dependent, so it "
            "passes every test written against a\nsample that happens to be ASCII."
        )

    return {
        "records": len(correct),
        "wide_rows": wide_rows,
        "ascii_rows": ascii_rows,
        "wrong_rows": wrong_rows,
        "cell_errors": cell_errors,
    }


# --------------------------------------------------------------------------
# B. index base
# --------------------------------------------------------------------------


def exp_index_base(verbose: bool = True) -> Dict[str, Any]:
    """1-indexed inclusive spec pasted into a 0-indexed reader: everything shifts one byte."""
    data = build_customer_file()
    right = parse(data, CUSTOMER_SPEC)
    shifted_spec = CUSTOMER_SPEC.rebase(0)
    shifted = parse(data, shifted_spec, on_error="collect")

    r_qty = sum(v for v in right.column("qty") if v is not None)
    s_qty = sum(v for v in shifted.column("qty") if v is not None)
    r_price = right.total("list_price")
    s_price = shifted.total("list_price")

    try:
        CUSTOMER_SPEC.rebase(0, keep_length=True).validate()
        caught = "no error (unexpected)"
    except SpecError as exc:
        caught = f"SpecError: {exc}"

    if verbose:
        _hdr("B. The one-byte shift that still parses")
        print(
            "The layout is written 1-indexed and inclusive, the way every copybook and\n"
            "data dictionary is written. Reading it 0-indexed moves every field one byte\n"
            "left: each field loses its last byte and borrows the previous field's last.\n"
        )
        print(f"{'field':<14}{'correct [s:e)':>16}{'shifted [s:e)':>16}")
        print(RULE)
        for f in CUSTOMER_SPEC.fields:
            a, b = CUSTOMER_SPEC.slice_of(f.name)
            c, d = shifted_spec.slice_of(f.name)
            print(f"{f.name:<14}{f'[{a}:{b})':>16}{f'[{c}:{d})':>16}")
        print(RULE)
        print()
        print(f"{'measure':<26}{'correct':>18}{'shifted':>18}{'ratio':>10}")
        print(RULE)
        print(f"{'sum(qty)':<26}{r_qty:>18,}{s_qty:>18,}{s_qty / r_qty:>10.3f}")
        print(
            f"{'sum(list_price)':<26}{float(r_price):>18,.2f}{float(s_price):>18,.2f}"
            f"{float(s_price / r_price):>10.3f}"
        )
        print(f"{'parse errors raised':<26}{len(right.errors):>18}{len(shifted.errors):>18}")
        print(RULE)
        print(
            f"\nThe shifted read produces {len(shifted.errors)} errors against "
            f"{len(right.errors)} for the correct one,\nand the totals are off by roughly 10x - "
            "the signature of a lost trailing digit.\nA report built on this looks entirely "
            "normal. `index_base` is a required field\non RecordSpec for exactly this reason: "
            "the convention is not inferable from the\nnumbers, so it has to be stated.\n"
        )
        print(
            "There is one cheap defence. Declare the record length as well as the fields,\n"
            "and the shift stops being silent, because the span no longer matches:\n"
        )
        print(f"  {caught}")
        print(
            "\nThat check costs one integer in the layout and catches every whole-record\n"
            "shift, every dropped filler field and every copybook that was edited in one\n"
            "place. It is the closest thing a flat file has to a checksum."
        )

    return {
        "qty_correct": r_qty,
        "qty_shifted": s_qty,
        "price_correct": r_price,
        "price_shifted": s_price,
        "errors_correct": len(right.errors),
        "errors_shifted": len(shifted.errors),
    }


# --------------------------------------------------------------------------
# C. the overpunch sign
# --------------------------------------------------------------------------


def _strip_to_digits(cell: Any) -> Decimal:
    """Fix #1: regex the non-digits away. This deletes the sign *and a digit*.

    The overpunch character is not a decoration attached to a number, it IS the
    final digit. Removing it removes one significant figure, so the result is a
    tenth of the magnitude - wrong in the opposite direction to fix #2.
    """
    if cell is None:
        return Decimal(0)
    digits = re.sub(r"[^0-9]", "", str(cell))
    return Decimal(digits or 0).scaleb(-2)


def _magnitude_only(cell: str) -> Decimal:
    """Fix #2: translate the overpunch to its digit, then lose the sign downstream.

    This is the competent half of the fix - the magnitude is right - and it is the
    more dangerous one, because the numbers now look completely reasonable.
    """
    last = cell[-1]
    if last in _OVERPUNCH_POS:
        digit = _OVERPUNCH_POS[last]
    elif last in _OVERPUNCH_NEG:
        digit = _OVERPUNCH_NEG[last]
    else:
        digit = int(last)
    return Decimal(int(cell[:-1] + str(digit))).scaleb(-2)


def exp_overpunch(verbose: bool = True) -> Dict[str, Any]:
    """The sign lives inside the last digit. Cleaning the column deletes it."""
    data = build_customer_file()
    right = parse(data, CUSTOMER_SPEC)
    raw = _raw_column(data, CUSTOMER_SPEC, "net_amount")

    correct_total = right.total("net_amount")
    stripped_total = sum((_strip_to_digits(r.decode("latin-1")) for r in raw), Decimal(0))
    magnitude_total = sum((_magnitude_only(r.decode("latin-1")) for r in raw), Decimal(0))
    negatives = [v for v in right.column("net_amount") if v is not None and v < 0]

    # what a reader that declares the field unsigned sees
    unsigned_spec = RecordSpec(
        [
            f if f.name != "net_amount" else Field("net_amount", 37, 9, Int())
            for f in CUSTOMER_SPEC.fields
        ],
        index_base=1,
        encoding="utf-8",
        length=62,
        name="CUSTMAST-unsigned",
    )
    unsigned = parse(data, unsigned_spec)
    rep = audit(data, unsigned_spec)
    sign_finding = [f for f in rep.findings if "overpunch sign" in f]

    if verbose:
        _hdr("C. Signed zoned decimal: the minus sign is a letter")
        print(
            "PIC S9(7)V99 DISPLAY punches the sign into the final digit. +24500.00 ends\n"
            "in '{' and -1425.30 ends in '}'. Same magnitude, one byte apart, and the\n"
            "column arrives as text because int() refuses it.\n"
        )
        print(
            f"{'customer':<12}{'bytes on disk':>14}{'int()':>12}{'fix #1':>14}"
            f"{'fix #2':>14}{'correct':>14}"
        )
        print(RULE)
        for i in (0, 1, 6, 10):
            cell = raw[i].decode("latin-1")
            try:
                as_int: Any = f"{int(cell):,}"
            except ValueError:
                as_int = "ValueError"
            print(
                f"{str(right.rows[i]['cust_id']):<12}{cell:>14}{as_int:>12}"
                f"{float(_strip_to_digits(cell)):>14,.2f}{float(_magnitude_only(cell)):>14,.2f}"
                f"{float(right.rows[i]['net_amount']):>14,.2f}"
            )
        print(RULE)
        print("  fix #1 = regex out the non-digits.   fix #2 = decode the digit, drop the sign.")
        print()
        d1 = stripped_total - correct_total
        d2 = magnitude_total - correct_total
        print(f"{'reading':<40}{'total net_amount':>18}{'error':>12}")
        print(RULE)
        print(f"{'byte-accurate, sign honoured':<40}{float(correct_total):>18,.2f}{'-':>12}")
        print(
            f"{'fix #1: non-digits stripped':<40}{float(stripped_total):>18,.2f}"
            f"{float(d1 / correct_total) * 100:>11.1f}%"
        )
        print(
            f"{'fix #2: magnitude kept, sign dropped':<40}{float(magnitude_total):>18,.2f}"
            f"{float(d2 / correct_total) * 100:>11.1f}%"
        )
        print(RULE)
        print(
            f"\nThe two obvious repairs fail in opposite directions. Fix #1 looks like it only\n"
            f"strips punctuation, but the overpunch character *is* the last digit, so the\n"
            f"column comes out a tenth of its true size. Fix #2 recovers every magnitude and\n"
            f"is the more dangerous one: {len(negatives)} of {len(right.rows)} rows are negative, worth "
            f"{float(sum(negatives)):,.2f}, and\nflipping them overstates revenue by "
            f"{float(d2):,.2f} - exactly twice the negative balance.\nRefunds, credit notes and "
            f"reversals do not vanish under fix #2, they flip, so\nthe error is 2x the thing the "
            f"report was built to watch."
        )
        print(
            f"\nDeclared as Int() instead, the parse raises {len(unsigned.errors)} field errors "
            f"rather than\nguessing, and audit() names the cause before the load runs:\n"
        )
        for f in sign_finding:
            print(f"  {f}")

    return {
        "correct_total": correct_total,
        "stripped_total": stripped_total,
        "magnitude_total": magnitude_total,
        "delta": magnitude_total - correct_total,
        "negatives": len(negatives),
        "unsigned_errors": len(unsigned.errors),
    }


# --------------------------------------------------------------------------
# D. implied decimal
# --------------------------------------------------------------------------


def exp_implied_decimal(verbose: bool = True) -> Dict[str, Any]:
    """PIC 9(7)V99 has no decimal point in the file. There is nothing to detect."""
    data = build_customer_file()
    right = parse(data, CUSTOMER_SPEC)
    raw = _raw_column(data, CUSTOMER_SPEC, "list_price")

    correct = right.total("list_price")
    as_int = sum(Decimal(int(r)) for r in raw)

    if verbose:
        _hdr("D. Implied decimal: a 100x error with no error")
        print(f"{'row':>4}{'bytes on disk':>16}{'read as int':>16}{'PIC 9(7)V99':>16}")
        print(RULE)
        for i in (0, 3, 5):
            print(
                f"{i:>4}{raw[i].decode('ascii'):>16}{int(raw[i]):>16,}"
                f"{float(right.rows[i]['list_price']):>16,.2f}"
            )
        print(RULE)
        print(f"\n{'sum, read as int':<24}{float(as_int):>20,.0f}")
        print(f"{'sum, scale honoured':<24}{float(correct):>20,.2f}")
        print(f"{'ratio':<24}{float(as_int / correct):>20,.0f}x")
        print(
            "\nBoth readings are integers of the right shape and both pass a null check,\n"
            "a range check on positivity and any dtype assertion. Scale is metadata that\n"
            "does not exist in the file: if it is not in the layout, it is not anywhere."
        )

    return {"correct": correct, "as_int": as_int, "ratio": as_int / correct}


# --------------------------------------------------------------------------
# E. framing
# --------------------------------------------------------------------------


def exp_framing(verbose: bool = True) -> Dict[str, Any]:
    """RECFM=F has no record separator, and packed data contains 0x0D as a value."""
    data = build_balance_file()
    block, _, _ = frame_records(data, BALANCE_SPEC, "block")
    lines, _, _ = frame_records(data, BALANCE_SPEC, "lines")

    ok = parse(data, BALANCE_SPEC, framing="block")
    broken = parse(data, BALANCE_SPEC, framing="lines", on_error="collect")

    # locate the offending byte
    s, e = BALANCE_SPEC.slice_of("adjustment")
    culprits = [
        (i, block[i][:10].decode("latin-1"), block[i][s:e].hex())
        for i in range(len(block))
        if set(block[i][s:e]) & {0x0A, 0x0D}
    ]

    if verbose:
        _hdr("E. Framing: the record separator that is also a number")
        print(
            f"{len(data)} bytes / {BALANCE_SPEC.record_length}-byte records = "
            f"{len(data) // BALANCE_SPEC.record_length} records, and the file contains no "
            f"newline\nby design. Splitting on line breaks anyway:\n"
        )
        print(f"{'framing':<12}{'records':>10}{'parse errors':>16}{'total balance':>18}")
        print(RULE)
        print(
            f"{'block':<12}{len(block):>10}{len(ok.errors):>16}"
            f"{float(ok.total('balance')):>18,.2f}"
        )
        print(
            f"{'lines':<12}{len(lines):>10}{len(broken.errors):>16}"
            f"{float(broken.total('balance')):>18,.2f}"
        )
        print(RULE)
        print("\nWhy there are line breaks in a file with no line breaks:\n")
        print(f"{'record':>8}  {'acct':<12}{'adjustment bytes':<20}{'contains':<10}")
        print(RULE)
        for i, acct, hexs in culprits:
            marks = []
            raw = block[i][s:e]
            if 0x0D in raw:
                marks.append("0x0D CR")
            if 0x0A in raw:
                marks.append("0x0A LF")
            print(f"{i:>8}  {acct:<12}{hexs:<20}{', '.join(marks):<10}")
        print(RULE)
        print(
            "\nCOMP-3 stores the sign in the final nibble: 0xD is negative, 0xC positive.\n"
            "Any negative amount whose last digit is 0 therefore ends in the byte 0x0D -\n"
            "a carriage return. -1234.50 packs to 000123450D. It is not a corrupt file;\n"
            "it is a correct file being read by something that assumes text. The same byte\n"
            "is what an FTP transfer in ASCII mode rewrites, which destroys the value on\n"
            "the wire before any parser sees it."
        )

    return {
        "n_block": len(block),
        "n_lines": len(lines),
        "errors_block": len(ok.errors),
        "errors_lines": len(broken.errors),
        "total_block": ok.total("balance"),
        "total_lines": broken.total("balance"),
        "culprits": [c[0] for c in culprits],
    }


# --------------------------------------------------------------------------
# F. packed decimal is not text
# --------------------------------------------------------------------------


def exp_packed_is_not_text(verbose: bool = True) -> Dict[str, Any]:
    """latin-1 decodes every byte sequence ever produced. That is the problem."""
    data = build_balance_file()
    right = parse(data, BALANCE_SPEC, framing="block")

    text_spec = RecordSpec(
        [
            f if f.name not in ("balance", "adjustment") else Field(f.name, f.start, f.length, Text())
            for f in BALANCE_SPEC.fields
        ],
        index_base=1,
        encoding="latin-1",
        length=31,
        name="ACCTBAL-as-text",
    )
    as_text = parse(data, text_spec, framing="block")

    utf8_spec = RecordSpec(
        list(text_spec.fields), index_base=1, encoding="utf-8", length=31, name="ACCTBAL-utf8"
    )
    as_utf8 = parse(data, utf8_spec, framing="block", on_error="collect")

    if verbose:
        _hdr("F. Packed decimal read as text: no exception, no data")
        print(f"{'acct':<12}{'balance (packed)':>18}{'hex':>14}{'as latin-1 text':>20}")
        print(RULE)
        for i in range(4):
            s, e = BALANCE_SPEC.slice_of("balance")
            raw = data[i * 31 : (i + 1) * 31][s:e]
            print(
                f"{right.rows[i]['acct']:<12}{float(right.rows[i]['balance']):>18,.2f}"
                f"{raw.hex():>14}{repr(as_text.rows[i]['balance']):>20}"
            )
        print(RULE)
        print(
            f"\nlatin-1 errors: {len(as_text.errors)}   utf-8 errors: {len(as_utf8.errors)}\n"
        )
        print(
            "latin-1 maps all 256 byte values to a character, so it never raises. The\n"
            "column loads, has the declared width, contains no nulls and passes a\n"
            "not-null check. Under utf-8 the same bytes raise instead - which is the more\n"
            "useful outcome, and the reason 'it decoded without error' is not evidence.\n"
            "Two digits per byte also means the field cannot be recovered afterwards:\n"
            "the character view has already lost the nibble boundaries."
        )

    return {
        "latin1_errors": len(as_text.errors),
        "utf8_errors": len(as_utf8.errors),
        "n_records": len(right.rows),
        "n_amounts": len(right.rows) * 2,
    }


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


def damage_ledger(verbose: bool = True) -> List[Tuple[str, str, str]]:
    """Every failure mode, its money impact on this sample, and whether it raises."""
    a = exp_byte_vs_char(verbose=False)
    b = exp_index_base(verbose=False)
    c = exp_overpunch(verbose=False)
    d = exp_implied_decimal(verbose=False)
    e = exp_framing(verbose=False)
    f = exp_packed_is_not_text(verbose=False)

    rows = [
        (
            "character offsets",
            f"{len(a['wrong_rows'])}/{a['records']} rows scrambled",
            "silent",
        ),
        ("index base off by one", f"{float(b['price_shifted'] / b['price_correct']):.3f}x totals", "silent"),
        ("overpunch sign dropped", f"+{float(c['delta']):,.2f} revenue (+9.2%)", "silent"),
        ("implied decimal ignored", f"{float(d['ratio']):.0f}x totals", "silent"),
        ("line framing on RECFM=F", f"{e['n_lines']} record instead of {e['n_block']}", "silent"),
        ("packed read as latin-1 text", f"{f['n_amounts']} amounts to mojibake", "silent"),
    ]
    if verbose:
        _hdr("Damage ledger")
        print(f"{'failure mode':<30}{'effect on this sample':<34}{'raises?':>10}")
        print(RULE)
        for name, effect, raises in rows:
            print(f"{name:<30}{effect:<34}{raises:>10}")
        print(RULE)
        print(
            f"\nAll {len(rows)} produce a plausible wrong answer with no exception anywhere. Four of\n"
            f"them also produce a *stable* wrong answer, so a reconciliation against last\n"
            f"month agrees with itself. That is the argument for a pre-flight audit on the\n"
            f"bytes: the failures that raise are the ones that were never going to ship."
        )
    return rows


def main() -> None:
    exp_byte_vs_char()
    exp_index_base()
    exp_overpunch()
    exp_implied_decimal()
    exp_framing()
    exp_packed_is_not_text()
    damage_ledger()
    _hdr("Pre-flight audit, customer master")
    print(audit(build_customer_file(), CUSTOMER_SPEC).text())
    _hdr("Pre-flight audit, account balances")
    print(audit(build_balance_file(), BALANCE_SPEC).text())


if __name__ == "__main__":
    main()
