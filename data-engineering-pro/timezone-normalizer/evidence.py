"""The experiments the README quotes. Every number below is produced here.

Run ``python3 evidence.py`` to regenerate all of it. Nothing is random. The one
thing that is not machine-independent is the tz database version, which is
printed with the results for exactly that reason.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from tznorm import (
    UTC,
    Reading,
    build_session_log,
    classify,
    etc_zone_is_inverted,
    find_alias_groups,
    ground_truth,
    local_day,
    normalize,
    resolve,
    same_rules,
    tzdata_version,
    utc_day,
)

RULE = "-" * 78


def _hdr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _sessions(rows: List[Dict[str, Any]], readings: List[Reading]) -> Dict[str, Dict[str, Reading]]:
    out: Dict[str, Dict[str, Reading]] = {}
    for row, r in zip(rows, readings):
        out.setdefault(row["session_id"], {})[row["event"]] = r
    return out


def _minutes(a: Optional[dt.datetime], b: Optional[dt.datetime]) -> Optional[float]:
    if a is None or b is None:
        return None
    return (b - a).total_seconds() / 60


# --------------------------------------------------------------------------
# A. the hour that happens twice
# --------------------------------------------------------------------------


def exp_ambiguous(verbose: bool = True) -> Dict[str, Any]:
    """Fall-back night, measured against the instants the events actually happened.

    The sample's local strings are rendered *from* true UTC instants, so this is a
    recovery error, not a comparison of two guesses.
    """
    rows = build_session_log()
    truth = ground_truth()
    results: Dict[str, Dict[str, Any]] = {}

    for policy in ("earlier", "later"):
        readings = normalize(rows, ambiguous=policy, nonexistent="flag")
        by_session = _sessions(rows, readings)
        for sid, ev in by_session.items():
            if "open" not in ev or "close" not in ev:
                continue
            results.setdefault(sid, {})[policy] = _minutes(ev["open"].utc, ev["close"].utc)

    for sid in list(results):
        results[sid]["truth"] = _minutes(truth.get((sid, "open")), truth.get((sid, "close")))

    contested = ["S-101", "S-104", "S-105"]
    negatives = [s for s in results if (results[s].get("earlier") or 0) < 0]

    if verbose:
        _hdr("A. The hour that happens twice")
        print(
            "New York, 2024-11-03. At 02:00 EDT the clock goes back to 01:00 EST, so the\n"
            "wall clock reads 01:00-01:59 twice, an hour apart. Three sessions in that\n"
            "window, with the durations the business thinks it is measuring:\n"
        )
        print(f"{'session':<10}{'open (local)':>16}{'close (local)':>16}{'truth':>10}{'fold=0':>10}{'fold=1':>10}")
        print(RULE)
        for sid in contested:
            ev = {r["event"]: r["local_ts"] for r in rows if r["session_id"] == sid}
            t = results[sid]
            f0 = t.get("earlier")
            f1 = t.get("later")
            print(
                f"{sid:<10}{ev['open'][11:]:>16}{ev['close'][11:]:>16}"
                f"{t['truth']:>10.0f}{f0:>10.0f}{f1:>10.0f}"
            )
        print(RULE)
        print("  minutes. fold=0 is Python's default for an ambiguous wall time.\n")
        print(
            "S-101 sits entirely before the transition, so every reading agrees - that is\n"
            "the control. S-104 really lasted 80 minutes and both fold policies report 20.\n"
            "S-105 really lasted 20 minutes and both report **-40**: a session that closed\n"
            "before it opened.\n"
        )
        print(
            "The important part is not that fold=0 is wrong. It is that **no single fold is\n"
            "right**. S-104 opened in the first pass of 01:00 and closed in the second, so\n"
            "recovering it needs fold=0 on one row and fold=1 on the other, chosen per row,\n"
            "from information the input does not contain. A global `ambiguous=` policy - in\n"
            "this module, in pytz, in pandas - cannot express the answer."
        )
        print(f"\nsessions with a negative duration under the default policy: {negatives}")

    return {"results": results, "negatives": negatives, "contested": contested}


# --------------------------------------------------------------------------
# B. the hour that never happens
# --------------------------------------------------------------------------


def exp_nonexistent(verbose: bool = True) -> Dict[str, Any]:
    """Spring-forward gaps, including a thirty-minute one, and the round trip that fails."""
    cases = [
        ("America/New_York", dt.datetime(2024, 3, 10, 2, 30), "one hour wide"),
        ("Australia/Lord_Howe", dt.datetime(2024, 10, 6, 2, 15), "THIRTY minutes wide"),
        ("Europe/London", dt.datetime(2024, 3, 31, 1, 30), "one hour wide"),
    ]
    out = []
    for zone, naive, width in cases:
        tz = ZoneInfo(zone)
        f0 = naive.replace(tzinfo=tz, fold=0)
        f1 = naive.replace(tzinfo=tz, fold=1)
        back = f0.astimezone(UTC).astimezone(tz).replace(tzinfo=None)
        out.append(
            {
                "zone": zone,
                "input": naive,
                "width": width,
                "fold0_utc": f0.astimezone(UTC),
                "fold1_utc": f1.astimezone(UTC),
                "round_trip": back,
                "classified": classify(naive, zone),
                "raised": False,
            }
        )

    if verbose:
        _hdr("B. The hour that never happens")
        print(
            "When a clock jumps forward the skipped wall-clock times do not exist. An\n"
            "upstream job that computes 'start + 45 minutes' in local terms writes one\n"
            "anyway. Python accepts it without complaint:\n"
        )
        print(f"{'zone':<22}{'input':<18}{'fold=0 -> UTC':<22}{'converted back':<18}")
        print(RULE)
        for c in out:
            print(
                f"{c['zone']:<22}{c['input'].strftime('%m-%d %H:%M'):<18}"
                f"{c['fold0_utc'].strftime('%Y-%m-%d %H:%M%z'):<22}"
                f"{c['round_trip'].strftime('%m-%d %H:%M'):<18}"
            )
        print(RULE)
        print(
            "\nNo exception is raised in any of these. The round trip is the tell: convert\n"
            "the wall time to UTC and back and you get a *different* wall time, because the\n"
            "one you started with is not on the clock. `classify()` uses precisely that\n"
            "test, which is why it needs no transition table:\n"
        )
        for c in out:
            print(f"  {c['zone']:<22}{c['input']}  ->  {c['classified']}   (gap is {c['width']})")
        print(
            "\nLord Howe Island is worth the trip: its DST shift is thirty minutes, so the\n"
            "gap is half an hour wide. Any check written as 'is the hour 02:xx suspicious'\n"
            "misses it, and any test suite built only on America/New_York never sees it.\n"
        )
        print(
            "pytz raised NonExistentTimeError here. zoneinfo, correctly, represents both\n"
            "readings and raises nothing - so the guard has to be yours."
        )

    return {"cases": out}


# --------------------------------------------------------------------------
# C. an offset in the payload decides it; an offset in the schema does not
# --------------------------------------------------------------------------


def exp_offset_vs_zone(verbose: bool = True) -> Dict[str, Any]:
    """The fix, and the limit of the fix."""
    rows = build_session_log(with_partner_feed=True)
    truth = ground_truth()
    readings = normalize(rows, ambiguous="earlier", nonexistent="flag")
    by_session = _sessions(rows, readings)

    recovered = {}
    for sid in ("S-104", "S-105"):
        wall = _minutes(by_session[sid]["open"].utc, by_session[sid]["close"].utc)
        api = _minutes(by_session[f"{sid}-api"]["open"].utc, by_session[f"{sid}-api"]["close"].utc)
        real = _minutes(truth[(sid, "open")], truth[(sid, "close")])
        recovered[sid] = {"wall": wall, "api": api, "truth": real}

    # the flip side: an offset cannot schedule
    ny = ZoneInfo("America/New_York")
    anchor = dt.datetime(2024, 10, 15, 9, 0, tzinfo=ny)  # 09:00 local, EDT
    fixed = dt.timezone(anchor.utcoffset())
    later_zone = (anchor + dt.timedelta(days=30)).astimezone(ny)
    later_fixed = (anchor.astimezone(fixed) + dt.timedelta(days=30)).astimezone(ny)

    if verbose:
        _hdr("C. An offset in the payload decides it. An offset in the schema does not.")
        print(
            "The same two contested sessions, from a partner API that transmits the UTC\n"
            "offset next to the wall clock. Identical instants, identical local times, one\n"
            "extra field:\n"
        )
        print(f"{'session':<12}{'wall clock only':>18}{'with offset':>14}{'truth':>10}")
        print(RULE)
        for sid, v in recovered.items():
            print(f"{sid:<12}{v['wall']:>18.0f}{v['api']:>14.0f}{v['truth']:>10.0f}")
        print(RULE)
        print(
            "\nExact, both rows, no policy required. `2024-11-03T01:30:00-04:00` and\n"
            "`2024-11-03T01:50:00-05:00` are different instants and say so. This is the\n"
            "whole fix, and it costs six characters per timestamp.\n"
        )
        print("But an offset is not a zone, and the difference shows up going forwards:\n")
        print(f"  09:00 local in America/New_York on {anchor.date()} is {anchor.strftime('%H:%M %Z (%z)')}")
        print(f"  + 30 days, carried as a zone         -> {later_zone.strftime('%Y-%m-%d %H:%M %Z (%z)')}")
        print(f"  + 30 days, carried as a fixed offset -> {later_fixed.strftime('%Y-%m-%d %H:%M %Z (%z)')}")
        print(
            f"\nAn hour apart, because the clock changed in between and a stored offset does\n"
            f"not know that. Store the offset to pin an instant that already happened.\n"
            f"Store the zone to answer anything about a clock that has not run yet -\n"
            f"reminders, SLAs, business-hours windows, market opens, batch schedules."
        )

    return {
        "recovered": recovered,
        "anchor": anchor,
        "zone_result": later_zone,
        "offset_result": later_fixed,
        "divergence_hours": (later_fixed - later_zone).total_seconds() / 3600,
    }


# --------------------------------------------------------------------------
# D. the identifier is not canonical
# --------------------------------------------------------------------------


def exp_identifiers(verbose: bool = True) -> Dict[str, Any]:
    """Two names for one place fragment a GROUP BY while every row converts correctly."""
    rows = build_session_log()
    zones = [r["zone"] for r in rows]
    groups = find_alias_groups(zones)

    by_zone: Dict[str, float] = {}
    for r in rows:
        by_zone[r["zone"]] = by_zone.get(r["zone"], 0.0) + r["amount"]

    merged: Dict[str, float] = {}
    canon = {z: g[0] for g in groups for z in g}
    for z, amt in by_zone.items():
        merged[canon.get(z, z)] = merged.get(canon.get(z, z), 0.0) + amt

    inverted = [m for z in sorted(set(zones)) if (m := etc_zone_is_inverted(z))]

    extra_pairs = [
        ("Europe/Kyiv", "Europe/Kiev"),
        ("America/Nuuk", "America/Godthab"),
        ("Asia/Yangon", "Asia/Rangoon"),
        ("US/Eastern", "America/New_York"),
    ]
    checked = [(a, b, same_rules(a, b)) for a, b in extra_pairs]

    if verbose:
        _hdr("D. The zone identifier is not canonical")
        print(
            "The tz database keeps old names working as links, so renamed zones keep\n"
            "resolving. Every conversion below is correct. The aggregate is not:\n"
        )
        print(f"{'zone as logged':<24}{'revenue':>12}   {'resolves to':<24}{'merged revenue':>16}")
        print(RULE)
        for z in sorted(by_zone):
            c = canon.get(z, z)
            print(
                f"{z:<24}{by_zone[z]:>12,.0f}   {c:<24}{merged[c]:>16,.0f}"
                + ("   <-- split" if z in canon else "")
            )
        print(RULE)
        for g in groups:
            print(f"\n  {' == '.join(g)}   (same rules 1990-2035, sampled every 6 hours)")
        print(
            "\nBengaluru appears twice and each half looks like a smaller office. Nothing\n"
            "raises, nothing is null, and the per-row timestamps are all right.\n"
        )
        print("Other pairs in the wild, checked the same way:\n")
        for a, b, ok in checked:
            print(f"  {a:<20} == {b:<20} {ok}")
        print("\nAnd the identifier that means the opposite of what it says:\n")
        for m in inverted:
            print(f"  WARNING {m}")
        print(
            "\n`Etc/GMT+5` is UTC-05:00. The sign follows the POSIX convention, which is\n"
            "inverted relative to ISO 8601 and to every human reading it. A vendor feed\n"
            "labelled `Etc/GMT+5` for a US Eastern office is off by ten hours, and the\n"
            "resulting timestamps are all perfectly valid."
        )

    return {
        "groups": groups,
        "by_zone": by_zone,
        "merged": merged,
        "inverted": inverted,
        "checked": checked,
    }


# --------------------------------------------------------------------------
# E. which day did it happen on
# --------------------------------------------------------------------------


def exp_day_bucketing(verbose: bool = True) -> Dict[str, Any]:
    """`GROUP BY date(ts)` asks a different question depending on the zone of ts."""
    rows = build_session_log()
    readings = normalize(rows, ambiguous="earlier", nonexistent="shift_forward")

    moved = []
    utc_rev: Dict[dt.date, float] = {}
    local_rev: Dict[dt.date, float] = {}
    for row, r in zip(rows, readings):
        u, l = utc_day(r), local_day(r)
        if u is None or l is None:
            continue
        utc_rev[u] = utc_rev.get(u, 0.0) + row["amount"]
        local_rev[l] = local_rev.get(l, 0.0) + row["amount"]
        if u != l:
            moved.append((row["session_id"], row["office"], row["local_ts"], u, l, row["amount"]))

    if verbose:
        _hdr("E. Which day did it happen on")
        print(
            f"{len(moved)} of {len(rows)} events fall on a different calendar day in UTC than\n"
            f"they do where they happened:\n"
        )
        print(f"{'session':<9}{'office':<13}{'local time':<18}{'UTC day':<13}{'local day':<13}{'amount':>9}")
        print(RULE)
        for sid, office, local, u, l, amt in moved:
            print(f"{sid:<9}{office:<13}{local:<18}{str(u):<13}{str(l):<13}{amt:>9,.0f}")
        print(RULE)
        print()
        days = sorted(set(list(utc_rev) + list(local_rev)))
        print(f"{'day':<14}{'revenue by UTC day':>20}{'revenue by local day':>22}{'delta':>12}")
        print(RULE)
        for d in days:
            u, l = utc_rev.get(d, 0.0), local_rev.get(d, 0.0)
            print(f"{str(d):<14}{u:>20,.0f}{l:>22,.0f}{l - u:>+12,.0f}")
        print(RULE)
        print(
            "\nNeither column is wrong. They answer different questions - 'what did the\n"
            "platform do in this 24-hour window' and 'what did each office do on its own\n"
            "Monday' - and a dashboard that does not say which one it is showing will be\n"
            "asked to reconcile with one that chose the other. The totals match; only the\n"
            "boundaries move. That is the version of this bug that survives longest,\n"
            "because the control total is always right."
        )

    return {"moved": moved, "utc_rev": utc_rev, "local_rev": local_rev}


# --------------------------------------------------------------------------
# F. offsets are not whole hours
# --------------------------------------------------------------------------


def exp_sub_hour(verbose: bool = True) -> Dict[str, Any]:
    """+05:45, +05:30, +12:45, and a DST shift of thirty minutes."""
    instant = dt.datetime(2024, 11, 3, 12, 0, tzinfo=UTC)
    zones = [
        "America/New_York",
        "Europe/London",
        "Asia/Kolkata",
        "Asia/Kathmandu",
        "Asia/Singapore",
        "Australia/Lord_Howe",
        "Pacific/Chatham",
    ]
    table = []
    for z in zones:
        tz = ZoneInfo(z)
        loc = instant.astimezone(tz)
        off = loc.utcoffset()
        assert off is not None
        jan = dt.datetime(2024, 1, 15, 12, tzinfo=UTC).astimezone(tz).utcoffset()
        jul = dt.datetime(2024, 7, 15, 12, tzinfo=UTC).astimezone(tz).utcoffset()
        assert jan is not None and jul is not None
        table.append(
            {
                "zone": z,
                "local": loc,
                "offset": off,
                "whole_hour": off.total_seconds() % 3600 == 0,
                "dst_shift": abs((jul - jan).total_seconds()) / 60,
            }
        )
    odd = [t for t in table if not t["whole_hour"]]
    half = [t for t in table if 0 < t["dst_shift"] < 60]

    if verbose:
        _hdr("F. Offsets are not whole hours, and DST shifts are not always an hour")
        print(f"One instant - {instant:%Y-%m-%d %H:%M UTC} - seen from seven zones:\n")
        print(f"{'zone':<24}{'local time':<22}{'offset':>10}{'whole hour':>13}{'DST shift':>12}")
        print(RULE)
        for t in table:
            hrs = t["offset"].total_seconds() / 3600
            print(
                f"{t['zone']:<24}{t['local'].strftime('%Y-%m-%d %H:%M'):<22}{hrs:>+10.2f}"
                f"{str(t['whole_hour']):>13}{t['dst_shift']:>10.0f}m"
            )
        print(RULE)
        print(
            f"\n{len(odd)} of {len(table)} are not a whole number of hours from UTC, and "
            f"{len(half)} shift by\nthirty minutes rather than sixty when their clocks change.\n"
        )
        print(
            "Two things break on this. Hour-of-day bucketing does not align across zones -\n"
            "an 'hourly' chart comparing Kathmandu with Singapore is comparing buckets\n"
            "offset by fifteen minutes. And any code that stores an offset as an integer\n"
            "number of hours, or derives a zone from `utcoffset().seconds // 3600`, silently\n"
            "truncates +05:45 to +05:00 - forty-five minutes, every row, one direction."
        )

    return {"table": table, "odd": [t["zone"] for t in odd], "half": [t["zone"] for t in half]}


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


def damage_ledger(verbose: bool = True) -> List[Tuple[str, str, str]]:
    a = exp_ambiguous(verbose=False)
    b = exp_nonexistent(verbose=False)
    c = exp_offset_vs_zone(verbose=False)
    d = exp_identifiers(verbose=False)
    e = exp_day_bucketing(verbose=False)
    f = exp_sub_hour(verbose=False)

    s105 = a["results"]["S-105"]
    rows = [
        ("ambiguous hour, default fold", f"S-105 lasts {s105['earlier']:.0f} min, truly {s105['truth']:.0f}", "silent"),
        ("no single fold is correct", f"S-104 off by {a['results']['S-104']['truth'] - a['results']['S-104']['earlier']:.0f} min either way", "silent"),
        ("nonexistent wall time", f"{len(b['cases'])}/{len(b['cases'])} accepted, round trip fails", "silent"),
        ("offset carried forward 30d", f"{abs(c['divergence_hours']):.0f}h drift vs the zone", "silent"),
        ("alias fragments GROUP BY", f"1 office split into {len(d['groups'][0]) if d['groups'] else 0} rows", "silent"),
        ("Etc/GMT+5 sign inversion", "10h error, all values valid", "silent"),
        ("UTC day vs local day", f"{len(e['moved'])} of 24 events change day", "silent"),
        ("sub-hour offsets", f"{len(f['odd'])} zones misbucket by 15-45 min", "silent"),
    ]
    if verbose:
        _hdr("Damage ledger")
        print(f"{'failure mode':<32}{'effect on this sample':<38}{'raises?':>8}")
        print(RULE)
        for name, effect, raises in rows:
            print(f"{name:<32}{effect:<38}{raises:>8}")
        print(RULE)
        print(
            f"\nAll {len(rows)} produce a plausible answer and none of them raise. Six produce a\n"
            "*stable* one, so re-running the pipeline reproduces the same wrong number and\n"
            "a reconciliation against yesterday agrees. Only the negative duration announces\n"
            "itself, and only if somebody is looking for negatives."
        )
    return rows


def main() -> None:
    print(f"tz database: {tzdata_version()}")
    exp_ambiguous()
    exp_nonexistent()
    exp_offset_vs_zone()
    exp_identifiers()
    exp_day_bucketing()
    exp_sub_hour()
    damage_ledger()
    _hdr("Pre-flight audit, session log")
    from tznorm import audit

    print(audit(build_session_log()).text())


if __name__ == "__main__":
    main()
