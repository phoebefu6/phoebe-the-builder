"""Every number in the README, computed here. Nothing is typed by hand.

Run:  python3 evidence.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import cron as C

UTC = timezone.utc
YEAR = 2026
TZ = "Europe/London"
START = datetime(YEAR, 1, 1)

# A crontab of the kind that actually exists in a repo: each line plausible,
# several of them wrong in a way that renders perfectly.
SAMPLE: List[Tuple[str, str]] = [
    ("0 0 13 * 5", "monthly close report, 'Friday the 13th'"),
    ("30 1 * * *", "nightly warehouse load"),
    ("*/7 * * * *", "queue drain, 'every 7 minutes'"),
    ("0 */5 * * *", "cache warm, 'every 5 hours'"),
    ("*/30 * * * *", "heartbeat"),
    ("0 9 * * 1-5", "weekday standup digest"),
    ("0 0 31 2 *", "quarter-end sweep"),
    ("0 3 * * 7", "weekly vacuum, 'Sunday'"),
    ("0 0 1 * *", "invoice run"),
    ("15 2 * * *", "backup rotate"),
]


def union_evidence() -> Dict[str, int]:
    c = C.parse("0 0 13 * 5")
    union = C._matching_days(c, START, 365)
    inter = C._matching_days(c, START, 365, force_intersection=True)
    fridays = [d for d in union if d.weekday() == 4]
    thirteenths = [d for d in union if d.day == 13]
    return {
        "union_days": len(union),
        "intersection_days": len(inter),
        "fridays": len(fridays),
        "thirteenths": len(thirteenths),
        "extra_runs": len(union) - len(inter),
        "factor": round(len(union) / max(len(inter), 1)),
    }


def step_evidence() -> Dict[str, object]:
    c = C.parse("*/7 * * * *")
    out = C.next_naive(c, datetime(YEAR, 5, 1, 0, 0), 20)
    gaps = [int((b - a).total_seconds() // 60) for a, b in zip(out, out[1:])]
    hourly = C.parse("0 */5 * * *")
    hout = C.next_naive(hourly, datetime(YEAR, 5, 1, 0, 0), 12)
    hgaps = [int((b - a).total_seconds() // 3600) for a, b in zip(hout, hout[1:])]
    return {
        "minute_values": list(C.parse("*/7 * * * *").minute.values),
        "minute_gaps": sorted(set(gaps)),
        "short_gap": min(gaps),
        "runs_per_hour": len(C.parse("*/7 * * * *").minute.values),
        "hour_values": list(hourly.hour.values),
        "hour_gaps": sorted(set(hgaps)),
        "runs_per_day": len(hourly.hour.values),
    }


def dst_evidence() -> Dict[str, object]:
    tz = C._zone(TZ)
    fixed = C.parse("30 1 * * *")
    interval = C.parse("*/30 * * * *")

    sk, rp = C._dst_hits(fixed, tz, START, 400)

    # Spring forward: what the fixed-time job actually does.
    spring = C.fires(fixed, datetime(YEAR, 3, 28, 12, 0), 2, TZ)
    spring_hit = [f for f in spring if f.kind == C.SKIPPED][0]

    # Fall back: fixed runs once, interval runs twice at the same wall clock.
    fall_fixed = C.fires(fixed, datetime(YEAR, 10, 24, 12, 0), 2, TZ)
    fall_int = C.fires(interval, datetime(YEAR, 10, 25, 0, 45), 6, TZ)
    rep_int = [f for f in fall_int if f.kind == C.REPEATED]
    rep_fixed = [f for f in fall_fixed if f.kind == C.REPEATED]

    # A whole DST day for an interval job: how many times does it run?
    def runs_on(day: datetime) -> int:
        got = C.fires(interval, day - timedelta(minutes=1), 60, TZ)
        return len([f for f in got if f.local.date() == day.date() and f.instant])

    normal_day = runs_on(datetime(YEAR, 6, 15))
    spring_day = runs_on(datetime(YEAR, 3, 29))
    fall_day = runs_on(datetime(YEAR, 10, 25))

    return {
        "skipped_count": len(sk),
        "repeated_count": len(rp),
        "skipped_local": f"{sk[0]:%Y-%m-%d %H:%M}" if sk else "-",
        "spring_runs_at_utc": f"{spring_hit.instant:%Y-%m-%d %H:%M} UTC",
        "spring_runs_at_local": f"{spring_hit.instant.astimezone(tz):%H:%M %Z}",
        "fixed_fallback_runs": len(rep_fixed),
        "interval_fallback_runs": len(rep_int),
        "interval_gap_hours": int(
            (rep_int[1].instant - rep_int[0].instant).total_seconds() // 3600
        ) if len(rep_int) > 1 else 0,
        "runs_normal_day": normal_day,
        "runs_spring_day": spring_day,
        "runs_fall_day": fall_day,
    }


def utc_scheduler_evidence() -> Dict[str, object]:
    """The same line on a UTC scheduler: no DST case, and an hour of drift."""
    tz = C._zone(TZ)
    c = C.parse("0 9 * * *")
    jan = C.fires(c, datetime(YEAR, 1, 10), 1, TZ, utc_scheduler=True)[0]
    jul = C.fires(c, datetime(YEAR, 7, 10), 1, TZ, utc_scheduler=True)[0]
    # And what the same expression does when the host keeps local time.
    ljan = C.fires(c, datetime(YEAR, 1, 10), 1, TZ)[0]
    ljul = C.fires(c, datetime(YEAR, 7, 10), 1, TZ)[0]
    return {
        "utc_jan_local": f"{jan.instant.astimezone(tz):%H:%M}",
        "utc_jul_local": f"{jul.instant.astimezone(tz):%H:%M}",
        "local_jan_utc": f"{ljan.instant:%H:%M}",
        "local_jul_utc": f"{ljul.instant:%H:%M}",
        "drift_minutes": abs(
            int((jul.instant.astimezone(tz).hour - jan.instant.astimezone(tz).hour) * 60)
        ),
    }


def crosscheck_evidence() -> Dict[str, int]:
    """Two independent searches over the sample, diffed fire by fire."""
    from test_cron import EXPRS

    exprs = EXPRS + [e for e, _ in SAMPLE]
    checked = agree = fires_compared = 0
    for e in exprs:
        try:
            c = C.parse(e)
        except C.CronError:
            continue
        checked += 1
        a = C.next_naive(c, datetime(YEAR, 2, 26, 23, 51), 15)
        b = C.brute_naive(c, datetime(YEAR, 2, 26, 23, 51), 15)
        fires_compared += len(b)
        if a == b:
            agree += 1
    return {
        "expressions": checked,
        "agreeing": agree,
        "fires_compared": fires_compared,
    }


def sample_audit() -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    rows, by_sev = [], {s: 0 for s in C.SEVERITIES}
    for expr, purpose in SAMPLE:
        c = C.parse(expr)
        f = C.audit(c, TZ, START)
        for x in f:
            by_sev[x.severity] += 1
        rows.append(
            {
                "expr": expr,
                "purpose": purpose,
                "findings": len(f),
                "codes": [x.code for x in f],
                "severities": [x.severity for x in f],
            }
        )
    return rows, by_sev


def zone_comparison() -> Dict[str, object]:
    """The same crontab, audited in a DST zone and in UTC.

    If the counts move, the findings are being computed from the time line and
    not pattern-matched off the text.
    """
    out = {}
    for zone in (TZ, "UTC"):
        n = clean = 0
        for expr, _ in SAMPLE:
            f = C.audit(C.parse(expr), zone, START)
            n += len(f)
            clean += 1 if not f else 0
        out[zone] = {"findings": n, "clean_lines": clean}
    never = C.next_naive(C.parse("0 0 31 2 *"), START, 1)
    out["never_fires_in_5_years"] = len(never) == 0
    return out


def main() -> None:
    bar = "=" * 78
    print(bar)
    print("CRON EXPLAINER - evidence")
    print(bar)

    u = union_evidence()
    print("\n1. '0 0 13 * 5' - the union day rule")
    print(f"   fires on            {u['union_days']} days in {YEAR}")
    print(f"   read as 'and'       {u['intersection_days']} days")
    print(f"   overrun             {u['extra_runs']} unintended runs (x{u['factor']})")
    print(f"   made of             {u['fridays']} Fridays + {u['thirteenths']} 13ths")

    s = step_evidence()
    print("\n2. steps that do not tile their field")
    print(f"   '*/7'  minutes      {s['minute_values'][:5]} ... {s['minute_values'][-1]}")
    print(f"          gaps         {s['minute_gaps']} minutes -> {s['runs_per_hour']}/hour, not 60/7")
    print(f"   '0 */5' hours       {s['hour_values']}")
    print(f"          gaps         {s['hour_gaps']} hours -> {s['runs_per_day']}/day, not 24/5")

    d = dst_evidence()
    print(f"\n3. '30 1 * * *' in {TZ}, 400 days")
    print(f"   wall times that do not exist   {d['skipped_count']}  ({d['skipped_local']})")
    print(f"   wall times that happen twice   {d['repeated_count']}")
    print(f"   the skipped run actually fires {d['spring_runs_at_utc']} = {d['spring_runs_at_local']}")
    print(f"   fall back: fixed-time runs     {d['fixed_fallback_runs']}x")
    print(f"   fall back: '*/30' runs         {d['interval_fallback_runs']}x, "
          f"{d['interval_gap_hours']}h apart, same wall clock")
    print(f"   '*/30' runs per day  normal {d['runs_normal_day']} | "
          f"spring {d['runs_spring_day']} | autumn {d['runs_fall_day']}")

    g = utc_scheduler_evidence()
    print("\n4. '0 9 * * *' - UTC scheduler vs a host on local time")
    print(f"   GitHub Actions      09:00 UTC -> {g['utc_jan_local']} local in Jan, "
          f"{g['utc_jul_local']} local in Jul")
    print(f"   local-time host     09:00 local -> {g['local_jan_utc']} UTC in Jan, "
          f"{g['local_jul_utc']} UTC in Jul")

    x = crosscheck_evidence()
    print("\n5. cross-check: field-jumping search vs minute-by-minute scan")
    print(f"   expressions         {x['expressions']}")
    print(f"   identical output    {x['agreeing']}/{x['expressions']}")
    print(f"   fire times compared {x['fires_compared']}")

    rows, by_sev = sample_audit()
    total = sum(r["findings"] for r in rows)
    clean = len([r for r in rows if r["findings"] == 0])
    print(f"\n6. the sample crontab ({len(rows)} lines)")
    print(f"   {'expression':<16} {'purpose':<34} findings")
    print("   " + "-" * 72)
    for r in rows:
        codes = ",".join(r["codes"]) if r["codes"] else "-"
        print(f"   {r['expr']:<16} {r['purpose'][:33]:<34} {codes}")
    print("   " + "-" * 72)
    print(f"   {total} findings across {len(rows) - clean} of {len(rows)} lines; "
          f"{clean} line(s) clean")
    print(f"   by severity: " + ", ".join(f"{k} {v}" for k, v in by_sev.items()))
    print(f"\n   Every one of these {len(rows)} lines is valid cron. None of them errors.")

    z = zone_comparison()
    print("\n7. the same crontab, audited against a different time line")
    for zone in (TZ, "UTC"):
        v = z[zone]
        print(f"   {zone:<16} {v['findings']:>3} findings, "
              f"{v['clean_lines']} of {len(SAMPLE)} lines clean")
    print("   the text did not change; the zone did")
    print(f"   '0 0 31 2 *' has no next fire within 5 years: {z['never_fires_in_5_years']}")
    print(bar)


if __name__ == "__main__":
    main()
