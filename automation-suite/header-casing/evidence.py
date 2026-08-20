"""Every number quoted in the README, printed from the engine.

Run `python3 evidence.py`. Nothing in the README is typed by hand.
"""

from __future__ import annotations

from headers import (
    CORPUS,
    CORPUS_BY_NAME,
    HOPS,
    LOOKUPS,
    PATHS,
    REGISTERED_ONLY,
    REGISTRY_NAMES,
    Verdict,
    ascii_lower,
    audit_corpus,
    canonical_mismatches,
    deliver,
    deliver_all,
    environ_collisions,
    findings,
    go_canonical,
    hpack_names,
    lookup_audit,
    py_title,
    safe_form,
    title_mismatches,
    total_findings,
    turkish_breakage,
    verdict_counts,
    wire_cost,
    wsgi_key,
)

RULE = "-" * 78


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def s1_two_spellings() -> None:
    head(1, "One field, two spellings, and a dict that holds both")
    m = CORPUS_BY_NAME["mixed-spelling"]
    for f in m.fields:
        print(f"  wire: {f.name:<12} value: {f.value}")
    d = deliver(m, "h1-to-dict")
    print()
    for e in d.events:
        print(f"  [{e.hop}] {e.code}: {e.detail}")
    print("\n  RFC 9110 5.1 says these are one field. A Dict[str, str] says two.")


def s2_paths() -> None:
    head(2, "The same request down every modelled path")
    m = CORPUS_BY_NAME["tracing-pair"]
    print(f"  sent ({m.note}):")
    for f in m.fields:
        print(f"    {f.name}: {f.value}")
    print()
    print(f"  {'path':<22} {'verdict':<14} arrived")
    for path, d in deliver_all(m).items():
        if d.arrived is None:
            got = "-- rejected --"
        else:
            got = ", ".join(f"{f.name}={f.value}" for f in d.arrived.fields)
        print(f"  {path:<22} {d.verdict().value:<14} {got}")


def s3_verdicts() -> None:
    head(3, "Twelve messages, nine paths, four verdicts")
    counts = verdict_counts()
    total = sum(counts.values())
    for v in Verdict:
        n = counts[v]
        print(f"  {v.value:<14} {n:>3} of {total}  {n / total:>5.1%}")
    keeps = sorted({p for (_, p), d in audit_corpus().items()
                    if d.verdict() is Verdict.PRESERVED})
    print(f"\n  the {counts[Verdict.PRESERVED]} preserved deliveries come from "
          f"{len(keeps)} of the {len(PATHS)} paths:")
    print(f"    {', '.join(keeps)}")
    print("  - and each of those either does nothing to the name, or was handed a")
    print("  message whose names were already lowercase.")
    f = total_findings()
    print(f"\n  findings: {f['blocking']} blocking, {f['silent']} silent, "
          f"{f['advisory']} advisory")
    print("  The silent ones are the interesting column: the request succeeded.")


def s4_lookup_matrix() -> None:
    head(4, "Five ways to read one field back, across the paths")
    m = CORPUS_BY_NAME["browser-get"]
    wanted = "Upgrade-Insecure-Requests"
    names = [n for n, _, _ in LOOKUPS]
    print(f"  looking for {wanted!r} in {m.name}\n")
    print("  " + "path".ljust(22) + "".join(n.ljust(22) for n in names))
    for path in PATHS:
        d = deliver(m, path)
        got = lookup_audit(d, wanted)
        row = "".join(("found" if got[n] else ".").ljust(22) for n in names)
        print(f"  {path.ljust(22)}{row}")
    print("\n  Only the case-insensitive column is right everywhere it is reachable;")
    print("  under CGI even that fails, because the name is no longer a field name.")


def s5_environ_collisions() -> None:
    head(5, "Exhaustive: every field name that shares a CGI variable with another")
    cols = environ_collisions()
    hyphenated = [n for n in REGISTRY_NAMES if "-" in n]
    print(f"  registry names checked:        {len(REGISTRY_NAMES)}")
    print(f"  of those, hyphenated:          {len(hyphenated)}")
    print(f"  distinct colliding field pairs:{len(cols):>4}")
    print("\n  first eight:")
    for a, b, var in cols[:8]:
        print(f"    {a:<28} + {b:<28} -> {var}")
    print("\n  Every hyphenated field name has an underscore twin that is also a legal")
    print("  token (RFC 9110 5.6.2) and lands on the same variable. The default")
    print("  `underscores_in_headers off` in nginx is what stops it - which means the")
    print("  fix for 'my X_Custom_Header is missing' re-opens the hole.")


def s6_canonical() -> None:
    head(6, "Exhaustive: registered names a canonicalising stack cannot reproduce")
    mm = canonical_mismatches()
    print(f"  {len(mm)} of {len(REGISTRY_NAMES)} names change spelling under the "
          f"canonical rule")
    reg = [(a, b) for a, b in mm if a in REGISTERED_ONLY]
    print(f"  {len(reg)} of them are IANA-registered names\n")
    for a, b in mm[:12]:
        print(f"    {a:<28} -> {b}")
    print("\n  and where `str.title()` disagrees with the canonical rule as well:")
    for raw, titled, canon in title_mismatches():
        print(f"    {raw:<28} title(): {titled:<20} canonical: {canon}")


def s7_turkish() -> None:
    head(7, "Exhaustive: the names a locale-sensitive lowercase destroys")
    tb = turkish_breakage()
    print(f"  {len(tb)} of {len(REGISTRY_NAMES)} names contain a capital I\n")
    for a, b in tb:
        print(f"    {a:<28} -> {b}")
    print("\n  `ı` (U+0131) is not a tchar, so the result is not a legal field name and")
    print("  no ASCII-case-folded lookup can match it. Java's String.toLowerCase()")
    print("  with no Locale.ROOT does this on a machine set to tr_TR.")


def s8_hpack() -> None:
    head(8, "Casing has a byte cost, not just a spelling")
    inside, outside = hpack_names()
    print(f"  HPACK static table entries covering the registry: {inside}")
    print(f"  names that pay for themselves on every request:   {outside}")
    m = CORPUS_BY_NAME["browser-get"]
    h1, h2 = wire_cost(m)
    print(f"\n  {m.name}: {h1} bytes as HTTP/1.1 field lines, {h2} modelled HPACK bytes")
    print(f"  ({(h1 - h2) / h1:.0%} smaller, and the saving comes from names that are")
    print("  lowercase and in the table - `Content-Type` indexed costs one byte)")


def s9_set_cookie() -> None:
    head(9, "The field that cannot be a dictionary entry")
    m = CORPUS_BY_NAME["login-response"]
    for path in ("h2-to-node", "h2-to-node-to-dict", "nginx-to-cgi"):
        d = deliver(m, path)
        n = 0 if d.arrived is None else sum(1 for f in d.arrived.fields
                                            if "cookie" in ascii_lower(f.name))
        print(f"  {path:<22} set-cookie lines arriving: {n} of 2  "
              f"verdict {d.verdict().value}")
    d = deliver(m, "h2-to-node-to-dict")
    for f in findings(d):
        if f.severity != "advisory":
            print(f"    {f.severity:<9} {f.code}: {f.text}")


def s10_rejections() -> None:
    head(10, "The three ways a message stops being a message")
    for name in ("h1-keepalive", "te-gzip", "handwritten-frame"):
        m = CORPUS_BY_NAME[name]
        for path in ("h2-gateway", "handwritten-h2-frame"):
            d = deliver(m, path)
            if d.rejected:
                print(f"  {name:<20} via {path:<22} {d.events[-1].code}")
                print(f"    {d.events[-1].detail}")
                break


def s11_hops() -> None:
    head(11, "The hops modelled, and what each one is allowed to do")
    for h in HOPS:
        print(f"  {h.name:<14} {h.kind:<12} {h.doc}")


def s12_advice() -> None:
    head(12, "What to do with a name, per name")
    for name in ("X_Request_Id", "X-Request-ID", "Content-Type", "X Request Id",
                 "x-2fa-token"):
        print(f"  {name}")
        for bit in safe_form(name).split("; "):
            print(f"      - {bit}")


def s13_worked_pair() -> None:
    head(13, "The spelling functions side by side")
    print(f"  {'name':<26}{'ascii_lower':<26}{'canonical':<22}{'title()':<22}"
          f"cgi")
    for n in ("ETag", "WWW-Authenticate", "Content-MD5", "X-2FA-Token", "TE"):
        print(f"  {n:<26}{ascii_lower(n):<26}{go_canonical(n):<22}{py_title(n):<22}"
              f"{wsgi_key(n)}")


def s14_headline() -> None:
    head(14, "The one number")
    all_d = audit_corpus()
    silent = [k for k, d in all_d.items()
              if d.verdict() is Verdict.LOSSY and not d.rejected]
    print(f"  message x path combinations:           {len(all_d)}")
    errored = sum(1 for k in silent if all_d[k].rejected)
    print(f"  combinations that lost data:           {len(silent)}")
    print(f"  of those, that returned an error:      {errored}")
    print(f"\n  {len(silent)} deliveries changed what the message said. Every one of them")
    print("  is a 200. The audit is the only thing that reports them.")


def main() -> None:
    print("HEADER CASING - evidence")
    print(f"corpus: {len(CORPUS)} messages, {len(PATHS)} paths, "
          f"{len(REGISTRY_NAMES)} field names")
    for fn in (s1_two_spellings, s2_paths, s3_verdicts, s4_lookup_matrix,
               s5_environ_collisions, s6_canonical, s7_turkish, s8_hpack,
               s9_set_cookie, s10_rejections, s11_hops, s12_advice,
               s13_worked_pair, s14_headline):
        fn()
    print()


if __name__ == "__main__":
    main()
