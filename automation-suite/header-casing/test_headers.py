"""Tests for the field-name audit.

Two kinds of test here. Most pin a documented rule (RFC 9110, RFC 9113, PEP 3333,
Go's canonicalisation) so a refactor cannot quietly change what the engine claims
about the standard. The rest pin the exhaustive searches, so the counts in the
README stay true.
"""

from __future__ import annotations

import pytest

from headers import (
    CORPUS,
    HPACK_STATIC_NAMES,
    LOOKUPS,
    PATHS,
    REGISTRY_NAMES,
    Field,
    Message,
    Verdict,
    ascii_lower,
    audit_corpus,
    canonical_mismatches,
    deliver,
    environ_collisions,
    findings,
    go_canonical,
    hpack_names,
    is_token,
    lookup_audit,
    py_title,
    safe_form,
    title_mismatches,
    turkish_breakage,
    turkish_lower,
    wire_cost,
    wsgi_key,
)


def msg(*pairs, version="1.1"):
    return Message("t", tuple(Field(n, v) for n, v in pairs), version)


# --- what a field name is --------------------------------------------------


@pytest.mark.parametrize("name", ["Content-Type", "x_foo", "X-2FA-Token", "a.b~c$d"])
def test_tchar_names_are_legal(name):
    assert is_token(name)


@pytest.mark.parametrize("name", ["", "Content Type", "x:y", "naïve", "a\nb"])
def test_non_token_names_are_illegal(name):
    assert not is_token(name)


def test_underscore_is_a_legal_field_name_character():
    """The nginx default drops a name that HTTP itself considers perfectly legal."""
    assert is_token("X_Request_Id")


def test_identity_is_case_insensitive():
    assert Field("Content-Type", "a").key == Field("CONTENT-TYPE", "a").key


# --- spelling functions ----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("content-type", "Content-Type"),
        ("CONTENT-TYPE", "Content-Type"),
        ("etag", "Etag"),
        ("x-2fa-token", "X-2fa-Token"),
        ("te", "Te"),
    ],
)
def test_go_canonical(raw, expected):
    assert go_canonical(raw) == expected


def test_go_canonical_leaves_illegal_names_alone():
    """Go returns the key unchanged when it is not a valid token."""
    assert go_canonical("Content Type") == "Content Type"


def test_title_capitalises_after_a_digit_and_canonical_does_not():
    assert py_title("x-2fa-token") == "X-2Fa-Token"
    assert go_canonical("x-2fa-token") == "X-2fa-Token"


def test_ascii_lower_leaves_non_ascii_alone():
    assert ascii_lower("X-Ünicode") == "x-Ünicode"


def test_turkish_lower_produces_a_name_that_is_no_longer_a_token():
    mangled = turkish_lower("If-Match")
    assert mangled == "ıf-match"
    assert not is_token(mangled)


def test_wsgi_key_is_one_way():
    assert wsgi_key("X-Request-Id") == wsgi_key("X_Request_Id") == "HTTP_X_REQUEST_ID"


# --- HTTP/2 rules ----------------------------------------------------------


def test_h2_lowercases_every_name():
    d = deliver(msg(("Content-Type", "a"), ("X-Trace", "b")), "h2-gateway")
    assert d.arrived is not None
    assert [f.name for f in d.arrived.fields] == ["content-type", "x-trace"]
    assert d.verdict() is Verdict.RENORMALIZED


def test_h2_rejects_connection_specific_fields():
    d = deliver(msg(("Host", "h"), ("Connection", "keep-alive")), "h2-gateway")
    assert d.verdict() is Verdict.REJECTED
    assert any(e.code == "h2-forbidden-field" for e in d.events)


def test_te_trailers_is_the_documented_exception():
    ok = deliver(msg(("TE", "trailers")), "h2-gateway")
    bad = deliver(msg(("TE", "gzip")), "h2-gateway")
    assert ok.verdict() is Verdict.RENORMALIZED
    assert bad.verdict() is Verdict.REJECTED


def test_strict_receiver_rejects_an_uppercase_name():
    d = deliver(msg(("Content-Type", "a")), "handwritten-h2-frame")
    assert d.verdict() is Verdict.REJECTED
    assert d.events[0].code == "malformed-uppercase"


def test_already_lowercase_survives_h2_untouched():
    d = deliver(msg(("content-type", "a")), "handwritten-h2-frame")
    assert d.verdict() is Verdict.PRESERVED


# --- proxies and runtimes --------------------------------------------------


def test_nginx_drops_underscored_names_silently():
    d = deliver(msg(("X_Request_Id", "v"), ("Host", "h")), "nginx-h1")
    assert d.arrived is not None
    assert d.lost() == ("x_request_id",)
    assert d.verdict() is Verdict.LOSSY


def test_cgi_collides_two_distinct_field_names():
    d = deliver(msg(("X-Request-Id", "proxy"), ("X_Request_Id", "client")), "nginx-to-cgi")
    codes = {e.code for e in d.events}
    assert "dropped-underscore" in codes
    assert d.arrived is not None
    assert len(d.arrived.fields) == 1


def test_cgi_collision_without_nginx_in_front_merges_the_values():
    m = msg(("X-Request-Id", "proxy"), ("X_Request_Id", "client"))
    arrived, events = __import__("headers").hop_cgi_environ(m)
    assert arrived is not None
    assert len(arrived.fields) == 1
    assert arrived.fields[0].value == "proxy, client"
    assert any(e.code == "environ-collision" for e in events)


def test_node_discards_a_duplicate_content_type():
    d = deliver(msg(("Content-Type", "application/json"), ("Content-Type", "text/plain")),
                "h2-to-node")
    assert d.arrived is not None
    assert [f.value for f in d.arrived.fields] == ["application/json"]
    assert any(e.code == "discarded-duplicate" for e in d.events)


def test_node_joins_cookies_with_a_semicolon():
    d = deliver(msg(("Cookie", "a=1"), ("Cookie", "b=2")), "h2-to-node")
    assert d.arrived is not None
    assert d.arrived.fields[0].value == "a=1; b=2"


def test_node_keeps_set_cookie_as_separate_lines():
    d = deliver(msg(("Set-Cookie", "a=1"), ("Set-Cookie", "b=2")), "h2-to-node")
    assert d.arrived is not None
    assert len(d.arrived.fields) == 2


def test_str_dict_splits_one_field_into_two_entries():
    d = deliver(msg(("X-Api-Key", "k1"), ("X-API-KEY", "k2")), "java-tr-locale")
    assert any(e.code == "split-identity" for e in d.events) or d.verdict() is Verdict.LOSSY


def test_go_sorts_and_canonicalises():
    d = deliver(msg(("x-zulu", "1"), ("etag", "2")), "h2-then-go")
    assert d.arrived is not None
    assert [f.name for f in d.arrived.fields] == ["Etag", "X-Zulu"]
    assert any(e.code == "reordered" for e in d.events)


def test_turkish_locale_breaks_the_lookup():
    d = deliver(msg(("If-Match", "\"v1\"")), "java-tr-locale")
    got = lookup_audit(d, "If-Match")
    assert got["CaseInsensitiveDict"] == []
    assert any(e.code == "turkish-dotless-i" for e in d.events)


# --- verdicts --------------------------------------------------------------


def test_direct_path_preserves_everything():
    for m in CORPUS:
        assert deliver(m, "direct-h1").verdict() is Verdict.PRESERVED


def test_a_rejected_delivery_loses_every_field():
    d = deliver(msg(("Host", "h"), ("Connection", "close")), "h2-gateway")
    assert d.rejected
    assert d.lost() == ("host", "connection")


def test_renormalized_means_same_meaning_different_bytes():
    d = deliver(msg(("Content-Type", "a")), "h2-gateway")
    assert d.verdict() is Verdict.RENORMALIZED
    assert d.arrived is not None
    assert d.arrived.identity() == d.sent.identity()


# --- lookups ---------------------------------------------------------------


def test_only_the_case_insensitive_lookup_survives_h2():
    d = deliver(msg(("X-Request-ID", "v")), "h2-gateway")
    got = lookup_audit(d, "X-Request-ID")
    assert got["h[name]"] == []
    assert got["CaseInsensitiveDict"] == ["v"]


def test_every_lookup_style_is_exercised():
    d = deliver(msg(("X-Request-ID", "v")), "nginx-to-cgi")
    got = lookup_audit(d, "X-Request-ID")
    assert set(got) == {name for name, _, _ in LOOKUPS}
    assert got["environ[HTTP_NAME]"] == ["v"]


# --- exhaustive searches ---------------------------------------------------


def test_every_hyphenated_name_has_a_colliding_underscore_twin():
    hyphenated = [n for n in REGISTRY_NAMES if "-" in n]
    collisions = environ_collisions()
    assert len(collisions) == len(hyphenated)


def test_collisions_never_pair_two_spellings_of_one_field():
    for a, b, _ in environ_collisions():
        assert ascii_lower(a) != ascii_lower(b)


def test_canonicalisation_cannot_reproduce_these_registered_names():
    pairs = dict(canonical_mismatches())
    assert pairs["ETag"] == "Etag"
    assert pairs["WWW-Authenticate"] == "Www-Authenticate"
    assert pairs["Content-MD5"] == "Content-Md5"


def test_title_and_canonical_disagree_only_after_a_digit_or_underscore():
    for raw, titled, canon in title_mismatches():
        assert any(c.isdigit() or c == "_" for c in raw)
        assert titled != canon


def test_turkish_breakage_is_exactly_the_names_with_a_capital_i():
    broken = {n for n, _ in turkish_breakage()}
    assert broken == {n for n in REGISTRY_NAMES if "I" in n}


def test_hpack_static_table_names_are_all_lowercase():
    assert all(n == ascii_lower(n) for n in HPACK_STATIC_NAMES)


def test_hpack_split_covers_the_registry():
    inside, outside = hpack_names()
    assert inside + outside == len(REGISTRY_NAMES)
    assert inside > 0 and outside > 0


def test_lowercasing_a_static_name_is_cheaper_on_the_wire():
    m = msg(("Content-Type", "application/json"))
    h1, h2 = wire_cost(m)
    assert h2 < h1


# --- findings and advice ---------------------------------------------------


def test_every_event_code_has_a_severity():
    from headers import SEVERITY_OF_CODE

    seen = {e.code for d in audit_corpus().values() for e in d.events}
    assert seen <= set(SEVERITY_OF_CODE)


def test_findings_are_sorted_worst_first():
    for d in audit_corpus().values():
        fs = findings(d)
        order = [{"blocking": 0, "silent": 1, "advisory": 2}[f.severity] for f in fs]
        assert order == sorted(order)


def test_safe_form_flags_the_underscore_and_the_capital_i():
    advice = safe_form("X_Request_ID")
    assert "nginx" in advice
    assert "`I`" in advice


def test_safe_form_refuses_an_illegal_name():
    assert "cannot be sent" in safe_form("X Request Id")


def test_corpus_produces_all_four_verdicts():
    seen = {d.verdict() for d in audit_corpus().values()}
    assert seen == set(Verdict)


def test_paths_are_all_reachable():
    assert set(PATHS) == {p for (_, p) in audit_corpus()}
