"""Engine tests: the truth column, the RFC edges, and the findings the README states."""

from __future__ import annotations

import explore as E
import pytest


@pytest.mark.parametrize("addr,group,hand", E.CORPUS)
def test_reference_agrees_with_the_hand_label(addr: str, group: str, hand: str) -> None:
    assert E.reference(addr)[0] == hand, (addr, E.reference(addr))


def test_length_limits_are_exact() -> None:
    assert E.reference("a" * 64 + "@example.com")[0] == "ok"
    assert E.reference("a" * 65 + "@example.com")[0] == "bad"
    at_limit = [a for a, g, _ in E.CORPUS if g == "at the length limit" and len(a) > 200][0]
    over = [a for a, g, _ in E.CORPUS if g == "over a length limit" and len(a) > 200][0]
    assert len(at_limit.encode()) == 254 and len(over.encode()) == 255


def test_every_share_group_is_in_the_corpus_and_deliverable() -> None:
    for g in E.SHARE:
        rows = [a for a, gg, _ in E.CORPUS if gg == g]
        assert rows and all(E.reference(a)[0] == "ok" for a in rows), g


def test_corpus_has_no_duplicates() -> None:
    addrs = [a for a, _, _ in E.CORPUS]
    assert len(set(addrs)) == len(addrs)


def test_headline_findings() -> None:
    r = E.study()
    assert r["tiers"] == {"ok": 36, "grey": 5, "bad": 27}
    assert r["perfect"] == []
    assert len(r["newline_accepted_by"]) == 9 and "email-validator 2.3" not in r["newline_accepted_by"]
    assert r["bad_accepted_by_k"]["a" * 65 + "@example.com"] == 10
    lib = r["per"]["email-validator 2.3"]
    assert len(lib["false_reject"]) == 3 and len(lib["false_accept"]) == 1


def test_a_stack_rejects_the_union() -> None:
    r = E.study()
    union = set()
    for n in r["stack"]:
        union |= set(r["per"][n]["false_reject"])
    assert set(r["stack_false_reject"]) == union


def test_signups_lost_responds_to_the_mix() -> None:
    only_apostrophe = {g: (1.0 if g == "apostrophe" else 0.0) for g in E.SHARE}
    assert E.score("tutorial regex", only_apostrophe)["signups_lost"] == pytest.approx(1.0)
    assert E.score("email-validator 2.3", only_apostrophe)["signups_lost"] == 0.0
