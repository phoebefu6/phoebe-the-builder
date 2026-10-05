"""Engine tests. Each checker is checked on hand-built claims, and the enumeration on its own arithmetic."""

from __future__ import annotations

import citation as T
import numpy as np

CTX = " ".join(T.PASSAGES)


def test_passages_are_true_and_self_supported() -> None:
    for i, t in enumerate(T.TUPLES):
        r = T.build(i, "none", "source", False)
        assert r["supported"] and r["true"] and all(r["checks"].values())


def test_citation_marker_is_not_content() -> None:
    """The defect this build shipped first: [1] read as the number 1 failed every cited claim."""
    assert T.numbers("Pro plan includes 5 seats. [1]") == {"5"}
    assert T.coverage("Pro plan includes 5 seats. [1]", T.PASSAGES[3]) == 1.0


def test_polarity_flip_is_lexically_identical_to_its_source() -> None:
    r = T.build(3, "polarity", "source", False)
    assert r["cov"] == 1.0 and r["checks"]["cited_lexical+numbers"] and not r["checks"]["cited_lexical+numbers+negation"]
    assert not r["supported"]


def test_neighbours_number_is_in_the_context_but_not_the_cited_passage() -> None:
    r = T.build(5, "value_near", "source", False)   # Pro refunds 30 -> Enterprise's 60
    assert "60 days" in r["claim"]
    assert r["checks"]["context_lexical"] and not r["checks"]["cited_lexical+numbers"]
    far = T.build(5, "value_far", "source", False)
    assert not far["checks"]["context_lexical"]


def test_entity_swap_can_produce_a_true_but_misattributed_claim() -> None:
    r = T.build(11, "entity", "source", False)      # Add-on packs 48h -> Basic plan 48h, which exists as [9]
    assert r["true"] and not r["supported"] and r["klass"] == "misattributed"
    other = T.build(8, "entity", "source", False)   # Basic 48h -> Pro 48h, which is false
    assert not other["true"] and other["klass"] == "entity"


def test_scope_and_entity_pass_every_lexical_rule() -> None:
    for edit in ("scope", "entity"):
        r = T.build(5, edit, "source", False)
        assert all(r["checks"].values()) and not r["supported"]


def test_enumeration_is_a_distribution_and_nothing_supported_fails_at_tau() -> None:
    for m in T.MODELS.values():
        rows = T.enumerate_claims(m)
        assert len(rows) == len(T.FACTS) * len(T.EDITS) * len(T.CITES) * 2
        assert abs(sum(r["p"] for r in rows) - 1) < 1e-12
        s = T.summarise(rows)
        assert all(v["false_fail"] == 0.0 for v in s["scorers"].values())
        assert s["p_supported"] < s["scorers"]["cited_lexical+numbers+negation"]["reported"]


def test_sweep_is_monotone_and_never_both_low() -> None:
    sw = T.threshold_sweep(T.MODELS["v1 cites when sure"], [0.3, 0.5, 0.7, 0.9, 1.0])
    fails = [r["fails_supported_paraphrase"] for r in sw]
    passes = [r["passes_polarity_flip"] for r in sw]
    assert fails == sorted(fails) and passes == sorted(passes, reverse=True)
    assert all(f + p >= 0.6 for f, p in zip(fails, passes))


def test_simulation_check_can_fail() -> None:
    """v2's simulated support must land outside v1's exact value, or the calibration proves nothing."""
    v1 = T.summarise(T.enumerate_claims(T.MODELS["v1 cites when sure"]))["p_supported"]
    mc = T.simulate(T.MODELS["v2 always cite"], 20_000, seed=3)["supported"]
    lo, hi = T.wilson(mc, 20_000)
    assert not lo <= v1 <= hi


def test_wilson_is_centred_and_shrinks() -> None:
    lo, hi = T.wilson(0.5, 1000)
    assert lo < 0.5 < hi and (hi - lo) > (T.wilson(0.5, 100000)[1] - T.wilson(0.5, 100000)[0])
    assert np.isclose(sum(T.wilson(0.5, 1000)) / 2, 0.5)


def test_parse_claims_rejects_bad_input() -> None:
    import pytest

    with pytest.raises(ValueError):
        T.parse_claims("", "x [1]")
    with pytest.raises(ValueError):
        T.parse_claims("[1] a", "x [2]")
    with pytest.raises(ValueError):
        T.parse_claims("[1] a", "")
    ps, cs = T.parse_claims("[1] a\n[2] b", "x [2]\ny")
    assert ps == ["a", "b"] and cs == [("x [2]", 1), ("y", None)]
