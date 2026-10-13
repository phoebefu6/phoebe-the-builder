"""Tests for the exact fairness engine. Each property is one a plausible bug would break."""

from __future__ import annotations

import fairness as F
import pytest


def test_rates_are_probabilities_and_selection_decomposes() -> None:
    for g in F.GROUPS:
        for t in (-1.0, 0.0, 0.79, 2.0):
            r = F.rates(g, t)
            for k in ("tpr", "fpr", "sel", "ppv", "acc"):
                assert 0 <= r[k] <= 1, (g, t, k)
            p = F.WORLD[g]["base"]
            assert abs(r["sel"] - (p * r["tpr"] + (1 - p) * r["fpr"])) < 1e-12


def test_chouldechova_identity_holds_at_every_threshold() -> None:
    """FPR = p/(1-p) * (1-PPV)/PPV * TPR: the algebra the impossibility rests on."""
    for g in F.GROUPS:
        p = F.WORLD[g]["base"]
        for t in (-1.5, -0.2, 0.79, 1.7):
            r = F.rates(g, t)
            assert abs(r["fpr"] - p / (1 - p) * (1 - r["ppv"]) / r["ppv"] * r["tpr"]) < 1e-12


def test_auc_closed_form_and_pooled_between_groups() -> None:
    assert F.auc("A") == pytest.approx(0.871, abs=0.001)
    assert F.auc("B") == pytest.approx(0.760, abs=0.001)
    assert F.auc("B") < F.pooled_auc() < F.auc("A")


def test_identical_groups_make_every_definition_hold_at_one_threshold() -> None:
    """The impossibility needs different base rates: with identical groups one threshold satisfies all three."""
    same = F.parse_world("B.base=0.40, B.mu_q=1.2, B.mu_u=-0.4")
    a, b = F.rates("A", 0.79, same), F.rates("B", 0.79, same)
    for m in ("sel", "tpr", "fpr", "ppv"):
        assert abs(a[m] - b[m]) < 1e-12


def test_each_policy_meets_its_own_definition() -> None:
    P = F.policies()
    assert P["demographic parity"]["overall"]["impact_ratio"] > 0.98
    assert abs(P["equal opportunity (TPR)"]["gaps"]["tpr"]) < 0.01
    assert abs(P["predictive parity (PPV)"]["gaps"]["ppv"]) < 0.01
    eo = P["equalized odds (TPR+FPR)"]["gaps"]
    assert abs(eo["tpr"]) < 0.004 and abs(eo["fpr"]) < 1e-9


def test_headline_numbers() -> None:
    P = F.policies()
    one = P["one threshold (max accuracy)"]
    assert one["overall"]["impact_ratio"] == pytest.approx(0.657, abs=0.001)
    assert one["per"]["A"]["tpr"] == pytest.approx(0.659, abs=0.001)
    assert one["per"]["B"]["tpr"] == pytest.approx(0.464, abs=0.001)
    eo = P["equalized odds (TPR+FPR)"]
    assert eo["overall"]["acc"] - one["overall"]["acc"] == pytest.approx(-0.063, abs=0.002)
    assert eo["per"]["A"]["tpr"] < one["per"]["A"]["tpr"] - 0.15, "equalized odds levels A down"
    pp = P["predictive parity (PPV)"]
    assert pp["per"]["B"]["sel"] < 0.05 and pp["overall"]["acc"] >= one["overall"]["acc"]


def test_no_threshold_pair_meets_all_three() -> None:
    imp = F.impossibility()
    assert imp["pairs"] > 50_000 and imp["all_three"] == 0 and imp["two_of_three"] > 0
    assert imp["closest"]["worst_gap"] > 0.10


def test_impossibility_check_can_pass() -> None:
    """The checker must find pairs when the theorem's premise fails (equal base rates, equal score quality)."""
    same = F.parse_world("B.base=0.40, B.mu_q=1.2, B.mu_u=-0.4")
    assert F.impossibility(same)["all_three"] > 0


def test_audit_agreement_rises_with_sample_size() -> None:
    au = F.audit_power(0.79, [25, 100, 500, 2000])
    ag = [r["agree"] for r in au]
    assert ag == sorted(ag)
    assert ag[1] == pytest.approx(0.845, abs=0.02)
    assert ag[-1] > 0.99


def test_data_fix_passes_four_fifths_and_raises_accuracy_but_keeps_ppv_gap() -> None:
    fix = F.data_fix()
    one0 = F.policies()["one threshold (max accuracy)"]
    one1 = fix["policies"]["one threshold (max accuracy)"]
    assert fix["auc_B"] == pytest.approx(F.auc("A"), abs=1e-9)
    assert one1["overall"]["impact_ratio"] >= 0.8 > one0["overall"]["impact_ratio"]
    assert one1["overall"]["acc"] > one0["overall"]["acc"]
    assert one1["gaps"]["ppv"] > 0.10, "base rates still differ, so precision still differs"
    assert fix["policies"]["equal opportunity (TPR)"]["overall"]["impact_ratio"] < 0.8


def test_simulation_matches_exact_inside_wilson() -> None:
    cal = F.calibrate()
    assert len(cal) == 40 and all(r["inside"] for r in cal)


def test_wilson_check_can_fail() -> None:
    pol = F.policies()["one threshold (max accuracy)"]
    k, n = F.simulate(pol)["B"]["tpr"]
    lo, hi = F.wilson(k, n)
    ex = pol["per"]["B"]["tpr"]
    assert lo <= ex <= hi and not (lo <= ex + 0.02 <= hi)


@pytest.mark.parametrize("bad", ["nope", "C.base=0.3", "B.base=x", "B.base=1.2", "A.share=0.9"])
def test_parse_world_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        F.parse_world(bad)
