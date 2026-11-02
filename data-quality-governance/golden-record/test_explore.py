"""Engine checks: each rule-set's known behaviour on a hand-built cluster, plus the study's headline."""

from __future__ import annotations

import json

import explore as E


def rec(source: str, day: int, **v: object) -> E.Record:
    r: E.Record = {"source": source, "updated_at": day, "_ts": {f: day for f in E.FIELDS}}
    r.update({f: v.get(f) for f in E.FIELDS})
    return r


OLD, NEW = "57 Lantern Way", "134 Mill Street"
CLUSTER = [
    rec("crm", 100, name="Robert Petrov", dob="1960-01-14", email="r@a.test", address=OLD),
    rec("web", 800, name="Rob Petrov", email="r@a.test", phone="+44 7700 900001", address=NEW),
    rec("billing", 500, name="ROBERT PETROV", email="accounts@x.test", phone="+44 20 7946 0001", address=OLD),
]


def test_each_rule_has_its_known_bias() -> None:
    assert E.most_recent(CLUSTER, "name") == "Rob Petrov"  # recency keeps the nickname
    assert E.source_priority(CLUSTER, "address") == OLD  # priority keeps CRM's old address
    assert E.majority(CLUSTER, "address") == OLD  # two stale copies outvote one fresh
    assert E.longest(CLUSTER, "email") == "accounts@x.test"
    assert E.per_field(CLUSTER, "name") == "Robert Petrov" and E.per_field(CLUSTER, "address") == NEW


def test_nulls_never_survive_when_a_value_exists() -> None:
    for k, rule in E.RULES.items():
        assert rule(CLUSTER, "dob") == "1960-01-14", k
        assert rule(CLUSTER, "phone") is not None, k
    assert E.most_recent([rec("crm", 1), rec("web", 2)], "dob") is None


def test_row_touch_misleads_recency_but_not_field_timestamp() -> None:
    touched = [dict(CLUSTER[0], updated_at=900), CLUSTER[1]]
    assert E.most_recent(touched, "address") == OLD
    assert E.most_recent_field_ts(touched, "address") == NEW


def test_lineage_sees_through_the_sync_stamp() -> None:
    synced = dict(CLUSTER[2], _ts={f: 900 for f in E.FIELDS}, _lineage=100)
    c = [CLUSTER[0], CLUSTER[1], synced]
    assert E.most_recent_field_ts(c, "address") == OLD
    assert E.most_recent_field_ts(c, "address", lineage=True) == NEW


def test_world_is_seeded_and_every_cluster_has_two_records() -> None:
    a, b = E.make_world(), E.make_world()
    assert a == b and all(len(c["records"]) >= 2 for c in a)


def test_no_rule_beats_the_ceiling_and_headline_holds() -> None:
    r = json.load(open("results.json")) if __import__("os").path.exists("results.json") else E.study()
    for k in E.RULES:
        for f in E.FIELDS:
            assert r["acc"][k][f] <= r["ceiling"][f] + 1e-12
        assert r["whole"][k] <= r["oracle_whole"] + 1e-12
    assert r == json.loads(json.dumps(E.study()))  # results.json is current
