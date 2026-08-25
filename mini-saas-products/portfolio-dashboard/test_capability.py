"""The capability map, asserted.

The old tool shipped without tests. These pin the taxonomy contract it now
depends on - if `build_site.py` grows a task or renames one, this fails rather
than the app rendering an empty tab.
"""

from __future__ import annotations

import pytest

import capability as C

TOOLS = C.load()


def test_catalog_loads_and_every_tool_is_classified():
    assert len(TOOLS) > 150
    assert C.unclassified(TOOLS) == []
    assert all(t.slug and t.name and t.task for t in TOOLS)


def test_every_tool_has_the_problem_it_was_built_for():
    """The problem line is the one column people read."""
    missing = [t.slug for t in TOOLS if not t.problem]
    assert missing == []


def test_task_order_matches_the_catalog_exactly():
    """Display order here must not drift from the taxonomy in build_site.py."""
    in_catalog = {t.task for t in TOOLS}
    assert in_catalog == set(C.TASK_ORDER)
    assert len(C.TASK_ORDER) == len(set(C.TASK_ORDER)) == 13


def test_every_task_has_at_least_one_tool():
    grouped = C.by_task(TOOLS)
    empty = [t for t in C.TASK_ORDER if not grouped[t]]
    assert empty == []


def test_grouping_partitions_the_catalog():
    grouped = C.by_task(TOOLS)
    assert sum(len(v) for v in grouped.values()) == len(TOOLS)
    seen = [t.slug for group in grouped.values() for t in group]
    assert len(seen) == len(set(seen))


def test_groups_are_sorted_by_name():
    for task, group in C.by_task(TOOLS).items():
        names = [t.name.lower() for t in group]
        assert names == sorted(names), task


# --------------------------------------------------------------------------
# Roles and scenarios reference real tasks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("role,_blurb,tasks", C.ROLES)
def test_every_role_maps_to_real_tasks_and_finds_tools(role, _blurb, tasks):
    assert set(tasks) <= set(C.TASK_ORDER), role
    tools = C.for_role(role, TOOLS)
    assert tools, role
    assert {t.task for t in tools} == set(tasks)


@pytest.mark.parametrize("name,_blurb,tasks", C.SCENARIOS)
def test_every_scenario_maps_to_real_tasks_in_order(name, _blurb, tasks):
    assert set(tasks) <= set(C.TASK_ORDER), name
    steps = C.for_scenario(name, TOOLS)
    assert [s[0] for s in steps] == list(tasks)
    assert all(step_tools for _t, step_tools in steps), name


def test_every_task_is_reachable_from_some_role():
    """A job no role owns would be invisible in the role tab."""
    covered = {t for _n, _b, tasks in C.ROLES for t in tasks}
    assert covered == set(C.TASK_ORDER)


def test_roles_overlap_where_the_handoffs_are():
    overlap = C.role_overlap(TOOLS)
    assert overlap[("Analytics engineer", "Data engineer")] > 0
    # The two warehouse-side roles share more than either shares with the
    # model-side one; that is the shape the handoff table is showing.
    ae_de = overlap[("Analytics engineer", "Data engineer")]
    ae_ml = overlap.get(("Analytics engineer", "ML / AI engineer"), 0)
    assert ae_de > ae_ml


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_search_matches_the_problem_text_not_just_the_name():
    hits = C.search("stale", TOOLS)
    assert any(t.slug == "data-freshness-monitor" for t in hits)
    # 'stale' appears in problem lines whose names do not contain it.
    assert any("stale" not in t.name.lower() for t in hits)


def test_search_is_case_insensitive_and_empty_returns_everything():
    assert len(C.search("", TOOLS)) == len(TOOLS)
    assert C.search("CRLF", TOOLS) == C.search("crlf", TOOLS)


def test_search_finds_nothing_for_nonsense():
    assert C.search("zzzznotathing", TOOLS) == []


# --------------------------------------------------------------------------
# The tombstone
# --------------------------------------------------------------------------


def test_the_old_tracker_parser_points_somewhere():
    """Removed, not vanished: importing an old name explains where it went."""
    import tracker_parser

    with pytest.raises(AttributeError, match="capability.load"):
        _ = tracker_parser.portfolio_stats


def test_missing_catalog_raises_something_actionable():
    with pytest.raises(C.CatalogMissing, match="build_site.py"):
        C.load.__wrapped__("/nonexistent/catalog.json")
