"""What rests on a skill, and the claim the report screen makes about it.

The number on the report is the one thing on that page a head of department
can act on - "one hour of teaching unblocks nineteen topics" is a business
case, and "the gap is fractions" is not. It is also the easiest number on the
page to get quietly wrong, because a transitive count over a hand-written
graph has no obvious right answer to eyeball. So the shape of the traversal is
pinned on graphs small enough to count by hand, and the real graph is only
checked for the properties that make the sentence on the page true.
"""

import pathlib

import pytest

from graph import (
    LOAD_BEARING,
    SKILLS,
    Skill,
    reach,
    validate,
    what_rests_on,
)

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"


def _graph(*rows) -> dict[str, Skill]:
    """A tiny graph from (id, needs, topic) triples."""
    skills = {}
    for skill_id, needs, topic in rows:
        skills[skill_id] = Skill(
            id=skill_id,
            name=skill_id.replace("_", " "),
            level=0,
            kind="skill",
            probe="?",
            needs=tuple(needs),
            topic=topic,
        )
    return validate(skills)


# A chain with a fork at the top:
#
#   top_a   top_b        <- both need middle
#       \   /
#       middle           <- needs floor
#         |
#       floor
CHAIN = (
    ("top_a", ("middle",), "Topic A"),
    ("top_b", ("middle",), "Topic B"),
    ("middle", ("floor",), None),
    ("floor", (), None),
)


# ---- the traversal -------------------------------------------------------


def test_the_whole_fan_out_is_counted_not_just_the_step_above():
    """The reason for the number: three things rest on the floor, and only one
    of them touches it directly. Counting immediate parents would say 1."""
    skills = _graph(*CHAIN)
    assert what_rests_on("floor", skills) == {"middle", "top_a", "top_b"}


def test_a_skill_does_not_rest_on_itself():
    skills = _graph(*CHAIN)
    assert "floor" not in what_rests_on("floor", skills)


def test_nothing_rests_on_the_top_of_the_graph():
    skills = _graph(*CHAIN)
    assert what_rests_on("top_a", skills) == set()


def test_a_skill_reached_two_ways_is_counted_once():
    """Diamond: both sides of the fork need the floor, and the thing above
    them needs both. It is one skill blocked, not two."""
    skills = _graph(
        ("roof", ("left", "right"), "Topic A"),
        ("left", ("floor",), None),
        ("right", ("floor",), None),
        ("floor", (), None),
    )
    assert what_rests_on("floor", skills) == {"left", "right", "roof"}


def test_a_skill_that_is_not_in_the_graph_blocks_nothing():
    """A typo must come back empty rather than raising on the one screen a
    student is already waiting on."""
    assert what_rests_on("no_such_skill", _graph(*CHAIN)) == set()
    assert reach("no_such_skill", _graph(*CHAIN))["skills"] == 0


# ---- what the report is handed -------------------------------------------


def test_reach_counts_topics_not_steps():
    """Topics are what a department recognises, and they only sit on the
    skills a paper can ask for. middle carries no topic, so it is a blocked
    skill and not a blocked kind of question."""
    found = reach("floor", _graph(*CHAIN))
    assert found["skills"] == 3
    assert found["topics"] == ["Topic A", "Topic B"]


def test_reach_says_how_big_the_map_is():
    """The page says "3 of the 4 skills we map", so the denominator has to
    come from the same place as the numerator."""
    skills = _graph(*CHAIN)
    assert reach("floor", skills)["of"] == len(skills) == 4


def test_a_small_fan_out_is_not_called_load_bearing():
    assert reach("floor", _graph(*CHAIN))["load_bearing"] is False


def test_the_load_bearing_line_is_drawn_where_the_graph_says():
    """Built to order: one floor with exactly LOAD_BEARING things on it is
    load-bearing, and one short of it is not."""
    def tower(height):
        rows = [("floor", (), None)]
        # One of them carries a topic, because a graph where nothing does is
        # a broken file rather than a small one.
        rows += [(f"s{n}", ("floor",), "Topic A" if n == 0 else None)
                 for n in range(height)]
        return _graph(*rows)

    assert reach("floor", tower(LOAD_BEARING))["load_bearing"] is True
    assert reach("floor", tower(LOAD_BEARING - 1))["load_bearing"] is False


# ---- the real graph ------------------------------------------------------


def test_every_real_skill_can_be_counted():
    """A dangling id or a cycle would make this hang or raise, and the graph
    check at import is what stops that - this is the proof it holds for the
    traversal going the other way too."""
    for skill_id in SKILLS:
        found = reach(skill_id)
        assert found["skills"] >= 0
        assert found["of"] == len(SKILLS)


def test_the_real_graph_has_both_kinds_of_gap():
    """The whole point of the line is that it distinguishes. A graph where
    everything is load-bearing, or nothing is, makes it noise."""
    heavy = [s for s in SKILLS if reach(s)["load_bearing"]]
    light = [s for s in SKILLS if not reach(s)["load_bearing"]]
    assert len(heavy) >= 5, "nothing is load-bearing, so the line says nothing"
    assert len(light) >= 20, "everything is load-bearing, so the line says nothing"


def test_a_blocked_skill_really_does_need_the_gap_somewhere_below_it():
    """Spot-check the claim rather than the count: walk back down from each
    blocked skill and the gap has to be reachable through needs."""
    gap = max(SKILLS, key=lambda s: reach(s)["skills"])

    for blocked in what_rests_on(gap):
        seen, queue = set(), [blocked]
        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(SKILLS[current].needs)
        assert gap in seen, f"{blocked} does not actually need {gap}"


# ---- the page ------------------------------------------------------------


def test_the_report_hands_the_page_what_it_needs():
    """The field name and the keys inside it, because the page reads them by
    name and a rename shows up as a missing line rather than an error."""
    import api

    found = api.reach(next(iter(SKILLS)))
    assert set(found) == {"skills", "of", "topics", "of_topics", "load_bearing"}


@pytest.mark.parametrize("name", ["index.html"])
def test_the_page_reads_the_field_the_server_sends(name):
    page = (WEB / name).read_text(encoding="utf-8")
    assert "rests_on_it" in page
    assert "costLine(named)" in page
    assert "load_bearing" in page


def test_the_page_says_nothing_when_nothing_rests_on_the_gap():
    """A skill at the top of the graph has nothing standing on it. Inventing
    a consequence for it would be a false claim on the one screen that has to
    be trusted, so the guard is pinned."""
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert "if (!rests || !rests.skills) return '';" in page


def test_the_page_can_tell_some_topics_from_all_of_them():
    """"Nineteen kinds of question" is read as a lot; "every kind of question
    we cover" is read as the whole subject, which is what it means. The page
    can only draw that distinction if it is handed the denominator."""
    from graph import topics

    gap = max(SKILLS, key=lambda s: len(reach(s)["topics"]))
    found = reach(gap)
    assert found["of_topics"] == len(topics())
    assert len(found["topics"]) <= found["of_topics"]


def test_the_page_has_the_sentence_for_a_gap_under_the_whole_subject():
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert "every kind of exam question we cover" in page
    assert "of_topics" in page
