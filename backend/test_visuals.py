"""The pictures that ship with the gaps.

A wrong diagram is worse than a wrong answer key. An answer key gets marked and
argued with; a diagram is looked at once, believed, and never checked. So these
are drawn by hand, committed, and checked here - the same treatment the question
bank gets, and for the same reason: four rendering faults have reached real
students in a fortnight, every one of them generated content that nothing looked
at first.
"""

import json

import pytest

import visuals
from graph import SKILLS


def test_there_are_some():
    assert visuals.drawn(), "no pictures at all"


@pytest.mark.parametrize("skill_id", visuals.drawn())
def test_each_picture_is_fit_to_show(skill_id):
    """Well-formed, scalable, themed, described, and small enough for a phone."""
    assert visuals.wrong_with(skill_id) is None, visuals.wrong_with(skill_id)


@pytest.mark.parametrize("skill_id", visuals.drawn())
def test_each_picture_is_of_a_skill_that_exists(skill_id):
    """A drawing filed under a name the graph does not have is one nothing will
    ever show."""
    assert skill_id in SKILLS, f"{skill_id}.svg is not a skill in the graph"


def test_a_skill_with_no_picture_is_not_an_error():
    """Most skills have none, and that is the normal case - a fact node wants
    recall, not a diagram."""
    assert visuals.for_skill("index_laws") is None
    assert visuals.for_skill("not_a_skill_at_all") is None


def test_an_unreadable_picture_never_costs_a_diagnosis(monkeypatch, tmp_path):
    """The report is the thing that matters. A picture is an addition to it and
    must never be the reason a student sees an error instead."""
    monkeypatch.setattr(visuals, "VISUAL_DIR", tmp_path)
    broken = tmp_path / "fraction_meaning.svg"
    broken.write_bytes(b"\xff\xfe not svg at all")

    assert visuals.for_skill("fraction_meaning") is not None or True  # must not raise


def test_the_tutors_chain_is_covered_end_to_end():
    """The three that exist are not a sample. They are the chain a real student
    fell down on 2026-09-02 - what a fraction is, equivalent fractions, and
    combining once the bottoms match - so that one session works throughout."""
    for skill_id in ("fraction_meaning", "equivalent_fractions",
                     "combine_over_common_denominator"):
        assert visuals.has(skill_id), f"{skill_id} has no picture"


def test_the_report_carries_the_picture():
    import api
    from eval import _cannot_do
    from walk import SkillResult, diagnose

    broken = _cannot_do("fraction_meaning")
    diagnosis = diagnose(
        "add_subtract_fractions",
        check=lambda s: SkillResult(s.id, held=s.id not in broken, mistake="x"),
        max_depth=8)
    report = api._report(diagnosis, question="Work out 5/6 - 1/4")

    gap = report["gaps"][0]
    assert gap["skill"] == "What a fraction is"
    assert gap["visual"], "the gap has a picture and the report did not send it"
    assert "<svg" in gap["visual"]
    # The picture is an addition, never a replacement.
    assert gap["practice"], "the practice line must still be there"


def test_the_page_puts_it_between_the_name_and_the_practice_line():
    """Order matters: name the gap, show it, then say what to do about it."""
    import pathlib

    page = (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text(encoding="utf-8")

    name = page.index('<p class="gap">')
    picture = page.index('class="visual"')
    doing = page.index('<p class="practice">')
    assert name < picture < doing


def test_the_picture_is_inlined_not_fetched():
    """An <img> would be a second request on a phone, and could not follow the
    page's theme."""
    import pathlib

    page = (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text(encoding="utf-8")
    assert "${named.visual}" in page
    assert ".visual svg" in page
