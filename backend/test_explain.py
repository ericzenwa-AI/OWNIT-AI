"""The explanations that ship with the graph.

These read data/explanations.json, which is committed, so they check what a
student will actually be shown rather than anything generated at run time. No
API calls.
"""

import json

import pytest

import explain
from graph import SKILLS


def written():
    return explain.load()


def test_every_skill_has_one():
    """A skill with no explanation is one the report can name and not explain,
    which is the state the whole product was in until today."""
    missing = sorted(s for s in SKILLS if s not in written())
    assert not missing, f"no explanation for: {', '.join(missing)}"


def test_none_is_for_a_skill_that_no_longer_exists():
    stale = sorted(s for s in written() if s not in SKILLS)
    assert not stale, f"explanations for skills not in the graph: {stale}"


def test_none_is_malformed():
    broken = {s: why for s, one in written().items() if (why := explain.looks_wrong(one))}
    assert not broken, f"malformed explanations: {broken}"


@pytest.mark.parametrize("field", ["what", "example", "wrong"])
def test_no_field_is_empty(field):
    """The bug that shipped six empty examples: _tidy trimmed backwards to the
    last sentence end, and an example that is pure maths has none, so every
    character was stripped."""
    empty = sorted(s for s, one in written().items() if not (one.get(field) or "").strip())
    assert not empty, f"{field} is empty for: {empty}"


def test_tidy_never_empties_a_field():
    """The fix, pinned. Trimming junk off the end must not be able to take the
    whole field with it."""
    assert explain._tidy("d/dx(3x^5) = 15x^4") == "d/dx(3x^5) = 15x^4"
    assert explain._tidy("5/8 means cut into 8 and take 5") == "5/8 means cut into 8 and take 5"
    # It still does the job it was written for.
    assert explain._tidy('Search "a thing".{}') == 'Search "a thing".'


def test_the_file_is_sorted_so_a_diff_is_readable():
    raw = json.loads(explain.EXPLAIN_FILE.read_text(encoding="utf-8"))
    assert list(raw) == sorted(raw)


def test_a_missing_explanation_does_not_raise():
    assert explain.for_skill("no_such_skill") is None


def test_the_report_explains_the_gap_and_every_step_of_the_route():
    """The point of the whole thing: a student can read what the gap is, and
    read every step between it and their own question."""
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
    assert gap["explanation"], "the gap itself must explain what it is"
    assert gap["explanation"]["what"]
    assert gap["explanation"]["wrong"]

    chain = gap["chain"]
    assert chain, "there is a route back up"
    for step in chain:
        assert step["id"] in SKILLS
        assert step["explanation"], f"{step['id']} on the route has no explanation"


def test_the_route_is_openable_on_the_page():
    import pathlib

    page = (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text(encoding="utf-8")
    assert "function openStep(" in page
    assert "function stepFor(" in page
    assert 'class="step"' in page
    # Everything needed came down with the report, so opening one costs nothing
    # and works with the connection dropped.
    opener = page[page.index("function openStep("):page.index("function stepFor(")]
    assert "fetch(" not in opener and "post(" not in opener


def test_what_goes_wrong_is_shown_not_just_stored():
    """The beat nothing else in the product has. It would be easy to generate it
    and never put it on screen."""
    import pathlib

    page = (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text(encoding="utf-8")
    assert page.count("What usually goes wrong") >= 2, (
        "shown for the gap and for an opened step")
