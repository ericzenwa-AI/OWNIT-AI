"""The two beats the finding page opens on, and the order it tells them in.

A student reaching that screen has just been told they are missing something.
What they need first is not a correct definition - it is the moment the name
stops being a label and starts meaning something. So the page leads with
`plain`, in words with no maths in them, then `like`, which puts the idea next
to something that happens in real life, and only then the accurate version.

Two things are worth more than the writing itself and both are tested here:
that `plain` really has no notation in it, and that `like` is allowed to be
empty. A comparison that nearly fits is worse than no comparison, because a
student keeps it and reasons with it on exactly the harder questions where it
breaks.
"""

import pathlib
import re

import pytest

import explain
from graph import SKILLS


def written():
    return explain.load()


def page():
    return (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text(encoding="utf-8")


def finding():
    """Just the finding card, not the whole file."""
    text = page()
    return text[text.index("function report("):text.index("function usefulCard(")]


# ---- what is written -------------------------------------------------------


def test_every_explanation_has_a_plain_line():
    missing = sorted(s for s, one in written().items() if explain.beats_missing(one))
    assert not missing, (
        "no plain-words line for: " + ", ".join(missing)
        + " - run `python backend/explain.py --beats`")


def test_no_plain_line_carries_maths_notation():
    """The beat exists for the student who cannot read the notation. Putting
    notation in it hands them the same wall a second time."""
    carrying = {}
    for skill_id, one in written().items():
        found = [c for c in ("^", "=", "<", ">", "*") if c in (one.get("plain") or "")]
        if found:
            carrying[skill_id] = found
    assert not carrying, f"notation in the plain line: {carrying}"


def test_a_real_life_comparison_is_allowed_to_be_absent():
    """Some things are not like anything. Forcing one teaches a wrong model,
    so an empty `like` has to stay a legal answer - and in practice a good
    number of them are, which is the evidence nothing was invented to fill a
    box."""
    have = written()
    empty = [s for s, one in have.items() if not (one.get("like") or "").strip()]
    assert empty, (
        "every single skill got a real-life comparison, which means they are "
        "being invented rather than found")


def test_the_comparisons_that_exist_are_not_one_liners():
    """A comparison has to say which part of the real thing is which part of
    the maths. "Like a pizza" on its own is decoration."""
    thin = {s: one["like"] for s, one in written().items()
            if (one.get("like") or "").strip() and len(one["like"]) < 40}
    assert not thin, f"comparisons too short to map onto anything: {thin}"


@pytest.mark.parametrize("field", ["plain", "like"])
def test_neither_beat_is_malformed(field):
    """Same guard as the other three: no scratch text, no non-ASCII, no wall
    of text on a phone."""
    broken = {}
    for skill_id, one in written().items():
        body = one.get(field)
        if not body:
            continue
        if any(ord(c) > 127 for c in body):
            broken[skill_id] = "not plain ASCII"
        elif explain.SCRATCH.search(body):
            broken[skill_id] = "carries scratch text"
        elif len(body) > 360:
            broken[skill_id] = f"too long at {len(body)}"
    assert not broken, f"malformed {field}: {broken}"


def test_the_guard_refuses_notation_in_the_plain_line():
    bad = {"what": "a", "example": "b", "wrong": "c",
           "plain": "It means x^2 = 4", "like": ""}
    assert "notation" in (explain.looks_wrong(bad) or "")


def test_the_guard_still_passes_an_explanation_with_no_beats_at_all():
    """They were added after 172 were written and checked. An older entry is
    unfinished, not broken - beats_missing is what reports those."""
    old = {"what": "a.", "example": "b.", "wrong": "c."}
    assert explain.looks_wrong(old) is None
    assert explain.beats_missing(old) is True


# ---- the order the page tells it in ----------------------------------------


def test_the_finding_opens_on_the_plain_words_not_the_maths():
    text = finding()
    said = text.index("explanation.plain")
    maths = text.index("explanation.what")
    assert said < maths, (
        "the accurate explanation is written for somebody who already knows; "
        "it must not be the first sentence on this screen")


def test_the_real_life_comparison_comes_before_the_maths_too():
    text = finding()
    assert text.index("explanation.like") < text.index("explanation.what")


def test_the_maths_is_still_all_there():
    """This is a reordering and an addition. Nothing that was on the page
    before it may have been dropped."""
    text = finding()
    for kept in ("explanation.what", "explanation.example", "explanation.wrong",
                 "named.practice", "named.visual"):
        assert kept in text, f"{kept} disappeared from the finding"


def test_an_explanation_with_no_plain_line_still_shows_its_explanation():
    """An older entry has no plain line, and must get the accurate sentence as
    an ordinary paragraph rather than a blank card or an empty toggle."""
    text = finding()
    assert "named.explanation ? `" in text
    assert ": `<p>${mathHtml(named.explanation.what)}</p>`" in text, (
        "the fallback branch must render `what` outright")


def test_the_accurate_sentence_is_tucked_away_only_when_plain_came_first():
    """On a skill whose plain line is already accurate the two say the same
    thing twice, and a page that repeats itself stops being read. Open it when
    there is nothing else; fold it away when there is."""
    text = finding()
    assert 'class="precise"' in text
    assert "Say that more precisely" in text
    assert "named.explanation.plain\n          ? `<details" in text or (
        "named.explanation.plain" in text and "<details" in text)


def test_the_worked_case_is_still_shown_outright():
    """The one thing nobody should have to open a toggle to reach."""
    text = finding()
    worked = text.index('class="worked"')
    assert text.index('class="precise"') < worked
    assert "</details>" in text[:worked], (
        "the worked example must sit outside the fold, not inside it")


def test_the_route_steps_tell_it_in_the_same_order():
    """Somebody opening a step of the route is opening it because they do not
    know what it is either."""
    text = page()
    body = text[text.index("function openStep("):text.index("function stepFor(")]
    assert body.index("said.plain") < body.index("said.what")


def test_the_page_has_somewhere_to_put_them():
    text = page()
    assert ".plain-said {" in text
    assert ".like {" in text
