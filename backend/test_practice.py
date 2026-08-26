"""Tests for the practice lines that ship with the graph.

These read data/practice.json, which is committed, so they are checking the
lines a student will actually be shown rather than anything generated at run
time. No API calls.
"""

import json

import practice
from graph import SKILLS


def committed():
    return practice.load()


def test_every_skill_has_a_line():
    """A skill added to the graph without one shows a student a bare label.

    This is the failure that matters most: the walk will happily diagnose a new
    skill, and the report would then name it and say nothing about what to do.
    """
    missing = sorted(s for s in SKILLS if s not in committed())
    assert not missing, f"no practice line for: {', '.join(missing)}"


def test_no_line_is_for_a_skill_that_no_longer_exists():
    stale = sorted(s for s in committed() if s not in SKILLS)
    assert not stale, f"practice lines for skills not in the graph: {stale}"


def test_no_line_is_malformed():
    """Structured output has leaked the model's own working into these before -
    a fragment of JSON, and once the words "No wait" mid-sentence."""
    broken = {s: why for s, line in committed().items() if (why := practice.looks_wrong(line))}
    assert not broken, f"malformed practice lines: {broken}"


def test_lines_are_short():
    """One or two sentences. Three is teaching, which is not this page's job."""
    long = {s: len(line) for s, line in committed().items() if len(line) > 340}
    assert not long, f"practice lines far too long: {long}"


def test_lines_are_plain_ascii():
    """The student sees the raw text, and the maths converter in front of it
    only understands ASCII notation."""
    odd = {
        s: [c for c in line if ord(c) > 127]
        for s, line in committed().items()
        if any(ord(c) > 127 for c in line)
    }
    assert not odd, f"non-ASCII in practice lines: {odd}"


def test_the_file_is_sorted_so_a_diff_is_readable():
    raw = json.loads(practice.PRACTICE_FILE.read_text(encoding="utf-8"))
    assert list(raw) == sorted(raw)


def test_a_missing_line_does_not_raise():
    """A report must render even for a skill nothing has been written for."""
    assert practice.line_for("no_such_skill") is None


def test_tidy_strips_what_gets_welded_on_the_end():
    assert practice._tidy('Practise this. Search "thing GCSE".{}') == \
        'Practise this. Search "thing GCSE".'
    assert practice._tidy('Explain it out loud.  ') == "Explain it out loud."


def test_scratch_text_is_caught():
    assert practice.looks_wrong('Fine sentence. Search "x".\',\'fix\':1}]}')
    assert practice.looks_wrong("Say what 34.host No wait, use 34.56 is worth.")
    assert practice.looks_wrong("") == "empty"


def test_a_good_line_passes():
    assert practice.looks_wrong(
        'Practise expanding two brackets, like (x + 3)(x - 2) = x^2 + x - 6. '
        'Search "expanding double brackets GCSE".'
    ) is None


def test_a_definite_integral_is_not_mistaken_for_scratch_text():
    """[x^3/3] is how the notation is written, and half the integration lines
    carry one. An earlier version of the check rejected all of them."""
    assert practice.looks_wrong(
        'Practise evaluating [x^3/3] between limits. Search "definite integrals".'
    ) is None
