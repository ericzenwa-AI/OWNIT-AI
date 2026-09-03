"""Doorways that name an act their children do not model.

Found by a tutor watching a student on 2026-09-02. The walk traced a
subtraction question correctly down to what a fraction is and back up through
equivalent fractions and finding a common denominator, and the student still
could not do the question. Nothing had tested the act of combining the
numerators over the shared denominator. Both children held; the question failed
anyway, and 10/12 - 3/12 = 7/24 had nowhere in the graph to live.

It generalises, and it is invisible three times over:

  the walk cannot find it - the act sits at the doorway, and the doorway is
  never asked about, because being stuck on it is the premise;
  the chain cannot show it - the chain is made of nodes;
  the ladder cannot scaffold it - the ladder only climbs steps the graph knows.

A lexical rule was tried first and is recorded here because it FAILED, and the
way it failed is the point. Flagging probes containing "then" or "and simplify"
catches 5 of 70 doorways and misses the founding case completely:
add_subtract_fractions reads "Add or subtract two fractions with different
denominators", which promises nothing and hides everything. No widening of the
word list reaches it. The signal is semantic, so the judgement is semantic, and
it lives in data/assembly_audit.json where it can be read and argued with.
"""

import json
from pathlib import Path

import pytest

from graph import SKILLS, entry_points

AUDIT_FILE = Path(__file__).resolve().parent.parent / "data" / "assembly_audit.json"


def audit():
    return json.loads(AUDIT_FILE.read_text(encoding="utf-8"))


def verdicts():
    return audit()["verdicts"]


# ---- the guard -------------------------------------------------------------


def test_every_doorway_has_been_judged():
    """The one that bites. A doorway nobody has looked at is the exact shape
    that let a student climb the whole chain and still fail the question."""
    doors = {s.id for s in entry_points()}
    unjudged = sorted(doors - set(verdicts()))

    assert not unjudged, (
        "these doorways have never been judged for whether their children cover "
        "the act they name: " + ", ".join(unjudged)
        + " - run scratchpad/audit_assembly.py, or add a verdict by hand to "
        "data/assembly_audit.json saying what the act is and whether a child "
        "covers it")


def test_the_audit_names_no_skill_that_does_not_exist():
    doors = {s.id for s in entry_points()}
    stale = sorted(set(verdicts()) - doors)
    assert not stale, f"verdicts for skills that are not doorways: {stale}"


def test_an_unmodelled_act_says_what_is_missing():
    """A verdict of "not modelled" with no explanation is not evidence, it is
    an accusation."""
    silent = sorted(
        s for s, v in verdicts().items() if not v["modelled"] and not v["missing"])
    assert not silent, f"marked unmodelled with no reason given: {silent}"


# ---- the two that have been acted on ---------------------------------------


def test_the_fraction_case_is_fixed():
    """The one the tutor found."""
    assert "combine_over_common_denominator" in SKILLS["add_subtract_fractions"].needs
    assert verdicts()["add_subtract_fractions"]["modelled"] is True


def test_the_stationary_points_case_is_fixed():
    """Cambridge 9709 March 2025 P1 Q2(b) - candidates differentiated dy/dx
    again instead of setting it to zero."""
    assert "set_derivative_zero" in SKILLS["find_stationary_points"].needs


def test_combining_is_not_hung_off_multiplication():
    """Multiplying fractions never uses a common denominator. That edge would
    be false, and a false link is worse than a missing one - it sends a
    diagnosis somewhere untrue, which is the failure this all exists to fix."""
    assert "combine_over_common_denominator" not in SKILLS["multiply_divide_fractions"].needs
    assert "equivalent_fractions" in SKILLS["multiply_divide_fractions"].needs


@pytest.mark.parametrize(
    "skill_id", ["combine_over_common_denominator", "set_derivative_zero"])
def test_each_new_node_says_where_it_came_from(skill_id):
    """Both were found by someone outside this repository looking at real
    students. That is the only kind of evidence the graph has, and it is worth
    nothing unwritten."""
    note = SKILLS[skill_id].note or ""
    assert note, f"{skill_id} has no note saying where it came from"
    assert any(w in note for w in ("tutor", "Cambridge", "examiner")), note


def test_the_new_probes_test_the_act_alone():
    """A probe the ingredients can answer is not testing the assembly."""
    combining = SKILLS["combine_over_common_denominator"].probe.lower()
    assert "already share" in combining
    assert "must not require finding the denominator" in combining

    zeroing = SKILLS["set_derivative_zero"].probe.lower()
    assert "already been worked out" in zeroing
    assert "not the differentiating" in zeroing


# ---- the backlog is a backlog, not a plan ----------------------------------


def test_the_audit_records_its_own_provenance():
    """Twenty-four of seventy doorways came back unmodelled. That is a third of
    the graph, and a reader a year from now needs to know it came from one
    model in one pass and not from students."""
    text = audit()
    for key in ("_what_this_is", "_how_it_was_made", "_what_it_is_not", "_calibration"):
        assert text.get(key), f"{key} is missing from the audit file"
    assert "claude-opus-5" in text["_how_it_was_made"]
    assert "tutor" in text["_how_it_was_made"]
