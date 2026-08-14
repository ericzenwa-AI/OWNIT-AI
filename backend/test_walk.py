"""Tests for the diagnostic walk.

No API calls: the walk takes a `check` callable, so the tests hand it a scripted
student instead of generating questions and asking them. The graph underneath is
the real one from data/skills.yaml.

Run from the repo root with:  pytest backend/
"""

import pytest

import walk
from graph import SKILLS
from walk import Narrowing, PresentationCheck, SkillResult, diagnose

# The branch these tests walk, from data/skills.yaml:
#
#   exact_form
#     |- surds
#     |    `- index_laws
#     |         |- index_notation   (floor)
#     |         |- fractions_arith  (floor)
#     |         `- negatives        (floor)
#     `- fractions_arith            (floor)

ENTRY = "exact_form"


def student(*, fails: set[str] = frozenset()):
    """A student who gets everything right except the named skills."""

    def check(skill):
        if skill.id in fails:
            return SkillResult(skill.id, held=False, mistake=f"muddled {skill.id}")
        return SkillResult(skill.id, held=True)

    return check


def asked(diagnosis) -> list[str]:
    return [r.skill_id for r in diagnosis.results if r.asked]


# ---- Without an attempt ---------------------------------------------------


def test_first_round_tests_the_whole_level_below():
    diagnosis = diagnose(ENTRY, check=student())
    assert set(asked(diagnosis)) == set(SKILLS[ENTRY].needs)


def test_attempt_is_optional():
    diagnosis = diagnose(ENTRY, check=student())
    assert diagnosis.had_attempt is False
    assert diagnosis.narrowed_to is None


def test_entry_node_is_never_asked():
    """Being stuck is the premise, so we don't test the question's own skill."""
    diagnosis = diagnose(ENTRY, check=student())
    entry_result = diagnosis.result_for(ENTRY)
    assert entry_result.asked is False
    assert entry_result.held is False


def test_entry_is_the_gap_when_everything_below_holds():
    diagnosis = diagnose(ENTRY, check=student())
    assert diagnosis.root_gaps == [ENTRY]
    assert diagnosis.chains == [[ENTRY]]


# ---- Descending -----------------------------------------------------------


def test_descends_into_what_failed():
    diagnosis = diagnose(ENTRY, check=student(fails={"surds"}))
    assert "index_laws" in asked(diagnosis)
    assert diagnosis.root_gaps == ["surds"]
    assert diagnosis.chains == [[ENTRY, "surds"]]


def test_does_not_descend_into_what_held():
    """A skill they have is the bottom of that branch - stop, don't dig."""
    diagnosis = diagnose(ENTRY, check=student())
    assert "index_laws" not in asked(diagnosis)


def test_dont_know_descends_like_any_other_failure():
    """Not held is not held - the walk does not treat it specially."""

    def check(skill):
        if skill.id == "surds":
            return SkillResult(skill.id, held=False, dont_know=True)
        return SkillResult(skill.id, held=True)

    diagnosis = diagnose(ENTRY, check=check)

    assert "index_laws" in asked(diagnosis)
    assert diagnosis.root_gaps == ["surds"]


def test_dont_know_is_recorded_apart_from_a_wrong_answer():
    def check(skill):
        if skill.id == "index_laws":
            return SkillResult(skill.id, held=False, dont_know=True)
        if skill.id == "surds":
            return SkillResult(skill.id, held=False, mistake="inverted the divisor")
        return SkillResult(skill.id, held=True)

    diagnosis = diagnose(ENTRY, check=check)

    wrong = diagnosis.result_for("surds")
    assert wrong.dont_know is False
    assert wrong.mistake == "inverted the divisor"

    blank = diagnosis.result_for("index_laws")
    assert blank.dont_know is True
    assert blank.mistake is None


def test_carries_the_mistake_through():
    diagnosis = diagnose(ENTRY, check=student(fails={"surds"}))
    assert diagnosis.result_for("surds").mistake == "muddled surds"


def test_stops_at_the_floor():
    """A failed skill with no prerequisites is as deep as the graph goes."""
    everything = {"surds", "index_laws", "index_notation", "fractions_arith", "negatives"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything))

    # fractions_arith is the most foundational of the entry node's needs, so it
    # is tried first, fails, and has nothing under it.
    assert diagnosis.root_gaps == ["fractions_arith"]
    assert diagnosis.chains == [[ENTRY, "fractions_arith"]]


def test_only_one_branch_is_walked():
    """Depth first buys its speed by not looking at the siblings."""
    everything = {"surds", "index_laws", "index_notation", "fractions_arith", "negatives"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything))

    assert asked(diagnosis) == ["fractions_arith"]
    assert "surds" not in asked(diagnosis)


def test_skipped_siblings_are_reported_not_dropped():
    """The gap is everything below the break, so silence here would mislead."""
    everything = {"surds", "index_laws", "index_notation", "fractions_arith", "negatives"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything))

    assert "surds" in diagnosis.unchecked


def test_nothing_is_unchecked_when_the_walk_completes():
    diagnosis = diagnose(ENTRY, check=student())
    assert diagnosis.unchecked == []
    assert diagnosis.stopped_early is None


def test_a_skill_is_only_asked_once():
    """fractions_arith sits under both exact_form and index_laws."""
    everything = {"surds", "index_laws", "fractions_arith"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything))
    assert asked(diagnosis).count("fractions_arith") == 1


# ---- With an attempt ------------------------------------------------------


def test_attempt_narrows_the_first_round(monkeypatch):
    monkeypatch.setattr(
        walk,
        "narrow_to_branch",
        lambda entry, attempt, **kw: Narrowing(
            branch_skill_ids=["surds"], reason="stopped at the root"
        ),
    )
    monkeypatch.setattr(
        walk,
        "check_presentation",
        lambda entry, attempt, **kw: PresentationCheck(
            presentation_only=False, note="method is wrong"
        ),
    )

    diagnosis = diagnose(ENTRY, "sqrt(8) = 4", check=student(fails={"surds"}))

    assert diagnosis.had_attempt is True
    assert diagnosis.narrowed_to == ["surds"]
    # fractions_arith is on the level below too, but the attempt ruled it out.
    assert asked(diagnosis)[0] == "surds"
    assert "fractions_arith" not in asked(diagnosis)


def test_everything_after_the_first_round_is_identical(monkeypatch):
    """Narrowing changes where the walk starts, not how it proceeds."""
    monkeypatch.setattr(
        walk,
        "narrow_to_branch",
        lambda entry, attempt, **kw: Narrowing(
            branch_skill_ids=["surds"], reason="stopped at the root"
        ),
    )
    monkeypatch.setattr(
        walk,
        "check_presentation",
        lambda entry, attempt, **kw: PresentationCheck(
            presentation_only=False, note=""
        ),
    )

    fails = {"surds", "index_laws"}
    with_attempt = diagnose(ENTRY, "sqrt(8) = 4", check=student(fails=fails))
    without = diagnose(ENTRY, check=student(fails=fails))

    # Same descent below surds, reached from a narrower start.
    assert [ENTRY, "surds", "index_laws"] in with_attempt.chains
    assert [ENTRY, "surds", "index_laws"] in without.chains


def test_a_held_branch_widens_to_the_rest_of_the_level(monkeypatch):
    """A wrong guess costs extra questions, never a wrong answer."""
    monkeypatch.setattr(
        walk,
        "narrow_to_branch",
        lambda entry, attempt, **kw: Narrowing(
            branch_skill_ids=["surds"], reason="looked like a surds problem"
        ),
    )
    monkeypatch.setattr(
        walk,
        "check_presentation",
        lambda entry, attempt, **kw: PresentationCheck(
            presentation_only=False, note=""
        ),
    )

    # surds holds, so the narrowing was wrong; fractions_arith is the real gap.
    diagnosis = diagnose(ENTRY, "sqrt(8) = 4", check=student(fails={"fractions_arith"}))

    assert diagnosis.widened is True
    assert "fractions_arith" in asked(diagnosis)
    assert diagnosis.root_gaps == ["fractions_arith"]


def test_no_widening_when_the_narrowed_branch_failed(monkeypatch):
    """A guess that pays off should not drag in the siblings."""
    monkeypatch.setattr(
        walk,
        "narrow_to_branch",
        lambda entry, attempt, **kw: Narrowing(branch_skill_ids=["surds"], reason=""),
    )
    monkeypatch.setattr(
        walk,
        "check_presentation",
        lambda entry, attempt, **kw: PresentationCheck(
            presentation_only=False, note=""
        ),
    )

    diagnosis = diagnose(ENTRY, "sqrt(8) = 4", check=student(fails={"surds"}))

    assert diagnosis.widened is False
    assert "fractions_arith" not in asked(diagnosis)


def test_presentation_check_only_runs_with_an_attempt(monkeypatch):
    called = []
    monkeypatch.setattr(
        walk,
        "check_presentation",
        lambda entry, attempt, **kw: called.append(attempt)
        or PresentationCheck(presentation_only=True, note="answer left as a decimal"),
    )
    monkeypatch.setattr(
        walk,
        "narrow_to_branch",
        lambda entry, attempt, **kw: Narrowing(branch_skill_ids=["surds"], reason=""),
    )

    without = diagnose(ENTRY, check=student())
    assert called == []
    assert without.presentation_note is None

    with_attempt = diagnose(ENTRY, "0.35", check=student())
    assert called == ["0.35"]
    assert with_attempt.presentation_note == "answer left as a decimal"


def test_presentation_finding_does_not_stop_the_walk(monkeypatch):
    """Writing it out badly doesn't prove the prerequisites are there."""
    monkeypatch.setattr(
        walk,
        "check_presentation",
        lambda entry, attempt, **kw: PresentationCheck(
            presentation_only=True, note="notation only"
        ),
    )
    monkeypatch.setattr(
        walk,
        "narrow_to_branch",
        lambda entry, attempt, **kw: Narrowing(branch_skill_ids=["surds"], reason=""),
    )

    diagnosis = diagnose(ENTRY, "0.35", check=student(fails={"surds"}))
    assert asked(diagnosis) == ["surds", "index_laws"]
    assert diagnosis.root_gaps == ["surds"]


# ---- Limits ---------------------------------------------------------------


def test_the_cap_stops_the_walk():
    diagnosis = diagnose("differentiate_function", check=student(fails=set(SKILLS)), cap=3)

    assert len(asked(diagnosis)) == 3
    assert "maximum of 3" in diagnosis.stopped_early


def test_a_capped_walk_reports_a_lead_not_a_diagnosis():
    """Its prerequisites were never checked, so it is not a confirmed gap."""
    diagnosis = diagnose("differentiate_function", check=student(fails=set(SKILLS)), cap=3)

    assert diagnosis.root_gaps == []
    assert diagnosis.deepest_failure is not None
    assert diagnosis.unchecked != []


def test_the_default_cap_clears_a_full_descent():
    """15 has to leave room for a straight walk to the floor plus siblings."""
    everything = set(SKILLS)
    diagnosis = diagnose("optimisation", check=student(fails=everything))
    assert diagnosis.stopped_early is None


def test_three_dont_knows_in_a_row_stops_the_walk():
    def check(skill):
        return SkillResult(skill.id, held=False, dont_know=True)

    diagnosis = diagnose("differentiate_function", check=check)

    assert len(asked(diagnosis)) == 3
    assert "I don't know" in diagnosis.stopped_early


def test_stopping_at_the_floor_still_confirms_the_gap():
    """A floor node has nothing beneath it, so an early stop costs nothing."""

    def check(skill):
        return SkillResult(skill.id, held=False, dont_know=True)

    diagnosis = diagnose("optimisation", check=check)

    assert diagnosis.stopped_early is not None
    assert diagnosis.deepest_failure == "negatives"
    assert SKILLS["negatives"].needs == ()
    # Confirmed despite the early stop - the descent had nowhere left to go.
    assert diagnosis.root_gaps == ["negatives"]


def test_the_dont_know_run_has_to_be_consecutive():
    """A right answer resets the run, so the walk is not cut off."""
    seen: list[str] = []

    def check(skill):
        seen.append(skill.id)
        if len(seen) == 3:
            return SkillResult(skill.id, held=True)
        return SkillResult(skill.id, held=False, dont_know=True)

    diagnosis = diagnose("optimisation", check=check)

    # Two don't-knows, then a correct answer. Never three running, so the walk
    # ends where the graph says, not where the limit says.
    assert len([r for r in diagnosis.results if r.dont_know]) == 2
    assert diagnosis.stopped_early is None
    assert diagnosis.root_gaps == ["expand_brackets"]


# ---- Narrowing fallback ---------------------------------------------------


def fake_client(parsed):
    """Stands in for the Anthropic client, returning a canned parse result."""

    class _Response:
        parsed_output = parsed

    class _Messages:
        def parse(self, **kwargs):
            return _Response()

    class _Client:
        messages = _Messages()

    return _Client()


def test_unusable_narrowing_falls_back_to_the_whole_level():
    """A wrong guess must not shrink the search to nothing."""
    client = fake_client(
        Narrowing(branch_skill_ids=["not_a_skill"], reason="confused")
    )
    narrowing = walk.narrow_to_branch(SKILLS[ENTRY], "anything", client=client)
    assert set(narrowing.branch_skill_ids) == set(SKILLS[ENTRY].needs)


def test_narrowing_keeps_only_real_prerequisites():
    client = fake_client(
        Narrowing(branch_skill_ids=["surds", "chain_rule"], reason="mixed")
    )
    narrowing = walk.narrow_to_branch(SKILLS[ENTRY], "anything", client=client)
    # chain_rule is a real skill, but it isn't below exact_form.
    assert narrowing.branch_skill_ids == ["surds"]


# ---- Bad input ------------------------------------------------------------


def test_unknown_entry_skill_raises():
    with pytest.raises(ValueError, match="not a skill"):
        diagnose("not_a_skill", check=student())
