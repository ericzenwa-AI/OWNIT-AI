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
#     |- surds              (level 3 - closer to the question, so asked first)
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


def test_the_closest_skill_is_asked_first():
    """Not the deepest. The first question has to look related to what they sent.

    Deepest-first sounds thorough and is worse: it opens with something that
    looks unrelated, and can name arithmetic as the gap without ever checking
    whether the student knows the method at all.
    """
    diagnosis = diagnose(ENTRY, check=student())
    # surds is level 3, fractions_arith level 4, so surds is nearer the question.
    assert asked(diagnosis)[0] == "surds"


def test_the_depth_limit_stops_the_descent():
    """Whatever the limit is set to, it is obeyed."""
    everything = {"surds", "index_laws", "index_notation", "fractions_arith", "negatives"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything), max_depth=2)

    assert asked(diagnosis) == ["surds", "index_laws"]
    assert "2 levels" in diagnosis.stopped_early
    # Nothing is confirmed, because what index_laws rests on was never asked.
    assert diagnosis.root_gaps == []
    assert diagnosis.deepest_failure == "index_laws"


def test_the_default_depth_reaches_the_floor():
    """Five levels is what finds a broken foundation, which is the case this
    exists for - a student whose real gap is years below the question."""
    everything = {"surds", "index_laws", "index_notation", "fractions_arith", "negatives"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything))

    gap = diagnosis.root_gaps[0]
    assert SKILLS[gap].needs == ()


def test_going_deeper_is_allowed_when_asked_for():
    everything = {"surds", "index_laws", "index_notation", "fractions_arith", "negatives"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything), max_depth=5)

    assert len(asked(diagnosis)) > 2
    assert diagnosis.root_gaps


def test_only_one_branch_is_walked():
    """Depth first buys its speed by not looking at every sibling."""
    everything = {"surds", "index_laws", "index_notation", "fractions_arith", "negatives"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything))

    assert diagnosis.unchecked


def test_skipped_siblings_are_reported_not_dropped():
    """The gap is everything below the break, so silence here would mislead."""
    everything = {"surds", "index_laws", "index_notation", "fractions_arith", "negatives"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything))

    assert diagnosis.unchecked


def test_nothing_asked_is_also_reported_as_unchecked():
    """A sibling queued as skipped can be reached later down another branch.
    Left alone it appears as both the answer and a thing we never looked at."""
    everything = {"surds", "index_laws", "index_notation", "fractions_arith", "negatives"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything))

    assert not set(diagnosis.unchecked) & set(asked(diagnosis))
    assert len(diagnosis.unchecked) == len(set(diagnosis.unchecked))


def test_nothing_is_unchecked_when_the_walk_completes():
    diagnosis = diagnose(ENTRY, check=student())
    assert diagnosis.unchecked == []
    assert diagnosis.stopped_early is None


def test_a_skill_is_only_asked_once():
    """fractions_arith sits under both exact_form and index_laws."""
    everything = {"surds", "index_laws", "fractions_arith"}
    diagnosis = diagnose(ENTRY, check=student(fails=everything), max_depth=5)
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
    with_attempt = diagnose(ENTRY, "sqrt(8) = 4", check=student(fails=fails), max_depth=5)
    without = diagnose(ENTRY, check=student(fails=fails), max_depth=5)

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
    diagnosis = diagnose(
        "differentiate_function", check=student(fails=set(SKILLS)), cap=1, max_depth=5
    )

    assert len(asked(diagnosis)) == 1
    assert "maximum of 1" in diagnosis.stopped_early


def test_a_capped_walk_reports_a_lead_not_a_diagnosis():
    """Its prerequisites were never checked, so it is not a confirmed gap."""
    diagnosis = diagnose(
        "differentiate_function", check=student(fails=set(SKILLS)), cap=1, max_depth=5
    )

    assert diagnosis.root_gaps == []
    assert diagnosis.deepest_failure is not None
    assert diagnosis.unchecked != []


def test_the_question_cap_is_not_what_stops_a_normal_walk():
    """The depth is reached long before fifteen questions are."""
    diagnosis = diagnose("optimisation", check=student(fails=set(SKILLS)))
    assert "maximum" not in (diagnosis.stopped_early or "")


def test_three_dont_knows_in_a_row_stops_the_walk():
    def check(skill):
        return SkillResult(skill.id, held=False, dont_know=True)

    diagnosis = diagnose("differentiate_function", check=check, max_depth=5)

    assert len(asked(diagnosis)) == 3
    assert "I don't know" in diagnosis.stopped_early


def test_a_shallow_limit_bites_before_the_dont_know_rule():
    """A failure always descends, so at two levels the walk stops after two in
    a row and the three-in-a-row rule can never fire."""

    def check(skill):
        return SkillResult(skill.id, held=False, dont_know=True)

    diagnosis = diagnose("differentiate_function", check=check, max_depth=2)

    assert len(asked(diagnosis)) == 2
    assert "2 levels" in diagnosis.stopped_early


def test_at_the_default_depth_the_dont_know_rule_does_fire():
    """Which is the point of it - a student saying it three times running is
    below the question, and grinding to the floor helps nobody."""

    def check(skill):
        return SkillResult(skill.id, held=False, dont_know=True)

    diagnosis = diagnose("differentiate_function", check=check)

    assert len(asked(diagnosis)) == 3
    assert "I don't know" in diagnosis.stopped_early


def test_reaching_the_floor_confirms_the_gap():
    """A floor skill has nothing beneath it, so the walk is genuinely finished
    rather than cut off, however far down it got."""

    def check(skill):
        return SkillResult(skill.id, held=False, dont_know=True)

    diagnosis = diagnose(ENTRY, check=check, max_depth=5)

    gap = diagnosis.root_gaps[0]
    assert SKILLS[gap].needs == ()
    # The run of "I don't know"s stopped this walk, but stopping cost nothing:
    # there was nothing beneath the gap left to check.
    assert diagnosis.stopped_early is not None
    assert diagnosis.root_gaps == [gap]


def test_the_dont_know_run_has_to_be_consecutive():
    """A right answer resets the run, so plenty of them never trip the rule."""
    seen: list[str] = []

    def check(skill):
        seen.append(skill.id)
        # Every third answer is right, so the run never reaches three.
        if len(seen) % 3 == 0:
            return SkillResult(skill.id, held=True)
        return SkillResult(skill.id, held=False, dont_know=True)

    diagnosis = diagnose("optimisation", check=check, max_depth=8)

    assert len([r for r in diagnosis.results if r.dont_know]) >= 3
    assert "I don't know" not in (diagnosis.stopped_early or "")


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


# ---- Carrying answers between parts of one question -----------------------


def test_a_settled_skill_is_not_asked_again():
    """Part (b) rests on the same foundations part (a) already tested."""
    asked_ids: list[str] = []

    def check(skill):
        asked_ids.append(skill.id)
        return SkillResult(skill.id, held=skill.id != "surds")

    from walk import reusing

    known: dict = {}
    diagnose(ENTRY, check=reusing(known, check))
    first_pass = list(asked_ids)

    asked_ids.clear()
    diagnose(ENTRY, check=reusing(known, check))

    assert first_pass, "the first walk should have asked something"
    assert asked_ids == [], "the second walk should have asked nothing new"


def test_a_carried_answer_says_so():
    def check(skill):
        return SkillResult(skill.id, held=True)

    from walk import reusing

    known: dict = {}
    reusing(known, check)(SKILLS["surds"])
    carried = reusing(known, check)(SKILLS["surds"])

    assert carried.reused is True
    assert carried.held is True


def test_carrying_does_not_change_the_diagnosis():
    from walk import reusing

    known: dict = {}
    plain = diagnose(ENTRY, check=student(fails={"surds"}))
    carried = diagnose(ENTRY, check=reusing(known, student(fails={"surds"})))

    assert plain.root_gaps == carried.root_gaps
    assert plain.chains == carried.chains


# ---- Taking one step at a time --------------------------------------------


def answer(skill_id, held=True, **kw):
    return SkillResult(skill_id, held=held, **kw)


def test_a_walk_with_no_answers_asks_the_first_question():
    from walk import step

    current = step(ENTRY)
    assert current.finished is False
    assert current.ask == _closest(ENTRY)[0]


def test_stepping_reaches_the_same_diagnosis_as_running_it_all_at_once():
    """The web version and the terminal version must not be able to differ."""
    from walk import step

    fails = {"surds", "index_laws"}
    at_once = diagnose(ENTRY, check=student(fails=fails))

    answers = []
    while True:
        current = step(ENTRY, answers)
        if current.finished:
            break
        answers.append(
            answer(current.ask, held=current.ask not in fails, mistake="muddled")
        )

    assert current.diagnosis.root_gaps == at_once.root_gaps
    assert current.diagnosis.chains == at_once.chains
    assert [r.skill_id for r in current.diagnosis.results] == [
        r.skill_id for r in at_once.results
    ]


def test_stepping_holds_nothing_between_calls():
    """Same answers in, same question out - no matter how often you ask.

    This is what lets a server forget a student entirely between requests.
    """
    from walk import step

    answers = [answer("fractions_arith", held=True)]
    first = step(ENTRY, answers)
    second = step(ENTRY, answers)
    third = step(ENTRY, list(answers))

    assert first.ask == second.ask == third.ask


def test_a_finished_walk_asks_for_nothing_more():
    from walk import step

    answers = [answer(need, held=True) for need in SKILLS[ENTRY].needs]
    current = step(ENTRY, answers)

    assert current.finished is True
    assert current.ask is None
    assert current.diagnosis.root_gaps == [ENTRY]


def test_a_walk_can_be_picked_up_later():
    """He closed the tab on Tuesday and came back on Thursday."""
    from walk import step

    tuesday = [answer("fractions_arith", held=True)]
    thursday = step(ENTRY, tuesday)

    assert thursday.finished is False
    assert thursday.ask == "surds"


def test_the_cap_still_applies_when_stepping():
    from walk import step

    given = [answer("classify_stationary", held=False, mistake="no")]
    current = step("optimisation", given, cap=1, max_depth=5)

    assert current.finished is True
    assert "maximum of 1" in current.diagnosis.stopped_early


def _closest(entry_id):
    from walk import _closest_first

    return _closest_first(SKILLS[entry_id].needs)
