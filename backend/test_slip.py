"""A gap is checked once more before it is named.

One wrong answer can be a slip - a misread sign, a mis-tap. So when the walk is
about to name a skill as the gap, it asks one more question on that skill with
different numbers first. Wrong again confirms it. Right this time records it as
"unsure, possibly a slip", treats it as holding, and the walk carries on.

No API calls. The walk is driven with scripted answers, and the API tests use
the same canned questions as test_api.py.
"""

import pytest
from fastapi.testclient import TestClient

import api
import bank
import store
from entry import EntryMatch
from questions import Distractor, MultipleChoiceQuestion
from walk import Reading, SkillResult, step

# The branch walked here, from data/skills.yaml:
#
#   exact_form
#     |- surds
#     |    `- index_laws
#     |         |- index_notation   (floor)
#     |         |- fractions_arith  (floor)
#     |         `- negatives        (floor)
#     `- fractions_arith            (floor)

ENTRY = "exact_form"
DOWN_TO_NEGATIVES = {"surds", "index_laws", "negatives"}


def walk_through(*, fails=frozenset(), right_second_time=frozenset(),
                 dont_know=frozenset(), reading=None, cap=15):
    """Answer every question the walk asks until it finishes.

    `fails` are got wrong the first time. A follow-up on anything in
    `right_second_time` is got right; any other follow-up is wrong again.
    Returns the diagnosis and every (skill, was_it_a_follow_up) asked.
    """
    answers, asked = [], []
    for _ in range(40):
        current = step(ENTRY, answers, reading=reading, cap=cap)
        if current.finished:
            return current.diagnosis, asked
        skill = current.ask
        asked.append((skill, current.follow_up))
        if current.follow_up:
            held = skill in right_second_time
        else:
            held = skill not in fails
        answers.append(SkillResult(
            skill, held=held,
            dont_know=(not held and not current.follow_up and skill in dont_know),
            mistake=None if held else f"muddled {skill}",
            follow_up=current.follow_up))
    raise AssertionError(f"the walk never finished: {asked}")


def follow_ups(asked):
    return [skill for skill, again in asked if again]


# ---- When it is asked -----------------------------------------------------


def test_a_gap_is_asked_about_again_before_it_is_named():
    _, asked = walk_through(fails=DOWN_TO_NEGATIVES)
    assert asked[-1] == ("negatives", True)


def test_only_the_gap_gets_a_second_question():
    """Not every wrong answer - only the one about to be named."""
    _, asked = walk_through(fails=DOWN_TO_NEGATIVES)
    assert follow_ups(asked) == ["negatives"]


def test_wrong_again_confirms_the_gap():
    diagnosis, _ = walk_through(fails=DOWN_TO_NEGATIVES)
    assert diagnosis.root_gaps == ["negatives"]
    assert diagnosis.unsure == []


def test_i_dont_know_is_never_asked_again():
    """Nobody mis-taps their way into "I don't know"."""
    diagnosis, asked = walk_through(fails=DOWN_TO_NEGATIVES,
                                    dont_know={"negatives"})
    assert follow_ups(asked) == []
    assert diagnosis.root_gaps == ["negatives"]


def test_the_question_they_came_in_with_is_never_asked_again():
    """Everything below held, so the question itself is the gap. It was never
    asked, so there is no answer to doubt."""
    diagnosis, asked = walk_through()
    assert follow_ups(asked) == []
    assert diagnosis.root_gaps == [ENTRY]


# ---- Right the second time -------------------------------------------------


def test_right_second_time_is_recorded_as_a_possible_slip():
    diagnosis, _ = walk_through(fails=DOWN_TO_NEGATIVES,
                                right_second_time={"negatives"})
    assert diagnosis.unsure == ["negatives"]
    assert "negatives" not in diagnosis.root_gaps


def test_a_slip_is_treated_as_holding_and_the_walk_goes_on():
    """With negatives holding, index_laws has everything beneath it - so it is
    the gap now, and it gets its own second question before being named."""
    diagnosis, asked = walk_through(fails=DOWN_TO_NEGATIVES,
                                    right_second_time={"negatives"})
    assert follow_ups(asked) == ["negatives", "index_laws"]
    assert diagnosis.root_gaps == ["index_laws"]


def test_slips_all_the_way_up_leave_no_clear_gap_below():
    """Every failure turned out to be possibly a slip. Nothing below explains
    being stuck, so the question itself is what is left - and each slip is
    still listed."""
    diagnosis, _ = walk_through(fails=DOWN_TO_NEGATIVES,
                                right_second_time=DOWN_TO_NEGATIVES)
    assert diagnosis.root_gaps == [ENTRY]
    assert sorted(diagnosis.unsure) == sorted(DOWN_TO_NEGATIVES)


def test_every_second_question_is_kept_for_the_report():
    diagnosis, _ = walk_through(fails=DOWN_TO_NEGATIVES,
                                right_second_time={"negatives"})
    assert [(f.skill_id, f.held) for f in diagnosis.follow_ups] == [
        ("negatives", True), ("index_laws", False)]


# ---- Their own working already shows it -------------------------------------


def test_no_second_question_when_their_working_went_wrong_right_there():
    """The attempt picked out this one skill out of several. That is the
    second piece of evidence already."""
    reading = Reading(had_attempt=True, narrowed_to=["fractions_arith"])
    diagnosis, asked = walk_through(fails={"fractions_arith"}, reading=reading)
    assert follow_ups(asked) == []
    assert diagnosis.root_gaps == ["fractions_arith"]


def test_without_that_the_same_gap_is_asked_again():
    _, asked = walk_through(fails={"fractions_arith"})
    assert follow_ups(asked) == ["fractions_arith"]


def test_working_that_pointed_somewhere_else_is_not_evidence():
    reading = Reading(had_attempt=True, narrowed_to=["surds"])
    _, asked = walk_through(fails={"fractions_arith"}, reading=reading)
    assert follow_ups(asked) == ["fractions_arith"]


# ---- The cap -----------------------------------------------------------------


def _first_questions(**kw):
    """How many questions the walk asks before it gets to the follow-up."""
    _, asked = walk_through(**kw)
    return len([a for a in asked if not a[1]])


def test_the_second_question_counts_towards_the_cap():
    """Slips all the way up is the walk with the most second questions in it.
    Squeeze the cap and the total still never goes over."""
    kw = dict(fails=DOWN_TO_NEGATIVES, right_second_time=DOWN_TO_NEGATIVES)
    _, uncapped = walk_through(**kw)
    for cap in range(1, len(uncapped) + 1):
        _, asked = walk_through(cap=cap, **kw)
        assert len(asked) <= cap


def test_with_no_room_left_the_gap_is_named_as_before():
    """The cap is used up by the time the gap is reached, so there is no room
    for a second question: the gap is named on the one answer, as it was."""
    full = _first_questions(fails={"fractions_arith"})
    diagnosis, asked = walk_through(fails={"fractions_arith"}, cap=full)
    assert follow_ups(asked) == []
    assert diagnosis.root_gaps == ["fractions_arith"]


def test_with_room_for_one_more_it_is_asked():
    full = _first_questions(fails={"fractions_arith"})
    _, asked = walk_through(fails={"fractions_arith"}, cap=full + 1)
    assert follow_ups(asked) == ["fractions_arith"]


# ---- A different question ------------------------------------------------------


def _q(text):
    return MultipleChoiceQuestion(
        question=text, correct_option="right",
        distractors=[Distractor(option=f"wrong {n}", mistake=f"m{n}")
                     for n in (1, 2, 3)])


def test_the_second_question_is_never_the_one_they_just_answered(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "slip.db")
    connection = store.connect()
    try:
        first = store.bank_question(connection, "negatives", _q("-4 - 5"))
        store.bank_question(connection, "negatives", _q("3 - 7"))
        session_id = store.open_walk(connection, entry_skill_id=ENTRY,
                                     reading=Reading(), question="q")
        store.record_answer(connection, session_id, skill_id="negatives",
                            question="-4 - 5", chosen="-1", held=False,
                            mistake="m", dont_know=False)

        banked_id, again = bank.another_question_for(
            "negatives", session_id, connection=connection)

        assert again.question == "3 - 7"
        assert banked_id != first
    finally:
        connection.close()


def test_with_nothing_unseen_one_is_written_and_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "slip.db")
    monkeypatch.setattr(bank, "generate_batch",
                        lambda skill_id, count: [_q("-4 - 5"), _q("2 - 9")])
    connection = store.connect()
    try:
        store.bank_question(connection, "negatives", _q("-4 - 5"))
        session_id = store.open_walk(connection, entry_skill_id=ENTRY,
                                     reading=Reading(), question="q")
        store.record_answer(connection, session_id, skill_id="negatives",
                            question="-4 - 5", chosen="-1", held=False,
                            mistake="m", dont_know=False)

        _, again = bank.another_question_for(
            "negatives", session_id, connection=connection)

        # Freshly written, and still not the one they already had.
        assert again.question == "2 - 9"
    finally:
        connection.close()


# ---- Over HTTP ---------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "api.db")
    monkeypatch.setattr(bank, "SHELF_FILE", tmp_path / "question_bank.jsonl")
    monkeypatch.setattr(bank, "_looked_in_the_file", False)
    monkeypatch.setattr(
        api, "identify_entry",
        lambda question, attachment=None, **kw: EntryMatch(
            skill_id="simplify_index_expression", confidence="high",
            plain_summary="", reason="", recognised_as="indices"))
    monkeypatch.setattr(bank, "question_for",
                        lambda skill_id, connection=None: (None, _q(f"first {skill_id}")))
    monkeypatch.setattr(bank, "another_question_for",
                        lambda skill_id, session_id, connection=None:
                            (None, _q(f"again {skill_id}")))
    return TestClient(api.app)


def _label(state, text):
    return next(o["label"] for o in state["asking"]["options"] if o["text"] == text)


def _run(client, *, right_on_follow_up):
    """Everything wrong the first time; follow-ups as asked."""
    state = client.post("/api/start", json={"question": "q"}).json()
    session_id, seen = state["session_id"], []
    for _ in range(20):
        if state.get("finished"):
            break
        again = state["asking"]["follow_up"]
        seen.append((state["asking"]["skill_name"], again))
        pick = "right" if (again and right_on_follow_up) else "wrong 1"
        state = client.post("/api/answer", json={
            "session_id": session_id, "label": _label(state, pick),
            "answered_before": state["asked_so_far"]}).json()
    return state, seen, session_id


def test_the_page_is_told_when_a_question_is_a_second_one(client):
    _, seen, _ = _run(client, right_on_follow_up=False)
    assert any(again for _, again in seen)
    assert not seen[0][1]


def test_a_follow_up_is_stored_as_one(client):
    _, _, session_id = _run(client, right_on_follow_up=False)
    connection = store.connect()
    try:
        flags = [row["follow_up"] for row in connection.execute(
            "SELECT follow_up FROM answers WHERE session_id = ? ORDER BY id",
            (session_id,))]
    finally:
        connection.close()
    assert 1 in flags and flags[0] == 0


def test_the_report_lists_a_possible_slip_on_its_own(client):
    state, _, _ = _run(client, right_on_follow_up=True)
    report = state["report"]
    assert report["unsure"]
    assert report["follow_ups"] and all(f["held"] for f in report["follow_ups"])
    for name in report["unsure"]:
        assert name not in [g["skill"] for g in report["gaps"]]
