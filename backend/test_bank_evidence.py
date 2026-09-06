"""What ordinary answering says about the questions, once anyone reads it.

Every answer has recorded the option picked, the misconception behind it and how
long it took since the day the table was written. The bank has counted how often
each question was asked and got right. None of it was read anywhere: one reader
was called from nowhere, another only from a command-line flag, and the numbers
page showed the funnel and nothing about the questions.

This is the half a student can answer without being asked anything. They cannot
tell you a question is filed under the wrong skill - that needs a tutor or a
model - but they can tell you it does not discriminate, that two of its options
are decoration, and that they guessed.
"""

import pathlib
import tempfile

import pytest

import store
from questions import Distractor, MultipleChoiceQuestion
from walk import Reading


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "evidence.db")
    connection = store.connect()
    yield connection
    connection.close()


def _a_question(connection):
    return store.bank_question(connection, "add_subtract_fractions",
        MultipleChoiceQuestion(
            question="Work out 1/4 + 1/6", correct_option="5/12",
            distractors=[
                Distractor(option="2/10", mistake="added the tops and the bottoms"),
                Distractor(option="1/10", mistake="added the bottoms only"),
                Distractor(option="2/24", mistake="multiplied the bottoms")]))


def _answered(connection, banked_id, *, right, chosen, seconds=45.0, mistake=None):
    session_id = store.open_walk(
        connection, entry_skill_id="add_subtract_fractions",
        reading=Reading(), question="q")
    connection.execute(
        """INSERT INTO answers (session_id, position, skill_id, question, chosen,
               outcome, misconception, seconds) VALUES (?, 1, ?, ?, ?, ?, ?, ?)""",
        (session_id, "add_subtract_fractions", "Work out 1/4 + 1/6", chosen,
         "correct" if right else "wrong", mistake, seconds))
    connection.execute(
        "UPDATE question_bank SET times_asked = times_asked + 1, "
        "times_correct = times_correct + ? WHERE id = ?",
        (1 if right else 0, banked_id))
    connection.commit()


def test_nothing_is_judged_before_there_is_enough_of_it(db):
    """A pass rate over four answers is one student having a bad day."""
    banked_id = _a_question(db)
    for _ in range(4):
        _answered(db, banked_id, right=True, chosen="5/12")

    got = store.bank_evidence(db, min_asked=10)

    assert got["judged"] == 0
    assert got["waiting"] == 1


def test_a_pass_rate_comes_out_once_there_is(db):
    banked_id = _a_question(db)
    for i in range(12):
        _answered(db, banked_id, right=i < 8, chosen="5/12" if i < 8 else "2/10",
                  mistake=None if i < 8 else "added the tops and the bottoms")

    one = store.bank_evidence(db, min_asked=10)["questions"][0]

    assert one["asked"] == 12
    assert one["correct"] == 8
    assert round(one["pass_rate"], 2) == 0.67


def test_options_nobody_picked_are_named(db):
    """The claim every distractor makes is that a real student would pick it.
    This is the only place that claim is ever tested."""
    banked_id = _a_question(db)
    for i in range(12):
        _answered(db, banked_id, right=i < 8, chosen="5/12" if i < 8 else "2/10",
                  mistake=None if i < 8 else "added the tops and the bottoms")

    one = store.bank_evidence(db, min_asked=10)["questions"][0]

    assert sorted(one["dead_options"]) == ["1/10", "2/24"]
    assert "2/10" not in one["dead_options"], "that one was picked four times"


def test_how_long_they_took_is_a_median(db):
    """One student who left the tab open over lunch must not move it."""
    banked_id = _a_question(db)
    for seconds in [30.0] * 11 + [4000.0]:
        _answered(db, banked_id, right=True, chosen="5/12", seconds=seconds)

    one = store.bank_evidence(db, min_asked=10)["questions"][0]

    assert one["median_seconds"] == 30.0


def test_the_mistakes_students_really_made_are_counted(db):
    banked_id = _a_question(db)
    for i in range(12):
        _answered(db, banked_id, right=False, chosen="2/10",
                  mistake="added the tops and the bottoms")

    seen = store.misconceptions_seen(db)

    assert seen[0]["misconception"] == "added the tops and the bottoms"
    assert seen[0]["times"] == 12


def test_an_empty_database_says_so_rather_than_nothing(db):
    got = store.bank_evidence(db)
    assert got["judged"] == 0
    assert got["questions"] == []
    assert store.misconceptions_seen(db) == []


def test_the_numbers_page_shows_it(tmp_path, monkeypatch):
    import api
    from fastapi.testclient import TestClient

    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "page.db")
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")

    connection = store.connect()
    banked_id = _a_question(connection)
    for i in range(12):
        _answered(connection, banked_id, right=i < 8,
                  chosen="5/12" if i < 8 else "2/10", seconds=3.0,
                  mistake=None if i < 8 else "added the tops and the bottoms")
    connection.close()

    page = TestClient(api.app).get("/admin/numbers", auth=("", "letmein")).text

    assert "What the questions are doing" in page
    assert "Options nobody ever picks" in page
    assert "1/10" in page, "the dead options should be named"
    assert "Answered too fast" in page
    assert "added the tops and the bottoms" in page


def test_the_page_is_honest_when_there_is_nothing_yet(tmp_path, monkeypatch):
    import api
    from fastapi.testclient import TestClient

    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "empty.db")
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")

    page = TestClient(api.app).get("/admin/numbers", auth=("", "letmein")).text

    assert "there is nothing to judge" in page
    assert "without anybody being asked anything" in page
