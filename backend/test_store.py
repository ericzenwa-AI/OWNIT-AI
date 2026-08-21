"""Tests for keeping what a diagnosis learned.

Every test writes to its own throwaway database file, so nothing here can touch
real student answers.

Run from the repo root with:  pytest backend/
"""

import pytest

import store
from walk import Diagnosis, SkillResult


@pytest.fixture
def db(tmp_path):
    connection = store.connect(tmp_path / "test.db")
    yield connection
    connection.close()


def answered(skill_id, *, held, mistake=None, dont_know=False):
    return SkillResult(
        skill_id,
        held=held,
        mistake=mistake,
        dont_know=dont_know,
        question=f"a question about {skill_id}",
        chosen="whatever they picked",
        seconds=12.5,
    )


def a_walk(results=None):
    """A finished diagnosis, shaped the way the walk produces one."""
    results = results or [
        answered("index_laws", held=False, mistake="treats a^0 as a"),
        answered("index_notation", held=True),
    ]
    return Diagnosis(
        entry_skill_id="differentiate_function",
        had_attempt=False,
        # The entry node is never asked - being stuck is the premise.
        results=[SkillResult("differentiate_function", held=False, asked=False)]
        + results,
        root_gaps=["index_laws"],
        chains=[["differentiate_function", "power_rule", "index_laws"]],
        unchecked=["surds"],
    )


# ---- Writing it down ------------------------------------------------------


def test_a_walk_is_saved(db):
    session_id = store.save_session(db, a_walk(), student_ref="student_7")

    row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert row["entry_skill_id"] == "differentiate_function"
    assert row["root_gaps"] == "index_laws"
    assert row["student_ref"] == "student_7"


def test_every_answer_is_saved_in_order(db):
    session_id = store.save_session(db, a_walk())

    rows = db.execute(
        "SELECT * FROM answers WHERE session_id = ? ORDER BY position", (session_id,)
    ).fetchall()
    assert [r["skill_id"] for r in rows] == ["index_laws", "index_notation"]
    assert rows[0]["outcome"] == "wrong"
    assert rows[0]["misconception"] == "treats a^0 as a"
    assert rows[1]["outcome"] == "correct"


def test_the_entry_node_is_not_stored_as_an_answer(db):
    """It was never asked, so there is no answer to record."""
    session_id = store.save_session(db, a_walk())

    skills = [
        r["skill_id"]
        for r in db.execute(
            "SELECT skill_id FROM answers WHERE session_id = ?", (session_id,)
        )
    ]
    assert "differentiate_function" not in skills


def test_dont_know_is_stored_apart_from_wrong(db):
    store.save_session(
        db, a_walk([answered("surds", held=False, dont_know=True)])
    )

    row = db.execute("SELECT * FROM answers WHERE skill_id = 'surds'").fetchone()
    assert row["outcome"] == "dont_know"
    assert row["misconception"] is None


def test_two_walks_do_not_collide(db):
    first = store.save_session(db, a_walk())
    second = store.save_session(db, a_walk())

    assert first != second
    assert db.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"] == 2


# ---- Reading it back ------------------------------------------------------


def test_misconceptions_come_back_commonest_first(db):
    for _ in range(3):
        store.save_session(
            db, a_walk([answered("index_laws", held=False, mistake="treats a^0 as a")])
        )
    store.save_session(
        db, a_walk([answered("index_laws", held=False, mistake="adds the indices")])
    )

    common = store.common_misconceptions(db, "index_laws")
    assert common[0] == ("treats a^0 as a", 3)
    assert common[1] == ("adds the indices", 1)


def test_misconceptions_are_per_skill(db):
    store.save_session(
        db, a_walk([answered("surds", held=False, mistake="a surds mistake")])
    )
    assert store.common_misconceptions(db, "index_laws") == []


def test_skill_stats_count_each_outcome(db):
    store.save_session(
        db,
        a_walk(
            [
                answered("index_laws", held=True),
                answered("surds", held=False, mistake="a mistake"),
            ]
        ),
    )
    store.save_session(db, a_walk([answered("index_laws", held=True)]))

    stats = {s.skill_id: s for s in store.skill_stats(db)}
    assert stats["index_laws"].asked == 2
    assert stats["index_laws"].held == 2
    assert stats["index_laws"].held_rate == 1.0
    assert stats["surds"].wrong == 1


def test_an_empty_store_reads_back_empty(db):
    assert store.skill_stats(db) == []
    assert store.common_misconceptions(db, "index_laws") == []


# ---- Feedback -------------------------------------------------------------


def test_feedback_records_the_real_gap(db):
    session_id = store.save_session(db, a_walk())
    store.record_feedback(
        db, session_id, "wrong", actual_gap="surds", note="he had index laws fine"
    )

    row = db.execute("SELECT * FROM feedback").fetchone()
    assert row["verdict"] == "wrong"
    assert row["actual_gap"] == "surds"


def test_feedback_rejects_anything_but_right_or_wrong(db):
    session_id = store.save_session(db, a_walk())
    with pytest.raises(ValueError, match="right"):
        store.record_feedback(db, session_id, "maybe")


# ---- The file itself ------------------------------------------------------


def test_opening_it_twice_keeps_what_was_there(tmp_path):
    path = tmp_path / "again.db"

    first = store.connect(path)
    store.save_session(first, a_walk())
    first.close()

    second = store.connect(path)
    assert second.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"] == 1
    second.close()


# ---- Coverage gaps --------------------------------------------------------


class _Match:
    """The shape record_unplaced reads off an EntryMatch."""

    def __init__(self, skill_id=None, confidence="high", looks_incomplete=False,
                 reason="binomial expansion is not covered"):
        self.skill_id = skill_id
        self.confidence = confidence
        self.looks_incomplete = looks_incomplete
        self.reason = reason


def test_an_unplaceable_question_is_filed(db):
    store.record_unplaced(db, "expand (1 - 9x)^(1/2)", _Match(), role="tutor")

    row = db.execute("SELECT * FROM unplaced").fetchone()
    assert row["question"] == "expand (1 - 9x)^(1/2)"
    assert row["reason"] == "binomial expansion is not covered"
    assert row["role"] == "tutor"


def test_the_backlog_reads_back_newest_first(db):
    store.record_unplaced(db, "first question", _Match())
    store.record_unplaced(db, "second question", _Match())

    backlog = store.unplaced_questions(db)
    assert [r["question"] for r in backlog] == ["second question", "first question"]


def test_a_gap_records_whether_it_came_from_a_photo(db):
    store.record_unplaced(db, "", _Match(), from_image=True)
    assert store.unplaced_questions(db)[0]["from_image"] == 1


def test_a_near_miss_keeps_what_was_guessed(db):
    """Nothing usable, but knowing what it reached for is worth having."""
    store.record_unplaced(db, "q", _Match(skill_id="index_laws", confidence="low"))

    row = db.execute("SELECT * FROM unplaced").fetchone()
    assert row["guessed"] == "index_laws"
    assert row["confidence"] == "low"


def test_an_empty_backlog_reads_back_empty(db):
    assert store.unplaced_questions(db) == []


# ---- Carrying answers between parts of one question -----------------------


def test_a_reused_answer_is_not_stored_twice(db):
    """Part (b) walks back through what part (a) already settled."""
    from dataclasses import replace

    first = answered("index_laws", held=False, mistake="treats a^0 as a")
    session = store.save_session(db, a_walk([first, replace(first, skill_id="surds")]))
    store.save_session(db, a_walk([replace(first, reused=True)]))

    rows = db.execute("SELECT skill_id FROM answers").fetchall()
    assert [r["skill_id"] for r in rows] == ["index_laws", "surds"]


def test_todays_count_includes_questions_we_could_not_place(db):
    """Both spend a call on the best model, and a run of unplaceable questions
    is exactly the shape an abusive one takes."""
    import walk
    from entry import EntryMatch

    assert store.starts_today(db) == 0

    store.open_walk(db, entry_skill_id="index_laws", reading=walk.Reading())
    assert store.starts_today(db) == 1

    store.record_unplaced(
        db,
        "something about vectors",
        EntryMatch(
            skill_id="",
            confidence="low",
            plain_summary="",
            reason="not on the map",
            recognised_as="vectors",
        ),
    )
    assert store.starts_today(db) == 2


def test_a_live_question_is_not_dropped_for_a_retired_twin(db):
    """A retired question and a live replacement can read exactly the same. The
    retired one comes first in the file, so matching on text alone drops the
    live one and quietly costs the skill a question."""
    same = "Factorise 2x^3 + 5x^2 - 4x - 3 completely."
    records = [
        {"skill_id": "factorise_cubic", "question": same,
         "correct_option": "wrong one", "distractors": [], "retired": 1},
        {"skill_id": "factorise_cubic", "question": same,
         "correct_option": "right one", "distractors": [], "retired": 0},
    ]

    assert store.restore_bank(db, records) == 2
    assert store.bank_counts(db)["factorise_cubic"] == 1


def test_the_same_question_live_and_retired_settles_on_the_file_ordering(db):
    """An export cannot produce this - a question is one row - but the file is
    committed and hand-editable, so a bad merge could. The last entry wins and
    the database stays on that answer rather than flipping between them."""
    live = {"skill_id": "surds", "question": "q", "correct_option": "a",
            "distractors": [], "retired": 0}
    records = [live, {**live, "retired": 1}]

    store.restore_bank(db, records)
    assert store.bank_counts(db) == {}

    store.restore_bank(db, records)
    assert store.bank_counts(db) == {}


def test_a_rewrite_is_not_mistaken_for_the_question_it_replaced(db):
    """A question and its rewrite can read identically and disagree about the
    answer - that is what a rewrite of a wrong question looks like. Matching on
    wording alone folds them into one."""
    same = "Factorise 2x^3 + 5x^2 - 4x - 3 completely."
    records = [
        {"skill_id": "factorise_cubic", "question": same,
         "correct_option": "(x + 1)(2x - 1)(x + 3)", "distractors": [], "retired": 1},
        {"skill_id": "factorise_cubic", "question": same,
         "correct_option": "(x + 1)(2x - 3)(x + 1)", "distractors": [], "retired": 1},
    ]

    assert store.restore_bank(db, records) == 2
    assert store.restore_bank(db, records) == 0


def test_a_question_retired_in_the_file_is_retired_here_too(db):
    """What the deployed machine was getting wrong.

    It restored while the question was live, the question was later found to
    have the wrong answer and retired, and the next restore only added a
    retired copy alongside. The live one is what gets served, so the server
    carried on handing out an answer we knew was wrong.
    """
    wrong = {"skill_id": "factorise_cubic", "question": "Factorise it.",
             "correct_option": "(x + 1)(2x - 3)(x + 1)", "distractors": [],
             "retired": 0}

    assert store.restore_bank(db, [wrong]) == 1
    assert store.bank_counts(db)["factorise_cubic"] == 1

    # The same question, now marked retired in the file.
    assert store.restore_bank(db, [{**wrong, "retired": 1}]) == 1
    assert store.bank_counts(db) == {}
    assert store.take_question(db, "factorise_cubic") is None


def test_a_question_brought_back_in_the_file_is_live_again(db):
    """The rule is that the file decides, which has to work both ways."""
    record = {"skill_id": "surds", "question": "q", "correct_option": "a",
              "distractors": [], "retired": 1}

    store.restore_bank(db, [record])
    assert store.bank_counts(db) == {}

    assert store.restore_bank(db, [{**record, "retired": 0}]) == 1
    assert store.bank_counts(db)["surds"] == 1


def test_duplicate_copies_left_by_the_old_rule_are_all_brought_into_line(db):
    """A machine that restored under the older rule has two copies of the same
    question, one live and one retired. Updating only one leaves it served."""
    record = {"skill_id": "surds", "question": "q", "correct_option": "a",
              "distractors": [], "retired": 0}
    store.restore_bank(db, [record])
    # The second copy the old rule would have inserted.
    db.execute(
        """INSERT INTO question_bank (skill_id, question, correct_option,
               distractors, created_at, retired)
           VALUES ('surds', 'q', 'a', '[]', '2026-01-01T00:00:00+00:00', 1)"""
    )
    db.commit()

    store.restore_bank(db, [{**record, "retired": 1}])

    live = db.execute(
        "SELECT COUNT(*) c FROM question_bank WHERE retired = 0"
    ).fetchone()["c"]
    assert live == 0


def test_restoring_an_unchanged_file_still_changes_nothing(db):
    records = [{"skill_id": "surds", "question": "q", "correct_option": "a",
                "distractors": [], "retired": 0}]

    assert store.restore_bank(db, records) == 1
    assert store.restore_bank(db, records) == 0
    assert store.restore_bank(db, records) == 0
