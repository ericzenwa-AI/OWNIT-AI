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


def answered(skill_id, *, held, mistake=None, dont_know=False, confidence="sure"):
    return SkillResult(
        skill_id,
        held=held,
        mistake=mistake,
        dont_know=dont_know,
        question=f"a question about {skill_id}",
        chosen="whatever they picked",
        confidence=confidence,
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
        db, a_walk([answered("surds", held=False, dont_know=True, confidence="guess")])
    )

    row = db.execute("SELECT * FROM answers WHERE skill_id = 'surds'").fetchone()
    assert row["outcome"] == "dont_know"
    assert row["misconception"] is None


def test_two_walks_do_not_collide(db):
    first = store.save_session(db, a_walk())
    second = store.save_session(db, a_walk())

    assert first != second
    assert db.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"] == 2


# ---- Lucky answers --------------------------------------------------------


def test_a_guessed_right_answer_is_flagged(db):
    """One in four guesses lands right, and a lucky one stops the walk."""
    store.save_session(db, a_walk([answered("index_laws", held=True, confidence="guess")]))

    lucky = store.lucky_answers(db)
    assert len(lucky) == 1
    assert lucky[0]["skill_id"] == "index_laws"


def test_a_confident_right_answer_is_not_flagged(db):
    store.save_session(db, a_walk([answered("index_laws", held=True, confidence="sure")]))
    assert store.lucky_answers(db) == []


def test_the_walk_knows_a_result_was_lucky():
    assert answered("index_laws", held=True, confidence="guess").lucky is True
    assert answered("index_laws", held=True, confidence="sure").lucky is False
    # Wrong and guessed is not lucky - nothing was gained by it.
    assert answered("index_laws", held=False, confidence="guess").lucky is False


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
                answered("index_laws", held=True, confidence="guess"),
                answered("surds", held=False, mistake="a mistake"),
            ]
        ),
    )
    store.save_session(db, a_walk([answered("index_laws", held=True)]))

    stats = {s.skill_id: s for s in store.skill_stats(db)}
    assert stats["index_laws"].asked == 2
    assert stats["index_laws"].held == 2
    assert stats["index_laws"].lucky == 1
    assert stats["index_laws"].held_rate == 1.0
    assert stats["surds"].wrong == 1


def test_an_empty_store_reads_back_empty(db):
    assert store.skill_stats(db) == []
    assert store.common_misconceptions(db, "index_laws") == []
    assert store.lucky_answers(db) == []


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
