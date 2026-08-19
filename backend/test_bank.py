"""Tests for the question bank.

No API calls: generation is replaced with canned questions, so these test the
shelving and the choosing.

Run from the repo root with:  pytest backend/
"""

import pytest

import bank
import store
from questions import Distractor, MultipleChoiceQuestion


@pytest.fixture
def db(tmp_path):
    connection = store.connect(tmp_path / "bank.db")
    yield connection
    connection.close()


def a_question(text="Simplify x^3 * x^4"):
    return MultipleChoiceQuestion(
        question=text,
        correct_option="x^7",
        distractors=[
            Distractor(option="x^12", mistake="multiplied the indices"),
            Distractor(option="x^1", mistake="subtracted the indices"),
            Distractor(option="2x^7", mistake="added the bases too"),
        ],
    )


# ---- Shelving -------------------------------------------------------------


def test_a_question_survives_the_round_trip(db):
    store.bank_question(db, "index_laws", a_question(), model="claude-haiku-4-5")

    _, taken = store.take_question(db, "index_laws")
    assert taken.question == "Simplify x^3 * x^4"
    assert taken.correct_option == "x^7"
    # The misconceptions are the point of a question, so they must come back too.
    assert taken.distractors[0].mistake == "multiplied the indices"


def test_an_empty_shelf_says_so(db):
    assert store.take_question(db, "index_laws") is None


def test_questions_are_kept_per_skill(db):
    store.bank_question(db, "index_laws", a_question())
    assert store.take_question(db, "surds") is None


def test_the_shelf_can_be_counted(db):
    store.bank_question(db, "index_laws", a_question("one"))
    store.bank_question(db, "index_laws", a_question("two"))
    store.bank_question(db, "surds", a_question("three"))

    assert store.bank_counts(db) == {"index_laws": 2, "surds": 1}


# ---- Choosing which to ask ------------------------------------------------


def test_the_least_used_question_is_offered_first(db):
    """Spreads students across the variants instead of hammering one."""
    first = store.bank_question(db, "index_laws", a_question("asked already"))
    second = store.bank_question(db, "index_laws", a_question("never asked"))
    store.mark_asked(db, first, correct=True)

    banked_id, question = store.take_question(db, "index_laws")
    assert banked_id == second
    assert question.question == "never asked"


def test_asking_is_counted(db):
    banked_id = store.bank_question(db, "index_laws", a_question())
    store.mark_asked(db, banked_id, correct=True)
    store.mark_asked(db, banked_id, correct=False)

    row = db.execute("SELECT * FROM question_bank WHERE id = ?", (banked_id,)).fetchone()
    assert row["times_asked"] == 2
    assert row["times_correct"] == 1


def test_a_retired_question_is_not_offered(db):
    banked_id = store.bank_question(db, "index_laws", a_question())
    store.retire_question(db, banked_id)

    assert store.take_question(db, "index_laws") is None


def test_retiring_keeps_the_question(db):
    """The answers already given against it still have to make sense."""
    banked_id = store.bank_question(db, "index_laws", a_question())
    store.retire_question(db, banked_id)

    assert db.execute("SELECT COUNT(*) c FROM question_bank").fetchone()["c"] == 1


# ---- Finding questions that teach us nothing ------------------------------


def test_a_question_everyone_passes_is_flagged(db):
    banked_id = store.bank_question(db, "index_laws", a_question())
    for _ in range(12):
        store.mark_asked(db, banked_id, correct=True)

    assert [r["id"] for r in store.weak_questions(db)] == [banked_id]


def test_a_question_nobody_passes_is_flagged(db):
    """More likely broken than hard."""
    banked_id = store.bank_question(db, "index_laws", a_question())
    for _ in range(12):
        store.mark_asked(db, banked_id, correct=False)

    assert [r["id"] for r in store.weak_questions(db)] == [banked_id]


def test_a_question_that_discriminates_is_left_alone(db):
    banked_id = store.bank_question(db, "index_laws", a_question())
    for index in range(12):
        store.mark_asked(db, banked_id, correct=index % 2 == 0)

    assert store.weak_questions(db) == []


def test_a_question_nobody_has_seen_is_not_judged(db):
    store.bank_question(db, "index_laws", a_question())
    assert store.weak_questions(db) == []


# ---- Filling --------------------------------------------------------------


def test_filling_shelves_what_was_written(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "fill.db")
    monkeypatch.setattr(
        bank, "generate_batch", lambda skill_id, count: [a_question()] * count
    )

    written = bank.fill(["index_laws", "surds"], per_skill=3, workers=2)

    assert written == 6
    connection = store.connect(tmp_path / "fill.db")
    assert connection.execute(
        "SELECT COUNT(*) c FROM question_bank"
    ).fetchone()["c"] == 6
    connection.close()


def test_one_failing_skill_does_not_stop_the_fill(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "fill.db")

    def sometimes(skill_id, count):
        if skill_id == "surds":
            raise RuntimeError("no")
        return [a_question()]

    monkeypatch.setattr(bank, "generate_batch", sometimes)
    assert bank.fill(["index_laws", "surds"], per_skill=1, workers=1) == 1


def test_an_empty_shelf_is_written_to_on_demand(monkeypatch, tmp_path):
    """A skill nobody has reached still works, and is cheaper next time."""
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "demand.db")
    monkeypatch.setattr(
        bank, "generate_batch", lambda skill_id, count: [a_question()] * count
    )

    banked_id, question = bank.question_for("index_laws")
    assert banked_id is not None
    assert question.correct_option == "x^7"

    connection = store.connect(tmp_path / "demand.db")
    assert connection.execute(
        "SELECT COUNT(*) c FROM question_bank"
    ).fetchone()["c"] == bank.PER_SKILL
    connection.close()


def test_the_shelf_is_used_when_it_has_something(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "shelf.db")

    connection = store.connect(tmp_path / "shelf.db")
    store.bank_question(connection, "index_laws", a_question("already written"))
    connection.close()

    def must_not_run(skill_id, count):
        raise AssertionError("wrote a new question when one was on the shelf")

    monkeypatch.setattr(bank, "generate_batch", must_not_run)

    _, question = bank.question_for("index_laws")
    assert question.question == "already written"


# ---- Surviving the database ------------------------------------------------
#
# The shelf was lost twice by ordinary accidents - a deleted file, a rebuilt
# database. These are about the file that stops that costing money again.


@pytest.fixture
def shelf(tmp_path, monkeypatch):
    """A database and a shelf file, both temporary."""
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "live.db")
    monkeypatch.setattr(bank, "SHELF_FILE", tmp_path / "question_bank.jsonl")
    return tmp_path


def test_the_shelf_survives_losing_the_database(shelf):
    """The whole point: delete the database, get the questions back."""
    connection = store.connect()
    store.bank_question(connection, "index_laws", a_question("Simplify x^2 * x^5"))
    store.bank_question(connection, "surds", a_question("Simplify root 50"))
    connection.close()

    assert bank.save() == 2

    (shelf / "live.db").unlink()

    assert bank.load() == 2
    connection = store.connect()
    try:
        assert store.bank_counts(connection) == {"index_laws": 1, "surds": 1}
    finally:
        connection.close()


def test_putting_the_shelf_back_twice_changes_nothing(shelf):
    """Someone will run it again to be sure. That must not duplicate."""
    connection = store.connect()
    store.bank_question(connection, "index_laws", a_question())
    connection.close()

    bank.save()
    assert bank.load() == 0
    assert bank.load() == 0

    connection = store.connect()
    try:
        assert store.bank_counts(connection) == {"index_laws": 1}
    finally:
        connection.close()


def test_a_retired_question_stays_retired(shelf):
    """Retiring is a judgement someone made about a bad question. Losing it
    means the next person reads the same bad question and judges again."""
    connection = store.connect()
    bad = store.bank_question(connection, "index_laws", a_question("Ambiguous"))
    store.retire_question(connection, bad)
    connection.close()

    bank.save()
    (shelf / "live.db").unlink()
    bank.load()

    connection = store.connect()
    try:
        # Retired, so nothing is on the shelf to hand out.
        assert store.bank_counts(connection) == {}
        assert store.take_question(connection, "index_laws") is None
    finally:
        connection.close()


def test_what_students_did_does_not_travel(shelf):
    """The question is ours to move around. How a particular class got on with
    it is theirs, and stays on the machine that collected it."""
    connection = store.connect()
    banked = store.bank_question(connection, "index_laws", a_question())
    store.mark_asked(connection, banked, correct=True)
    store.mark_asked(connection, banked, correct=False)
    connection.close()

    bank.save()

    import json

    records = None
    with open(shelf / "question_bank.jsonl", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert records, "nothing was written"
    assert "times_asked" not in records[0]
    assert "times_correct" not in records[0]


def test_loading_nothing_is_not_an_error(shelf):
    """A fresh clone has no shelf file yet. That is a normal state."""
    assert bank.load() == 0
