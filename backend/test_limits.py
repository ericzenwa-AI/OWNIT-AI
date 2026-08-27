"""The two daily ceilings, and that neither one spends the other's budget.

They guard different things. A reading calls the best model on a photo and is
the only part of a session that costs money, so its ceiling is a spending limit
set low. A reuse - a retake, or moving to another part of the same question -
calls nothing at all, so its ceiling is only there to stop a loop filling the
database.

The failure that matters is a leak: a student retaking a question four times
and finding they have used up the day's real budget, or a loop of retakes
quietly running the bill up under a limit that was supposed to cap it.
"""

import pytest
from fastapi.testclient import TestClient

import api
import bank
import store
from entry import EntryMatch
from questions import Distractor, MultipleChoiceQuestion

DOOR = "simplify_index_expression"


@pytest.fixture(autouse=True)
def never_the_committed_shelf(tmp_path, monkeypatch):
    monkeypatch.setattr(bank, "SHELF_FILE", tmp_path / "question_bank.jsonl")
    monkeypatch.setattr(bank, "_looked_in_the_file", False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "limits.db")
    monkeypatch.setattr(
        api, "identify_entry",
        lambda question, attachment=None, **kw: EntryMatch(
            skill_id=DOOR, confidence="high", plain_summary="",
            reason="", recognised_as="indices and surds"),
    )
    monkeypatch.setattr(
        bank, "question_for",
        lambda skill_id, connection=None: (None, MultipleChoiceQuestion(
            question=f"About {skill_id}", correct_option="right",
            distractors=[Distractor(option=o, mistake="m")
                         for o in ("a", "b", "c")])))
    return TestClient(api.app)


def fresh(client, question="a new question"):
    return client.post("/api/start", json={"question": question})


def reuse(client, question="the same question"):
    return client.post("/api/start", json={"question": question, "start_at": DOOR})


def counters():
    connection = store.connect()
    try:
        return store.starts_today(connection), store.reuses_today(connection)
    finally:
        connection.close()


# ---- each counter counts its own -------------------------------------------


def test_a_new_session_counts_against_the_starts_limit(client):
    assert counters() == (0, 0)
    assert fresh(client).status_code == 200
    assert counters() == (1, 0)


def test_a_retake_counts_against_the_reuse_limit_only(client):
    fresh(client)
    before_starts, _ = counters()

    assert reuse(client).status_code == 200

    starts, reuses = counters()
    assert reuses == 1
    assert starts == before_starts, "a retake spent the new-session budget"


def test_many_retakes_never_touch_the_starts_counter(client):
    """The whole point. A student working through one question properly may
    retake it several times, and must not run the day's real budget down."""
    fresh(client)
    for _ in range(10):
        assert reuse(client).status_code == 200

    starts, reuses = counters()
    assert starts == 1
    assert reuses == 10


def test_many_new_sessions_never_touch_the_reuse_counter(client):
    for i in range(5):
        fresh(client, question=f"question {i}")

    starts, reuses = counters()
    assert starts == 5
    assert reuses == 0


# ---- each ceiling stops its own, and only its own ---------------------------


def test_the_starts_ceiling_stops_new_sessions(client, monkeypatch):
    monkeypatch.setattr(api, "DAILY_STARTS", 2)

    assert fresh(client, "one").status_code == 200
    assert fresh(client, "two").status_code == 200
    refused = fresh(client, "three")

    assert refused.status_code == 429
    assert "as many questions as it can today" in refused.json()["detail"]


def test_a_retake_still_works_when_the_starts_ceiling_is_reached(client, monkeypatch):
    """The reason the two are separate. Somebody part way through a question
    must not be locked out because the day's new questions ran out."""
    monkeypatch.setattr(api, "DAILY_STARTS", 1)
    assert fresh(client).status_code == 200
    assert fresh(client, "another").status_code == 429

    assert reuse(client).status_code == 200


def test_the_reuse_ceiling_stops_retakes(client, monkeypatch):
    monkeypatch.setattr(api, "DAILY_REUSES", 2)
    fresh(client)

    assert reuse(client).status_code == 200
    assert reuse(client).status_code == 200
    refused = reuse(client)

    assert refused.status_code == 429
    assert "repeat attempts" in refused.json()["detail"]


def test_a_new_session_still_works_when_the_reuse_ceiling_is_reached(client, monkeypatch):
    monkeypatch.setattr(api, "DAILY_REUSES", 1)
    fresh(client)
    assert reuse(client).status_code == 200
    assert reuse(client).status_code == 429

    assert fresh(client, "a different question").status_code == 200


# ---- what each counter is actually counting --------------------------------


def test_the_starts_counter_follows_readings_not_sessions(client):
    """It counts the thing that costs money. A question read but never walked
    still counts; a walk that reads nothing does not."""
    fresh(client)
    starts, _ = counters()

    connection = store.connect()
    read = connection.execute("SELECT COUNT(*) AS n FROM question_read").fetchone()["n"]
    connection.close()

    assert starts == read == 1


def test_a_reuse_is_marked_on_the_session_row(client):
    fresh(client)
    reuse(client)

    connection = store.connect()
    rows = connection.execute(
        "SELECT reused_reading FROM sessions ORDER BY id").fetchall()
    connection.close()

    assert [r["reused_reading"] for r in rows] == [0, 1]


def test_a_reuse_never_calls_the_model(client, monkeypatch):
    """If this ever fires, a retake has started costing money and the ceiling
    protecting it is set for something that is free."""
    def refuse(*args, **kwargs):
        raise AssertionError("a retake read the question again")

    monkeypatch.setattr(api, "identify_entry", refuse)

    assert reuse(client).status_code == 200


# ---- the column reaching a database that predates it ------------------------


def test_an_older_database_gets_the_column_added(tmp_path, monkeypatch):
    """The deployed database was made before this column existed, and the
    schema runs as CREATE TABLE IF NOT EXISTS - which would leave it behind."""
    import re
    import sqlite3

    path = tmp_path / "old.db"
    older = re.sub(
        r"\n    -- True when this walk started.*?\n"
        r"    reused_reading    INTEGER NOT NULL DEFAULT 0,",
        "", store.SCHEMA, flags=re.S)
    raw = sqlite3.connect(path)
    raw.executescript(older)
    raw.execute(
        "INSERT INTO sessions (created_at, entry_skill_id) VALUES (?, ?)",
        ("2020-01-01T00:00:00+00:00", DOOR))
    raw.commit()
    raw.close()

    connection = store.connect(path)
    try:
        columns = {r["name"] for r in connection.execute("PRAGMA table_info(sessions)")}
        assert "reused_reading" in columns
        kept = connection.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        assert kept == 1, "migrating dropped a session"
        assert store.reuses_today(connection) == 0
    finally:
        connection.close()

    # Opening it again must not try to add the column a second time.
    store.connect(path).close()
