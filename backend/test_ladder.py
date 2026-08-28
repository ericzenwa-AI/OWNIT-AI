"""Tests for the closing ladder.

No model is called anywhere here - the writer is stubbed. What is being tested
is everything around it: that a climb is offered only when there is something to
climb from, that it is capped apart from starts, that the answer key never
reaches the browser, and above all that a student who clears every rung and then
misses their own question is recorded as exactly that and not as a new gap.
"""

import json

import pytest
from fastapi.testclient import TestClient

import api
import ladder
import store
from questions import Distractor, MultipleChoiceQuestion
from walk import Diagnosis, Reading


def _mcq(stem: str) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question=stem,
        correct_option="right",
        distractors=[
            Distractor(option=f"wrong {n}", mistake=f"slip {n}") for n in (1, 2, 3)
        ],
    )


FOUR = [_mcq("rung one"), _mcq("rung two"), _mcq("rung three"), _mcq("their question")]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "ladder.db")
    monkeypatch.setattr(ladder, "build", lambda **kwargs: list(FOUR))
    return TestClient(api.app)


def _finished_walk(*, question="3^x = 81", gap="index_laws", summary=None):
    connection = store.connect()
    match = type("M", (), {"confidence": "high", "plain_summary": summary or ""})()
    session_id = store.open_walk(
        connection, entry_skill_id="solve_index_equation", reading=Reading(),
        question=question, match=match)
    diagnosis = Diagnosis(entry_skill_id="solve_index_equation", had_attempt=False)
    diagnosis.root_gaps = [gap] if gap else []
    diagnosis.chains = [["solve_index_equation", "equate_indices", gap]] if gap else []
    store.close_walk(connection, session_id, diagnosis)
    connection.close()
    return session_id


def _answer(client, session_id, position, label):
    return client.post("/api/ladder/answer", json={
        "session_id": session_id, "position": position, "label": label}).json()


def _label_of(rung, text):
    return next(o["label"] for o in rung["options"] if o["text"] == text)


# ---- being offered at all --------------------------------------------------


def test_a_climb_starts_at_the_first_rung(client):
    session_id = _finished_walk()

    rung = client.post("/api/ladder", json={"session_id": session_id}).json()["rung"]

    assert rung["position"] == 1
    assert rung["of"] == 4
    assert rung["is_final"] is False
    assert len(rung["options"]) == 4


def test_the_answer_key_never_reaches_the_browser(client):
    session_id = _finished_walk()

    body = client.post("/api/ladder", json={"session_id": session_id}).text

    assert "mistake" not in body
    assert "correct" not in body


def test_a_walk_that_confirmed_nothing_has_nothing_to_climb_from(client):
    """Three don't-knows name no gap. There is no bottom to start at."""
    session_id = _finished_walk(gap=None)

    response = client.post("/api/ladder", json={"session_id": session_id})

    assert response.status_code == 400
    assert "nothing to climb" in response.json()["detail"]


def test_asking_twice_does_not_pay_twice(client, monkeypatch):
    session_id = _finished_walk()
    client.post("/api/ladder", json={"session_id": session_id})

    def explode(**kwargs):
        raise AssertionError("the ladder was written a second time")

    monkeypatch.setattr(ladder, "build", explode)
    again = client.post("/api/ladder", json={"session_id": session_id})

    assert again.status_code == 200
    assert again.json()["rung"]["position"] == 1


# ---- the cap, which is its own ---------------------------------------------


def test_ladders_are_capped_apart_from_starts(client, monkeypatch):
    monkeypatch.setattr(api, "DAILY_LADDERS", 1)
    client.post("/api/ladder", json={"session_id": _finished_walk()})

    refused = client.post("/api/ladder", json={"session_id": _finished_walk()})

    assert refused.status_code == 429
    # And the front door is still open, because the two are counted apart.
    assert client.post("/api/start", json={"question": "leave it exact"}).status_code == 200


def test_a_ladder_does_not_use_up_a_start(client, monkeypatch):
    monkeypatch.setattr(api, "DAILY_STARTS", 1)
    session_id = _finished_walk()

    assert client.post("/api/ladder", json={"session_id": session_id}).status_code == 200


# ---- climbing it -----------------------------------------------------------


def test_a_wrong_step_still_moves_on(client):
    """A climb, not a second diagnosis. Stopping to mark it would make it one."""
    session_id = _finished_walk()
    rung = client.post("/api/ladder", json={"session_id": session_id}).json()["rung"]

    got = _answer(client, session_id, 1, _label_of(rung, "wrong 1"))

    assert got["right"] is False
    assert got["mistake"] == "slip 1"
    assert got["rung"]["position"] == 2


def test_clearing_every_rung_ends_climbed(client):
    session_id = _finished_walk()
    rung = client.post("/api/ladder", json={"session_id": session_id}).json()["rung"]
    for position in (1, 2, 3, 4):
        got = _answer(client, session_id, position, _label_of(rung, "right"))
        rung = got["rung"]

    assert got["finished"] is True
    assert got["right"] is True
    assert got["climbed"] is True


def test_missing_the_top_after_clearing_everything_below(client):
    """The signal the whole table exists for: the pieces were there and putting
    them together was not. That is not a gap - the walk already found and
    cleared that one."""
    session_id = _finished_walk()
    rung = client.post("/api/ladder", json={"session_id": session_id}).json()["rung"]
    for position in (1, 2, 3):
        rung = _answer(client, session_id, position, _label_of(rung, "right"))["rung"]

    got = _answer(client, session_id, 4, _label_of(rung, "wrong 2"))

    assert got["finished"] is True
    assert got["right"] is False
    assert got["climbed"] is True, "they cleared everything below it"
    assert got["mistake"] == "slip 2"


def test_that_signal_is_recorded_and_readable(client):
    session_id = _finished_walk()
    rung = client.post("/api/ladder", json={"session_id": session_id}).json()["rung"]
    for position in (1, 2, 3):
        rung = _answer(client, session_id, position, _label_of(rung, "right"))["rung"]
    _answer(client, session_id, 4, _label_of(rung, "wrong 2"))

    connection = store.connect()
    missed = store.missed_the_top(connection)
    connection.close()

    assert len(missed) == 1
    assert missed[0]["chosen"] == "wrong 2"
    assert missed[0]["mistake"] == "slip 2"


def test_someone_who_stumbled_lower_down_is_not_in_that_list(client):
    """They did not clear everything below, so missing the top says nothing
    new - it is the gap they already have, showing up again."""
    session_id = _finished_walk()
    rung = client.post("/api/ladder", json={"session_id": session_id}).json()["rung"]
    rung = _answer(client, session_id, 1, _label_of(rung, "wrong 1"))["rung"]
    for position in (2, 3):
        rung = _answer(client, session_id, position, _label_of(rung, "right"))["rung"]
    _answer(client, session_id, 4, _label_of(rung, "wrong 3"))

    connection = store.connect()
    missed = store.missed_the_top(connection)
    connection.close()

    assert missed == []


# ---- the photographed question ---------------------------------------------


def test_a_photo_question_is_marked_as_a_reading(client, monkeypatch):
    """The image is deleted, so the top rung is the model's paraphrase. The page
    has to be able to say so."""
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return list(FOUR)

    monkeypatch.setattr(ladder, "build", spy)
    session_id = _finished_walk(question="", summary="Find x where 2^x = 128")

    rung = client.post("/api/ladder", json={"session_id": session_id}).json()["rung"]

    assert seen["verbatim"] is False
    assert seen["question"] == "Find x where 2^x = 128"
    assert rung["verbatim"] is False


def test_a_typed_question_is_marked_verbatim(client, monkeypatch):
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return list(FOUR)

    monkeypatch.setattr(ladder, "build", spy)
    session_id = _finished_walk(question="3^x = 81")

    rung = client.post("/api/ladder", json={"session_id": session_id}).json()["rung"]

    assert seen["verbatim"] is True
    assert rung["verbatim"] is True


def test_a_session_with_neither_words_nor_a_reading_is_refused(client):
    session_id = _finished_walk(question="", summary="")

    response = client.post("/api/ladder", json={"session_id": session_id})

    assert response.status_code == 400
    assert "no record of the original question" in response.json()["detail"]
