"""Asking how they got it, and keeping what they say.

The walk already takes one bit of evidence per question - right or wrong, and
which wrong option, each of which carries a named misconception. That is more
than most things do and it is still a guess made from four boxes written in
advance. Two students who pick the same wrong option for entirely different
reasons are recorded identically.

These tests are about the layer that asks, and about the promises it has to
keep: that it never blocks the walk, never changes the diagnosis, and never
punishes an honest "I don't know".
"""

import pathlib

import pytest
from fastapi.testclient import TestClient

import api
import reasoning
import store
from questions import Distractor, MultipleChoiceQuestion
from walk import Reading


QUESTION = "Find the gradient of y = x^2 - 4x at the point where x = 3."


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "why.db")
    connection = store.connect()
    yield connection
    connection.close()


@pytest.fixture
def client(db):
    return TestClient(api.app)


def _answered(connection, *, chosen="-3", outcome="wrong"):
    session_id = store.open_walk(
        connection, entry_skill_id="gradient_at_point", reading=Reading(), question="q")
    connection.execute(
        """INSERT INTO answers (session_id, position, skill_id, question, chosen,
               outcome, seconds) VALUES (?, 1, ?, ?, ?, ?, 40.0)""",
        (session_id, "gradient_at_point", QUESTION, chosen, outcome))
    store.bank_question(connection, "gradient_at_point", MultipleChoiceQuestion(
        question=QUESTION, correct_option="2",
        distractors=[Distractor(option="-3", mistake="used y rather than dy/dx"),
                     Distractor(option="6", mistake="a"),
                     Distractor(option="0", mistake="b")]))
    connection.commit()
    return session_id


class _Reading:
    pattern = "pieces_not_connected"
    evidence = "i put x=3 into y = x^2-4x"
    got_right = "They differentiated correctly."
    said_back = "Your differentiation was right; you then used the wrong expression."


# ---- what it must never do -------------------------------------------------


def test_too_little_to_read_is_not_an_error(client, db):
    """A student who types "idk" has not failed at anything and must not be
    shown a failure."""
    session_id = _answered(db)

    response = client.post("/api/why", json={
        "session_id": session_id, "position": 1, "said": "idk"})

    assert response.status_code == 200
    assert response.json()["read"] is False


def test_a_reader_that_falls_over_leaves_the_answer_alone(client, db, monkeypatch):
    """Explaining is optional and must never cost them their place."""
    session_id = _answered(db)

    def boom(**kwargs):
        raise reasoning.NotWorthReading("the reader is down")

    monkeypatch.setattr(reasoning, "read", boom)
    response = client.post("/api/why", json={
        "session_id": session_id, "position": 1,
        "said": "i differentiated and then substituted into the wrong thing"})

    assert response.status_code == 200
    assert response.json()["read"] is False
    kept = db.execute("SELECT outcome FROM answers WHERE session_id = ?",
                      (session_id,)).fetchone()
    assert kept["outcome"] == "wrong", "the answer itself is untouched"


def test_what_counts_as_correct_does_not_come_from_the_browser(client, db, monkeypatch):
    """The page could say anything. The right answer is read from the bank."""
    session_id = _answered(db)
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return _Reading()

    monkeypatch.setattr(reasoning, "read", capture)
    client.post("/api/why", json={
        "session_id": session_id, "position": 1,
        "said": "i differentiated and got 2x - 4 then substituted wrongly"})

    assert seen["correct"] == "2"


def test_an_answer_that_does_not_exist_is_refused(client, db):
    response = client.post("/api/why", json={
        "session_id": 9999, "position": 1, "said": "some words here"})
    assert response.status_code == 404


# ---- what it keeps ---------------------------------------------------------


def test_it_stores_their_words_and_what_was_read(client, db, monkeypatch):
    session_id = _answered(db)
    monkeypatch.setattr(reasoning, "read", lambda **kw: _Reading())

    client.post("/api/why", json={
        "session_id": session_id, "position": 1,
        "said": "i differentiated and got 2x - 4 then put 3 into the original"})

    kept = store.reasoning_in(db, session_id)
    assert len(kept) == 1
    assert "2x - 4" in kept[0]["said"], "their own words are kept"
    assert kept[0]["pattern"] == "pieces_not_connected"
    assert kept[0]["evidence"], "a conclusion with no evidence is not evidence"
    assert kept[0]["got_right"]


def test_patterns_add_up_across_students(client, db, monkeypatch):
    """The distractors claim to know how students go wrong. This is the version
    that came from students describing it themselves."""
    monkeypatch.setattr(reasoning, "read", lambda **kw: _Reading())
    for _ in range(3):
        session_id = _answered(db)
        client.post("/api/why", json={
            "session_id": session_id, "position": 1,
            "said": "i differentiated then substituted into the original"})

    seen = store.patterns_seen(db)
    assert seen[0]["pattern"] == "pieces_not_connected"
    assert seen[0]["times"] == 3


def test_the_pattern_is_not_sent_to_the_student(client, db, monkeypatch):
    """A label for how somebody is thinking is a thing to hand a teacher, not a
    sixteen year old in the middle of trying."""
    session_id = _answered(db)
    monkeypatch.setattr(reasoning, "read", lambda **kw: _Reading())

    got = client.post("/api/why", json={
        "session_id": session_id, "position": 1,
        "said": "i differentiated and then used the wrong expression"}).json()

    assert "said_back" in got
    assert "pattern" not in got
    assert "evidence" not in got


# ---- the reader itself -----------------------------------------------------


def test_a_reply_of_one_word_is_refused_before_it_costs_anything():
    with pytest.raises(reasoning.NotWorthReading):
        reasoning.read(question=QUESTION, chosen="-3", correct="2",
                       said="idk", skill_id="gradient_at_point")


def test_the_prompt_carries_what_the_skill_rests_on():
    """So "prerequisite gap" can be told from "misconception" - the difference
    is whether the thing underneath is there."""
    prompt = reasoning.build_prompt(
        question=QUESTION, chosen="-3", correct="2",
        said="i differentiated then substituted into y", skill_id="gradient_at_point")

    assert "WHAT THAT RESTS ON" in prompt
    assert "Differentiate a given function" in prompt


def test_being_right_is_one_of_the_answers():
    """Our answer key can be wrong, and a student whose reasoning is sound must
    never be told otherwise."""
    assert "actually_right" in reasoning.PATTERNS


# ---- the page --------------------------------------------------------------


def _page():
    return (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text(encoding="utf-8")


def test_it_is_skippable():
    page = _page()
    assert "function skipWhy(" in page
    assert "Skip" in page


def test_it_is_never_asked_after_an_honest_dont_know():
    """There is nothing to investigate about "I don't know", and asking would
    punish the answer the page spends effort encouraging."""
    page = _page()
    hook = page[page.index("async function answer(label)"):page.index("function draw()")]
    assert "!last.dont_know" in hook


def test_the_walk_is_not_waiting_on_it():
    """It sits between two questions and the next one is already decided."""
    page = _page()
    assert "if (asking_why) return;" in page, (
        "draw() must stand aside while the question is open, not re-enter")
