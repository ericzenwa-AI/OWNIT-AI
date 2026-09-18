"""The front door written for the student, and the voice behind it.

Two doors on purpose. A tutor or parent at / is deciding whether to trust a
method. A sixteen year old with a question in front of them is not evaluating
anything - they are stuck - and the only line that makes them press a button is
one they recognise themselves in.
"""

import pathlib

import pytest
from fastapi.testclient import TestClient

import api
import store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "student.db")
    return TestClient(api.app)


def page():
    return (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "student.html").read_text(encoding="utf-8")


def diagnostic():
    return (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text(encoding="utf-8")


# ---- the door --------------------------------------------------------------


def test_it_serves(client):
    assert client.get("/student").status_code == 200


def test_the_two_doors_are_different_pages(client):
    assert client.get("/student").text != client.get("/").text


def test_it_opens_on_the_student_and_not_on_the_buyer():
    text = page()
    assert "Stuck on a maths question?" in text
    # "Your student" is the tutor page's opening and must not be here.
    assert "Your student isn't bad at maths" not in text


def test_the_way_in_is_obvious(client):
    text = page()
    assert text.count('href="/start"') >= 2
    assert "Find my gap" in text
    assert "Free. No sign-up." in text


def test_it_says_what_is_not_covered_before_they_start():
    """A student who brings trigonometry and is turned away after typing it out
    does not come back."""
    assert "Not trigonometry, statistics or mechanics yet" in page()


def test_it_does_not_promise_the_answer():
    text = page()
    assert "It won't do your homework" in text
    assert "never gives you the answer" in text


def test_a_student_can_say_it_got_them_wrong(client):
    """The one thing a student can report without knowing any maths: that it
    told them something that did not match."""
    assert 'id="comment-form"' in page()
    response = client.post("/api/comment", json={
        "comment": "It said indices but I didn't get the fractions bit",
        "about": "Fractions"})
    assert response.status_code == 200


def test_the_tutor_page_is_still_reachable_from_it():
    assert 'href="/"' in page()


def test_it_falls_back_rather_than_erroring(client, monkeypatch):
    """A missing file must not be a 500 on the page traffic is pointed at."""
    monkeypatch.setattr(api, "STUDENT_PAGE", pathlib.Path("nowhere.html"))
    assert client.get("/student").status_code == 200


# ---- the voice while answering ---------------------------------------------


def test_the_first_question_says_why_it_is_being_asked():
    """Being asked something that is not your question, with no reason given,
    is the moment a student decides this is a test."""
    assert "Before your question &mdash; something underneath it" in diagnostic()


def test_no_invented_denominator():
    """The walk is adaptive - median three questions, anywhere from one to
    thirteen. "Question 1 of 6" would be made up, and it would be the only
    number on the screen."""
    import re

    # Comments are stripped first: the comment explaining why there is no
    # denominator necessarily contains one.
    text = re.sub(r"<!--.*?-->", "", diagnostic(), flags=re.S)
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)

    for lie in ("of 6", "/6", "of six"):
        assert lie not in text, f"a fixed question count appeared: {lie}"
    assert "Question ${state.asked_so_far + 1}" in text


def test_it_says_nothing_is_marked():
    assert "nothing here is marked" in diagnostic()


# ---- the finding -----------------------------------------------------------


def test_the_finding_opens_on_what_they_got_right():
    """"You did the differentiation fine, it was the bracket" is heard.
    "You cannot expand brackets" is not."""
    text = diagnostic()
    assert "We found something" in text
    assert 'class="held-up"' in text
    assert "const heldUp" in text


def test_it_only_says_that_when_it_is_true():
    """A student who got nothing right must not be told they did."""
    text = diagnostic()
    assert "gap && heldUp.length ?" in text, (
        "the credit must be conditional on there being some")
