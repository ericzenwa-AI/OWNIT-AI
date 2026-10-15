"""The paid test: one free diagnosis, then a code or a parent.

This is a measurement, not a lock. The free-diagnosis flag lives in the
browser and a code works wherever it is typed, both on purpose - the question
is whether a parent will pay, and a gate that shut a sixteen year old out of
their homework at ten at night would answer a different one.

So what is worth testing is not "can it be got round". It is that a code is
judged correctly, that a paid-for code cannot be taken away by a script meant
for clearing numbers, that nothing about a student is ever collected, and
that the page still lets a retake through.
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import api
import paywall
import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "paywall.db")
    connection = store.connect()
    yield connection
    connection.close()


@pytest.fixture
def client(db):
    return TestClient(api.app)


def _code(connection, *, days=30, email="parent@example.com"):
    code = paywall.new_code()
    store.add_code(connection, code, expires_at=paywall.expires_in(days),
                   parent_email=email)
    return code


# ---- judging a code --------------------------------------------------------


def test_a_good_code_unlocks(client, db):
    code = _code(db)

    got = client.post("/api/unlock", json={"code": code}).json()

    assert got["ok"] is True
    assert got["until"]


def test_a_code_nobody_made_is_refused(client, db):
    got = client.post("/api/unlock", json={"code": "OWNIT-AAAA-BBBB"}).json()

    assert got["ok"] is False
    assert "not one of ours" in got["why"]


def test_an_expired_code_is_refused(client, db):
    code = paywall.new_code()
    gone = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    store.add_code(db, code, expires_at=gone, parent_email="p@example.com")

    got = client.post("/api/unlock", json={"code": code}).json()

    assert got["ok"] is False
    assert "run out" in got["why"]


def test_a_code_read_aloud_and_typed_badly_still_works(client, db):
    """It arrives lower case off a phone, or pasted with a space on the end."""
    code = _code(db)

    got = client.post("/api/unlock", json={"code": f"  {code.lower()} "}).json()

    assert got["ok"] is True


def test_an_empty_code_is_refused_without_a_database_trip(client, db):
    assert client.post("/api/unlock", json={"code": "   "}).json()["ok"] is False


def test_a_refusal_gives_nothing_away(client, db):
    """A wrong code must not hint at what a right one looks like."""
    got = client.post("/api/unlock", json={"code": "NOPE"}).json()

    assert "OWNIT-" not in got["why"]
    assert len(got["why"]) < 60


def test_using_a_code_is_counted(client, db):
    """So a code being passed round a class is visible rather than silent."""
    code = _code(db)
    for _ in range(3):
        client.post("/api/unlock", json={"code": code})

    assert store.find_code(db, code)["times_used"] == 3


def test_a_redemption_is_recorded(client, db):
    client.post("/api/unlock", json={"code": _code(db)})

    assert "code_redeemed" in [row["name"] for row in store.clicks(db)]


def test_a_code_that_cannot_be_read_counts_as_expired():
    """Rather than letting a malformed date through as valid for ever."""
    assert paywall.has_expired("not a date") is True
    assert paywall.has_expired("") is True


def test_codes_avoid_the_characters_people_mistype():
    """A code gets read off a screen and typed by somebody else, so the
    random part carries no I, O, 0 or 1. The OWNIT- prefix is a word rather
    than something to transcribe, so it is exempt."""
    for _ in range(40):
        random_part = paywall.new_code().removeprefix("OWNIT-")
        assert not set(random_part) & set("IO01")


# ---- the free one ----------------------------------------------------------


def test_finishing_a_diagnosis_is_recorded_by_the_server(client, db):
    """The free-diagnosis count is the denominator of the whole test, so it
    is a fact the server holds rather than something the page reports."""
    from questions import Distractor, MultipleChoiceQuestion
    from walk import Reading

    session_id = store.open_walk(db, entry_skill_id="place_value",
                                 reading=Reading(), question="q")
    store.bank_question(db, "place_value", MultipleChoiceQuestion(
        question="what is the 4 worth in 0.47?", correct_option="four tenths",
        distractors=[Distractor(option="four", mistake="a"),
                     Distractor(option="forty", mistake="b"),
                     Distractor(option="four hundredths", mistake="c")]))
    db.commit()

    state = client.get(f"/api/session/{session_id}").json()
    while state.get("asking"):
        state = client.post("/api/answer", json={
            "session_id": session_id,
            "label": state["asking"]["options"][0]["label"],
            "answered_before": state.get("asked_so_far", 0)}).json()

    assert "free_diagnosis_completed" in [row["name"] for row in store.clicks(db)]


def _page():
    import pathlib

    return (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text(encoding="utf-8")


def test_the_upload_screen_stands_aside_once_the_free_one_is_gone():
    page = _page()
    assert "if (usedTheirFreeOne() && !unlocked()) return paywall();" in page


def test_the_free_one_is_spent_when_the_finding_is_shown():
    """Not when the walk finishes: somebody whose browser died mid-report has
    not had theirs."""
    page = _page()
    report = page[page.index("function report() {"):]
    assert "remember(FREE_USED, '1');" in report[:400]


def test_a_retake_never_meets_the_paywall():
    """It starts from a skill already known and never asks for a new
    question, so checking the gap closed stays free - that loop is the only
    way a student sees the work landed."""
    page = _page()
    retake = page[page.index("async function retake()"):page.index("function canRetake(")]
    assert "askScreen()" not in retake
    assert "paywall()" not in retake


def test_the_paywall_offers_both_ways_out():
    page = _page()
    assert "Ask a parent to unlock it" in page
    assert "Already paid?" in page
    assert "navigator.share" in page


def test_there_is_a_fallback_when_sharing_is_not_supported():
    """Firefox has none, and nor do some in-app browsers."""
    page = _page()
    assert "Copy the message" in page
    assert "wa.me" in page and "mailto:" in page


def test_backing_out_of_the_share_sheet_is_not_treated_as_a_failure():
    page = _page()
    ask = page[page.index("if (ask) ask.onclick"):page.index("const copy =")]
    assert "they changed their mind" in ask


def test_the_message_to_a_parent_says_what_it_costs_and_where_to_go():
    page = _page()
    message = page[page.index("function messageToParent()"):page.index("let paywallTerms")]
    assert "/parent?ref=student" in message
    assert "can you unlock it for me?" in message


# ---- what is never collected ----------------------------------------------


def test_the_paywall_asks_a_student_for_nothing(client, db):
    """No email, no name, no account. The only box on it takes a code."""
    page = _page()
    wall = page[page.index("async function paywall()"):page.index("function wirePaywall()")]

    # The only input on the whole screen takes a code.
    inputs = [line for line in wall.splitlines() if "<input" in line]
    assert len(inputs) == 1 and 'id="code"' in inputs[0]
    assert 'type="email"' not in wall
    # An email LINK is how they message a parent, which is the opposite of
    # collecting one - but nothing may ask them for theirs.
    for asking in ("your email", "your name", "sign up", "create an account"):
        assert asking not in wall.lower()


def test_the_only_email_anywhere_is_the_parents(db):
    code = _code(db, email="parent@example.com")

    kept = store.find_code(db, code)
    assert kept["parent_email"] == "parent@example.com"
    assert "student" not in str(kept).lower()


def test_a_code_can_be_made_without_an_email_at_all(db):
    """For whoever would rather hold nothing."""
    code = paywall.new_code()
    store.add_code(db, code, expires_at=paywall.expires_in())

    assert store.find_code(db, code)["parent_email"] is None


# ---- what must never be wiped ---------------------------------------------


def test_clearing_the_numbers_cannot_take_away_paid_access():
    """reset_analytics runs in a hurry, on production, by somebody who wants
    a clean funnel. It must not be able to delete what people bought."""
    import reset_analytics

    assert "access_code" in reset_analytics.NEVER
    assert "access_code" not in reset_analytics.NOISE
    assert "access_code" not in reset_analytics.SAID


# ---- the price and the link ------------------------------------------------


def test_the_link_comes_from_the_environment(client, monkeypatch):
    monkeypatch.setenv("PAYSTACK_LINK", "https://example.com/pay")

    got = client.get("/api/unlock").json()

    assert got["link"] == "https://example.com/pay"
    assert got["open"] is True


def test_no_link_set_is_a_normal_state_not_an_error(client, monkeypatch):
    """Better to say payment is not open than send somebody trying to pay you
    to a page that does not exist."""
    monkeypatch.delenv("PAYSTACK_LINK", raising=False)

    got = client.get("/api/unlock").json()

    assert got["open"] is False
    assert got["link"] == ""


def test_the_price_is_written_in_exactly_one_place(client):
    """A page saying one price and a checkout charging another is the fastest
    way to lose somebody mid-payment."""
    import pathlib
    import re

    got = client.get("/api/unlock").json()
    assert got["price"] == paywall.PRICE

    web = pathlib.Path(__file__).resolve().parent.parent / "web"
    for name in ("index.html", "parent.html"):
        text = (web / name).read_text(encoding="utf-8")
        # The parent page prints a fallback in its markup, which the server's
        # answer replaces on load. Anything else quoting a price would be a
        # second source of truth.
        prices = set(re.findall(r"&pound;\d+|£\d+", text))
        assert prices <= {"£6", "&pound;6"}, f"{name} quotes {prices}"


# ---- the parent page -------------------------------------------------------


def test_the_parent_page_serves(client):
    assert client.get("/parent").status_code == 200


def test_opening_it_is_counted(client, db):
    client.get("/parent?ref=student")

    assert "parent_page_opened" in [row["name"] for row in store.clicks(db)]


def test_it_is_written_for_the_adult_not_the_student(client):
    text = client.get("/parent").text

    assert "Your child asked you to unlock this" in text
    assert "never gives them the answer" in text.lower()
    assert "no account" in text.lower()


def test_it_falls_back_rather_than_erroring(client, monkeypatch):
    import pathlib

    monkeypatch.setattr(api, "PARENT_PAGE", pathlib.Path("nowhere.html"))
    assert client.get("/parent").status_code == 200
