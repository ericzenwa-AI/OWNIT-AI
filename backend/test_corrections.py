"""Tests for turning tutor verdicts into evidence about the graph's links.

Every `needs` link in the graph is a guess, and nothing else in this repository
can check one. test_graph.py proves the shape is well formed - no cycles, no
dangling ids - which a completely wrong link passes. eval.py cannot help either:
it builds its simulated student by propagating up the same links the walk comes
down, so a wrong link makes the simulated student wrong in the same direction
and eval still scores well.

A tutor saying "no, it was this instead" is the only signal from outside that
reasoning. These tests are about not losing it.
"""

import pytest

import store
from walk import Diagnosis


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "corrections.db")
    connection = store.connect()
    yield connection
    connection.close()


def _judged(connection, *, chain, named, verdict, actually=None):
    """One finished walk down `chain`, judged by a tutor."""
    from walk import Reading

    session_id = store.open_walk(
        connection, entry_skill_id=chain[0], reading=Reading(), question="q")

    diagnosis = Diagnosis(entry_skill_id=chain[0], had_attempt=False)
    diagnosis.root_gaps = [named]
    diagnosis.chains = [list(chain)]
    store.close_walk(connection, session_id, diagnosis)

    store.record_feedback(
        connection, session_id, verdict,
        actual_gap=actually, note=None)
    connection.commit()
    return session_id


CHAIN = ["differentiate_function", "power_rule", "index_laws"]


def test_nothing_judged_reads_as_nothing(db):
    found = store.corrections(db)
    assert found["verdicts"] == {"right": 0, "wrong": 0}
    assert found["named"] == {}
    assert found["pairs"] == {}
    assert found["edges"] == {}


def test_a_right_verdict_is_counted_against_the_skill_named(db):
    _judged(db, chain=CHAIN, named="index_laws", verdict="right")

    found = store.corrections(db)
    assert found["verdicts"]["right"] == 1
    assert found["named"]["index_laws"] == {"right": 1, "wrong": 0}


def test_a_wrong_verdict_records_what_it_actually_was(db):
    _judged(db, chain=CHAIN, named="index_laws", verdict="wrong",
            actually="surds")

    found = store.corrections(db)
    assert found["named"]["index_laws"]["wrong"] == 1
    assert found["pairs"][("index_laws", "surds")] == 1


def test_every_link_the_walk_came_down_is_counted(db):
    """The chain is stored top-down, so consecutive pairs are the links."""
    _judged(db, chain=CHAIN, named="index_laws", verdict="right")

    edges = store.corrections(db)["edges"]
    assert edges[("differentiate_function", "power_rule")] == {"judged": 1, "wrong": 0}
    assert edges[("power_rule", "index_laws")] == {"judged": 1, "wrong": 0}


def test_a_link_on_a_wrong_diagnosis_is_marked_wrong(db):
    _judged(db, chain=CHAIN, named="index_laws", verdict="wrong", actually="surds")

    edges = store.corrections(db)["edges"]
    assert edges[("power_rule", "index_laws")] == {"judged": 1, "wrong": 1}


def test_a_link_that_has_only_ever_led_somewhere_wrong_is_suspect(db):
    """The thing worth acting on. Two tutors, same link, both corrections."""
    _judged(db, chain=CHAIN, named="index_laws", verdict="wrong", actually="surds")
    _judged(db, chain=CHAIN, named="index_laws", verdict="wrong", actually="surds")

    suspects = store.suspect_edges(db)
    assert ("power_rule", "index_laws", 2, 2) in suspects


def test_a_link_that_usually_works_is_not_suspect(db):
    """The denominator is the whole point. A link on three right answers and
    one wrong is a wrong answer, not a wrong link."""
    for _ in range(3):
        _judged(db, chain=CHAIN, named="index_laws", verdict="right")
    _judged(db, chain=CHAIN, named="index_laws", verdict="wrong", actually="surds")

    assert store.suspect_edges(db) == []
    edges = store.corrections(db)["edges"]
    assert edges[("power_rule", "index_laws")] == {"judged": 4, "wrong": 1}


def test_two_tutors_correcting_the_same_way_stack_up(db):
    _judged(db, chain=CHAIN, named="index_laws", verdict="wrong", actually="surds")
    _judged(db, chain=CHAIN, named="index_laws", verdict="wrong", actually="surds")
    _judged(db, chain=CHAIN, named="index_laws", verdict="wrong", actually="fractions")

    pairs = store.corrections(db)["pairs"]
    assert pairs[("index_laws", "surds")] == 2
    assert pairs[("index_laws", "fractions")] == 1


def test_a_correction_naming_something_not_in_the_graph_is_still_kept(db):
    """A tutor types their own words when we do not recognise the skill. That
    is a coverage gap named by someone who teaches it, and losing it would be
    worse than the untidiness of keeping it."""
    _judged(db, chain=CHAIN, named="index_laws", verdict="wrong",
            actually="rearranging before differentiating")

    pairs = store.corrections(db)["pairs"]
    assert pairs[("index_laws", "rearranging before differentiating")] == 1


def test_a_verdict_with_no_named_gap_does_not_vanish(db):
    """A walk that stopped early names nothing. A tutor can still say it was
    wrong, and that is still evidence."""
    from walk import Reading

    session_id = store.open_walk(
        db, entry_skill_id="differentiate_function", reading=Reading(), question="q")
    diagnosis = Diagnosis(
        entry_skill_id="differentiate_function", had_attempt=False)
    diagnosis.root_gaps = []
    diagnosis.chains = []
    store.close_walk(db, session_id, diagnosis)
    store.record_feedback(db, session_id, "wrong", actual_gap="surds")
    db.commit()

    found = store.corrections(db)
    assert found["verdicts"]["wrong"] == 1
    assert found["pairs"][("(nothing named)", "surds")] == 1


def test_the_admin_page_shows_the_corrections(tmp_path, monkeypatch):
    import api
    from fastapi.testclient import TestClient

    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "page.db")
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")

    connection = store.connect()
    _judged(connection, chain=CHAIN, named="index_laws", verdict="wrong",
            actually="surds")
    connection.close()

    page = TestClient(api.app).get("/admin/feedback", auth=("", "letmein")).text

    assert "What tutors corrected" in page
    assert "Laws of indices" in page          # named, by its human name
    assert "1 called wrong" in page
