"""The backlog of suspected assembly gaps, and the rule that keeps it honest.

Two things are being protected here.

The standing rule: a new doorway is not finished until somebody has asked
whether passing its prerequisites would actually let a student do the question.
test_assembly.py enforces that a verdict exists; this file enforces that the
machinery around it cannot quietly start lying.

The ordering: a suspected gap is a suspicion. Twenty-four of them came out of a
single model pass and none has been seen in a student. The order they are fixed
in has to come from outside this repository - a tutor, an examiner report, the
same thing said twice - and never from a number this code made up. A backlog
that sorts itself by confidence would look like progress and be worth nothing.
"""

import json

import pytest

import assembly
import store
from walk import Diagnosis, Reading


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "backlog.db")
    connection = store.connect()
    yield connection
    connection.close()


@pytest.fixture
def audit(tmp_path, monkeypatch):
    """A copy of the real file, so tests can write to it without touching it."""
    copy = tmp_path / "assembly_audit.json"
    copy.write_text(assembly.AUDIT_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(assembly, "AUDIT_FILE", copy)
    return copy


# ---- the standing rule -----------------------------------------------------


def test_the_rule_is_written_where_someone_adding_a_skill_will_see_it():
    """A rule that lives only in a test is a rule nobody reads before breaking
    it. skills.yaml is the file you open to add a doorway."""
    import pathlib

    import yaml

    text = pathlib.Path(assembly.AUDIT_FILE).resolve().parent.parent / "data" / "skills.yaml"
    meta = yaml.safe_load(text.read_text(encoding="utf-8"))["meta"]
    rule = "\n".join(meta["notes"])

    assert "THE RULE FOR ADDING A DOORWAY" in rule
    assert "could they now do the actual question" in rule
    assert "assembly_audit.json" in rule


def test_every_doorway_is_judged_right_now():
    assert assembly.unjudged() == []


def test_the_audit_records_the_rule_and_how_the_order_is_made():
    got = assembly.load()
    assert "_the_standing_rule" in got
    assert "_how_the_backlog_is_ordered" in got
    assert "outside confirmation only" in got["_how_the_backlog_is_ordered"]


# ---- the ordering ----------------------------------------------------------


def test_with_nothing_confirmed_nothing_outranks_anything(audit):
    """The honest state. Every suspicion at zero, so the order is alphabetical
    and carries no claim."""
    got = json.loads(audit.read_text(encoding="utf-8"))
    for verdict in got["verdicts"].values():
        verdict["evidence"] = []
    audit.write_text(json.dumps(got), encoding="utf-8")

    rows = assembly.backlog()

    assert rows, "there are suspected gaps to list"
    assert all(row[0] == 0 for row in rows), "nothing should have a score"
    assert [r[3] for r in rows] == sorted(r[3] for r in rows)


def test_recorded_evidence_lifts_something_to_the_top(audit):
    first_before = assembly.backlog()[0][3]
    other = next(s for s in assembly.suspected() if s != first_before)

    assembly.confirm(other, "tutor session", "watched a student fail exactly here")

    assert assembly.backlog()[0][3] == other


def test_a_tutor_marking_a_diagnosis_wrong_counts_as_confirmation(db, audit):
    """The live half. This fault looks, from outside, like a diagnosis that was
    wrong on a session which started at that doorway - every prerequisite
    answered and the question still not doable."""
    doorway = next(iter(assembly.suspected()))

    session_id = store.open_walk(
        db, entry_skill_id=doorway, reading=Reading(), question="q")
    diagnosis = Diagnosis(entry_skill_id=doorway, had_attempt=False)
    diagnosis.root_gaps = []
    store.close_walk(db, session_id, diagnosis)
    store.record_feedback(db, session_id, "wrong", actual_gap="something else")
    db.commit()

    top = assembly.backlog(db)[0]
    assert top[3] == doorway
    assert top[2] == 1, "the wrong verdict should be counted"


def test_confirmation_has_to_name_a_source_we_recognise(audit):
    doorway = next(iter(assembly.suspected()))
    with pytest.raises(ValueError):
        assembly.confirm(doorway, "a hunch", "felt right")


def test_confirming_something_never_judged_is_refused(audit):
    with pytest.raises(KeyError):
        assembly.confirm("not_a_skill", "tutor session", "x")


def test_the_two_already_confirmed_say_who_confirmed_them():
    """These are the only two with outside evidence, and the whole argument for
    the backlog is that the rest do not have any."""
    fractions = assembly.confirmations("add_subtract_fractions")
    stationary = assembly.confirmations("find_stationary_points")

    assert any(e["source"] == "tutor session" for e in fractions)
    assert any(e["source"] == "examiner report" for e in stationary)
    assert any("9709" in e["detail"] for e in stationary)


def test_a_fixed_doorway_leaves_the_backlog():
    """add_subtract_fractions was the one the tutor found, and it is fixed, so
    it must not still be listed as work to do."""
    assert "add_subtract_fractions" not in assembly.suspected()


def test_stationary_points_is_still_on_it():
    """Only half of it was fixed. set_derivative_zero covers 'set it to zero';
    substituting the roots back for the y-coordinates is still unmodelled, and
    pretending otherwise would lose the examiner's other finding."""
    assert "find_stationary_points" in assembly.suspected()
