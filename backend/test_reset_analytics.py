"""Tests for clearing the analytics.

This one runs against production, in a hurry, by someone who wants a clean
funnel. So the tests are mostly about what it must refuse to do.
"""

import pytest

import reset_analytics
import store


@pytest.fixture
def connection(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "reset.db")
    connection = store.connect()
    yield connection
    connection.close()


def _some_of_everything(connection):
    store.record_page_view(connection, "/")
    store.join_waitlist(connection, "someone@example.com")
    from walk import Reading
    session_id = store.open_walk(
        connection, entry_skill_id="index_laws", reading=Reading(), question="q"
    )
    connection.execute(
        "INSERT INTO answers (session_id, position, skill_id, outcome) VALUES (?, 1, ?, ?)",
        (session_id, "index_laws", "wrong"),
    )
    from questions import Distractor, MultipleChoiceQuestion

    store.bank_question(
        connection,
        "index_laws",
        MultipleChoiceQuestion(
            question="q",
            correct_option="a",
            distractors=[Distractor(option=o, mistake="m") for o in ("b", "c", "d")],
        ),
    )
    connection.commit()
    return session_id


def test_the_waitlist_survives(connection):
    """The one table with no copy anywhere. Somebody typed their address in and
    there is no way to ask them again."""
    _some_of_everything(connection)

    reset_analytics.clear(connection, reset_analytics.NOISE + reset_analytics.SAID)

    assert store.waitlist_size(connection) == 1


def test_the_question_bank_survives(connection):
    """It is committed too, so it would come back - but only after paying a
    model to rewrite everything the file does not hold."""
    _some_of_everything(connection)

    reset_analytics.clear(connection, reset_analytics.NOISE + reset_analytics.SAID)

    left = connection.execute("SELECT COUNT(*) AS n FROM question_bank").fetchone()["n"]
    assert left == 1


def test_it_refuses_to_clear_a_protected_table(connection):
    """Not reachable through the command line, so this is about someone editing
    NOISE later and not noticing what they added."""
    for table in sorted(reset_analytics.NEVER):
        with pytest.raises(AssertionError):
            reset_analytics.clear(connection, [table])


def test_the_protected_tables_are_not_in_either_list():
    overlap = reset_analytics.NEVER & set(reset_analytics.NOISE + reset_analytics.SAID)
    assert not overlap, f"a protected table is also on a clear list: {overlap}"


def test_the_counting_is_actually_cleared(connection):
    _some_of_everything(connection)

    reset_analytics.clear(connection, reset_analytics.NOISE)

    for table in reset_analytics.NOISE:
        left = connection.execute(
            f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608
        ).fetchone()["n"]
        assert left == 0, f"{table} still has rows"


def test_what_someone_typed_is_kept_unless_asked_for(connection):
    session_id = _some_of_everything(connection)
    store.leave_comment(connection, "this was wrong", session_id=session_id)

    reset_analytics.clear(connection, reset_analytics.NOISE)

    left = connection.execute("SELECT COUNT(*) AS n FROM comments").fetchone()["n"]
    assert left == 1


def test_showing_deletes_nothing(connection, capsys):
    _some_of_everything(connection)

    reset_analytics.main([])

    assert store.waitlist_size(connection) == 1
    rows = connection.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
    assert rows == 1
    assert "Nothing was deleted" in capsys.readouterr().out
