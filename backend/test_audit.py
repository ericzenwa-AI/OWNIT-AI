"""Tests for the answer-key audit.

No API calls: the checker is replaced with canned verdicts, so these test the
bookkeeping rather than the maths.

Run from the repo root with:  pytest backend/
"""

import json

import audit


def a_row(banked_id=1, skill_id="index_laws", correct="x^7"):
    """A bank row, as sqlite hands it over - subscriptable by column name."""
    return {
        "id": banked_id,
        "skill_id": skill_id,
        "question": "Simplify x^3 * x^4",
        "correct_option": correct,
        "distractors": json.dumps(
            [
                {"option": "x^12", "mistake": "multiplied the indices"},
                {"option": "x^1", "mistake": "subtracted the indices"},
                {"option": "2x^7", "mistake": "added the bases too"},
            ]
        ),
    }


# ---- Laying the options out ------------------------------------------------


def test_every_option_is_offered_to_the_checker():
    options = audit._options_for(a_row())

    assert sorted(options) == sorted(["x^7", "x^12", "x^1", "2x^7"])


def test_the_order_is_the_same_every_run():
    """A re-run has to ask the same question the same way, or a question that
    passed yesterday can fail today for no reason anyone can see."""
    assert audit._options_for(a_row()) == audit._options_for(a_row())


def test_different_questions_are_laid_out_differently():
    """The correct option must not sit in the same place every time."""
    orders = {tuple(audit._options_for(a_row(banked_id=i))) for i in range(1, 12)}

    assert len(orders) > 1


# ---- Agreeing, disagreeing, and not being able to tell ---------------------


def _checked(**kw):
    base = dict(
        banked_id=1,
        skill_id="index_laws",
        question="q",
        stored="x^7",
        picked="x^7",
        working="",
        note="",
    )
    return audit.Checked(**{**base, **kw})


def test_picking_the_stored_answer_is_agreement():
    one = _checked()

    assert one.agrees
    assert not one.disagrees


def test_picking_a_different_answer_is_disagreement():
    one = _checked(picked="x^12")

    assert not one.agrees
    assert one.disagrees


def test_picking_none_of_them_is_disagreement():
    one = _checked(picked=None)

    assert one.disagrees
    assert one.none_correct


def test_a_check_that_could_not_run_is_not_a_disagreement():
    """The dangerous one. An expired key made every question look wrong, and
    under --fix that would have retired the entire bank."""
    one = _checked(picked=None, failed=True, note="check failed: no api key")

    assert not one.agrees
    assert not one.disagrees
    assert not one.none_correct


# ---- Grouping the damage ---------------------------------------------------


def test_a_skill_is_grouped_under_its_topic():
    assert audit.topic_of("first_principles") == "differentiation"


def test_a_shared_skill_is_not_attributed_to_a_topic():
    """index_laws turns up under logs, surds, binomials and differentiation.
    Filing its broken questions under any one of them would be a lie."""
    assert audit.topic_of("index_laws") == "(shared, no topic)"


def test_an_unknown_skill_does_not_crash_the_report():
    assert audit.topic_of("not_a_skill") == "unknown"


# ---- The report ------------------------------------------------------------


def test_failures_are_not_counted_as_wrong_answers(capsys):
    checked = [
        _checked(banked_id=1),
        _checked(banked_id=2, picked=None, failed=True),
        _checked(banked_id=3, picked="x^12"),
    ]

    audit.report(checked)
    printed = capsys.readouterr().out

    # One judged wrong out of the two that were actually judged - not one of
    # three, and not two of three.
    assert "1 of 2 wrong" in printed
    assert "1 could not be checked" in printed


def test_a_clean_bank_says_so(capsys):
    audit.report([_checked(banked_id=1), _checked(banked_id=2)])

    assert "confirmed correct" in capsys.readouterr().out


def test_nothing_checked_is_not_reported_as_a_clean_bank(capsys):
    """Every check failing must not read as every answer being right."""
    audit.report([_checked(picked=None, failed=True)])
    printed = capsys.readouterr().out

    assert "Nothing was actually checked" in printed
    assert "confirmed correct" not in printed
