"""Every new question is solved blind before it is kept.

A wrong answer key is the worst failure this system has: a student who answers
correctly is recorded as not holding the skill, and the walk goes looking for
a gap that is not there. The audit catches those, but only after the fact and
only when someone runs it. These pin the check that now happens on the way in.

No API calls. The solver is replaced with one that answers however the test
says, so what is pinned is the decision made from its answers.
"""

import pytest

import audit
import bank
import store
from questions import BadQuestionError, Distractor, MultipleChoiceQuestion

SKILL = "negatives"


def q(text: str, key: str = "15") -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question=text,
        correct_option=key,
        distractors=[
            Distractor(option="-15", mistake="kept one minus sign"),
            Distractor(option="-8", mistake="added instead"),
            Distractor(option="8", mistake="added and dropped the sign"),
        ],
    )


def solver(first: dict, second: dict | None = None, seen: list | None = None):
    """A fake solve. Each dict maps question text to what that pass does:
    "agree", "disagree", or "fail" (the call errors). Unlisted means agree."""
    second = second or {}

    def check(row, task):
        table = first if task is audit.FIRST_PASS else second
        outcome = table.get(row["question"], "agree")
        if seen is not None:
            seen.append((row["question"], task.model, row))
        if outcome == "fail":
            raise RuntimeError("the API is down")
        picked = row["correct_option"] if outcome == "agree" else "-15"
        return audit.Checked(row["id"], row["skill_id"], row["question"],
                             row["correct_option"], picked, "", "")
    return check


# ---- the decision --------------------------------------------------------


def test_a_question_the_solver_agrees_with_is_kept():
    result = audit.verify(SKILL, [q("a")], check=solver({}))
    assert [x.question for x in result.passed] == ["a"]
    assert not result.rejected and not result.unchecked


def test_one_bad_solve_does_not_cost_a_good_question():
    """First pass disagrees, the stronger second pass agrees with the key."""
    result = audit.verify(SKILL, [q("a")],
                          check=solver({"a": "disagree"}, {"a": "agree"}))
    assert [x.question for x in result.passed] == ["a"]


def test_a_question_both_passes_reject_is_thrown_away():
    result = audit.verify(SKILL, [q("a")],
                          check=solver({"a": "disagree"}, {"a": "disagree"}))
    assert [x.question for x in result.rejected] == ["a"]
    assert not result.passed


def test_a_check_that_never_ran_is_not_a_pass():
    """An API error says nothing about the question. It must not be kept as
    if it had been checked - that is how unchecked keys get onto the shelf."""
    result = audit.verify(SKILL, [q("a")], check=solver({"a": "fail"}))
    assert [x.question for x in result.unchecked] == ["a"]
    assert not result.passed and not result.rejected


def test_a_second_pass_that_never_ran_is_not_a_pass_either():
    result = audit.verify(SKILL, [q("a")],
                          check=solver({"a": "disagree"}, {"a": "fail"}))
    assert [x.question for x in result.unchecked] == ["a"]
    assert not result.passed


def test_only_disputed_questions_go_to_the_dearer_model():
    seen = []
    audit.verify(SKILL, [q("a"), q("b"), q("c")],
                 check=solver({"b": "disagree"}, {}, seen=seen))
    second = [text for text, model, _ in seen if model == audit.SECOND_PASS.model]
    assert second == ["b"]


def test_the_solver_is_never_told_which_option_is_the_key():
    """Showing the key would only measure how agreeable the model is. The row
    carries it for comparing afterwards, so check what the prompt is built
    from: four options, and nothing saying which."""
    class Client:
        class messages:
            sent = []

            @classmethod
            def parse(cls, **kwargs):
                cls.sent.append(kwargs)

                class R:
                    parsed_output = audit.Verdict(working="", answer="A")
                return R

    row = audit._as_row(0, SKILL, q("Work out -3 x -5"))
    audit.check_one(row, audit.FIRST_PASS, Client)

    prompt = Client.messages.sent[0]["messages"][0]["content"]
    for word in ("correct", "key", "answer is", "mistake", "kept one minus"):
        assert word not in prompt.lower()
    assert all(opt in prompt for opt in ("15", "-15", "-8", "8"))


def test_nothing_written_means_nothing_checked():
    assert audit.verify(SKILL, [], check=solver({})) == audit.Verified([], [], [])


# ---- what the shelf gets -------------------------------------------------


@pytest.fixture
def empty_shelf(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "live.db")
    monkeypatch.setattr(bank, "SHELF_FILE", tmp_path / "question_bank.jsonl")
    monkeypatch.setattr(bank, "_looked_in_the_file", False)


def _with(monkeypatch, batches, first, second=None):
    """Serve these batches in turn, and check them with this solver."""
    calls = []

    def write(skill_id, count):
        calls.append(skill_id)
        return batches[len(calls) - 1]

    check = solver(first, second)
    monkeypatch.setattr(bank, "generate_batch", write)
    monkeypatch.setattr(bank, "verify",
                        lambda skill_id, qs: audit.verify(skill_id, qs, check=check))
    return calls


def _shelved():
    connection = store.connect()
    try:
        return sorted(r["question"] for r in connection.execute(
            "SELECT question FROM question_bank"))
    finally:
        connection.close()


def test_only_checked_questions_reach_the_shelf(empty_shelf, monkeypatch):
    _with(monkeypatch, [[q("good"), q("bad"), q("also good")]],
          first={"bad": "disagree"}, second={"bad": "disagree"})

    _, asked = bank.question_for(SKILL)

    assert asked.question in ("good", "also good")
    assert _shelved() == ["also good", "good"]


def test_a_batch_that_all_fails_is_written_again(empty_shelf, monkeypatch):
    calls = _with(monkeypatch, [[q("wrong")], [q("right")]],
                  first={"wrong": "disagree"}, second={"wrong": "disagree"})

    _, asked = bank.question_for(SKILL)

    assert len(calls) == 2
    assert asked.question == "right"
    assert _shelved() == ["right"]


def test_a_question_judged_wrong_is_never_asked(empty_shelf, monkeypatch):
    """Twice over, everything wrong. Stopping the session is better than
    marking a student wrong for being right."""
    calls = _with(monkeypatch, [[q("wrong 1")], [q("wrong 2")]],
                  first={"wrong 1": "disagree", "wrong 2": "disagree"},
                  second={"wrong 1": "disagree", "wrong 2": "disagree"})

    with pytest.raises(BadQuestionError):
        bank.question_for(SKILL)
    assert len(calls) == bank.ATTEMPTS
    assert _shelved() == []


def test_an_unreachable_checker_still_lets_the_session_go_on(empty_shelf, monkeypatch):
    """The API being down is not evidence against the question, so one is
    asked - but it is not kept, so it is never served again unchecked."""
    _with(monkeypatch, [[q("unseen")], [q("unseen too")]],
          first={"unseen": "fail", "unseen too": "fail"})

    banked_id, asked = bank.question_for(SKILL)

    assert banked_id is None
    assert asked.question in ("unseen", "unseen too")
    assert _shelved() == []


def test_filling_keeps_only_what_passed(empty_shelf, monkeypatch, capsys):
    _with(monkeypatch, [[q("good"), q("bad"), q("unchecked")]],
          first={"bad": "disagree", "unchecked": "fail"},
          second={"bad": "disagree"})

    kept = bank.fill([SKILL], per_skill=3, workers=1)

    assert kept == 1
    assert _shelved() == ["good"]
    said = capsys.readouterr().out
    assert "kept 1 of 3" in said
    assert "1 judged wrong" in said and "1 could not be checked" in said
