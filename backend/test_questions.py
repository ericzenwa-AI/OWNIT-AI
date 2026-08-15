"""Tests for presenting a question and reading the answer.

No API calls: these build a question by hand and drive ask_in_terminal with a
scripted input function.

Run from the repo root with:  pytest backend/
"""

import pytest

from questions import (
    DONT_KNOW_LABEL,
    DONT_KNOW_OPTION,
    Distractor,
    MultipleChoiceQuestion,
    ask_in_terminal,
    build_prompt,
    shuffled_options,
)
from graph import SKILLS

QUESTION = MultipleChoiceQuestion(
    question="Differentiate x^2.",
    correct_option="2x",
    distractors=[
        Distractor(option="x", mistake="dropped the coefficient"),
        Distractor(option="2", mistake="differentiated twice"),
        Distractor(option="x^3/3", mistake="integrated instead"),
    ],
)


def typing(*picks):
    """A student who types these answers, in order."""
    answers = iter(picks)
    return lambda prompt: next(answers)


@pytest.fixture
def unshuffled(monkeypatch):
    """Pin the option order so a test can name a specific option by label.

    Without this, every call reshuffles and A is a different option each time.
    Order becomes: A = correct, then the distractors as declared.
    """
    monkeypatch.setattr("random.shuffle", lambda seq: None)


# ---- The fixed option -----------------------------------------------------


def test_dont_know_is_shown_last(capsys):
    ask_in_terminal(QUESTION, input_fn=typing("A", "1"))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    option_lines = [line for line in lines if line.strip()[1:3] == ") "]

    assert len(option_lines) == 5
    assert option_lines[-1].strip() == f"{DONT_KNOW_LABEL}) {DONT_KNOW_OPTION}"


def test_dont_know_is_never_shuffled(capsys):
    """Its position must not move between questions - students rely on that."""
    for _ in range(20):
        ask_in_terminal(QUESTION, input_fn=typing("A", "1"))
        lines = capsys.readouterr().out.splitlines()
        option_lines = [line for line in lines if line.strip()[1:3] == ") "]
        assert option_lines[-1].strip().startswith(f"{DONT_KNOW_LABEL})")


def test_dont_know_is_not_a_generated_option():
    """It is bolted on at display time, not written by the model."""
    options = [text for text, _ in shuffled_options(QUESTION)]
    assert len(options) == 4
    assert DONT_KNOW_OPTION not in options


def test_the_model_is_never_asked_for_it():
    prompt = build_prompt(SKILLS["deriv_sin_cos"])
    assert "don't know" not in prompt.lower()
    assert "exactly 3 entries" in prompt or "exactly 3" in prompt


# ---- What a pick means ----------------------------------------------------


def test_picking_dont_know_is_recorded_separately():
    answered = ask_in_terminal(QUESTION, input_fn=typing(DONT_KNOW_LABEL))

    assert answered.dont_know is True
    assert answered.correct is False
    # Nothing to name: they didn't reach for a rule at all.
    assert answered.mistake is None
    assert answered.chosen == DONT_KNOW_OPTION


def test_a_wrong_answer_is_not_a_dont_know(unshuffled):
    """Both fail, but one of them names a rule the student is holding."""
    expected = [d.mistake for d in QUESTION.distractors]

    for label, mistake in zip("BCD", expected):
        answered = ask_in_terminal(QUESTION, input_fn=typing(label, "1"))
        assert answered.correct is False
        assert answered.dont_know is False
        assert answered.mistake == mistake


def test_a_correct_answer_carries_no_mistake(unshuffled):
    answered = ask_in_terminal(QUESTION, input_fn=typing("A", "1"))

    assert answered.correct is True
    assert answered.mistake is None
    assert answered.dont_know is False
    assert answered.chosen == "2x"


# ---- How sure they were ---------------------------------------------------


def test_confidence_is_recorded(unshuffled):
    """A right answer they guessed is not the same as one they knew."""
    guessed = ask_in_terminal(QUESTION, input_fn=typing("A", "3"))
    assert guessed.correct is True
    assert guessed.confidence == "guess"

    known = ask_in_terminal(QUESTION, input_fn=typing("A", "1"))
    assert known.confidence == "sure"


def test_being_sure_and_wrong_is_captured(unshuffled):
    """The most useful signal there is - a rule they believe and it is wrong."""
    answered = ask_in_terminal(QUESTION, input_fn=typing("B", "1"))
    assert answered.correct is False
    assert answered.confidence == "sure"
    assert answered.mistake == "dropped the coefficient"


def test_dont_know_is_not_asked_how_sure_it_is():
    """They have already told us. Asking again is friction for no signal."""
    answered = ask_in_terminal(QUESTION, input_fn=typing(DONT_KNOW_LABEL))
    assert answered.confidence == "guess"


def test_an_unreadable_confidence_is_asked_again(unshuffled):
    answered = ask_in_terminal(QUESTION, input_fn=typing("A", "9", "2"))
    assert answered.confidence == "think"


def test_how_long_they_took_is_recorded(unshuffled):
    answered = ask_in_terminal(QUESTION, input_fn=typing("A", "1"))
    assert answered.seconds is not None
    assert answered.seconds >= 0


# ---- Input handling -------------------------------------------------------


def test_rejects_a_label_that_is_not_offered():
    """F is past the end; the student is asked again rather than crashing."""
    answered = ask_in_terminal(QUESTION, input_fn=typing("F", DONT_KNOW_LABEL))
    assert answered.dont_know is True


def test_accepts_lower_case_and_stray_spaces():
    answered = ask_in_terminal(QUESTION, input_fn=typing(f" {DONT_KNOW_LABEL.lower()} "))
    assert answered.dont_know is True
