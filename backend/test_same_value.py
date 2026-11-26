"""A question must not have two right answers.

The one that was served: "Work out 3/4 x 2/5 + 1/10", key 4/10, with
"6/20 + 1/10" offered as a wrong answer. Same number. No API calls here -
the check is plain arithmetic, and it runs before the paid one.
"""

import json
import pathlib
from fractions import Fraction

import audit
from questions import Distractor, MultipleChoiceQuestion
from same_value import two_right_answers, value_of


def caught(question, key, *wrong):
    return two_right_answers(question, key, list(wrong))


# ---- the ones that were on the shelf --------------------------------------


def test_the_one_that_was_served():
    assert caught("Work out 3/4 * 2/5 + 1/10", "4/10",
                  "7/19", "6/20 + 1/10", "5/10") == ["6/20 + 1/10"]


def test_an_unsimplified_right_answer_is_still_right():
    assert caught("Work out 2/3 * 9/5", "6/5", "18/15", "11/8") == ["18/15"]


def test_the_same_option_twice():
    assert caught("Find the median of 9, 7, 3, 7, 5, 2, 9, 7", "7", "7", "6", "8") == ["7"]


def test_the_formula_left_unworked_is_the_answer():
    assert caught("Find the sum to infinity.", "25", "20/0.8", "24", "100") == ["20/0.8"]


def test_written_in_latex_it_is_still_caught():
    assert caught("Work out", r"\frac{4}{10}", r"\frac{6}{20} + \frac{1}{10}") == [
        r"\frac{6}{20} + \frac{1}{10}"]


def test_same_words_twice_is_caught_whatever_they_are():
    assert caught("Which is true?", "Add the powers", "add the  powers", "Multiply") == [
        "add the  powers"]


# ---- fair wrong answers that must be let through ---------------------------


def test_simplest_form_makes_an_unsimplified_answer_wrong():
    assert caught("Write 18/24 in its simplest form.", "3/4", "9/12", "6/8") == []


def test_standard_form_makes_an_unnormalised_one_wrong():
    assert caught("Give your answer in standard form.", "3*10^8", "30*10^7") == []


def test_a_named_denominator_makes_another_equivalent_wrong():
    assert caught("Which fraction is equivalent to 5/8, with denominator 40?",
                  "25/40", "40/64", "5/40") == []


def test_genuinely_different_numbers_are_left_alone():
    assert caught("Work out -4 - 5", "-9", "-1", "9", "1") == []


def test_letters_are_not_judged():
    """Not proven equal is not the same as different - and not this check's job."""
    assert caught("Expand 2(x + 3)", "2x + 6", "2x + 3", "x + 6") == []
    assert value_of("2x + 6") is None


def test_values_are_exact_not_rounded():
    assert value_of("1/3") == Fraction(1, 3)
    assert value_of("0.333") != value_of("1/3")
    assert value_of("2^-2") == Fraction(1, 4)
    assert value_of("−3 × 4") == -12


# ---- it runs first, and costs nothing --------------------------------------


def q(text, key, *wrong):
    return MultipleChoiceQuestion(question=text, correct_option=key, distractors=[
        Distractor(option=w, mistake="m") for w in wrong])


def test_a_question_with_two_right_answers_never_reaches_the_paid_check():
    solved = []

    def check(row, task):
        solved.append(row["question"])
        return audit.Checked(row["id"], row["skill_id"], row["question"],
                             row["correct_option"], row["correct_option"], "", "")

    bad = q("Work out 3/4 * 2/5 + 1/10", "4/10", "7/19", "6/20 + 1/10", "5/10")
    good = q("Work out -4 - 5", "-9", "-1", "9", "1")
    result = audit.verify("fractions_arith", [bad, good], check=check)

    assert result.rejected == [bad]
    assert result.passed == [good]
    assert solved == ["Work out -4 - 5"]


def test_nothing_left_on_the_shelf_has_two_right_answers():
    shelf = pathlib.Path(__file__).resolve().parent.parent / "data" / "question_bank.jsonl"
    live = [json.loads(line) for line in shelf.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    broken = [r["question"] for r in live if not r.get("retired") and two_right_answers(
        r["question"], r["correct_option"], [d["option"] for d in r["distractors"]])]
    assert broken == []
