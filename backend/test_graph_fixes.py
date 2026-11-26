"""Places in the graph that were found sending a student the wrong way.

Each was found by testing as a student on 2026-11-26. These hold them fixed.
"""

import json
import pathlib

from graph import SKILLS


def test_binomial_coefficients_is_about_using_them():
    """Asked "which formula gives it", a student who uses nCr or Pascal's
    triangle says "I don't know" and is walked down for nothing."""
    skill = SKILLS["binomial_coefficients"]
    assert skill.kind == "procedure"
    assert "which formula" in skill.probe and "must not ask" in skill.probe


def test_under_binomial_coefficients_is_pascals_triangle_not_fractions():
    needs = SKILLS["binomial_coefficients"].needs
    assert list(needs) == ["pascal_triangle"]
    assert list(SKILLS["pascal_triangle"].needs) == []


def test_factorials_were_left_out_on_purpose():
    """A student using nCr or the triangle never touches them."""
    assert not any("factorial" in s.id for s in SKILLS.values())


def test_a_frequency_table_does_not_need_algebra():
    assert "multiply_terms" not in SKILLS["multiply_by_frequency"].needs


def test_clearing_a_surd_does_not_need_algebraic_fractions():
    needs = SKILLS["rationalise_denominator"].needs
    assert "algebraic_fractions" not in needs
    assert "equivalent_fractions" in needs


NUMBER_SKILLS = ["fractions_arith", "common_denominator", "equivalent_fractions",
                 "reciprocal", "add_subtract_fractions", "combine_over_common_denominator",
                 "multiply_divide_fractions", "fraction_of_amount", "multiply_by_frequency"]


def test_plain_number_skills_say_so_to_the_question_writer():
    """Without it, "Finding a common denominator" was written as 3/(2x) and
    5/(3x) - algebraic fractions, harder than the skill above it."""
    for skill_id in NUMBER_SKILLS:
        assert "no letters" in SKILLS[skill_id].probe, skill_id


def test_the_questions_that_tested_wording_are_off_the_shelf():
    shelf = pathlib.Path(__file__).resolve().parent.parent / "data" / "question_bank.jsonl"
    live = [json.loads(line) for line in shelf.read_text(encoding="utf-8").splitlines()
            if line.strip() and not json.loads(line)["retired"]]
    recall = [r["question"] for r in live if r["skill_id"] == "binomial_coefficients"
              and "formula" in r["question"].lower()]
    assert recall == []
