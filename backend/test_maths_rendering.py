"""Tests the notation converter that sits in front of KaTeX.

The bank stores maths the way a person types it - x^(-2), sqrt(136),
(4x^2 + 3x)/(x - 2) - and KaTeX needs LaTeX, so web/index.html converts at the
moment of display. That conversion is regex work on real student-facing text,
which is exactly the kind of code that looks right and is not.

These run the actual JavaScript out of the page rather than a Python
translation of it, so what is tested is what ships. KaTeX itself is stubbed to
hand back the LaTeX it was given, which is the part worth asserting on - that
it typesets correctly is KaTeX's problem, not ours.

Skipped rather than failed when no JavaScript engine is installed, so the suite
still runs on a machine without one.

    pip install quickjs
"""

import re
from pathlib import Path

import pytest

dukpy = pytest.importorskip("dukpy", reason="needs a JS engine: pip install dukpy")

PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"

STUBS = """
  // Stand-ins for the two things the converter borrows from the page.
  function escape(text) {
    return String(text === null || text === undefined ? '' : text)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  // Hands back the LaTeX it was given, wrapped so a test can see which parts
  // of a sentence were treated as maths.
  var katex = {renderToString: function (latex) { return '[[' + latex + ']]'; }};
  var window = {katex: katex};
"""


def _the_maths_code() -> str:
    """Lift the converter out of the page, so the test cannot drift from it."""
    source = PAGE.read_text(encoding="utf-8")
    start = source.index("// ---- Maths ---")
    end = source.index("function escape(text) {", start)
    return source[start:end]


@pytest.fixture(scope="module")
def js():
    return STUBS + _the_maths_code()


def render(js, text: str) -> str:
    """Run the page's own converter over one string."""
    return dukpy.evaljs(js + "\nmathHtml(dukpy['input']);", input=text)


def latex_in(rendered: str) -> list[str]:
    """Just the bits that were handed to KaTeX."""
    return re.findall(r"\[\[(.*?)\]\]", rendered, re.S)


# ---- Powers ----------------------------------------------------------------


def test_a_bare_power_is_braced(js):
    """x^12 without braces typesets as x^1 followed by a loose 2."""
    assert "x^{12}" in latex_in(render(js, "Evaluate x^12"))[0]


def test_negative_power_in_brackets(js):
    assert "x^{-2}" in latex_in(render(js, "Differentiate 2x^(-2)"))[0]


def test_subscript(js):
    assert "S_{n}" in latex_in(render(js, "The sum S_n of the series"))[0]


# ---- Roots and fractions ---------------------------------------------------


def test_square_root(js):
    assert "\\sqrt{136}" in latex_in(render(js, "distance sqrt(136)"))[0]


def test_a_simple_fraction_is_stacked(js):
    assert "\\frac{3}{4}" in latex_in(render(js, "Calculate 3/4 exactly"))[0]


def test_a_bracketed_quotient_loses_its_redundant_brackets(js):
    latex = latex_in(render(js, "Differentiate (4x^2 + 3x)/(x - 2) fully"))[0]

    assert "\\frac{" in latex
    assert "4x^{2} + 3x" in latex
    assert "x - 2" in latex


# ---- Leaving the sentence alone --------------------------------------------


def test_prose_is_not_typeset(js):
    """The dangerous failure: a sentence swallowed into an expression."""
    rendered = render(js, "Which of the following states the meaning of a negative index?")

    assert latex_in(rendered) == []
    assert "Which of the following" in rendered


def test_only_the_notation_inside_a_sentence_is_typeset(js):
    rendered = render(js, "Differentiate 2x^(-2) with respect to x and simplify")
    handed_over = latex_in(rendered)

    assert len(handed_over) == 1
    assert "2x^{-2}" in handed_over[0]
    # The words on either side stayed words.
    assert "with respect to" in rendered
    assert "simplify" in rendered


def test_short_words_do_not_get_dragged_into_an_expression(js):
    """'of' and 'to' are two letters, which is how a run swallows a sentence."""
    rendered = render(js, "Find the derivative of (4x^2 + 3x)/(x - 2) and simplify.")

    assert "of" in rendered
    assert "and simplify" in rendered
    for latex in latex_in(rendered):
        assert " of " not in latex
        assert "and" not in latex


def test_a_full_stop_is_not_part_of_the_expression(js):
    rendered = render(js, "Evaluate 2^6.")

    assert latex_in(rendered)[0].endswith("2^{6}")
    assert rendered.rstrip().endswith(".")


def test_an_expression_with_no_notation_is_left_as_text(js):
    """Re-rendering something already readable only risks making it worse."""
    assert latex_in(render(js, "Expand (3 - x)(2 + 5x)")) == []


# ---- Cases the real bank turned up -----------------------------------------


def test_a_root_with_a_coefficient_in_front(js):
    """4sqrt(3) is how the bank writes it. Matching sqrt on a word boundary
    missed this while catching sqrt(12) in the same sentence, which reads worse
    than typesetting neither."""
    latex = latex_in(render(js, "The answer is 4sqrt(3) exactly"))[0]

    assert "4\\sqrt{3}" in latex


def test_a_quotient_with_spaces_around_the_slash(js):
    """An integrand arrives as 'sin(x) / cos(x)'. Requiring non-space either
    side of the slash missed exactly the shape a quotient comes in."""
    latex = latex_in(render(js, "Consider sin(x) / cos(x) dx here"))[0]

    assert "\\frac" in latex


def test_a_differential_stays_in_the_expression(js):
    """dx has no digit and no operator, so nothing else would let it join a
    run, and it sits in the middle of one."""
    assert "dx" in latex_in(render(js, "Consider sin(x) / cos(x) dx here"))[0]


def test_a_log_with_a_base(js):
    """log_3 - the trailing word boundary fails because _ is not a letter, so
    the log rendered as three italic variables."""
    latex = latex_in(render(js, "Solve log_3(2x + 1) = 2"))[0]

    assert "\\log" in latex


def test_a_function_name_inside_a_longer_word_is_left_alone(js):
    """Dropping the word boundary must not start rewriting English."""
    assert "\\log" not in "".join(latex_in(render(js, "The logarithm of x^2")))


def test_dy_by_dx_becomes_a_stacked_fraction(js):
    latex = latex_in(render(js, "Find dy/dx where y = x^2"))[0]

    assert "\\frac{dy}{dx}" in latex


# ---- Not breaking ----------------------------------------------------------


def test_empty_text_is_fine(js):
    assert render(js, "") == ""


def test_a_tag_never_reaches_the_page_intact(js):
    """Question text comes from a model and goes into innerHTML, so no path
    through here may leave a tag standing.

    There are two paths and they neutralise it differently: prose is escaped,
    and a run that gets typeset has its angle brackets turned into LaTeX
    relations. What matters is that neither leaves '<script>' behind, not
    which of the two did the work.
    """
    rendered = render(js, "Is 3 < 5 and 7 > 2 <script>alert(1)</script>")

    assert "<script>" not in rendered
    assert "</script>" not in rendered
    assert "<" not in rendered.replace("&lt;", "")


def test_a_tag_is_escaped_when_it_is_plainly_prose(js):
    """The other path. A closing tag carries a slash, which is enough to look
    like a fraction and get typeset, so this uses an opening tag alone to be
    sure the escaping route is the one under test."""
    rendered = render(js, "Careful with <b>bold text")

    assert "<b>" not in rendered
    assert "&lt;b&gt;" in rendered


def test_a_plain_number_option_survives(js):
    assert "64" in render(js, "64")


# ---- The one that reached a student ----------------------------------------


def test_a_fraction_inside_an_exponent(js):
    """Seen on a student's screen: x^(1/2) came out as \frac{x^{1}{2}}, which
    KaTeX prints in red, and the equals sign after it was swallowed into the
    denominator.

    An operand may contain braces, because a fraction can sit over something
    already converted. So once the exponent rule had turned this into x^{1/2},
    the fraction rule matched x^{1 over 2}. Fractions run first now, while the
    brackets are still brackets.
    """
    assert r"x^{\frac{1}{2}}" in latex_in(render(js, "what x^(1/2) means"))[0]


def test_the_whole_option_as_it_was_stored(js):
    """The text in the bank, start to finish, with nothing left over."""
    written = latex_in(render(js, "x^(1/2) = sqrt(x)"))[0]

    assert r"x^{\frac{1}{2}}" in written
    assert r"\sqrt{x}" in written
    # The equals stayed where it was rather than becoming a denominator.
    assert "=" in written.split(r"\sqrt")[0]


def test_a_negative_fractional_power(js):
    """The sign comes out in front of the fraction rather than inside it, which
    is how it is written on paper."""
    assert r"x^{-\frac{1}{2}}" in latex_in(render(js, "x^(-1/2)"))[0]


def test_a_fraction_over_a_power_still_works(js):
    """The order changed, so what was already right has to stay right. Here the
    division is outside the exponent rather than inside it."""
    assert r"\frac{3}{x^{2}}" in latex_in(render(js, "3/x^2"))[0]


def test_a_power_over_a_number_still_works(js):
    assert r"\frac{x^{2}}{3}" in latex_in(render(js, "x^2/3"))[0]
