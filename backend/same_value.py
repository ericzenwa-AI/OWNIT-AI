"""Catch a wrong option that is not actually wrong, or not actually an answer.

"Work out 3/4 x 2/5 + 1/10" was served with 4/10 as the answer and
"6/20 + 1/10" as a distractor. Those are the same number. A student who
picked the second had made no mistake - they had stopped one step early -
and was marked as not holding fractions.

The blind solve does not catch this: it only asks whether a solver lands on
the key, and it does. So this asks the other question, with plain arithmetic
and no model: does any distractor have exactly the same value as the key?

Only options that are pure numbers - integers, decimals, fractions, + - x /,
whole-number powers, brackets - are judged. Anything with a letter in it, a
root, or words is left alone, because "not proven equal" is not "different".

A question that asks for a form ("simplest form", "fully", "standard
form") is let off: there an unsimplified equal value is a fair wrong answer.
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction

# Wording that makes the form of the answer part of the question.
ASKS_FOR_A_FORM = re.compile(
    r"simplest|simplif|lowest terms|fully|standard form|mixed number|"
    r"improper|as a (fraction|decimal|percentage|mixed)|in the form|"
    r"exact|surd|recurring|significant|decimal place|nearest|round|"
    r"denominator|numerator",
    re.IGNORECASE)


def _tidy(text: str) -> str:
    s = text.strip().strip("$").strip()
    # \frac{a}{b} and \dfrac{a}{b}, innermost first.
    frac = re.compile(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}")
    while frac.search(s):
        s = frac.sub(r"((\1)/(\2))", s)
    s = (s.replace("\\left", "").replace("\\right", "")
          .replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
          .replace("×", "*").replace("÷", "/").replace("·", "*")
          .replace("−", "-").replace("–", "-").replace("^", "**")
          .replace("{", "(").replace("}", ")"))
    return s


def value_of(text: str) -> Fraction | None:
    """The exact value of a purely numeric option, or None if it is not one."""
    s = _tidy(text)
    if not s or re.search(r"[A-Za-z\\=<>%,√π]", s):
        return None
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                if b == 0:
                    raise ValueError("division by zero")
                return a / b
            if isinstance(node.op, ast.Pow):
                if b.denominator != 1 or abs(b) > 12:
                    raise ValueError("not a whole power")
                return a ** int(b)
        raise ValueError("not plain arithmetic")

    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def same_as_the_key(question: str, correct: str, distractors: list[str]) -> list[str]:
    """Distractors whose value is exactly the key's. Empty means none found."""
    if ASKS_FOR_A_FORM.search(question):
        return []
    key = value_of(correct)
    if key is None:
        return []
    return [d for d in distractors if value_of(d) == key]


def two_right_answers(question: str, correct: str, distractors: list[str]) -> list[str]:
    """Distractors that are also right: the key word for word, or its value.

    The same text twice is wrong whatever the question asks. The same value
    is wrong unless the question asks for a particular form.
    """
    def plain(t):
        return re.sub(r"\s+", "", t).lower()

    repeated = [d for d in distractors if plain(d) == plain(correct)]
    equal = same_as_the_key(question, correct, distractors)
    return repeated + [d for d in equal if d not in repeated]


# One number: a whole number, a decimal, or one fraction - the shape a
# finished answer to "work out" takes.
_ONE_NUMBER = re.compile(r"^\(?-?\d+(\.\d+)?\)?(/\(?-?\d+(\.\d+)?\)?)?$")


def _one_number(text: str) -> bool:
    return bool(_ONE_NUMBER.match(_tidy(text).replace(" ", "")))


def unfinished(question: str, correct: str, distractors: list[str]) -> list[str]:
    """Wrong options that are a step of working rather than an answer.

    When the answer is one number, every option has to be one number too.
    "6/20 + 1/10" beside "4/10" is not a wrong answer - it is the working,
    stopped early - and whether or not it adds up to the key, it gives away
    that something still needs doing, or marks a student down for stopping.
    """
    if not _one_number(correct):
        return []
    return [d for d in distractors
            if value_of(d) is not None and not _one_number(d)]


def what_is_wrong(question: str, correct: str, distractors: list[str]) -> str | None:
    """Why this question cannot be asked, in a few words, or None if it can."""
    if two_right_answers(question, correct, distractors):
        return "a wrong option is also right"
    if unfinished(question, correct, distractors):
        return "a wrong option is working, not an answer"
    return None
