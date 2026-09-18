"""Why they answered that, not just whether it was right.

The walk already takes one bit of evidence per question: right or wrong, and
which wrong option. That is more than most things do - each distractor carries
a named misconception - but it is still a guess about the student made from four
prepared boxes. A student who picks "5" because they substituted into the
original function and a student who picks "5" because they added instead of
multiplying are recorded identically.

So: ask them. "How did you get that?" in their own words, and read the answer.

WHAT THIS IS NOT

It does not decide the diagnosis and it does not steer the walk. The walk is
deterministic from the answers given, which is what lets a student close the tab
and come back, and what makes eval.py able to measure it at all. Letting a
free-text reading move the walk is a real change to that and is deliberately not
this.

What it does is record what actually happened, so the report can say "you
differentiated correctly and then substituted into the wrong expression" instead
of "wrong", and so there is finally evidence about the thing every distractor
claims: that the wrong options are mistakes real students make.

WHY SONNET

This is reading, not deciding. It never moves the walk, so being wrong costs a
sentence in a report rather than a diagnosis - which is the opposite of the
entry match, where a misread sends the whole session somewhere untrue. Three
times cheaper, and the student is waiting while it runs.

ALWAYS OPTIONAL

Typing an explanation is work, and most students will not. The walk must be
identical whether they do or not - skipping is not a lesser path, it is the
normal one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import llm
from graph import SKILLS

load_dotenv(Path(__file__).resolve().parent / ".env")

READ = llm.Task(llm.SONNET, effort="medium")

# What a student can be doing when they get something wrong. Written as things
# a teacher would recognise rather than as categories, because the point is to
# say something true about this student and not to file them.
PATTERNS = (
    "arithmetic_slip",          # knew the method, dropped a sign or a number
    "wrong_operation",          # did something else to the right objects
    "right_rule_wrong_place",   # a real rule, applied where it does not hold
    "misread_the_question",     # answered a different question correctly
    "pieces_not_connected",     # holds each step, could not assemble them
    "no_starting_point",        # did not know what to do first at all
    "prerequisite_gap",         # something underneath is missing
    "misconception",            # believes something that is not true
    "guessed",                  # no reasoning offered or claimed
    "actually_right",           # their reasoning is sound; the option or our
                                # key is what is wrong
)

# How long a student's explanation may be. Long enough for "i differentiated and
# got 2x-4 then put 3 in", short enough that nobody is pasting an essay.
MAX_WORDS = 400


class Reading(BaseModel):
    """What their explanation says about how they were thinking."""

    pattern: Literal[PATTERNS] = Field(  # type: ignore[valid-type]
        description="The single closest match. Use guessed only when they say so "
                    "or offer no reasoning at all.")
    # Quoted or closely paraphrased from what they wrote. A conclusion with no
    # evidence is the thing that makes an AI diagnosis untrustworthy, and this
    # is shown to a tutor who will check it.
    evidence: str = Field(
        description="The part of what they wrote that shows this, in their words.")
    # What they demonstrably did right on the way. Usually the more useful half.
    got_right: str = Field(
        default="",
        description="What their explanation shows they CAN do. Empty if nothing.")
    # Said to the student, so it has to be true and worth reading.
    said_back: str = Field(
        description="One or two sentences telling them what happened, in plain "
                    "words. Not a mark and not praise. Name the step where it "
                    "went, and what was fine before it.")


SYSTEM = (
    "A student has just answered a maths question wrongly and said how they got "
    "their answer. Work out what they were actually doing.\n"
    "You are not marking them and you are not teaching them. You are reading "
    "one piece of evidence about how somebody thinks.\n"
    "Be exact about where it went. 'They differentiated correctly and then "
    "substituted into the original function' is worth something; 'they made an "
    "error' is not. If what they wrote shows they can do most of it, say so - "
    "that is usually the more useful half.\n"
    "If their reasoning is actually sound and the option they picked is right, "
    "say so with actually_right. Our answer key can be wrong and a student "
    "should never be told they are wrong when they are not."
)


class NotWorthReading(RuntimeError):
    """Too little to read. Never fatal - an explanation is always optional."""


def build_prompt(*, question: str, chosen: str, correct: str,
                 said: str, skill_id: str) -> str:
    skill = SKILLS[skill_id]
    rests_on = "\n".join(
        f"- {SKILLS[n].name}: {SKILLS[n].probe}"
        for n in skill.needs if n in SKILLS) or "- nothing; this is a floor skill"

    return (
        f"THE QUESTION\n{question}\n\n"
        f"WHAT THEY PICKED\n{chosen}\n\n"
        f"THE ANSWER THAT WAS RIGHT\n{correct}\n\n"
        f"HOW THEY SAY THEY GOT IT\n{said}\n\n"
        f"WHAT THIS QUESTION IS TESTING\n{skill.name}: {skill.probe}\n\n"
        f"WHAT THAT RESTS ON\n{rests_on}\n\n"
        "Read their explanation and say what they were doing. Quote the part of "
        "it that shows you. Say what they got right on the way, if anything.\n"
        "Then say it back to them in one or two sentences - plain words, no "
        "jargon they would have to look up, and nothing that implies they should "
        "already have known this."
    )


def read(*, question: str, chosen: str, correct: str, said: str,
         skill_id: str, client: Anthropic | None = None) -> Reading:
    """Read one explanation. Raises NotWorthReading if there is nothing in it."""
    words = (said or "").split()
    if len(words) < 2:
        raise NotWorthReading("too short to read")
    if len(words) > MAX_WORDS:
        said = " ".join(words[:MAX_WORDS])

    client = client or Anthropic()
    response = client.messages.parse(
        **READ.kwargs(),
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(
            question=question, chosen=chosen, correct=correct,
            said=said, skill_id=skill_id)}],
        output_format=Reading,
    )

    if response.stop_reason == "refusal":
        raise NotWorthReading("the reader declined")
    got = response.parsed_output
    if got is None:
        raise NotWorthReading(f"nothing came back ({response.stop_reason})")
    return got
