"""Generates one multiple-choice question that tests a single skill.

The skill's `probe` says what a question testing that skill alone must require,
and its `kind` says what shape the question has to take. A recall question and a
do-it question look nothing alike, so generating the same shape for all three
kinds would tell us nothing about which skill is actually missing.

Run it for one skill:

    python backend/questions.py deriv_sin_cos
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

import llm
from graph import SKILLS, Skill

# The key lives in backend/.env, next to this file.
load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "claude-opus-5"
MAX_TOKENS = 16000
DISTRACTOR_COUNT = 3


class UnknownSkillError(ValueError):
    """Raised when the requested skill id is not in the graph."""


class BadQuestionError(RuntimeError):
    """Raised when the model returns something we can't use."""


class Distractor(BaseModel):
    """A wrong option, paired with the mistake that leads a student to it."""

    option: str
    mistake: str


class MultipleChoiceQuestion(BaseModel):
    question: str
    correct_option: str
    distractors: list[Distractor]


@dataclass(frozen=True)
class Answered:
    """What a student picked, and what picking it means."""

    chosen: str
    correct: bool
    # The misconception behind the option they picked. None when they got it
    # right, and None when they said they didn't know - there is no wrong rule
    # to name in that case.
    mistake: str | None
    # A wrong answer and "I don't know" are both failures, but they are not the
    # same failure. A wrong answer means the student has a rule in their head
    # and it is the wrong one. This means there is nothing there to correct yet.
    dont_know: bool = False
    # How long they took. Recorded because it is free and invisible to the
    # student; nothing reads it yet.
    seconds: float | None = None


# What each kind of skill has to be asked about. These are the three question
# shapes; the probe supplies the content.
QUESTION_SHAPES = {
    "procedure": (
        "Pose a task the student has to carry out, with a numerical or "
        "algebraic answer. The correct option is the result of doing it "
        "properly; each wrong option is the result a student actually reaches "
        "after one specific slip in the working."
    ),
    "fact": (
        "Ask the student to state a rule, standard result, or definition. The "
        "correct option is the rule as it is; each wrong option is a rule they "
        "have confused it with or half-remembered."
    ),
    "concept": (
        "Give a claim, a piece of working, or a situation, and ask which "
        "judgement about it is right and why. The correct option gives the "
        "real reason; each wrong option is a reason that sounds plausible but "
        "rests on a specific misconception. It must not be answerable by "
        "carrying out a procedure."
    ),
}


SYSTEM_PROMPT = (
    "You write diagnostic multiple-choice questions for GCSE and A-level "
    "maths. Pitch each one at the skill you are given rather than at a "
    "qualification: the same graph holds \"say what the top and bottom of a "
    "fraction each tell you\" and \"differentiate from first principles\", and "
    "a question about the first written for an A-level audience tests nothing. "
    "Each "
    "question tests exactly one skill, and every wrong option corresponds to "
    "one specific mistake a real student makes, so that the option a student "
    "picks tells you what they are missing."
)


def _prerequisite_names(skill: Skill) -> list[str]:
    return [SKILLS[need].name for need in skill.needs if need in SKILLS]


def _dependent_names(skill_id: str) -> list[str]:
    """Skills that sit above this one, and so must not be needed to answer."""
    return sorted(s.name for s in SKILLS.values() if skill_id in s.needs)


def build_prompt(skill: Skill) -> str:
    """Assemble the user turn for one skill."""
    prerequisites = _prerequisite_names(skill)
    dependents = _dependent_names(skill.id)

    assumed = ", ".join(prerequisites) if prerequisites else "basic arithmetic only"
    forbidden = (
        ", ".join(dependents) if dependents else "any skill built on top of this one"
    )

    return (
        "SKILL\n"
        f"name: {skill.name}\n"
        f"kind: {skill.kind}\n"
        f"probe: {skill.probe}\n\n"
        "The probe describes what a question testing this skill alone must "
        "require. Write one question that does exactly that.\n\n"
        f"SHAPE (this skill is a {skill.kind})\n"
        f"{QUESTION_SHAPES[skill.kind]}\n\n"
        "SCOPE\n"
        f"- Assume the student already has: {assumed}.\n"
        f"- The question must not require: {forbidden}.\n"
        "- A student who holds this skill and nothing above it should be able "
        "to answer it.\n\n"
        "OPTIONS\n"
        "- The question has four options in total: one correct, in "
        "`correct_option`, and three wrong, in `distractors`. The "
        f"`distractors` array must hold exactly {DISTRACTOR_COUNT} entries, "
        "not four.\n"
        "- Each wrong option must be what a student would actually arrive at "
        "by making one specific mistake. Say what that mistake is, in terms of "
        "what the student did or believed. A wrong option nobody would pick "
        "teaches us nothing.\n"
        "- Do not label the options, order them, or say which is correct in "
        "the option text.\n"
        "- Write maths in plain ASCII: / for division, ^ for powers, sqrt() "
        "for roots, * for multiplication. No LaTeX, no non-ASCII symbols, and "
        "never an escape sequence like \\u00d7 - the student sees the raw text."
    )


def generate_question(
    skill_id: str,
    *,
    client: Anthropic | None = None,
    model: str | None = None,
) -> MultipleChoiceQuestion:
    """Ask Claude for one question testing `skill_id`, and nothing else."""
    skill = SKILLS.get(skill_id)
    if skill is None:
        raise UnknownSkillError(f"'{skill_id}' is not a skill in the graph")

    client = client or Anthropic()

    # Rebuild the task rather than patching the model into settings already
    # built. Effort is not universal - Haiku rejects it outright - so swapping
    # the model alone leaves a parameter behind that the new model refuses,
    # and the override fails with a 400 instead of just using another model.
    settings = (llm.QUESTION if not model else llm.for_model(model)).kwargs()

    response = client.messages.parse(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(skill)}],
        output_format=MultipleChoiceQuestion,
        **settings,
    )

    if response.stop_reason == "refusal":
        raise BadQuestionError(f"The model declined to answer for '{skill_id}'")

    question = response.parsed_output
    if question is None:
        raise BadQuestionError(
            f"No structured output for '{skill_id}' (stop_reason: {response.stop_reason})"
        )

    if len(question.distractors) != DISTRACTOR_COUNT:
        raise BadQuestionError(
            f"Expected {DISTRACTOR_COUNT} wrong options for '{skill_id}', "
            f"got {len(question.distractors)}"
        )

    return question


class QuestionSet(BaseModel):
    questions: list[MultipleChoiceQuestion]


def generate_batch(
    skill_id: str,
    count: int = 5,
    *,
    client: Anthropic | None = None,
    model: str | None = None,
) -> list[MultipleChoiceQuestion]:
    """Write several questions for one skill in a single call.

    Asking once for five is both cheaper than five calls and better, because
    the model can see them together and make them differ. Generated one at a
    time they come out as the same question with the numbers changed, which is
    no protection against a student meeting the skill twice.
    """
    skill = SKILLS.get(skill_id)
    if skill is None:
        raise UnknownSkillError(f"'{skill_id}' is not a skill in the graph")

    # Rebuild the task rather than patching the model into settings already
    # built. Effort is not universal - Haiku rejects it outright - so swapping
    # the model alone leaves a parameter behind that the new model refuses,
    # and the override fails with a 400 instead of just using another model.
    settings = (llm.QUESTION if not model else llm.for_model(model)).kwargs()

    response = client_or_new(client).messages.parse(
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{build_prompt(skill)}\n\n"
                    f"Write {count} questions like that, not one.\n"
                    "- They must be genuinely different, not the same question "
                    "with new numbers. Vary what is being asked and how it is "
                    "set out.\n"
                    "- Between them they should cover different mistakes. If "
                    "one is built around a sign error, another should not be.\n"
                    "- Every one must still test this skill alone, and still "
                    "have exactly one correct option and three wrong ones."
                ),
            }
        ],
        output_format=QuestionSet,
        **settings,
    )

    produced = response.parsed_output
    if produced is None:
        raise BadQuestionError(f"No questions came back for '{skill_id}'")

    # Keep the well-formed ones rather than losing the batch to one bad entry.
    return [
        question
        for question in produced.questions
        if len(question.distractors) == DISTRACTOR_COUNT
    ]


def client_or_new(client: Anthropic | None) -> Anthropic:
    return client or Anthropic()


# ---- Asking -------------------------------------------------------------


LABELS = "ABCD"

# Not a distractor. The model never writes this one and never sees it - it is
# fixed text bolted on at display time, always last, never shuffled in among
# the others, so a student always knows where to find it.
DONT_KNOW_LABEL = "E"
DONT_KNOW_OPTION = "I don't know"


def shuffled_options(
    question: MultipleChoiceQuestion,
) -> list[tuple[str, str | None]]:
    """The four generated options in random order, each paired with its mistake.

    The correct one carries None, which is what makes an answer scoreable.
    """
    options: list[tuple[str, str | None]] = [(question.correct_option, None)]
    options += [(d.option, d.mistake) for d in question.distractors]
    random.shuffle(options)
    return options


def ask_in_terminal(
    question: MultipleChoiceQuestion,
    header: str | None = None,
    input_fn=input,
) -> Answered:
    """Put the question to the student, without giving away which one is right."""
    options = shuffled_options(question)
    labels = LABELS[: len(options)]
    started = time.monotonic()

    if header:
        print(header)
    print(question.question)
    print()
    for label, (text, _) in zip(labels, options):
        print(f"  {label}) {text}")
    print(f"  {DONT_KNOW_LABEL}) {DONT_KNOW_OPTION}")
    print()

    valid = labels + DONT_KNOW_LABEL
    while True:
        pick = input_fn(
            f"Your answer ({labels[0]}-{labels[-1]}, "
            f"or {DONT_KNOW_LABEL} if you don't know): "
        ).strip().upper()
        if pick in valid:
            break
        print(f"Type one of {', '.join(valid)}.")

    seconds = round(time.monotonic() - started, 1)

    if pick == DONT_KNOW_LABEL:
        return Answered(
            chosen=DONT_KNOW_OPTION,
            correct=False,
            mistake=None,
            dont_know=True,
            seconds=seconds,
        )

    text, mistake = options[labels.index(pick)]
    return Answered(
        chosen=text, correct=mistake is None, mistake=mistake, seconds=seconds
    )


# ---- Command line ---------------------------------------------------------


def _print_skill_list() -> None:
    for skill in sorted(SKILLS.values(), key=lambda s: (s.level, s.id)):
        print(f"  level {skill.level}  {skill.kind:<9}  {skill.id:<28}  {skill.name}")


def _print_question(skill: Skill, question: MultipleChoiceQuestion) -> None:
    print(f"skill  {skill.id} - {skill.name}")
    print(f"       {skill.kind}, level {skill.level}")
    print(f"probe  {skill.probe}")
    print()
    print(question.question)
    print()

    # Shuffle so the correct answer isn't always first when eyeballing.
    for label, (text, mistake) in zip(LABELS, shuffled_options(question)):
        print(f"  {label}) {text}")
        print(f"     {'[correct]' if mistake is None else '[wrong] ' + mistake}")
        print()

    # Shown so this preview matches what a student actually sees.
    print(f"  {DONT_KNOW_LABEL}) {DONT_KNOW_OPTION}")
    print("     [not held, with no mistake to name]")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate one multiple-choice question for a single skill."
    )
    parser.add_argument("skill_id", nargs="?", help="a skill id from data/skills.yaml")
    parser.add_argument("--list", action="store_true", help="list every skill id")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead")
    parser.add_argument("--model", default=MODEL, help=f"default: {MODEL}")
    args = parser.parse_args(argv)

    if args.list:
        _print_skill_list()
        return 0

    if not args.skill_id:
        parser.error("give a skill id, or --list to see them all")

    try:
        question = generate_question(args.skill_id, model=args.model)
    except (UnknownSkillError, BadQuestionError) as error:
        print(f"error: {error}", file=sys.stderr)
        if isinstance(error, UnknownSkillError):
            print("run with --list to see the available ids", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(question.model_dump(), indent=2))
    else:
        _print_question(SKILLS[args.skill_id], question)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
