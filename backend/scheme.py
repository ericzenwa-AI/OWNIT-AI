"""Mines mark schemes for evidence about the skill graph.

Every `needs` link in data/skills.yaml is currently a guess. Plausible ones,
mostly mine, and a wrong link does not announce itself - it produces a fluent
diagnosis pointing at the wrong skill.

A mark scheme is the one document an exam board publishes that breaks a question
into the steps that earn marks. Those steps are the sub-skills, written down by
the people who set the paper. So they are evidence for exactly the thing the
graph is guessing at, and they cost nothing to read.

Two things come out:

  - Steps that map to no skill we have. Each is a candidate node, named by an
    examiner rather than by me, ranked by how often it is actually needed.
  - Steps that map to skills we do have. That is confirmation, and it shows
    which parts of the graph carry real weight.

    python backend/scheme.py "data/papers/markscheme.pdf"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from anthropic import Anthropic
from pydantic import BaseModel

from entry import attachment_block
from graph import SKILLS
from questions import MAX_TOKENS, MODEL


class MarkStep(BaseModel):
    """One thing a candidate has to do to earn a mark."""

    mark: str
    does: str
    # The underlying skill in a few plain words, not the specific working.
    # "differentiate a power", not "gets 6x^2 - 24x".
    skill_hint: str


class SchemeQuestion(BaseModel):
    number: str
    topic: str
    steps: list[MarkStep]


class Scheme(BaseModel):
    questions: list[SchemeQuestion]


class HintMapping(BaseModel):
    hint: str
    # A skill id from the graph, or empty when nothing in the graph covers it.
    skill_id: str


class Mapped(BaseModel):
    mappings: list[HintMapping]


def read_scheme(
    pdf: Path, *, client: Anthropic | None = None, model: str = MODEL
) -> list[SchemeQuestion]:
    """Break a mark scheme into the steps that earn marks, per question."""
    client = client or Anthropic()

    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=(
            "You read A-level maths mark schemes and say what a candidate has "
            "to be able to DO to earn each mark. You describe skills, not "
            "answers."
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    attachment_block(pdf),
                    {
                        "type": "text",
                        "text": (
                            "For every question in this mark scheme, list the "
                            "steps that earn marks, in order.\n"
                            "- One entry per mark or closely grouped marks, "
                            "with its code (M1, A1, B1, dM1 and so on).\n"
                            "- In `does`, say plainly what the candidate has "
                            "to do at that step.\n"
                            "- In `skill_hint`, name the underlying skill in a "
                            "few words - the thing they would need to have "
                            "been taught. Say 'differentiate a power', not "
                            "'gets 6x^2 - 24x'. Two students on different "
                            "questions using the same skill must produce the "
                            "same hint.\n"
                            "- Skip general marking guidance, notation notes "
                            "and the front matter."
                        ),
                    },
                ],
            }
        ],
        output_format=Scheme,
    )

    scheme = response.parsed_output
    return scheme.questions if scheme else []


def map_hints(
    hints: list[str], *, client: Anthropic | None = None, model: str = MODEL
) -> dict[str, str | None]:
    """Match each skill named by the mark scheme to a skill in the graph.

    Anything that matches nothing is a candidate node - a skill an examiner
    thinks is worth a mark and the graph has never heard of.
    """
    if not hints:
        return {}

    client = client or Anthropic()
    catalogue = "\n".join(
        f"- {skill.id}: {skill.name} - {skill.probe}" for skill in SKILLS.values()
    )

    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=(
            "You match skills named in a mark scheme against a fixed list of "
            "skills, and say plainly when the list has no equivalent."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "These skills came out of a mark scheme:\n"
                    + "\n".join(f"- {hint}" for hint in hints)
                    + "\n\nThis is the skill graph:\n"
                    + catalogue
                    + "\n\nFor each mark scheme skill, give the id of the graph "
                    "skill that means the same thing.\n"
                    "- Match on what the student must be able to do, not on "
                    "wording.\n"
                    "- Use an empty string when the graph has nothing "
                    "equivalent. Do not stretch to the nearest one: a wrong "
                    "match hides a missing skill, which is the thing we are "
                    "looking for.\n"
                    "- Return one entry for every skill listed above."
                ),
            }
        ],
        output_format=Mapped,
    )

    mapped = response.parsed_output
    if mapped is None:
        return {hint: None for hint in hints}

    known = {
        item.hint: (item.skill_id if item.skill_id in SKILLS else None)
        for item in mapped.mappings
    }
    return {hint: known.get(hint) for hint in hints}


# ---- Reporting ------------------------------------------------------------


def report(questions: list[SchemeQuestion], mapping: dict[str, str | None]) -> None:
    steps = [step for question in questions for step in question.steps]
    covered = [s for s in steps if mapping.get(s.skill_hint)]
    missing = [s for s in steps if not mapping.get(s.skill_hint)]

    print()
    print("=" * 68)
    print(f"{len(questions)} questions, {len(steps)} marked steps")
    print(f"{len(covered)} rest on a skill we have, {len(missing)} do not")
    print()

    if missing:
        print("Skills an examiner awards marks for that the graph does not have,")
        print("commonest first. Each is a candidate node, named by the exam board:")
        for hint, times in Counter(s.skill_hint for s in missing).most_common():
            print(f"  {times:>3}x  {hint}")
        print()

    if covered:
        print("Skills the graph already has, by how often marks depend on them:")
        counted = Counter(mapping[s.skill_hint] for s in covered)
        for skill_id, times in counted.most_common(15):
            print(f"  {times:>3}x  {SKILLS[skill_id].name}")
        print()

    print("What each question actually needs, in order:")
    for question in questions:
        print(f"\n  Q{question.number}  ({question.topic})")
        for step in question.steps:
            skill_id = mapping.get(step.skill_hint)
            marker = SKILLS[skill_id].name if skill_id else f"** {step.skill_hint}"
            print(f"    {step.mark:<5} {marker}")


# ---- Command line ---------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine a mark scheme for evidence about the skill graph."
    )
    parser.add_argument("pdf", help="a mark scheme PDF")
    args = parser.parse_args(argv)

    path = Path(args.pdf)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    print(f"Reading {path.name}...")
    questions = read_scheme(path)
    if not questions:
        print("error: no questions found in that mark scheme", file=sys.stderr)
        return 2

    hints = sorted({step.skill_hint for q in questions for step in q.steps})
    print(f"{len(questions)} questions, {len(hints)} distinct skills. Matching...")
    mapping = map_hints(hints)

    report(questions, mapping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
