"""The way back up to the question they came in with.

The diagnostic finds the gap and stops. That is the hard half and it is free -
banked questions, no model call - but it leaves a student holding a skill name
and their original question, with the distance between the two unbridged.

This is that bridge: four questions climbing from the gap to their own problem,
the last one being the problem itself. Written fresh for the question they
actually sent, which is the whole point and also why it cannot come off the
shelf - a banked question is about a skill, and this has to be about their
question.

Deliberately not cached. Caching this would mean deciding what counts as "the
same question", which is a real problem worth solving with usage data rather
than a guess. Until then every ladder is written from scratch, and the cost of
that is why it is offered rather than run automatically.

    python backend/ladder.py 42        # build one for a session, print it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent / ".env")

import llm
import store
from graph import SKILLS
from questions import MultipleChoiceQuestion

# Same model and effort the bank uses, for the same reason: writing the question
# was never the hard part, working out the answer was. A wrong key here is worse
# than a wrong key anywhere else - it lands after a student has climbed four
# steps and is being told whether they got there.
WRITE = llm.Task(llm.SONNET, effort="medium")

RUNGS = 4


class Rung(BaseModel):
    """One step of the climb."""

    question: MultipleChoiceQuestion
    # What this step adds to the one before it, in a few words. Not shown to the
    # student - it is here so the model has to commit to the steps differing.
    adds: str = Field(description="What this step adds that the last one did not.")


class Ladder(BaseModel):
    rungs: list[Rung]


SYSTEM = (
    "You build a short ladder of multiple-choice questions that carries a "
    "student from a skill they have just been found to be missing up to the "
    "exam question they were originally stuck on.\n"
    "Each step adds exactly one piece of the structure of their question. The "
    "last step is their question. You are not teaching - there is no "
    "explanation, no worked example, and no encouragement. Each question has to "
    "stand alone and be answerable by someone who has just done the one before."
)


def build_prompt(
    *, question: str, verbatim: bool, gap_id: str, entry_id: str, chain: list[str]
) -> str:
    gap = SKILLS[gap_id]
    entry = SKILLS[entry_id]
    between = "\n".join(
        f"- {SKILLS[s].name}: {SKILLS[s].probe}" for s in chain if s in SKILLS
    )

    # A photographed question is never stored - the image is read once and
    # deleted - so what comes back here is the model's own reading of it. The
    # ladder is built from that, and the page says so rather than passing a
    # paraphrase off as their words.
    how = (
        "THE STUDENT'S OWN QUESTION"
        if verbatim
        else "THE STUDENT'S QUESTION, AS IT WAS READ FROM A PHOTO (a paraphrase, "
        "not their exact wording)"
    )

    return (
        f"{how}\n{question}\n\n"
        "WHAT THE DIAGNOSTIC FOUND\n"
        f"They could not do: {gap.name} - {gap.probe}\n"
        f"They came in stuck on: {entry.name}\n\n"
        f"THE STEPS BETWEEN THE TWO\n{between or '- nothing in between'}\n\n"
        f"WRITE {RUNGS} QUESTIONS\n"
        f"- Question 1 sits squarely on {gap.name}, using the numbers from their "
        "own question wherever that is natural.\n"
        "- Each one after it adds exactly one more piece of the structure of "
        "their question. Say in `adds` what that piece is.\n"
        f"- Question {RUNGS} IS their question, put as multiple choice, worded as "
        "closely to the original as the format allows.\n"
        "- Four options each: one correct, three wrong. Every wrong option must "
        "be what a student actually reaches after one specific slip, and the "
        "mistake must say what that slip was.\n"
        "- Do not teach, explain or encourage. Do not refer to the other "
        "questions, to the diagnosis, or to 'the gap'.\n"
        "- Plain ASCII maths: / for division, ^ for powers, sqrt() for roots. No "
        "LaTeX, no characters outside ASCII."
    )


def build(
    *, question: str, verbatim: bool, gap_id: str, entry_id: str,
    chain: list[str], client: Anthropic | None = None,
) -> list[MultipleChoiceQuestion]:
    """One call, all four rungs. Asked together so they can differ and climb."""
    client = client or Anthropic()
    response = client.messages.parse(
        **WRITE.kwargs(),
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": build_prompt(
                question=question, verbatim=verbatim,
                gap_id=gap_id, entry_id=entry_id, chain=chain),
        }],
        output_format=Ladder,
    )

    if response.stop_reason == "max_tokens":
        raise LadderError("The ladder was cut off before it finished.")

    ladder = response.parsed_output
    if ladder is None or not ladder.rungs:
        raise LadderError(
            f"No ladder came back (stop_reason: {response.stop_reason})."
        )

    good = [r.question for r in ladder.rungs if len(r.question.distractors) == 3]
    if len(good) < 2:
        raise LadderError("The ladder came back too short to be worth showing.")
    return good[:RUNGS]


class LadderError(RuntimeError):
    """The ladder could not be written. Never fatal - the report still stands."""


def for_session(session_id: int, *, connection=None, client=None) -> list:
    """Build and store the ladder for one finished walk."""
    own = connection is None
    connection = connection or store.connect()
    try:
        walk_row = store.walk_state(connection, session_id)
        if walk_row is None:
            raise LadderError("No such session.")

        gaps = [g for g in (walk_row["root_gaps"] or "").split(",") if g]
        if not gaps:
            raise LadderError(
                "This walk did not confirm a gap, so there is nothing to climb from."
            )

        typed = (walk_row["question"] or "").strip()
        question = typed or (walk_row["plain_summary"] or "").strip()
        if not question:
            raise LadderError("There is no record of the original question.")

        chain = [s for s in (walk_row["chain"] or "").split(" -> ") if s]
        rungs = build(
            question=question,
            verbatim=bool(typed),
            gap_id=gaps[0],
            entry_id=walk_row["entry_skill_id"],
            chain=list(reversed(chain)),
            client=client,
        )
        store.save_ladder(connection, session_id, rungs, verbatim=bool(typed))
        return rungs
    finally:
        if own:
            connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a closing ladder.")
    parser.add_argument("session_id", type=int)
    args = parser.parse_args(argv)

    try:
        rungs = for_session(args.session_id)
    except LadderError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for number, rung in enumerate(rungs, start=1):
        print(f"\n{number}. {rung.question}")
        print(f"   correct: {rung.correct_option}")
        for distractor in rung.distractors:
            print(f"   wrong  : {distractor.option}  ({distractor.mistake})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
