"""Works out which skill a student's question is asking for.

A student has a question they are stuck on. They do not know that
`find_stationary_points` is a thing that exists, so asking them for a skill id
is asking them for something only we know. This takes the question as they
would actually give it - pasted text, or a photo of it - and matches it to one
of the entry points in the graph.

Everything downstream depends on getting this right: pick the wrong entry node
and the walk tests the wrong prerequisites and names the wrong gap, confidently.
So the student confirms the match before any of that starts. One yes/no is
cheap next to a diagnosis built on the wrong question.

The walk itself is untouched - this hands it a skill id and steps out of the way.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from pydantic import BaseModel

from graph import SKILLS, entry_points
from questions import MAX_TOKENS, MODEL
from walk import diagnose

# A match we are prepared to act on. Anything less and we say so instead of
# forcing the student down a branch we do not believe in.
USABLE_CONFIDENCE = ("high", "medium")

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

OUT_OF_SCOPE = """I could not match that to anything I am able to diagnose.

Right now I only cover differentiation: differentiating a function, gradients
and tangents, stationary points, optimisation, and rates of change. If your
question is about integration, trigonometry, vectors or anything else, it is
outside what I can help with yet - that is a gap in me, not in your question."""


class EntryMatch(BaseModel):
    """Which entry skill the question is asking for, if any."""

    # The id of the matched skill, or None when nothing fits.
    skill_id: str | None
    confidence: Literal["high", "medium", "low"]
    # What the question is asking for, said plainly enough for a student to
    # recognise or reject. This is what we show them, not the reasoning.
    plain_summary: str
    reason: str


def _image_block(image_path: str | Path) -> dict:
    """Wrap a photo of a question so the API can read it.

    The Messages API takes images natively as base64 content blocks, so a photo
    needs no separate transcription step - it goes in the same request as the
    text and Claude reads it directly.
    """
    path = Path(image_path)
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(
            f"Cannot read '{path.suffix}' images. Use one of: "
            f"{', '.join(sorted(MEDIA_TYPES))}"
        )

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(path.read_bytes()).decode("utf-8"),
        },
    }


def build_prompt(question: str) -> str:
    """The instructions for matching a question to an entry skill."""
    options = "\n".join(
        f"- {skill.id}: {skill.name} - {skill.probe}" for skill in entry_points()
    )

    return (
        "A student is stuck on this maths question:\n\n"
        f"{question or '(the question is in the attached image)'}\n\n"
        "These are the skills a question can be asking for:\n"
        f"{options}\n\n"
        "Say which one this question asks for.\n"
        "- Match on what the student has to DO, not on the words that happen "
        "to appear. A question mentioning a tangent may still be asking for "
        "stationary points.\n"
        "- If the question needs several of these, pick the one it is really "
        "testing - usually the last step, the thing the marks are for.\n"
        "- All of these are differentiation. If the question is about "
        "integration, trigonometry, vectors, statistics or anything else, set "
        "skill_id to null and say so in `reason`. Do not stretch to the "
        "nearest option: a wrong match sends the student down a diagnosis "
        "built on the wrong question.\n"
        "- Set confidence to low if you are guessing.\n\n"
        "In `plain_summary`, say what the question is asking the student to do, "
        "in one sentence a 17-year-old would recognise. No jargon they would "
        "have to look up, and do not name the skill id."
    )


def identify_entry(
    question: str,
    image_path: str | Path | None = None,
    *,
    client: Anthropic | None = None,
    model: str = MODEL,
) -> EntryMatch:
    """Match a pasted question, or a photo of one, to an entry skill."""
    if not question.strip() and image_path is None:
        raise ValueError("Give me the question text, or a photo of it.")

    client = client or Anthropic()

    content: list[dict] = []
    if image_path is not None:
        # Images go before the text they relate to.
        content.append(_image_block(image_path))
    content.append({"type": "text", "text": build_prompt(question)})

    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=(
            "You identify which A-level maths skill a question is testing. You "
            "are matching against a fixed list and saying plainly when nothing "
            "on it fits."
        ),
        messages=[{"role": "user", "content": content}],
        output_format=EntryMatch,
    )

    match = response.parsed_output
    if match is None:
        return EntryMatch(
            skill_id=None,
            confidence="low",
            plain_summary="",
            reason="Could not read the question.",
        )
    return match


def is_usable(match: EntryMatch) -> bool:
    """Do we believe this enough to build a diagnosis on it?

    Guards against a skill that doesn't exist, one that isn't a valid starting
    point, and a match the model itself was unsure of.
    """
    if match.skill_id is None:
        return False
    if match.skill_id not in {skill.id for skill in entry_points()}:
        return False
    return match.confidence in USABLE_CONFIDENCE


# ---- Confirming with the student ------------------------------------------


def describe(match: EntryMatch) -> str:
    """How we put the match to the student, in words they can judge."""
    skill = SKILLS[match.skill_id]
    opening = (
        "This question is asking you to:"
        if match.confidence == "high"
        else "I think this question is asking you to (I am not certain):"
    )
    return f"{opening}\n\n  {match.plain_summary}\n\n  In short: {skill.name}."


def confirm_in_terminal(match: EntryMatch, input_fn=input) -> str | None:
    """Show the match, and let the student accept it, correct it, or stop.

    Returns the skill id to walk from, or None if they want to stop.
    """
    print()
    print(describe(match))
    print()

    while True:
        answer = input_fn("Is that what you are stuck on? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return match.skill_id
        if answer in ("n", "no"):
            return _pick_by_hand(input_fn)
        print("Type y or n.")


def _pick_by_hand(input_fn=input) -> str | None:
    """We guessed wrong, so let the student say which it is themselves."""
    options = entry_points()

    print()
    print("Which of these is it?")
    for number, skill in enumerate(options, start=1):
        print(f"  {number}. {skill.name} - {skill.probe}")
    print("  0. None of these")
    print()

    while True:
        answer = input_fn("Number: ").strip()
        if answer.isdigit() and 0 <= int(answer) <= len(options):
            choice = int(answer)
            return None if choice == 0 else options[choice - 1].id
        print(f"Type a number from 0 to {len(options)}.")


# ---- Command line ---------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start from a question rather than a skill id."
    )
    parser.add_argument("question", nargs="?", default="", help="the question text")
    parser.add_argument("--image", help="a photo of the question instead")
    parser.add_argument("--attempt", help="what the student tried, if anything")
    parser.add_argument("--attempt-file", help="read the attempt from a file instead")
    args = parser.parse_args(argv)

    try:
        match = identify_entry(args.question, args.image)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not is_usable(match):
        print()
        print(OUT_OF_SCOPE)
        if match.reason:
            print()
            print(f"({match.reason})")
        return 1

    entry_skill_id = confirm_in_terminal(match)
    if entry_skill_id is None:
        print()
        print(OUT_OF_SCOPE)
        return 1

    attempt = args.attempt
    if args.attempt_file:
        attempt = Path(args.attempt_file).read_text(encoding="utf-8")

    from walk import _print_diagnosis

    _print_diagnosis(diagnose(entry_skill_id, attempt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
