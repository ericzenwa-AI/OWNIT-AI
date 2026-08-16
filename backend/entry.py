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

from graph import SKILLS, entry_points, topics
from questions import MAX_TOKENS, MODEL
from walk import diagnose

# Who is sitting at the keyboard. It decides what we can usefully ask when the
# match fails: a tutor can name the skill outright, a student cannot - being
# unable to say what a question is testing is most of what being stuck means.
ROLES = ("student", "tutor", "teacher", "parent")
EXPERT_ROLES = ("tutor", "teacher")

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

def out_of_scope() -> str:
    """What we say when we cannot place a question.

    Read off the graph rather than written down, so it stops being a lie the
    moment a topic is added or removed.
    """
    covered = topics()
    listed = (
        covered[0]
        if len(covered) == 1
        else ", ".join(covered[:-1]) + " and " + covered[-1]
    )
    return (
        "I could not match that to anything I am able to diagnose.\n\n"
        f"Right now I cover {listed}. If your question is about something "
        "else, it is outside what I can help with yet - that is a gap in me, "
        "not in your question."
    )


class QuestionPart(BaseModel):
    """One lettered part of a question we are not starting with."""

    label: str
    skill_id: str | None
    plain_summary: str


class EntryMatch(BaseModel):
    """Which entry skill the question is asking for, if any."""

    # The id of the matched skill, or None when nothing fits. On a question with
    # lettered parts this is part (a) - we start there and work forwards.
    skill_id: str | None
    confidence: Literal["high", "medium", "low"]
    # What the question is asking for, said plainly enough for a student to
    # recognise or reject. This is what we show them, not the reasoning.
    plain_summary: str
    reason: str
    # The question looks cut off, or refers to an earlier part we cannot see.
    # Missing text is the one failure a student can actually fix, because we
    # need the words themselves rather than their reading of them.
    looks_incomplete: bool = False
    # (b), (c) and so on. Someone photographing a whole question needs help with
    # all of it, so these are offered in turn rather than thrown away.
    other_parts: list[QuestionPart] = []


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


def build_prompt(question: str, topic: str | None = None) -> str:
    """The instructions for matching a question to an entry skill.

    Given a topic, only that topic's entry points are offered - which is what
    makes narrowing worth anything once there is more than one.
    """
    points = entry_points()
    if topic:
        points = [skill for skill in points if skill.topic == topic]

    options = "\n".join(
        f"- {skill.id}: {skill.name} - {skill.probe}" for skill in points
    )
    covered = ", ".join(sorted({skill.topic for skill in points}))

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
        f"- Everything on this list is {covered}. If the question is about "
        "anything else, set skill_id to null and say so in `reason`. Do not "
        "stretch to the nearest option: a wrong match sends the student down a "
        "diagnosis built on the wrong question.\n"
        "- Set confidence to low if you are guessing.\n"
        "- Set `looks_incomplete` true if the question appears cut off, or "
        "refers to a part, a diagram, or a previous answer that is not here. "
        "Missing text is worth asking for; a reading of the text is not.\n"
        "- If the question has lettered parts, start with part (a): put its "
        "skill in `skill_id` and its summary in `plain_summary`. Put every "
        "later part in `other_parts`, in order, each with its letter. Someone "
        "who sends a whole question needs help with all of it.\n\n"
        "In `plain_summary`, say what the question is asking the student to do, "
        "in one sentence a 17-year-old would recognise. No jargon they would "
        "have to look up, and do not name the skill id."
    )


def identify_entry(
    question: str,
    image_path: str | Path | None = None,
    *,
    topic: str | None = None,
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
    content.append({"type": "text", "text": build_prompt(question, topic)})

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


def confirm_in_terminal(match: EntryMatch, input_fn=input) -> bool:
    """Show the match in plain words and ask whether it is right."""
    print()
    print(describe(match))
    print()

    while True:
        answer = input_fn("Is that what you are stuck on? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Type y or n.")


def resolve_entry(
    question: str,
    image_path: str | Path | None = None,
    *,
    role: str = "student",
    client: Anthropic | None = None,
    input_fn=input,
) -> tuple[str | None, EntryMatch]:
    """Get to a skill we can walk from, or admit we cannot place the question.

    When the match fails we only ask for things the person in front of us
    actually holds. A tutor can name the skill. A student cannot - not knowing
    what a question is testing is most of what being stuck means - but they do
    know what topic they are on and whether there was more to the question. So
    those are what we ask them for, and never for their reading of the question.
    """
    match = identify_entry(question, image_path, client=client)
    placed = is_usable(match)

    if placed and confirm_in_terminal(match, input_fn):
        return match.skill_id, match

    # An expert can just tell us, and the vocabulary means something to them.
    # Say why we are asking first - a wall of skill names with no explanation
    # leaves them unable to tell a misread question from one we do not cover.
    if role in EXPERT_ROLES:
        print()
        if placed:
            print("Right - which is it then?")
        else:
            print("I could not place that question.")
            if match.reason:
                print(f"  {match.reason}")
        return _pick_by_hand(input_fn), match

    # Missing text is the one gap a student can genuinely close for us.
    if match.looks_incomplete:
        print()
        print("Some of that question seems to be missing.")
        extra = input_fn("Paste the whole question, including any earlier parts: ")
        if extra.strip():
            match = identify_entry(f"{question}\n{extra}", image_path, client=client)
            if is_usable(match) and confirm_in_terminal(match, input_fn):
                return match.skill_id, match

    # They may not know what the question tests, but they know what they are on.
    topic = ask_for_topic(input_fn)
    if topic:
        match = identify_entry(question, image_path, topic=topic, client=client)
        if is_usable(match) and confirm_in_terminal(match, input_fn):
            return match.skill_id, match

    return None, match


def ask_for_topic(input_fn=input) -> str | None:
    """What are they covering in class? A fact they hold, not a judgement.

    Returns None when there is only one topic, because narrowing to the only
    option asks the student a question and learns nothing.
    """
    choices = topics()
    if len(choices) < 2:
        return None

    print()
    print("What are you covering in class at the moment?")
    for number, topic in enumerate(choices, start=1):
        print(f"  {number}. {topic}")
    print("  0. Not sure")
    print()

    while True:
        answer = input_fn("Number: ").strip()
        if answer.isdigit() and 0 <= int(answer) <= len(choices):
            chosen = int(answer)
            return None if chosen == 0 else choices[chosen - 1]
        print(f"Type a number from 0 to {len(choices)}.")


def listed_entry_points() -> list:
    """Doorways in the order an expert is shown them, grouped by topic.

    A flat run of every doorway was readable at eleven and is a wall at
    twenty-four, and it only grows from here.
    """
    return sorted(entry_points(), key=lambda skill: (skill.topic, skill.name))


def _pick_by_hand(input_fn=input) -> str | None:
    """Let an expert name the skill outright.

    Only ever shown to a tutor or teacher. These are our internal names, and
    they mean something to someone who teaches the subject and nothing to a
    sixteen-year-old who is stuck.
    """
    options = listed_entry_points()

    print()
    print("Which of these is it?")
    current_topic = None
    for number, skill in enumerate(options, start=1):
        if skill.topic != current_topic:
            current_topic = skill.topic
            print(f"\n  {current_topic}")
        print(f"    {number:>2}. {skill.name}")
    print("\n     0. None of these")
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
    parser.add_argument("--student", help="anonymous reference, e.g. student_7")
    parser.add_argument(
        "--role",
        default="student",
        choices=ROLES,
        help="who is at the keyboard (default: student)",
    )
    parser.add_argument("--no-save", action="store_true", help="do not record this walk")
    args = parser.parse_args(argv)

    try:
        entry_skill_id, match = resolve_entry(
            args.question, args.image, role=args.role
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if entry_skill_id is None:
        print()
        print(out_of_scope())
        if match.reason:
            print()
            print(f"({match.reason})")
        if not args.no_save:
            _record_gap(args, match)
        return 1

    attempt = args.attempt
    if args.attempt_file:
        attempt = Path(args.attempt_file).read_text(encoding="utf-8")

    from walk import SkillResult, _print_diagnosis, reusing, save_walk

    # Someone who sends a whole question needs help with all of it, so we start
    # at (a) and work forwards. What one part establishes carries to the next -
    # the parts of a question rest on the same foundations, and asking twice
    # reads as not having listened.
    known: dict[str, SkillResult] = {}

    for label, skill_id, summary in _parts(entry_skill_id, match):
        if label is not None:
            print()
            print(f"Part ({label}): {summary}")
            if skill_id is None:
                print("  I cannot place this part, so I am skipping it.")
                continue
            answer = input(f"Work on part ({label}) now? (y/n): ").strip().lower()
            if answer not in ("y", "yes"):
                break

        diagnosis = diagnose(
            skill_id,
            # The attempt is working on part (a); it says nothing about (b).
            attempt if label is None else None,
            check=reusing(known),
        )
        _print_diagnosis(diagnosis)

        if not args.no_save:
            save_walk(
                diagnosis,
                question=args.question,
                attempt=attempt if label is None else None,
                student_ref=args.student,
                role=args.role,
                entry_confidence=match.confidence,
                # False when our first guess was rejected and we got here
                # another way.
                entry_confirmed=entry_skill_id == match.skill_id,
            )

    return 0


def _parts(entry_skill_id: str, match: EntryMatch) -> list[tuple]:
    """Part (a) first, then whatever else the question contained."""
    return [(None, entry_skill_id, match.plain_summary)] + [
        (part.label, part.skill_id, part.plain_summary) for part in match.other_parts
    ]


def _record_gap(args, match: EntryMatch) -> None:
    """File a question nobody could place, as a coverage gap.

    Costs whoever pasted it nothing - they have already done the only thing
    required, which is to try. Never let filing it break the run.
    """
    import store

    try:
        connection = store.connect()
        try:
            store.record_unplaced(
                connection,
                args.question,
                match,
                from_image=bool(args.image),
                role=args.role,
                student_ref=args.student,
            )
        finally:
            connection.close()
    except Exception as error:  # noqa: BLE001 - never lose the run over filing
        print(f"\n(could not record this gap: {error})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
