"""Teaching the gap, one exchange at a time.

The diagnostic ends by naming a skill and handing over an explanation to read.
That was always the weak half: reading an explanation is not learning a thing,
and a student who loses the thread in a minute - which is most of the ones
this is for - gets nothing from a page of correct prose.

So after the diagnosis there is a lesson. It is a conversation with a shape:

    connect   why this skill is what stopped them on THEIR question
    show      the idea on a small example, broken into steps
    try       they do one step themselves
    fade      bigger steps, less help
    apply     back to their own question, guided, never solved for them

WHAT MAKES IT A LESSON RATHER THAN A CHATBOT

Three things, and none of them is the prompt.

The STAGE IS HELD BY THE SERVER. The model proposes; `advance` decides. It
cannot jump to the end because a student sounded confident, and it cannot
quietly stay in `show` forever either.

MOVING ON TAKES EVIDENCE. Two understood replies in a stage, not one. One
right answer is a coin that landed the right way up; the brief says so, and
making it a counter rather than an instruction is what makes it true.

THE ANSWER IS NEVER GIVEN. Not to the student's own question, at any stage.
That is the line the whole product is on the right side of, and `apply` is
exactly where it would be easiest to cross.

WHY SONNET

A turn is short, grounded in an explanation we already wrote and checked, and
read by somebody who will say so if it makes no sense. It runs many times in
one lesson, where the entry match runs once, so the cost difference is the
whole cost of the feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import explain
import llm
from graph import SKILLS

load_dotenv(Path(__file__).resolve().parent / ".env")

TEACH = llm.Task(llm.SONNET, effort="medium")

# In order. A lesson only ever moves one of these at a time, and only forwards.
STAGES = ("connect", "show", "try", "fade", "apply", "done")

# What each stage is for, handed to the model as the job for this turn rather
# than left for it to infer from the conversation.
THE_JOB = {
    "connect": (
        "Say what the thing they are missing is, and why it is what stopped "
        "them on their own question. Two or three sentences. End by asking "
        "whether that matches what happened, so they say something back."
    ),
    "show": (
        "SHOW it. Fill in `demo` with the transformation line by line on ONE "
        "small example - not their question - and stop one step short, so the "
        "line they have to produce is the step they got wrong. Keep `say` to "
        "a sentence: the demonstration is doing the explaining."
    ),
    "try": (
        "Give them one short thing to do themselves, close to the example you "
        "just showed. One step, not a whole problem. No working given away."
    ),
    "fade": (
        "A harder one, with less help: no reminder of the rule unless they "
        "ask. If they have got two right, make this one look different from "
        "the last so it is the idea being tested and not the pattern."
    ),
    "apply": (
        "Back to THEIR question. Point at the place in it where this skill is "
        "needed and ask them to do that part - that part only. Never work the "
        "whole question, never state its final answer, even if asked."
    ),
}

# How many understood replies it takes to leave a stage. Two, not one: a
# single right answer is a coin that landed the right way up, and the whole
# complaint about the old report was that it moved on from one.
TO_ADVANCE = 2

# A lesson that has run this long is not working, and the student should be
# told so rather than kept going round.
MAX_TURNS = 40

# How much of the conversation the model is shown. Enough to follow what has
# been tried; short enough that turn forty does not cost ten times turn one.
RECENT = 12


class DemoStep(BaseModel):
    """One line of a worked demonstration, as it stands at that moment."""

    show: str = Field(
        description="The WHOLE expression as it now reads, not just the bit "
                    "that moved. Plain ASCII maths, short.")
    changed: str = Field(
        default="",
        description="The part of `show` that is new this step, copied from it "
                    "exactly. Empty on the first line.")
    note: str = Field(
        default="",
        description="One short line: what changed and why. Not a sentence "
                    "about the whole method.")


class Move(BaseModel):
    """One turn of the lesson."""

    # What the last thing the student said tells us. "first" on the opening
    # turn, when there is nothing to judge yet.
    read: Literal["first", "understood", "partly", "muddled", "no_answer"] = Field(
        description="How their last reply went. Be strict: 'understood' means "
                    "they showed the reasoning, not that the number was right.")
    why: str = Field(
        default="",
        description="The part of what they said that shows it, in their words. "
                    "Kept for the tutor, never shown to the student.")
    say: str = Field(
        description="What you say to them now. Plain words, short - three "
                    "sentences at the very most. No preamble, no praise.")
    ask: str = Field(
        description="The ONE thing you want them to do or answer now, as a "
                    "question. Never more than one. Empty only when the lesson "
                    "is finished.")
    # The showing half. A lesson that only talks is the explanation we already
    # had, in instalments.
    demo: list[DemoStep] = Field(
        default_factory=list,
        description="A small worked transformation, one line per step, when "
                    "showing beats telling. Empty when there is nothing to "
                    "show - most turns.")
    finished: bool = Field(
        default=False,
        description="True only when they have applied the skill to their own "
                    "question themselves.")


# A demonstration is a few lines, not a worked solution. More than this and it
# is the thing we took out: a wall handed to somebody who stopped reading.
MAX_DEMO = 5
MAX_LINE = 70
MAX_NOTE = 140


def unusable_demo(steps: list) -> str | None:
    """Why these steps must not be shown, or None if they are fine.

    Dropped rather than fatal: the turn still has words, and a lesson that
    fails because a picture was malformed is worse than one without a picture.

    The project's rule about generated pictures is that they do not ship -
    four rendering faults reached real students in a fortnight, and a wrong
    diagram is worse than a wrong answer key because nobody re-reads it. This
    is the reason a demonstration here is TEXT, rendered by the same maths
    pipeline as everything else on the page, rather than drawn markup: the
    only new thing being generated is which line comes next.
    """
    if not steps:
        return None
    if len(steps) < 2:
        return "one line is not a demonstration"
    if len(steps) > MAX_DEMO:
        return f"{len(steps)} steps is a worked solution, not a demonstration"

    for n, step in enumerate(steps, start=1):
        show = (getattr(step, "show", "") or "").strip()
        changed = (getattr(step, "changed", "") or "").strip()
        note = (getattr(step, "note", "") or "").strip()

        if not show:
            return f"step {n} shows nothing"
        if len(show) > MAX_LINE:
            return f"step {n} is {len(show)} characters, too long to follow"
        if len(note) > MAX_NOTE:
            return f"step {n}'s note is {len(note)} characters"
        for text, what in ((show, "line"), (changed, "highlight"), (note, "note")):
            if any(ord(c) > 127 for c in text):
                return f"step {n}'s {what} is not plain ASCII"
        # The highlight is how a student sees WHAT changed. One that is not
        # actually in the line would underline nothing, or the wrong thing.
        if changed and changed not in show:
            return f"step {n} highlights '{changed[:20]}', which is not in the line"
    return None


def tidy_demo(steps: list) -> list:
    """Drop highlights that mark nothing.

    A step whose "what changed" is the entire line - which is what the first
    line of a demonstration usually comes back as - is pointing at everything,
    which is the same as pointing at nothing, and puts a chip on screen that
    only repeats the line above it.
    """
    for step in steps:
        changed = (getattr(step, "changed", "") or "").strip()
        if changed and changed == (getattr(step, "show", "") or "").strip():
            step.changed = ""
    return steps


class CannotTeach(RuntimeError):
    """The lesson cannot continue. Never silent, never fatal to the session."""


def advance(stage: str, read: str, understood_here: int,
            replied: bool = True) -> tuple[str, int]:
    """Where the lesson goes next. The server decides this, not the model.

    Forwards one stage at a time and only on evidence; back never - repeating
    a stage is done by staying in it, which keeps the transcript honest about
    how long something took.
    """
    if stage not in STAGES:
        stage = "connect"
    if stage == "done":
        return "done", understood_here

    # Connecting is agreeing that this is what happened, not demonstrating
    # anything - so any reply at all leaves it. Requiring understanding here
    # would strand the confused student in the one stage that teaches nothing,
    # which is the opposite of who this is for.
    #
    # And it turns on whether they actually said something, which is a fact we
    # hold, rather than on the model's reading of it. Asked to judge "yes,
    # that is the bit" it kept answering "first", and the lesson sat on the
    # opening stage for three turns agreeing with itself.
    if stage == "connect":
        return ("show", 0) if replied else ("connect", 0)

    if read == "understood":
        understood_here += 1
    elif read in ("muddled", "no_answer"):
        # Lost ground. Not negative - a student who gets one right then one
        # wrong has not earned the next stage, but has not undone the first.
        understood_here = max(0, understood_here - 1)

    if understood_here >= TO_ADVANCE:
        return STAGES[STAGES.index(stage) + 1], 0
    return stage, understood_here


def stuck_for(transcript: list, stage: str) -> int:
    """How many turns running they have failed to get the thing being asked.

    Counted here rather than left to the model to notice, because it does not
    notice. Put a student in front of it who answers something other than the
    question, and it will ask the same question again, and again - which is
    the exact thing a tutor must not do, and the reason "do not repeat
    yourself" cannot live in a prompt alone.
    """
    running = 0
    for turn in reversed(transcript):
        if turn.get("stage") != stage:
            break
        if turn.get("read") in ("understood",):
            break
        if turn.get("read") in ("partly", "muddled", "no_answer"):
            running += 1
    return running


# After this many turns going nowhere, asking again is not teaching.
PATIENCE = 2


def build_prompt(*, skill_id: str, question: str, stage: str,
                 transcript: list, said: str) -> str:
    skill = SKILLS[skill_id]
    one = explain.for_skill(skill_id) or {}

    # The explanation is already written and already checked by a model that
    # did not write it, so the lesson is grounded in that rather than inventing
    # the maths again every turn.
    known = "\n".join(filter(None, [
        f"IN PLAIN WORDS: {one.get('plain', '')}",
        f"LIKE THIS: {one.get('like', '')}" if one.get("like") else "",
        f"MORE PRECISELY: {one.get('what', '')}",
        f"A WORKED CASE: {one.get('example', '')}",
        f"WHAT USUALLY GOES WRONG: {one.get('wrong', '')}",
    ]))

    # .get rather than [] throughout: a turn missing a field is a row we would
    # rather teach around than crash a lesson over.
    history = "\n".join(
        f"{'YOU' if turn.get('say') else 'THEM'}: {turn.get('say') or turn.get('said') or ''}"
        for turn in transcript[-RECENT:]) or "(nothing yet)"

    stuck = stuck_for(transcript, stage)
    if stuck >= PATIENCE:
        change = (
            f"\n\nTHEY HAVE NOT GOT THIS IN {stuck} TRIES. Asking it again is "
            "not teaching. Do not repeat your last question in any form. "
            + ("Drop to an easier parallel example with smaller numbers - "
               "never work any part of their own question for them - and ask "
               "about that instead."
               if stage == "apply" else
               "Work this one step out for them, say in one line what the "
               "move was, then ask them a DIFFERENT and easier question about "
               "it.")
        )
    else:
        change = ""

    return (
        f"THE QUESTION THEY CAME IN STUCK ON\n{question or '(not recorded)'}\n\n"
        f"WHAT THEY ARE MISSING\n{skill.name}: {skill.probe}\n\n"
        f"WHAT IS ALREADY WRITTEN ABOUT IT\n{known}\n\n"
        f"THE LESSON SO FAR\n{history}\n\n"
        f"WHAT THEY JUST SAID\n{said or '(nothing yet - this is the first turn)'}\n\n"
        f"YOUR JOB THIS TURN ({stage})\n{THE_JOB.get(stage, THE_JOB['try'])}"
        f"{change}\n\n"
        "Judge their last reply first, then write the turn."
    )


SYSTEM = (
    "You are sitting next to a sixteen year old who is stuck on a maths "
    "question, and you have just worked out what they are actually missing. "
    "You are teaching them that one thing.\n"
    "\n"
    "HOW YOU TALK. Short. Three sentences at most before you hand back to "
    "them. No lists, no headings, no 'great question', no praise for ordinary "
    "answers, nothing that implies they should already know this. Plain words "
    "a ten year old would follow, even though they are sixteen - whatever "
    "they were taught in the usual words did not stick.\n"
    "\n"
    "WHAT YOU NEVER DO. You never work their own question. You never state "
    "its final answer, however they ask - not as a check, not 'just to "
    "confirm', not at the end. Helping them do one step of it is the job; "
    "doing it for them is the one thing this tool exists not to do.\n"
    "\n"
    "You never give the answer to a question you have just asked them. If "
    "they are stuck, make it smaller, or ask a different way, or show the "
    "same thing on easier numbers - then ask again.\n"
    "\n"
    "SHOWING BEATS TELLING. When a thing is easier to watch than to read, put "
    "it in `demo`: one line per step, each line the WHOLE expression as it "
    "then reads, with `changed` copied exactly from the part that is new and "
    "`note` saying in a few words what moved and why. Two to five lines. For "
    "'why is sqrt(x) the same as x^(1/2)' that is: sqrt(x) * sqrt(x) = x, "
    "then x^(1/2) * x^(1/2) = x^1, then the line that makes them equal.\n"
    "Stop the demonstration one step short of the end whenever you can, so "
    "the next line is theirs to write. Leave `demo` empty on turns that are "
    "just a question or a nudge, which is most of them.\n"
    "Never demonstrate on their own question - that is working it for them.\n"
    "\n"
    "EVERY TURN ENDS WITH ONE THING FOR THEM TO DO. One. Not two, not a "
    "question with a list of parts.\n"
    "\n"
    "BEING STRICT ABOUT 'understood'. Say understood only when what they "
    "wrote shows the reasoning. A bare right number after a hint is 'partly'. "
    "A right answer they could have got by guessing between two options is "
    "'partly'. You are deciding whether to stop teaching this, so be hard to "
    "convince.\n"
    "\n"
    "Maths in plain ASCII: / for divide, ^ for powers, sqrt() for roots."
)


def next_move(*, skill_id: str, question: str, stage: str, transcript: list,
              said: str, client: Anthropic | None = None) -> Move:
    """The tutor's next turn. Raises CannotTeach if nothing usable came back."""
    if skill_id not in SKILLS:
        raise CannotTeach(f"{skill_id} is not a skill we hold")

    client = client or Anthropic()
    response = client.messages.parse(
        **TEACH.kwargs(),
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(
            skill_id=skill_id, question=question, stage=stage,
            transcript=transcript, said=said)}],
        output_format=Move,
    )

    if response.stop_reason == "refusal":
        raise CannotTeach("the tutor declined")
    move = response.parsed_output
    if move is None:
        raise CannotTeach(f"nothing came back ({response.stop_reason})")
    return move
