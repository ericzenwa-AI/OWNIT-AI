"""What a skill actually means, in words a student uses.

The graph has three things to say about a skill and none of them is an
explanation. `probe` says what a question testing it must require - that is for
writing questions. `practice` says what to go and do about it - that is a task.
Neither says what the thing IS, so a student who does not know is told to go and
practise something they cannot name.

That was tolerable while this was aimed at tutors, who supply the explaining
themselves. It is the whole product when it is not.

THE SHAPE

Three beats, because they do three different jobs:

  what     what it is, in the words a person would use out loud. Not a
           definition - a definition is what you read when you already know.
  example  one concrete case, small numbers, worked. The thing itself rather
           than a description of the thing.
  wrong    what students actually do instead, and why it survives. This is the
           beat nothing else in the product has, and it is the one a tutor would
           have supplied: "reading 3/4 as three and four" is worth more than any
           amount of correct exposition, because it names the thing the student
           is probably doing right now.

Written per skill rather than per student's question. A skill does not change,
so one explanation can be checked once and served free forever - the same trade
the question bank makes, and that one has held: 831 questions, every answer key
confirmed by a model that did not write it.

    python backend/explain.py                 # what is written, what is missing
    python backend/explain.py --fill          # write the missing ones
    python backend/explain.py --check         # verify with another model
    python backend/explain.py --show surds    # read one
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import llm
from graph import SKILLS, Skill

load_dotenv(Path(__file__).resolve().parent / ".env")

EXPLAIN_FILE = Path(__file__).resolve().parent.parent / "data" / "explanations.json"
# Not committed - it is the record of one check run, not part of the product.
CHECK_FILE = Path(__file__).resolve().parent / "explanations_checked.json"

BATCH = 6
WORKERS = 5

# Same model as the bank and the practice lines, for the same reason: the
# writing is the easy half and being right about the maths is the hard one.
# 16000 is the ceiling for a non-streaming request - the SDK refuses anything
# that could run past ten minutes. Six skills at three short fields each is
# nowhere near it, and streaming for this would be machinery for nothing.
WRITE = llm.Task(llm.SONNET, effort="medium")

# Judgement about someone else's writing, so not the model that wrote it.
CHECK = llm.Task(llm.SONNET, effort="medium")


class Explanation(BaseModel):
    skill_id: str
    what: str = Field(description="What it is, in plain spoken words. One or two sentences.")
    example: str = Field(description="One small worked case. Plain ASCII maths.")
    wrong: str = Field(
        description="What students do instead, and why it survives. One or two sentences.")


class Batch(BaseModel):
    explanations: list[Explanation]


class Judgement(BaseModel):
    problem: str = Field(default="", description="What is wrong with it. Empty if fine.")
    accurate: bool = Field(description="True if the maths, including the example, is right.")
    plain: bool = Field(description="True if a struggling student could read it unaided.")
    real_mistake: bool = Field(
        description="True if `wrong` names something students actually do, not a strawman.")


SYSTEM_WRITE = (
    "You explain one maths skill to the student who has just been told it is "
    "the thing standing between them and the question they were stuck on.\n"
    "They are on their own, probably on a phone, possibly late, and possibly "
    "years behind. They did not ask for a lesson - they asked for help with a "
    "question - so every word has to earn its place.\n"
    "Write the way a good tutor talks, not the way a textbook writes. Short "
    "sentences. No 'simply', no 'just', no 'obviously', no encouragement, and "
    "nothing that implies they should already know this."
)


def _describe(skill: Skill) -> str:
    return (f"id: {skill.id}\nname: {skill.name}\nkind: {skill.kind}\n"
            f"what a question on it must require: {skill.probe}\n")


def build_prompt(skills: list[Skill]) -> str:
    listed = "\n".join(_describe(s) for s in skills)
    return (
        "Explain each of these.\n\n" + listed + "\n"
        "FOR EACH, THREE PARTS\n"
        "- `what`: what it is, in the words someone would say out loud. One or "
        "two sentences. Not a definition - a definition is what you read when "
        "you already know. If there is an idea underneath it, that idea is what "
        "goes here.\n"
        "- `example`: one small worked case with real numbers, short enough to "
        "follow in one read. Show the thing rather than describe it. Check it "
        "comes out right before you write it down.\n"
        "- `wrong`: what students actually DO instead, and why it survives. Be "
        "specific - name the wrong answer they get. This is the most useful "
        "part, so it must be a real mistake a teacher would recognise, not an "
        "invented one. If a wrong method still gets easy questions right, say "
        "so, because that is why nobody has caught it.\n\n"
        "RULES\n"
        "- A 'concept' is understood, a 'procedure' is done, a 'fact' is known. "
        "Explain each accordingly.\n"
        "- Plain ASCII maths: / for division, ^ for powers, sqrt() for roots. "
        "No LaTeX, no characters outside ASCII.\n"
        "- Do not mention the diagnosis, the gap, the other skills, or this "
        "system. They are reading about the thing itself.\n"
        "- Use the id exactly as given."
    )


# Structured output has leaked the model's own working into a field before -
# a fragment of JSON, a note about the schema, once the words "No wait". A
# checker reading for meaning waves those through, because the sentence in
# front of the rubbish is fine.
SCRATCH = re.compile(r"}\]|'\]|':|\\|\bper schema\b|\bLet's write\b|\bNo wait\b", re.I)
WELDED = re.compile(r"(?<![a-z])(?<!\d)\.(?!\d)[a-z]{3,}")


def _tidy(text: str) -> str:
    """Trim junk welded onto the end, without eating the whole field.

    The first version trimmed backwards to the last sentence end. That is safe
    for a practice line, which always finishes with a search phrase in quotes,
    and destroys an example - `d/dx(3x^5) = 15x^4` contains no full stop at all,
    so every character got stripped and the field came back empty. Six of them
    shipped that way before the checker caught it.
    """
    text = " ".join((text or "").split())
    trimmed = text
    while trimmed and trimmed[-1] not in '.!?"':
        trimmed = trimmed[:-1].rstrip()
    return trimmed or text


def looks_wrong(one: dict) -> str | None:
    """Why this must never be shown, or None if it is safe."""
    for field in ("what", "example", "wrong"):
        body = one.get(field) or ""
        if not body.strip():
            return f"{field} is empty"
        if any(ord(c) > 127 for c in body):
            return f"{field} is not plain ASCII"
        if SCRATCH.search(body):
            return f"{field} carries the model's own scratch text"
        if WELDED.search(body):
            return f"{field} has a full stop welded to the next word"
        # 450 rather than a round 400. The cap is about a student on a phone
        # not being handed a wall, and at that size the difference between
        # the two is nothing - whereas the best-written `wrong` in the set,
        # on writing an expression from words, came in at 403 and would have
        # been thrown away to satisfy a number that was a guess.
        if len(body) > 450:
            return f"{field} is too long at {len(body)} characters"
    return None


def write_batch(skills: list[Skill], client: Anthropic | None = None) -> dict:
    client = client or Anthropic()
    response = client.messages.parse(
        **WRITE.kwargs(), system=SYSTEM_WRITE,
        messages=[{"role": "user", "content": build_prompt(skills)}],
        output_format=Batch)

    parsed = response.parsed_output
    if parsed is None:
        return {}
    wanted = {s.id for s in skills}
    return {
        one.skill_id: {
            "what": _tidy(one.what),
            "example": _tidy(one.example),
            "wrong": _tidy(one.wrong),
        }
        for one in parsed.explanations if one.skill_id in wanted
    }


# The console this is run from is often cp1252, which cannot print a root sign
# or a minus sign - and this file is full of both. A print that raises after 147
# calls have been made throws away work that has already been paid for.
def _say(line: str = "") -> None:
    try:
        print(line)
    except UnicodeEncodeError:
        print(str(line).encode("ascii", "replace").decode("ascii"))


def load() -> dict:
    if not EXPLAIN_FILE.exists():
        return {}
    with EXPLAIN_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def save(written: dict) -> None:
    EXPLAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EXPLAIN_FILE.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(written.items())), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


_loaded: dict | None = None


def for_skill(skill_id: str) -> dict | None:
    """The explanation for one skill, or None. Never raises - a missing
    explanation must not cost a student their diagnosis."""
    global _loaded
    if _loaded is None:
        try:
            _loaded = load()
        except Exception:  # noqa: BLE001
            _loaded = {}
    return _loaded.get(skill_id)


def fill(skill_ids: list[str], workers: int = WORKERS) -> dict:
    batches = [[SKILLS[s] for s in skill_ids[i:i + BATCH]]
               for i in range(0, len(skill_ids), BATCH)]
    written: dict = {}
    client = Anthropic()

    def work(batch):
        try:
            return write_batch(batch, client)
        except Exception as error:  # noqa: BLE001 - one bad batch must not stop the run
            print(f"  batch failed ({error})", file=sys.stderr)
            return {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(work, batches):
            written.update(got)
            print(f"  {len(written)}/{len(skill_ids)}")

    # Every batch failing looks exactly like every batch returning nothing, and
    # the difference matters: one is a bug and the other is a model having a bad
    # day. Saying so beats writing an empty file and reporting success.
    if skill_ids and not written:
        raise RuntimeError(
            f"all {len(batches)} batches came back empty - nothing was written. "
            "The errors above say why.")
    return written


# ---- Checking --------------------------------------------------------------

SYSTEM_CHECK = (
    "You are checking an explanation of one maths skill, written for a student "
    "who has just been told this is what is standing between them and their "
    "question. You did not write it. Say plainly what is wrong before deciding."
)


def check_one(skill: Skill, one: dict, client: Anthropic | None = None) -> Judgement:
    client = client or Anthropic()
    response = client.messages.parse(
        **CHECK.kwargs(), system=SYSTEM_CHECK,
        messages=[{"role": "user", "content": (
            f"THE SKILL\n{_describe(skill)}\n"
            f"WHAT IT SAYS IT IS\n{one['what']}\n\n"
            f"THE EXAMPLE\n{one['example']}\n\n"
            f"WHAT IT SAYS GOES WRONG\n{one['wrong']}\n\n"
            "Decide three things.\n"
            "- accurate: is the maths right, including the example? An example "
            "that does not come out to what it claims is the worst failure "
            "here, because it is the part a student will copy.\n"
            "- plain: could a student who does NOT know this read it unaided? "
            "Jargon they would have to look up fails this.\n"
            "- real_mistake: is the mistake one students actually make, that a "
            "teacher would recognise? An invented or strawman error fails."
        )}],
        output_format=Judgement)
    return response.parsed_output or Judgement(
        problem="the checker returned nothing",
        accurate=False, plain=False, real_mistake=False)


def check_all(written: dict, workers: int = WORKERS) -> list:
    client = Anthropic()
    items = [(s, one) for s, one in sorted(written.items()) if s in SKILLS]

    def work(item):
        skill_id, one = item
        try:
            return skill_id, check_one(SKILLS[skill_id], one, client)
        except Exception as error:  # noqa: BLE001
            return skill_id, Judgement(
                problem=f"check failed: {error}",
                accurate=False, plain=False, real_mistake=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, items))


# ---- Command line ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explanations, one per skill.")
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--all", action="store_true", help="rewrite every one")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--show")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)

    written = load()

    if args.show:
        one = written.get(args.show)
        if not one:
            _say(f"Nothing written for '{args.show}'.")
            return 1
        for field in ("what", "example", "wrong"):
            _say(f"{field:>8}: {one[field]}")
        return 0

    if args.check:
        if not written:
            _say("Nothing written yet.")
            return 1

        broken = {s: why for s, one in written.items() if (why := looks_wrong(one))}
        if broken:
            _say(f"{len(broken)} are malformed and must not ship:")
            for skill_id, why in broken.items():
                _say(f"  [{why}] {skill_id}")
            _say()

        _say(f"Checking {len(written)} explanations on {CHECK.model}...")
        judged = check_all(written, args.workers)

        # To disk first. A check that cannot be printed is still a check that
        # was paid for, and the answer is worth more than the screen.
        CHECK_FILE.write_text(
            json.dumps({s: v.model_dump() for s, v in judged}, indent=2),
            encoding="utf-8")
        _say(f"(verdicts saved to {CHECK_FILE.name})")

        bad = [(s, v) for s, v in judged
               if not (v.accurate and v.plain and v.real_mistake)]

        _say()
        if not bad and not broken:
            _say(f"All {len(written)} were confirmed accurate, plain, and about "
                  "a mistake students really make.")
            return 0
        for skill_id, verdict in bad:
            flags = ",".join(name for name, ok in (
                ("inaccurate", verdict.accurate), ("not plain", verdict.plain),
                ("strawman mistake", verdict.real_mistake)) if not ok)
            _say(f"  [{flags}] {SKILLS[skill_id].name}")
            _say(f"      -> {verdict.problem[:150]}")
        return 1

    if not args.fill:
        missing = [s for s in SKILLS if s not in written]
        _say(f"{len(written)} written, {len(missing)} of {len(SKILLS)} missing")
        for skill_id in missing[:12]:
            _say(f"  {SKILLS[skill_id].name}")
        if len(missing) > 12:
            _say(f"  ... and {len(missing) - 12} more")
        return 0

    wanted = list(SKILLS) if args.all else [s for s in SKILLS if s not in written]
    if not wanted:
        _say("Every skill already has one. Use --all to rewrite them.")
        return 0

    _say(f"Writing {len(wanted)} explanations...")
    written.update(fill(wanted, args.workers))
    save(written)
    _say(f"\nWrote {EXPLAIN_FILE.name}. Run --check before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
