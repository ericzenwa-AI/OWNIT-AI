"""What to actually go and do about a gap, in a sentence or two.

Naming the gap is only half an answer. "Highest common factor" is the diagnosis;
it is not something a fifteen year old can act on tonight. A student who reads
it still has to work out what that phrase means, what practising it looks like,
and what to type into a search box - and the ones this is built for are exactly
the ones least likely to guess right.

So every skill carries a line telling them what to go and do. One or two
sentences, a concrete example, and something searchable. Not teaching: the
tutor, the textbook and the internet all do that better than a diagnostic can.

The lines live in data/practice.json rather than in skills.yaml, for the same
reason the questions live in their own file: skills.yaml is the hand-written
shape of the subject, and burying 143 generated sentences in it would make a
change to the graph impossible to see in a diff.

    python backend/practice.py                # what is written, what is missing
    python backend/practice.py --fill         # write the missing ones
    python backend/practice.py --fill --all   # rewrite every one
    python backend/practice.py --check        # verify them with another model
    python backend/practice.py --show surds   # read one
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

PRACTICE_FILE = Path(__file__).resolve().parent.parent / "data" / "practice.json"

# Written in batches so the model can see them together and avoid opening every
# one the same way. Small enough that one bad batch is cheap to redo.
BATCH = 8
WORKERS = 5

# Writing one is a small, well-specified job, but it has to be right about the
# maths and pitched at the right age, which is the same reason question
# generation is not on the cheapest model.
WRITE = llm.Task(llm.SONNET, effort="medium")

# Judgement about someone else's writing. Must not be the model that wrote it.
CHECK = llm.Task(llm.SONNET, effort="medium")


class Practice(BaseModel):
    skill_id: str
    line: str = Field(description="One or two sentences. What to go and do.")


class Batch(BaseModel):
    lines: list[Practice]


class Judgement(BaseModel):
    """Ordered so the model commits to reasons before a verdict."""

    problem: str = Field(
        default="",
        description="What is wrong with it, if anything. Empty if it is fine.",
    )
    accurate: bool = Field(description="True if the maths and any example are correct.")
    on_target: bool = Field(description="True if it describes THIS skill, not a neighbour.")
    right_level: bool = Field(description="True if pitched at a student meeting this skill.")


SYSTEM_WRITE = (
    "You tell a school student what to go and practise. They have just been "
    "told which single skill is in the way of the question they were stuck on, "
    "and they are reading it alone, possibly at eleven at night, possibly "
    "having been behind for years.\n"
    "Write to them, not about them. Plain words, no jargon they would have to "
    "look up, and never a sentence that implies they should already know this."
)


def _describe(skill: Skill) -> str:
    return (
        f"id: {skill.id}\n"
        f"name: {skill.name}\n"
        f"kind: {skill.kind}\n"
        f"what a question on it must require: {skill.probe}\n"
    )


def build_prompt(skills: list[Skill]) -> str:
    listed = "\n".join(_describe(s) for s in skills)
    return (
        "Write one practice line for each skill below.\n\n"
        f"{listed}\n"
        "EACH LINE\n"
        "- One or two sentences. Never three.\n"
        "- Say what to DO, starting with the action: \"Practise...\", "
        "\"Get comfortable...\", \"Make sure you can...\".\n"
        "- Include one concrete example in plain ASCII, so they can see what "
        "the thing looks like - \"the highest common factor of 18 and 24\" "
        "rather than \"two numbers\". Invent an example that fits the skill in "
        "front of you, and check it works before you write it down: an equation "
        "must actually have a solution, and any answer you state must be right.\n"
        "- End with something searchable, in quotes, that would actually return "
        "practice questions - a phrase a student would type, with the "
        "qualification if it helps: search \"expanding brackets GCSE\".\n"
        "- A 'procedure' skill is practised by doing lots of them. A 'fact' is "
        "learned by recall. A 'concept' is not drilled at all - for those, say "
        "what they should be able to explain, and to whom.\n"
        "- Do not teach the method. Do not explain why it matters. Do not "
        "praise, reassure, or mention the diagnosis. They are reading this to "
        "find out what to open next.\n"
        "- Plain ASCII maths: / for division, ^ for powers, sqrt() for roots. "
        "No LaTeX and no symbols outside ASCII.\n"
        "- Use the id exactly as given, so the lines can be matched back up."
    )


def write_batch(skills: list[Skill], client: Anthropic | None = None) -> dict[str, str]:
    client = client or Anthropic()
    response = client.messages.parse(
        system=SYSTEM_WRITE,
        messages=[{"role": "user", "content": build_prompt(skills)}],
        output_format=Batch,
        **WRITE.kwargs(),
    )
    parsed = response.parsed_output
    if parsed is None:
        return {}
    wanted = {s.id for s in skills}
    return {p.skill_id: _tidy(p.line) for p in parsed.lines if p.skill_id in wanted}


# Structured output sometimes leaks the model's own working into the line - a
# fragment of the JSON it was assembling, a note to itself, once the phrase
# "No wait". The model check reads these for meaning and passes them, because
# the sentence in front of the rubbish is fine. So they are caught here instead.
#
# Square brackets are not a junk signal on their own: [x^3/3] is how a definite
# integral is written, and half the integration lines contain one.
SCRATCH = re.compile(
    r"}\]|'\]|':|\\|\bper schema\b|\bLet's write\b|\bNo wait\b", re.I
)

# A full stop welded to the next word, which is how leaked text joins on:
# "34.host", ".dthis". Decimals and e.g./i.e. are the legitimate cases.
WELDED = re.compile(r"(?<![a-z])(?<!\d)\.(?!\d)[a-z]{3,}")


def looks_wrong(line: str) -> str | None:
    """Why this line must never be shown to anyone, or None if it is safe.

    Damage only. A line that is merely long, or that ends without a search
    phrase, is thin rather than broken - see thin_reason.
    """
    if not line.strip():
        return "empty"
    if any(ord(c) > 127 for c in line):
        return "not plain ASCII"
    if SCRATCH.search(line):
        return "carries the model's own scratch text"
    if WELDED.search(line):
        return "a full stop welded to the next word"
    if len(line) > 340:
        return f"far too long at {len(line)} characters"
    return None


def thin_reason(line: str) -> str | None:
    """Worth rewriting, but safe to ship in the meantime."""
    if not re.search(r'"[^"]+"', line):
        return "no searchable phrase in quotes"
    if len(line) > 300:
        return f"long at {len(line)} characters"
    return None


# Lines have come back with stray characters welded on the end - a lone brace,
# a quote, once the word "dispatch". Harmless to the model, and the first thing
# a student would notice.
def _tidy(line: str) -> str:
    line = line.strip()
    while line and line[-1] not in '.!?"':
        line = line[:-1].rstrip()
    return line


def load() -> dict[str, str]:
    if not PRACTICE_FILE.exists():
        return {}
    with PRACTICE_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def save(lines: dict[str, str]) -> None:
    """Sorted, one key per line, so a diff shows which lines changed."""
    PRACTICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PRACTICE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(lines.items())), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def line_for(skill_id: str) -> str | None:
    """The practice line for one skill, or None if it has not been written."""
    return _cache().get(skill_id)


_loaded: dict[str, str] | None = None


def _cache() -> dict[str, str]:
    global _loaded
    if _loaded is None:
        try:
            _loaded = load()
        except Exception:  # noqa: BLE001 - a missing line must never break a report
            _loaded = {}
    return _loaded


def fill(skill_ids: list[str], workers: int = WORKERS) -> dict[str, str]:
    batches = [
        [SKILLS[s] for s in skill_ids[i : i + BATCH]]
        for i in range(0, len(skill_ids), BATCH)
    ]
    written: dict[str, str] = {}
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
    return written


# ---- Checking --------------------------------------------------------------

SYSTEM_CHECK = (
    "You are checking advice written for a school student about what to "
    "practise. You did not write it. Judge it against the skill it is meant to "
    "describe, and say plainly what is wrong before you decide."
)


def check_one(skill: Skill, line: str, client: Anthropic | None = None) -> Judgement:
    client = client or Anthropic()
    response = client.messages.parse(
        system=SYSTEM_CHECK,
        messages=[{
            "role": "user",
            "content": (
                f"THE SKILL\n{_describe(skill)}\n"
                f"THE LINE WRITTEN FOR IT\n{line}\n\n"
                "Decide three things.\n"
                "- accurate: is the maths right, including any worked example? "
                "An example that does not come out to what it says is the worst "
                "failure here.\n"
                "- on_target: does it describe THIS skill, rather than one just "
                "above or below it in difficulty?\n"
                "- right_level: would a student who is meeting this skill "
                "understand it, without it assuming what they are missing?"
            ),
        }],
        output_format=Judgement,
        **CHECK.kwargs(),
    )
    return response.parsed_output or Judgement(
        problem="the checker returned nothing", accurate=False, on_target=False, right_level=False
    )


def check_all(lines: dict[str, str], workers: int = WORKERS) -> list[tuple[str, Judgement]]:
    client = Anthropic()
    items = [(s, l) for s, l in sorted(lines.items()) if s in SKILLS]

    def work(item):
        skill_id, line = item
        try:
            return skill_id, check_one(SKILLS[skill_id], line, client)
        except Exception as error:  # noqa: BLE001
            return skill_id, Judgement(
                problem=f"check failed: {error}", accurate=False, on_target=False, right_level=False
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, items))


# ---- Command line ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Practice lines for each skill.")
    parser.add_argument("--fill", action="store_true", help="write missing lines")
    parser.add_argument("--all", action="store_true", help="rewrite every line")
    parser.add_argument("--check", action="store_true", help="verify with another model")
    parser.add_argument("--show", help="print the line for one skill")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)

    lines = load()

    if args.show:
        print(lines.get(args.show) or f"Nothing written for '{args.show}'.")
        return 0

    if args.check:
        if not lines:
            print("Nothing written yet.")
            return 1
        broken = [(s, why) for s, l in sorted(lines.items())
                  if (why := looks_wrong(l))]
        if broken:
            print(f"{len(broken)} lines are malformed and must not ship:\n")
            for skill_id, why in broken:
                print(f"  [{why}] {SKILLS[skill_id].name}")
                print(f"      {lines[skill_id]}")
            print()

        thin = [(s, why) for s, l in sorted(lines.items()) if (why := thin_reason(l))]
        if thin:
            print(f"{len(thin)} lines are thin - worth rewriting, safe meanwhile: "
                  + ", ".join(s for s, _ in thin))
            print()


        print(f"Checking {len(lines)} lines on {CHECK.model}...")
        bad = []
        for skill_id, verdict in check_all(lines, args.workers):
            if not (verdict.accurate and verdict.on_target and verdict.right_level):
                bad.append((skill_id, verdict))
        print()
        if not bad:
            print(f"All {len(lines)} lines were confirmed accurate, on target and "
                  "at the right level.")
            return 0
        print(f"{len(bad)} of {len(lines)} need looking at:\n")
        for skill_id, verdict in bad:
            flags = ",".join(
                name for name, ok in (
                    ("inaccurate", verdict.accurate),
                    ("off-target", verdict.on_target),
                    ("wrong level", verdict.right_level),
                ) if not ok
            )
            print(f"  [{flags}] {SKILLS[skill_id].name}")
            print(f"      {lines[skill_id]}")
            print(f"      -> {verdict.problem}")
        return 1

    if not args.fill:
        missing = [s for s in SKILLS if s not in lines]
        print(f"{len(lines)} lines written, {len(missing)} of {len(SKILLS)} skills missing")
        for skill_id in missing[:15]:
            print(f"  {SKILLS[skill_id].name}")
        if len(missing) > 15:
            print(f"  ... and {len(missing) - 15} more")
        return 0

    wanted = list(SKILLS) if args.all else [s for s in SKILLS if s not in lines]
    if not wanted:
        print("Every skill already has a line. Use --all to rewrite them.")
        return 0

    print(f"Writing {len(wanted)} lines...")
    lines.update(fill(wanted, args.workers))
    save(lines)
    print(f"\nWrote {PRACTICE_FILE.name}. Run --check before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
