"""Checks that the answer marked correct in the bank actually is correct.

A wrong answer key is the worst failure this system has. The walk can be
perfect and the diagnosis still wrong: a student who answers correctly is
recorded as not holding the skill, and the descent goes hunting a gap that is
not there. eval.py cannot catch it, because eval tests the walk's reasoning
and takes the questions on trust.

How it checks, and why this way:

The model is never told which option is marked correct. It is given the
question and all four options and asked to work it out and pick one. Its pick
is then compared with the stored key. Showing it the key first would only
measure how agreeable the model is, and comparing free-text answers would
drown in false alarms, since 15x^4 - 4x^(-3) and 15x^4 - 4/x^3 are the same
answer written two ways. Picking a letter has neither problem.

Anything the two disagree on goes to a second, stronger model before being
touched, so one model having a bad day does not retire a good question.

    python backend/audit.py                 # check everything, report only
    python backend/audit.py --skill surds   # check one skill
    python backend/audit.py --fix           # retire the bad ones and refill
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import llm
import store
from graph import SKILLS

# Same place questions.py reads it from. Without this the key is missing and
# every check errors, which used to look exactly like every answer being wrong.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Solving is judgement, not generation. The whole point is to be a second
# opinion, so it must not be the model that wrote the question.
FIRST_PASS = llm.Task(llm.SONNET, effort="medium")

# Only for the disagreements, which should be few - but "few" is not "cheap".
# This was Opus at high effort, which is the most expensive setting in the
# codebase, and running it over two audits was most of one month's credit. A
# tiebreak between a checker and a written answer does not need extended
# thinking; it needs a better reader, which is the model rather than the dial.
SECOND_PASS = llm.Task(llm.OPUS, effort="low")

WORKERS = 8

LETTERS = "ABCD"

SYSTEM_PROMPT = (
    "You are checking A-level mathematics multiple-choice questions.\n"
    "Work the question out yourself, carefully, and pick the option that is "
    "correct.\n"
    "Show your working before you commit to a letter.\n"
    "If none of the options is correct, say so with the letter X and explain "
    "what the right answer actually is."
)


class Verdict(BaseModel):
    """Ordered so the model has to work before it commits to an answer."""

    working: str = Field(description="Brief working, a line or two.")
    answer: str = Field(description="The letter of the correct option, or X if none is.")
    note: str = Field(
        default="",
        description="If X, what the right answer is. Otherwise leave empty.",
    )


@dataclass
class Checked:
    banked_id: int
    skill_id: str
    question: str
    stored: str
    picked: str | None
    working: str
    note: str
    # A check that could not run is not a check that disagreed. Conflating the
    # two once made an expired key look like every answer in the bank being
    # wrong, and would have retired good questions under --fix.
    failed: bool = False

    @property
    def agrees(self) -> bool:
        return self.picked is not None and self.picked == self.stored

    @property
    def disagrees(self) -> bool:
        """Genuinely judged wrong, as opposed to never judged at all."""
        return not self.failed and not self.agrees

    @property
    def none_correct(self) -> bool:
        return not self.failed and self.picked is None


def _options_for(row) -> list[str]:
    """The four options, in a fixed order that does not leak the answer.

    Seeded on the row id so a re-run asks the same question the same way, and
    so the correct one is not always in the same place.
    """
    options = [row["correct_option"]] + [
        d["option"] for d in json.loads(row["distractors"])
    ]
    random.Random(row["id"]).shuffle(options)
    return options


def check_one(row, task: llm.Task, client: Anthropic | None = None) -> Checked:
    client = client or Anthropic()
    options = _options_for(row)

    listed = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, options))
    prompt = f"{row['question']}\n\n{listed}"

    response = client.messages.parse(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_format=Verdict,
        **task.kwargs(),
    )

    verdict = response.parsed_output
    if verdict is None:
        # Treated as a disagreement so it gets looked at rather than passed.
        return Checked(
            row["id"], row["skill_id"], row["question"], row["correct_option"],
            None, "", "the checker returned nothing", failed=True,
        )

    letter = (verdict.answer or "").strip().upper()[:1]
    picked = options[LETTERS.index(letter)] if letter in LETTERS else None

    return Checked(
        banked_id=row["id"],
        skill_id=row["skill_id"],
        question=row["question"],
        stored=row["correct_option"],
        picked=picked,
        working=verdict.working,
        note=verdict.note,
    )


def check_many(rows, task: llm.Task, workers: int = WORKERS) -> list[Checked]:
    client = Anthropic()

    def work(row):
        try:
            return check_one(row, task, client)
        except Exception as error:  # noqa: BLE001 - one failure must not stop the pass
            print(f"  [{row['id']}] {row['skill_id']}: {error}", file=sys.stderr)
            return Checked(
                row["id"], row["skill_id"], row["question"], row["correct_option"],
                None, "", f"check failed: {error}", failed=True,
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, rows))


def banked(connection, skill_id: str | None = None) -> list:
    if skill_id:
        return connection.execute(
            "SELECT * FROM question_bank WHERE retired = 0 AND skill_id = ? ORDER BY id",
            (skill_id,),
        ).fetchall()
    return connection.execute(
        "SELECT * FROM question_bank WHERE retired = 0 ORDER BY skill_id, id"
    ).fetchall()


# ---- Reporting -------------------------------------------------------------


def topic_of(skill_id: str) -> str:
    """Which topic a skill sits under, for grouping the damage.

    Shared skills carry no topic of their own, so they are reported together
    rather than being attributed to a topic they do not belong to.
    """
    skill = SKILLS.get(skill_id)
    if skill is None:
        return "unknown"
    return skill.topic or "(shared, no topic)"


def report(checked: list[Checked], confirmed: list[Checked] | None = None) -> None:
    bad = confirmed if confirmed is not None else [c for c in checked if c.disagrees]
    broke = [c for c in checked if c.failed]
    judged = [c for c in checked if not c.failed]

    print()
    print("=" * 70)
    print(f"{len(checked)} questions checked")
    if broke:
        print(f"{len(broke)} could not be checked at all - see the errors above")
    print()
    if not judged:
        print("  Nothing was actually checked. Fix the errors and re-run.")
        return
    if not bad:
        print(f"  Every one of the {len(judged)} stored answers was confirmed correct.")
        return

    rate = len(bad) / len(judged)
    print(f"  {len(bad)} of {len(judged)} wrong ({rate:.0%})")
    print()

    by_topic = defaultdict(list)
    for one in bad:
        by_topic[topic_of(one.skill_id)].append(one)

    counts = defaultdict(int)
    for one in judged:
        counts[topic_of(one.skill_id)] += 1

    print("  By topic:")
    for topic in sorted(by_topic, key=lambda t: -len(by_topic[t])):
        wrong = len(by_topic[topic])
        print(f"    {topic:<26} {wrong:>3} of {counts[topic]:<4} ({wrong / counts[topic]:.0%})")

    print()
    print("  Worst skills:")
    by_skill = defaultdict(list)
    for one in bad:
        by_skill[one.skill_id].append(one)
    for skill_id in sorted(by_skill, key=lambda s: -len(by_skill[s]))[:12]:
        print(f"    {SKILLS[skill_id].name:<38} {len(by_skill[skill_id])}")


def show(examples: list[Checked], limit: int = 8) -> None:
    print()
    print("  Examples:")
    for one in examples[:limit]:
        print()
        print(f"    [{one.banked_id}] {SKILLS[one.skill_id].name}")
        print(f"       Q       : {one.question[:96]}")
        print(f"       bank say: {one.stored}")
        print(f"       check   : {one.picked or 'none of the options is correct'}")
        if one.note:
            print(f"       note    : {one.note[:96]}")


# ---- Command line ----------------------------------------------------------


# Rough, and rough is the point: enough to notice that a number is bigger than
# expected before it is spent rather than after.
PENCE_PER_CHECK = 2


def _worth_it(count: int, agreed: bool) -> bool:
    """Say what this will cost, and stop unless told to carry on.

    Written after an audit run quietly cost several pounds. Nothing here knows
    the real price - it only knows the order of magnitude, which is all that is
    needed to catch "I meant to check twelve, not six hundred".
    """
    pounds = count * PENCE_PER_CHECK / 100
    print(f"About to check {count} questions - roughly £{pounds:.2f}, "
          f"more if many disagree.")
    if agreed:
        return True
    answer = input("Go ahead? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Nothing checked.")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the bank's answer keys.")
    parser.add_argument("--skill", help="check one skill only")
    parser.add_argument("--limit", type=int, help="check only the first N")
    parser.add_argument(
        "--fix", action="store_true", help="retire confirmed-wrong questions and refill"
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--yes", action="store_true", help="skip the what-will-this-cost prompt"
    )
    args = parser.parse_args(argv)

    connection = store.connect()
    try:
        rows = banked(connection, args.skill)
    finally:
        connection.close()

    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Nothing on the shelf to check.")
        return 0

    if not _worth_it(len(rows), args.yes):
        return 0

    print(f"Checking {len(rows)} questions on {FIRST_PASS.model}...")
    checked = check_many(rows, FIRST_PASS, args.workers)

    disputed = [c for c in checked if c.disagrees]
    if not disputed:
        report(checked)
        return 0

    print(f"\n{len(disputed)} disagreed. Re-checking those on {SECOND_PASS.model}...")
    second = check_many(
        [r for r in rows if r["id"] in {d.banked_id for d in disputed}],
        SECOND_PASS,
        args.workers,
    )

    # Only questions both models reject are treated as broken. One model having
    # a bad day should not retire a question that is fine.
    confirmed = [c for c in second if c.disagrees]
    rescued = len(disputed) - len(confirmed) - len([c for c in second if c.failed])

    report(checked, confirmed)
    if rescued:
        print(f"\n  ({rescued} disputed by the first pass but upheld by the second)")
    show(confirmed)

    if not args.fix:
        print("\nNothing changed. Re-run with --fix to retire these and write replacements.")
        return 0

    connection = store.connect()
    try:
        for one in confirmed:
            store.retire_question(connection, one.banked_id)
    finally:
        connection.close()
    print(f"\nRetired {len(confirmed)} questions.")

    import bank

    short = sorted({c.skill_id for c in confirmed})
    print(f"Writing replacements for {len(short)} skills...")

    # Top each affected skill back up to its usual depth.
    connection = store.connect()
    try:
        held = store.bank_counts(connection)
    finally:
        connection.close()

    for skill_id in short:
        missing = max(0, bank.PER_SKILL - held.get(skill_id, 0))
        if missing:
            bank.fill([skill_id], per_skill=missing, workers=1)

    print("\nRe-run this to check the replacements before trusting them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
