"""Serves questions off the shelf instead of writing a new one every time.

A question belongs to a skill, not to a student. Writing a fresh one for every
teenager who reaches index laws pays repeatedly for the same thing, and
generation is most of what a session costs.

The shelf fills itself. Ask for a skill nobody has reached before and it writes
a batch, keeps them, and hands one back - so a new skill works immediately and
gets cheaper on its own. Filling it up front is only an optimisation, worth
doing before a busy week rather than being a thing you must remember.

    python backend/bank.py                 # what is on the shelf
    python backend/bank.py --fill          # write questions for empty skills
    python backend/bank.py --fill --all    # top every skill up
    python backend/bank.py --weak          # questions that teach us nothing
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import store
from graph import SKILLS
from questions import MultipleChoiceQuestion, generate_batch

# Enough that a student meeting a skill twice probably sees a different
# question, few enough that filling the whole graph stays cheap.
PER_SKILL = 5

WORKERS = 5


def question_for(skill_id: str, *, connection=None) -> tuple[int | None, MultipleChoiceQuestion]:
    """A question for this skill: off the shelf if there is one, else written.

    Returns the bank id alongside it so the answer can be recorded against the
    question that was actually asked. The id is None when banking failed, which
    must never stop a student being asked something.
    """
    own_connection = connection is None
    try:
        connection = connection or store.connect()
    except Exception:  # noqa: BLE001 - a broken shelf is not a broken session
        from questions import generate_question

        return None, generate_question(skill_id)

    try:
        taken = store.take_question(connection, skill_id)
        if taken is not None:
            return taken

        # Nothing on the shelf. Write a batch, keep them, hand one back - the
        # next student through this skill costs nothing.
        written = generate_batch(skill_id, PER_SKILL)
        if not written:
            from questions import generate_question

            return None, generate_question(skill_id)

        ids = [store.bank_question(connection, skill_id, q) for q in written]
        choice = random.randrange(len(written))
        return ids[choice], written[choice]
    finally:
        if own_connection:
            connection.close()


def fill(skill_ids: list[str], per_skill: int = PER_SKILL, workers: int = WORKERS) -> int:
    """Write questions for these skills and shelve them."""
    written = 0
    connection = store.connect()

    def work(skill_id: str):
        try:
            return skill_id, generate_batch(skill_id, per_skill)
        except Exception as error:  # noqa: BLE001 - one bad skill must not stop the fill
            print(f"  {skill_id}: failed ({error})", file=sys.stderr)
            return skill_id, []

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for skill_id, questions in pool.map(work, skill_ids):
                for question in questions:
                    store.bank_question(connection, skill_id, question)
                written += len(questions)
                print(f"  {skill_id}: {len(questions)}")
    finally:
        connection.close()

    return written


# ---- Command line ---------------------------------------------------------


def show_shelf() -> None:
    connection = store.connect()
    try:
        held = store.bank_counts(connection)
    finally:
        connection.close()

    empty = [s for s in SKILLS if s not in held]
    total = sum(held.values())

    print(f"{total} questions on the shelf, across {len(held)} of {len(SKILLS)} skills")
    if empty:
        print(f"{len(empty)} skills have none yet:")
        for skill_id in sorted(empty)[:15]:
            print(f"  {SKILLS[skill_id].name}")
        if len(empty) > 15:
            print(f"  ... and {len(empty) - 15} more")


def show_weak() -> None:
    connection = store.connect()
    try:
        weak = store.weak_questions(connection)
    finally:
        connection.close()

    if not weak:
        print("No question has been asked enough times to judge yet.")
        return

    print("Questions that are not discriminating:")
    for row in weak:
        rate = row["pass_rate"]
        verdict = "everyone gets it right" if rate > 0.5 else "nobody does"
        print(f"  [{row['id']}] {SKILLS[row['skill_id']].name}: {verdict}")
        print(f"      {row['question'][:90]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the question bank.")
    parser.add_argument("--fill", action="store_true", help="write missing questions")
    parser.add_argument(
        "--all", action="store_true", help="top up every skill, not just empty ones"
    )
    parser.add_argument("--weak", action="store_true", help="show useless questions")
    parser.add_argument("--per-skill", type=int, default=PER_SKILL)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)

    if args.weak:
        show_weak()
        return 0

    if not args.fill:
        show_shelf()
        return 0

    connection = store.connect()
    try:
        held = store.bank_counts(connection)
    finally:
        connection.close()

    wanted = list(SKILLS) if args.all else [s for s in SKILLS if s not in held]
    if not wanted:
        print("Every skill already has questions. Use --all to top them up.")
        return 0

    print(f"Writing {args.per_skill} questions for {len(wanted)} skills...")
    written = fill(wanted, args.per_skill, args.workers)
    print(f"\nShelved {written} questions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
