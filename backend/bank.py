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
    python backend/bank.py --save          # write the shelf out to be committed
    python backend/bank.py --load          # put a committed shelf back
    python backend/bank.py --dry           # skills that cost a live generation
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import store
from audit import Verified, verify
from graph import SKILLS
from questions import BadQuestionError, MultipleChoiceQuestion, generate_batch

# Falling back to a live generation costs money, so it goes to the log where
# the host will show it, as well as to the database where it can be counted.
log = logging.getLogger("ownit.bank")

# Enough that a student meeting a skill twice probably sees a different
# question, few enough that filling the whole graph stays cheap.
PER_SKILL = 5

WORKERS = 5

# Where the shelf lives when it is not in the database. One question per line,
# so a commit shows which questions changed rather than "binary file differs",
# and two people adding questions do not collide over the whole file.
SHELF_FILE = Path(__file__).resolve().parent.parent / "data" / "question_bank.jsonl"


def save(path: Path | None = None) -> int:
    """Write the shelf to a file that can be committed."""
    path = Path(path or SHELF_FILE)
    connection = store.connect()
    try:
        records = store.export_bank(connection)
    finally:
        connection.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)


def load(path: Path | None = None) -> int:
    """Put the committed shelf back and make the database agree with it.

    Returns how many rows changed - added, or retired because the file says so.
    """
    path = Path(path or SHELF_FILE)
    if not path.exists():
        return 0

    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    connection = store.connect()
    try:
        return store.restore_bank(connection, records)
    finally:
        connection.close()


# Whether this process has already tried the file. Restocking reads the whole
# shelf in one go, so it is worth doing once rather than per skill, and worth
# not repeating when the file genuinely has nothing for a skill.
_looked_in_the_file = False


def restock(force: bool = False) -> int:
    """Fill an empty database from the committed shelf.

    This is what stands between a deploy and a large bill. The database is the
    machine's, and on a host with an ephemeral disk it is empty every time the
    app restarts - so without this, the first student through each skill pays
    to have five questions written that are already sitting in the repository.
    """
    global _looked_in_the_file
    if _looked_in_the_file and not force:
        return 0
    _looked_in_the_file = True

    try:
        added = load()
    except Exception as error:  # noqa: BLE001 - a bad shelf file must not stop a session
        log.warning("could not read the shelf file: %s", error)
        return 0

    if added:
        log.info("restocked %s questions from %s", added, SHELF_FILE.name)
    return added


def question_for(skill_id: str, *, connection=None) -> tuple[int | None, MultipleChoiceQuestion]:
    """A question for this skill: off the shelf if there is one, else written.

    Three places are tried, cheapest first: the database, then the committed
    shelf file, then the model. The model is the only one that costs anything,
    and reaching it means this skill has nothing anywhere - which is worth
    knowing about, so it is recorded rather than done quietly.

    Returns the bank id alongside it so the answer can be recorded against the
    question that was actually asked. The id is None when banking failed, which
    must never stop a student being asked something.
    """
    own_connection = connection is None
    try:
        connection = connection or store.connect()
    except Exception:  # noqa: BLE001 - a broken shelf is not a broken session
        # Nowhere to keep anything, but what is asked still gets checked.
        return None, _one_to_ask(skill_id, *_written_and_checked(skill_id))

    try:
        taken = store.take_question(connection, skill_id)
        if taken is not None:
            return taken

        # Nothing in the database for this skill. Before paying for a question,
        # look in the file - after a deploy the database is empty and the whole
        # shelf is sitting in the repository, unread.
        if restock():
            taken = store.take_question(connection, skill_id)
            if taken is not None:
                return taken

        # Nothing anywhere. This is the only path that costs money, so say so
        # loudly and write it down: a skill arriving here is either new or has
        # had everything retired under it, and both are worth seeing.
        log.warning(
            "no banked question for %r - generating live, which costs money", skill_id
        )
        try:
            store.record_live_generation(connection, skill_id)
        except Exception:  # noqa: BLE001 - bookkeeping must not break a session
            pass

        passed, spare = _written_and_checked(skill_id)
        if not passed:
            return None, _one_to_ask(skill_id, passed, spare)

        # Only what a blind solve agreed with goes on the shelf, because the
        # shelf is what every later student is served from.
        ids = [store.bank_question(connection, skill_id, q) for q in passed]
        choice = random.randrange(len(passed))
        return ids[choice], passed[choice]
    finally:
        if own_connection:
            connection.close()


def another_question_for(skill_id: str, session_id: int, *, connection
                         ) -> tuple[int | None, MultipleChoiceQuestion]:
    """A second question on a skill, different from any this session has seen.

    Asked when a skill is about to be named as the gap, to tell a real gap
    from a slip. The same question again would only test whether they
    remember what they picked, so anything already answered here is out.

    The shelf first, least-used first. Written live - and checked, like
    every new question - only when the shelf has nothing unseen left.
    """
    from questions import Distractor

    unseen = store.unasked_for(connection, skill_id, session_id)
    if unseen:
        row = unseen[0]
        return row["id"], MultipleChoiceQuestion(
            question=row["question"],
            correct_option=row["correct_option"],
            distractors=[Distractor(**d) for d in json.loads(row["distractors"])],
        )

    log.warning("nothing unseen on the shelf for %r - writing a follow-up live, "
                "which costs money", skill_id)
    try:
        store.record_live_generation(connection, skill_id)
    except Exception:  # noqa: BLE001 - bookkeeping must not break a session
        pass

    seen = {row["question"] for row in connection.execute(
        "SELECT question FROM answers WHERE session_id = ?", (session_id,))}
    passed, spare = _written_and_checked(skill_id)
    passed = [q for q in passed if q.question not in seen]
    spare = [q for q in spare if q.question not in seen]
    if not passed:
        return None, _one_to_ask(skill_id, passed, spare)

    ids = [store.bank_question(connection, skill_id, q) for q in passed]
    return ids[0], passed[0]


# Two goes before giving up. A whole batch failing its check is rare enough
# that a second batch nearly always has something in it, and a third would
# just be paying again for a skill whose questions keep coming out wrong.
ATTEMPTS = 2


def _written_and_checked(skill_id: str, count: int = PER_SKILL):
    """Write a batch, check it, and try once more if nothing survived.

    Returns (passed, spare). `passed` were solved blind to the same answer as
    their key. `spare` could not be checked at all - an API error, not a
    wrong answer - and are only ever a last resort, never kept. Anything
    judged wrong is in neither.
    """
    spare: list = []
    for _ in range(ATTEMPTS):
        written = generate_batch(skill_id, count)
        checked: Verified = verify(skill_id, written)
        if checked.rejected:
            log.warning("%s of %s questions written for %r were judged wrong "
                        "and thrown away", len(checked.rejected), len(written),
                        skill_id)
        if checked.passed:
            return checked.passed, spare
        spare.extend(checked.unchecked)
    return [], spare


def _one_to_ask(skill_id: str, passed: list, spare: list) -> MultipleChoiceQuestion:
    """Something to put in front of the student, in order of trust."""
    if passed:
        return random.choice(passed)
    if spare:
        # The checker could not be reached, which says nothing about the
        # question. Asked once so the session can go on, and not kept, so it
        # is never served again without being checked.
        log.warning("could not check any question for %r - asking one "
                    "unchecked and not keeping it", skill_id)
        return random.choice(spare)
    # Everything written was solved to a different answer than its key, twice
    # over. Asking one of those is how a student gets marked wrong for being
    # right, which is worse than this session stopping.
    raise BadQuestionError(
        f"every question written for {skill_id!r} was judged wrong when checked")


def fill(skill_ids: list[str], per_skill: int = PER_SKILL, workers: int = WORKERS) -> int:
    """Write questions for these skills and shelve them."""
    written = 0
    connection = store.connect()

    def work(skill_id: str):
        try:
            batch = generate_batch(skill_id, per_skill)
            return skill_id, len(batch), verify(skill_id, batch)
        except Exception as error:  # noqa: BLE001 - one bad skill must not stop the fill
            print(f"  {skill_id}: failed ({error})", file=sys.stderr)
            return skill_id, 0, Verified([], [], [])

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for skill_id, made, checked in pool.map(work, skill_ids):
                # Only what was checked and agreed with goes on the shelf.
                for question in checked.passed:
                    store.bank_question(connection, skill_id, question)
                written += len(checked.passed)
                dropped = []
                if checked.rejected:
                    dropped.append(f"{len(checked.rejected)} judged wrong")
                if checked.unchecked:
                    dropped.append(f"{len(checked.unchecked)} could not be checked")
                print(f"  {skill_id}: kept {len(checked.passed)} of {made}"
                      + (f" ({', '.join(dropped)})" if dropped else ""))
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


def show_dry() -> None:
    """Skills that have cost money because the shelf had nothing for them."""
    connection = store.connect()
    try:
        dry = store.ran_dry(connection)
        held = store.bank_counts(connection)
    finally:
        connection.close()

    if not dry:
        print("No question has ever had to be written live. The shelf has held.")
        return

    print("Skills that ran dry, and what they hold now:")
    for row in dry:
        name = SKILLS[row["skill_id"]].name if row["skill_id"] in SKILLS else row["skill_id"]
        now = held.get(row["skill_id"], 0)
        print(f"  {row['times']:>3}x  {name:<40} holds {now} now")
        print(f"        last: {row['last_time'][:16]}")
    print()
    print("Run --fill to top these up, then --save and commit.")


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
    parser.add_argument(
        "--save", action="store_true", help="write the shelf to a committable file"
    )
    parser.add_argument("--load", action="store_true", help="put a saved shelf back")
    parser.add_argument(
        "--dry", action="store_true", help="skills that have cost a live generation"
    )
    parser.add_argument("--per-skill", type=int, default=PER_SKILL)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)

    if args.weak:
        show_weak()
        return 0

    if args.dry:
        show_dry()
        return 0

    if args.save:
        count = save()
        # Shown relative to where you are standing when that says anything, and
        # in full when it does not. Run from backend/ this used to raise, after
        # the file had already been written - a traceback that looked like the
        # save had failed when it had in fact just succeeded.
        try:
            where = SHELF_FILE.relative_to(Path.cwd())
        except ValueError:
            where = SHELF_FILE
        print(f"Wrote {count} questions to {where}")
        print("Commit it - that is what stops this costing money twice.")
        return 0

    if args.load:
        if not SHELF_FILE.exists():
            print(f"No saved shelf at {SHELF_FILE}. Nothing to put back.")
            return 1
        added = load()
        print(f"Put back {added} questions." if added else "Already up to date.")
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
    print("Run --save and commit, or these are lost with the database.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
