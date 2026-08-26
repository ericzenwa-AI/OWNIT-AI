"""Clear the numbers, without clearing anything that matters.

Most of the early traffic on a thing like this is the person who built it,
clicking through their own app. Those sessions sit in the funnel looking like
users, and the one number you actually want - did a stranger finish a walk -
is buried under thirty of your own.

So this empties the counting. It does not empty the two things that would hurt
to lose, and it cannot be talked into it:

  waitlist        people who typed their email in. Irreplaceable - there is no
                  copy of that anywhere else, and no way to ask again.
  question_bank   the shelf. Committed to the repository as well, so it would
                  come back, but only after paying a model to rewrite every
                  question it could not find.

Nothing is deleted without --clear. On its own it prints what is there and
stops, because the failure this guards against is running it while assuming it
would ask first.

    python backend/reset_analytics.py              # what is there
    python backend/reset_analytics.py --clear      # clear the testing noise
    python backend/reset_analytics.py --clear --and-feedback
"""

from __future__ import annotations

import argparse
import sys

import store

# Emptied by --clear. Ordered so a child goes before its parent, because
# answers and ratings both point at sessions.
NOISE = [
    "pending",
    "answers",
    "rating",
    "sessions",
    "page_view",
    "question_read",
    "live_generation",
]

# Things somebody typed, rather than something we counted. Kept unless asked
# for, because "most of it was me" is not "all of it was me".
SAID = ["feedback", "comments", "unplaced"]

# Not deletable from here at any flag. If you genuinely want these gone, do it
# deliberately somewhere else - this file is run in a hurry, on production, by
# someone who wants a clean funnel.
NEVER = {"waitlist", "question_bank"}


def counts(connection, tables) -> dict[str, int]:
    got = {}
    for table in tables:
        try:
            got[table] = connection.execute(
                f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608 - names are ours
            ).fetchone()["n"]
        except Exception:  # noqa: BLE001 - a table that does not exist yet is empty
            got[table] = 0
    return got


def show(connection) -> None:
    everything = counts(connection, NOISE + SAID + sorted(NEVER))
    print("Counting, cleared by --clear:")
    for table in NOISE:
        print(f"  {everything[table]:>6}  {table}")
    print()
    print("Typed by someone, cleared only with --and-feedback:")
    for table in SAID:
        print(f"  {everything[table]:>6}  {table}")
    print()
    print("Never cleared by this script:")
    for table in sorted(NEVER):
        print(f"  {everything[table]:>6}  {table}")


def clear(connection, tables) -> dict[str, int]:
    removed = {}
    for table in tables:
        assert table not in NEVER, f"refusing to clear {table}"
        before = counts(connection, [table])[table]
        connection.execute(f"DELETE FROM {table}")  # noqa: S608 - names are ours
        removed[table] = before
    connection.commit()
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clear the analytics, keep the rest.")
    parser.add_argument("--clear", action="store_true", help="actually delete")
    parser.add_argument(
        "--and-feedback",
        action="store_true",
        help="also clear comments, tutor verdicts and unplaced questions",
    )
    args = parser.parse_args(argv)

    connection = store.connect()
    try:
        if not args.clear:
            show(connection)
            print()
            print("Nothing was deleted. Add --clear to do it.")
            return 0

        tables = NOISE + (SAID if args.and_feedback else [])
        removed = clear(connection, tables)

        print("Cleared:")
        for table, n in removed.items():
            print(f"  {n:>6}  {table}")
        kept = counts(connection, sorted(NEVER) + ([] if args.and_feedback else SAID))
        print()
        print("Kept:")
        for table, n in sorted(kept.items()):
            print(f"  {n:>6}  {table}")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
