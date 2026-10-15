"""How the paid test is going.

    python backend/paywall_numbers.py
    python backend/paywall_numbers.py --days 7

Six counts, in the order somebody actually passes through them, so the place
it falls apart is the line where the number collapses. The drop from "saw the
paywall" to "pressed pay" is the number this whole test exists to produce.

The counts come out of the same events table as everything else rather than
somewhere of their own, so one reset clears one set of numbers.
"""

from __future__ import annotations

import argparse

import paywall
import store

# In the order a person meets them, not alphabetical: the shape of the drop
# is the finding.
STEPS = [
    ("free_diagnosis_completed", "Finished the free diagnosis"),
    ("paywall_shown", "Hit the paywall"),
    ("ask_parent_clicked", "Asked a parent"),
    ("parent_page_opened", "A parent opened the page"),
    ("payment_link_clicked", "Pressed pay"),
    ("code_redeemed", "Typed in a code"),
]


def counts(connection, days: int | None) -> dict:
    return {row["name"]: row["times"] for row in store.clicks(connection, days)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The paid test, in counts.")
    parser.add_argument("--days", type=int, help="only the last N days")
    args = parser.parse_args(argv)

    connection = store.connect()
    try:
        seen = counts(connection, args.days)
        sold = store.codes(connection)
    finally:
        connection.close()

    when = f"last {args.days} days" if args.days else "all time"
    print(f"\nThe paid test - {when}\n")

    top = seen.get("free_diagnosis_completed", 0)
    for name, label in STEPS:
        n = seen.get(name, 0)
        # A share of the people who finished a free diagnosis, which is the
        # only denominator that means anything here. Shown from the second
        # line on, because the first IS the denominator.
        share = ""
        if top and name != "free_diagnosis_completed":
            share = f"   {n / top:.0%} of those who finished one"
        print(f"  {label:<34} {n:>5}{share}")

    live = [c for c in sold if not paywall.has_expired(c["expires_at"])]
    print(f"\n  {'Codes made':<34} {len(sold):>5}")
    print(f"  {'Still valid':<34} {len(live):>5}")

    if not top:
        print("\nNobody has finished a free diagnosis yet, so there is nothing "
              "to read into the rest.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
