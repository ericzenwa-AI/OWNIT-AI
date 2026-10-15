"""Make an access code for somebody who has paid.

Run after a payment lands, with the email off their receipt:

    python backend/make_code.py --email parent@example.com

It prints one code. Send that to them; they give it to their child, who types
it into the box on the paywall. It works on any device, for 30 days.

    python backend/make_code.py --list          # every code and how it is doing
    python backend/make_code.py --days 90       # a longer one

WHOSE EMAIL

The parent's, and only because a code has to be sendable to somebody and
matched to a payment if they write in. No student's email is collected
anywhere in this product, and none should ever be typed in here.

The email is optional. If you would rather hold nothing at all, leave it out
and keep the receipt yourself - the code works either way.
"""

from __future__ import annotations

import argparse
import sys

import paywall
import store


def make(email: str | None, days: int, note: str | None) -> str:
    connection = store.connect()
    try:
        # Collisions are not a real risk at this scale, but a code that
        # already exists would raise rather than quietly overwrite somebody
        # else's access, so it is worth the three lines.
        for _ in range(5):
            code = paywall.new_code()
            if store.find_code(connection, code) is None:
                store.add_code(connection, code,
                               expires_at=paywall.expires_in(days),
                               parent_email=email, note=note)
                return code
        raise RuntimeError("could not find an unused code")
    finally:
        connection.close()


def show() -> None:
    connection = store.connect()
    try:
        rows = store.codes(connection)
    finally:
        connection.close()

    if not rows:
        print("No codes yet.")
        return

    print(f"{'code':<18} {'expires':<22} {'used':>5}  who")
    for row in rows:
        gone = " (expired)" if paywall.has_expired(row["expires_at"]) else ""
        print(f"{row['code']:<18} {row['expires_at'][:19]:<22} "
              f"{row['times_used']:>5}  {row['parent_email'] or '-'}{gone}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Access codes for ownIT.")
    parser.add_argument("--email", help="the PARENT's email, off their receipt")
    parser.add_argument("--days", type=int, default=paywall.DAYS)
    parser.add_argument("--note", help="anything worth remembering about it")
    parser.add_argument("--list", action="store_true", help="show every code")
    args = parser.parse_args(argv)

    if args.list:
        show()
        return 0

    if args.days < 1:
        print("--days has to be at least 1", file=sys.stderr)
        return 2

    code = make(args.email, args.days, args.note)
    print()
    print(f"  {code}")
    print()
    print(f"Good for {args.days} days. Send it to "
          f"{args.email or 'whoever paid'} and they give it to their child.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
