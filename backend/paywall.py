"""Access codes, the price, and the payment link.

A deliberately small thing. This is a test of one question - will a UK parent
pay for this - and the cheapest honest way to ask it is a link to a payment
page and a code typed in afterwards. So there is no payment API here, no
accounts, no card details, and nothing that could leak a student's identity:
the only personal detail anywhere is the parent's own email, typed in by hand
off their receipt.

WHAT IT DOES NOT DO

It does not keep anybody out. The free-diagnosis flag lives in the browser,
so clearing it starts again, and a code works on any browser it is typed
into. Both were decided deliberately - a paywall that measured willingness to
pay is the point, and one that locked a sixteen year old out of their
homework while a parent was asleep is not.

THE LINK

Read from the environment rather than written down, so it can be changed
without a deploy. Nothing here knows or cares which company is on the other
end of it: a Paystack link and a Stripe link are both just a URL. The
variable keeps the name PAYSTACK_LINK because that is what was asked for, but
it will hold either.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone

# What a parent is asked for. One place, because it is written on two screens
# and they must never disagree - a page that says one price and a checkout
# that charges another is the fastest way to lose somebody mid-payment.
PRICE = "£6/month"

# How long a code lasts from the moment it is made.
DAYS = 30

# No I, O, 0 or 1. A code gets read off a screen and typed by somebody else,
# often a tired parent reading it out, and those four are where that goes
# wrong.
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
BLOCK = 4
BLOCKS = 2


def payment_link() -> str:
    """Where the pay button goes, or empty if it has not been set up yet.

    Empty is a normal state, not an error: the page then says payment is not
    open rather than sending somebody to a broken link.
    """
    return (os.environ.get("PAYSTACK_LINK") or "").strip()


def new_code() -> str:
    """A fresh code, in the shape OWNIT-7K2F-9QX4."""
    blocks = [
        "".join(secrets.choice(ALPHABET) for _ in range(BLOCK))
        for _ in range(BLOCKS)
    ]
    return "OWNIT-" + "-".join(blocks)


def tidy(typed: str) -> str:
    """What somebody typed, in the form the database holds.

    Upper-cased and stripped of spaces, because a code arrives having been
    read aloud, pasted out of an email with a trailing space, or typed in
    lower case on a phone that capitalises nothing.
    """
    return "".join((typed or "").split()).upper()


def expires_in(days: int = DAYS) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(
        timespec="seconds")


def has_expired(expires_at: str, now: datetime | None = None) -> bool:
    """True when a code's day has passed. A code we cannot read has."""
    try:
        when = datetime.fromisoformat(expires_at)
    except (TypeError, ValueError):
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when <= (now or datetime.now(timezone.utc))
