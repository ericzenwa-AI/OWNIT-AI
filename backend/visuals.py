"""Showing the gap, instead of telling them to go and look it up.

The diagnostic finds the gap with real machinery - a graph of 147 skills, a walk
down it, banked questions with a checked answer key - and then every path out of
the report says the same two things: "Practise..." and "Search...". For a tutor
that is enough; they take it from there. For a student on their own at eleven at
night it is where this product stops and the internet begins.

A picture is the part that cannot be replaced by a sentence. "The bottom says how
many parts the whole is cut into" is a sentence a student can read twice and
still not see. A bar cut into eight with five shaded is the same claim, and it
lands.

WHY THESE ARE FILES AND NOT GENERATED

Four separate rendering faults have reached real students in a fortnight - LaTeX
stored in the bank, a fraction inside an exponent, a negative exponent in a
denominator, stray Unicode. Every one was generated content that nothing checked
before somebody saw it. A wrong diagram is worse than a wrong answer key: it
looks authoritative, it is not marked, and nobody re-reads it.

Skills are finite and they do not change. So these are drawn once, committed,
and checked - the same treatment the question bank gets. It also makes them free
to serve, which matters when the day has a spend ceiling.

WHICH SKILLS GET ONE

Not all of them, and the graph already says which. `kind` is concept, procedure
or fact. A concept is a thing you can see and can barely say - that is where a
picture does work nothing else does. A procedure wants steps. A fact wants
recall, and a picture on one is decoration.

The three here are not a sample. They are the exact chain a real student fell
down on 2026-09-02, so that one session works end to end.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree

VISUAL_DIR = Path(__file__).resolve().parent.parent / "data" / "visuals"

# Kept small deliberately. These are inlined into the page, so every byte is on
# the wire for a student who may be on a phone on mobile data.
MAX_BYTES = 12_000


def path_for(skill_id: str) -> Path:
    return VISUAL_DIR / f"{skill_id}.svg"


def has(skill_id: str) -> bool:
    return path_for(skill_id).is_file()


def drawn() -> list[str]:
    """Every skill with a picture, in a fixed order."""
    if not VISUAL_DIR.is_dir():
        return []
    return sorted(p.stem for p in VISUAL_DIR.glob("*.svg"))


def for_skill(skill_id: str) -> str | None:
    """The SVG markup, ready to inline. None when there is no picture.

    Never raises. A missing or unreadable picture must not cost a student their
    diagnosis - the report is the thing that matters and this is an addition to
    it.
    """
    try:
        found = path_for(skill_id)
        if not found.is_file():
            return None
        markup = found.read_text(encoding="utf-8").strip()
        return markup or None
    except Exception:  # noqa: BLE001 - a picture is never worth an error page
        return None


# ---- Checking ---------------------------------------------------------------
#
# What a picture has to be before it is allowed near a student. These run in the
# test suite rather than at request time, because the files do not change
# between deploys and a student should not pay to have them re-checked.

FORBIDDEN = re.compile(r"<script|<foreignObject|javascript:|onload=|onclick=", re.I)


def wrong_with(skill_id: str) -> str | None:
    """Why this picture must not ship, or None if it is fine."""
    found = path_for(skill_id)
    if not found.is_file():
        return "there is no file"

    raw = found.read_bytes()
    if len(raw) > MAX_BYTES:
        return f"too big at {len(raw)} bytes, cap is {MAX_BYTES}"

    markup = raw.decode("utf-8", "replace")
    if "�" in markup:
        return "not valid UTF-8"

    try:
        root = ElementTree.fromstring(markup)
    except ElementTree.ParseError as error:
        return f"not well-formed XML: {error}"

    if not root.tag.endswith("svg"):
        return f"the root element is {root.tag}, not svg"
    if "viewBox" not in root.attrib:
        return "no viewBox, so it cannot scale to a phone"
    if "width" in root.attrib or "height" in root.attrib:
        return "a fixed width or height overrides the page's own sizing"

    if FORBIDDEN.search(markup):
        return "carries script or an event handler"

    # It is inlined into the page, so it inherits the page's colours. A picture
    # that hardcodes black is invisible on the dark theme, which is the failure
    # nobody notices until somebody reports a blank box.
    if "currentColor" not in markup:
        return "nothing uses currentColor, so it will not follow the theme"

    if "<title" not in markup:
        return "no <title>, so it says nothing to a screen reader"

    return None
