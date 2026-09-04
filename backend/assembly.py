"""Does passing every prerequisite actually mean you can do the question?

For most doorways the answer is yes. For some it is not, and that gap is the
hardest kind to see, because it is invisible to everything else here:

  the walk cannot find it - the act sits AT the doorway, and the doorway is
  never asked about, since being stuck on it is the premise;
  the chain cannot show it - a chain is made of nodes;
  the ladder cannot scaffold it - it only climbs steps the graph knows about.

Found on 2026-09-02 by a tutor watching a student. The walk traced a subtraction
question correctly down to what a fraction is and back up through equivalent
fractions and finding a common denominator, and the student still could not do
it. Nothing had tested combining the numerators over the shared denominator.
Both children held; 10/12 - 3/12 = 7/24 had nowhere to live.

WHAT THIS FILE IS FOR

Two jobs. It keeps the backlog of suspected gaps in an order that comes from
evidence rather than from opinion, and it is the check a new doorway has to
pass before it is finished.

The order is derived, never hand-written. A suspected gap is a suspicion until
somebody outside this repository confirms it - a tutor marking a diagnosis
wrong, an examiner report naming the error, or the same thing said twice in
feedback. Those are counted here and the backlog sorts by them. With no
evidence, everything sits at zero and the honest report is "nothing is
confirmed yet", not a ranking invented to look like progress.

    python backend/assembly.py             # the backlog, most-confirmed first
    python backend/assembly.py --unjudged  # doorways nobody has judged
    python backend/assembly.py --judge     # judge those (one model call, costs)
    python backend/assembly.py --confirm find_stationary_points \\
        --source "tutor session" --detail "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

AUDIT_FILE = Path(__file__).resolve().parent.parent / "data" / "assembly_audit.json"

# What counts as somebody outside this repository saying the gap is real.
SOURCES = ("tutor session", "examiner report", "feedback pattern")


def load() -> dict:
    return json.loads(AUDIT_FILE.read_text(encoding="utf-8"))


def save(audit: dict) -> None:
    audit["verdicts"] = dict(sorted(audit["verdicts"].items()))
    AUDIT_FILE.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def doorways() -> set[str]:
    from graph import entry_points

    return {s.id for s in entry_points()}


def unjudged(audit: dict | None = None) -> list[str]:
    """Doorways nobody has asked the question about. This is the standing rule:
    a new doorway is not finished until it is judged."""
    audit = audit or load()
    return sorted(doorways() - set(audit["verdicts"]))


def suspected(audit: dict | None = None) -> dict:
    audit = audit or load()
    return {
        skill_id: verdict
        for skill_id, verdict in audit["verdicts"].items()
        if not verdict["modelled"]
    }


def confirmations(skill_id: str, audit: dict | None = None) -> list:
    audit = audit or load()
    return (audit["verdicts"].get(skill_id) or {}).get("evidence") or []


def backlog(connection=None, audit: dict | None = None) -> list[tuple]:
    """Suspected gaps, most-confirmed first.

    Two kinds of confirmation, counted together. Recorded evidence is written
    into the audit file by hand when a tutor says something or an examiner
    report names the error. Live evidence is read out of the database: a
    diagnosis marked wrong on a session that STARTED at this doorway is exactly
    what this fault looks like from the outside - every prerequisite answered,
    the question still not doable, so the tutor says the diagnosis was wrong.
    """
    audit = audit or load()
    live: dict = {}
    if connection is not None:
        import store

        live = store.verdicts_by_entry(connection)

    rows = []
    for skill_id, verdict in suspected(audit).items():
        recorded = len(verdict.get("evidence") or [])
        wrong = (live.get(skill_id) or {}).get("wrong", 0)
        rows.append((recorded + wrong, recorded, wrong, skill_id, verdict))
    return sorted(rows, key=lambda r: (-r[0], r[3]))


def confirm(skill_id: str, source: str, detail: str, on: str | None = None) -> dict:
    """Record that somebody outside this repository saw this gap."""
    audit = load()
    if skill_id not in audit["verdicts"]:
        raise KeyError(f"{skill_id} has not been judged - run --judge first")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")

    entry = audit["verdicts"][skill_id]
    entry.setdefault("evidence", []).append(
        {"source": source, "on": on or date.today().isoformat(), "detail": detail})
    save(audit)
    return entry


# ---- Command line ----------------------------------------------------------


def _report(connection=None) -> None:
    audit = load()
    todo = unjudged(audit)
    rows = backlog(connection, audit)

    print(f"{len(audit['verdicts'])} doorways judged, {len(todo)} not yet")
    if todo:
        print("  never judged:", ", ".join(todo))
    print()

    confirmed = [r for r in rows if r[0]]
    print(f"{len(rows)} suspected gaps, {len(confirmed)} with outside confirmation")
    if not confirmed:
        print()
        print("  Nothing is confirmed. Every one of these came from a single")
        print("  model pass and none has been seen in a real student, so the")
        print("  order below is alphabetical and means nothing. Fix them as")
        print("  confirmation arrives, not from the top.")
    print()

    for total, recorded, wrong, skill_id, verdict in rows:
        mark = f"[{total}]" if total else "[ ]"
        print(f"  {mark} {skill_id}")
        print(f"        act: {verdict['act']}")
        if recorded:
            for got in verdict["evidence"]:
                print(f"        seen: {got['source']}, {got['on']} - {got['detail'][:70]}")
        if wrong:
            print(f"        {wrong} diagnosis(es) here marked wrong by a tutor")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The assembly-gap backlog.")
    parser.add_argument("--unjudged", action="store_true",
                        help="list doorways nobody has judged")
    parser.add_argument("--judge", action="store_true",
                        help="judge the unjudged ones with a model (costs money)")
    parser.add_argument("--confirm", metavar="SKILL",
                        help="record outside evidence for a suspected gap")
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument("--detail", default="")
    args = parser.parse_args(argv)

    if args.unjudged:
        todo = unjudged()
        print("\n".join(todo) if todo else "Every doorway has been judged.")
        return 0

    if args.confirm:
        if not args.source or not args.detail:
            print("error: --confirm needs --source and --detail", file=sys.stderr)
            return 2
        try:
            confirm(args.confirm, args.source, args.detail)
        except (KeyError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(f"Recorded. {args.confirm} now has "
              f"{len(confirmations(args.confirm))} confirmation(s).")
        return 0

    if args.judge:
        todo = unjudged()
        if not todo:
            print("Every doorway has been judged. Nothing to do.")
            return 0
        print(f"{len(todo)} doorways to judge: {', '.join(todo)}")
        print("Run scratchpad/audit_assembly.py, which sends the whole graph at "
              "once so a partial view cannot invent absences.")
        return 0

    connection = None
    try:
        import store

        connection = store.connect()
        _report(connection)
    finally:
        if connection is not None:
            connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
