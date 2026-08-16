"""Keeps what every diagnosis learned, so the tool gets better at doing them.

A walk currently prints to a terminal and evaporates. What it knew at that
moment - that this student picked "3 to the power 0 is 3" - is the one thing
nobody else has and nobody can copy out of the code, because it is not in the
code. It only exists if it is written down.

Four things come out of keeping it:

  - Distractors stop being invented. After a few hundred walks we know which
    wrong beliefs are real and how common, and can ask for those instead.
  - The graph can be checked. A prerequisite that never turns out to be the gap
    is either wrong or unreachable, and the data says which.
  - Fewer questions. If failing one skill reliably predicts failing another,
    the second can be inferred rather than asked.
  - The teaching that comes later has something to teach against. "Missing
    index laws" is a topic. "Believes a^0 = a" is a lesson.

This is a plain SQLite file - one file on disk, no server, nothing to install.
Moving to a proper database later is routine and changes nothing written here.

Nothing in this module is student-identifying. Sessions carry a reference you
choose, like "student_7", and the point is that it should not be a name.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "ownit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                INTEGER PRIMARY KEY,
    created_at        TEXT    NOT NULL,
    student_ref       TEXT,
    role              TEXT,
    question          TEXT,
    attempt           TEXT,
    entry_skill_id    TEXT    NOT NULL,
    entry_confidence  TEXT,
    entry_confirmed   INTEGER,
    root_gaps         TEXT,
    chain             TEXT,
    unchecked         TEXT,
    stopped_early     TEXT
);

CREATE TABLE IF NOT EXISTS answers (
    id            INTEGER PRIMARY KEY,
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    position      INTEGER NOT NULL,
    skill_id      TEXT    NOT NULL,
    question      TEXT,
    chosen        TEXT,
    outcome       TEXT    NOT NULL,
    misconception TEXT,
    seconds       REAL
);

-- Filled in afterwards by whoever knows: was this diagnosis actually right?
-- Without it, beta testers can only have opinions about the graph. With it,
-- they can correct it.
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    created_at  TEXT    NOT NULL,
    verdict     TEXT    NOT NULL,
    actual_gap  TEXT,
    note        TEXT
);

CREATE INDEX IF NOT EXISTS answers_by_skill ON answers(skill_id);
"""


def connect(path: str | Path = DEFAULT_PATH) -> sqlite3.Connection:
    """Open the store, creating the file and tables the first time."""
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _outcome(result) -> str:
    if result.held:
        return "correct"
    return "dont_know" if result.dont_know else "wrong"


def save_session(
    connection: sqlite3.Connection,
    diagnosis,
    *,
    question: str | None = None,
    attempt: str | None = None,
    student_ref: str | None = None,
    role: str | None = None,
    entry_confidence: str | None = None,
    entry_confirmed: bool | None = None,
) -> int:
    """Write one finished walk, and every answer in it. Returns the session id."""
    cursor = connection.execute(
        """INSERT INTO sessions (created_at, student_ref, role, question, attempt,
               entry_skill_id, entry_confidence, entry_confirmed,
               root_gaps, chain, unchecked, stopped_early)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            _now(),
            student_ref,
            role,
            question,
            attempt,
            diagnosis.entry_skill_id,
            entry_confidence,
            None if entry_confirmed is None else int(entry_confirmed),
            ",".join(diagnosis.root_gaps),
            " -> ".join(diagnosis.chains[0]) if diagnosis.chains else "",
            ",".join(diagnosis.unchecked),
            diagnosis.stopped_early,
        ),
    )
    session_id = cursor.lastrowid

    # The entry node is not asked about - being stuck is the premise - so there
    # is no answer to record for it.
    asked = [r for r in diagnosis.results if r.asked]
    connection.executemany(
        """INSERT INTO answers (session_id, position, skill_id, question,
               chosen, outcome, misconception, seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                session_id,
                position,
                result.skill_id,
                result.question,
                result.chosen,
                _outcome(result),
                result.mistake,
                result.seconds,
            )
            for position, result in enumerate(asked, start=1)
        ],
    )

    connection.commit()
    return session_id


def record_feedback(
    connection: sqlite3.Connection,
    session_id: int,
    verdict: str,
    *,
    actual_gap: str | None = None,
    note: str | None = None,
) -> None:
    """Say whether a diagnosis was right, and what the real gap was if not.

    This is how a tutor corrects the graph rather than just disagreeing with it.
    """
    if verdict not in ("right", "wrong"):
        raise ValueError("verdict must be 'right' or 'wrong'")

    connection.execute(
        """INSERT INTO feedback (session_id, created_at, verdict, actual_gap, note)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, _now(), verdict, actual_gap, note),
    )
    connection.commit()


# ---- Reading it back ------------------------------------------------------


@dataclass(frozen=True)
class SkillStats:
    skill_id: str
    asked: int
    held: int
    wrong: int
    dont_know: int

    @property
    def held_rate(self) -> float:
        return self.held / self.asked if self.asked else 0.0


def skill_stats(connection: sqlite3.Connection) -> list[SkillStats]:
    """How every skill is performing, most-asked first."""
    rows = connection.execute(
        """SELECT skill_id,
                  COUNT(*)                                          AS asked,
                  SUM(outcome = 'correct')                          AS held,
                  SUM(outcome = 'wrong')                            AS wrong,
                  SUM(outcome = 'dont_know')                        AS dont_know
           FROM answers
           GROUP BY skill_id
           ORDER BY asked DESC"""
    ).fetchall()

    return [
        SkillStats(
            skill_id=row["skill_id"],
            asked=row["asked"],
            held=row["held"] or 0,
            wrong=row["wrong"] or 0,
            dont_know=row["dont_know"] or 0,
        )
        for row in rows
    ]


def common_misconceptions(
    connection: sqlite3.Connection, skill_id: str, limit: int = 5
) -> list[tuple[str, int]]:
    """The wrong beliefs students actually hold about one skill, commonest first.

    This is what eventually replaces invented distractors, and what the teaching
    layer will need in order to correct a belief rather than fill a blank.
    """
    rows = connection.execute(
        """SELECT misconception, COUNT(*) AS times
           FROM answers
           WHERE skill_id = ? AND outcome = 'wrong' AND misconception IS NOT NULL
           GROUP BY misconception
           ORDER BY times DESC
           LIMIT ?""",
        (skill_id, limit),
    ).fetchall()

    return [(row["misconception"], row["times"]) for row in rows]
