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

import json
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

-- Questions we could not place. Every one is a coverage gap named by someone
-- who teaches the subject, and it costs them nothing to report - they report it
-- by pasting a question. Fifty tutors using this writes the backlog by itself,
-- ordered by what students actually bring.
CREATE TABLE IF NOT EXISTS unplaced (
    id               INTEGER PRIMARY KEY,
    created_at       TEXT    NOT NULL,
    question         TEXT,
    from_image       INTEGER,
    role             TEXT,
    student_ref      TEXT,
    guessed          TEXT,
    confidence       TEXT,
    looks_incomplete INTEGER,
    recognised_as    TEXT,
    reason           TEXT
);

-- Questions we have already written, ready to ask again. A question belongs to
-- a skill, not to a student, so writing a fresh one for every teenager who
-- reaches index laws is paying repeatedly for the same thing. Generating is
-- most of the cost of a session; taking one off the shelf is free.
--
-- times_asked and times_correct are what let a bad question be found later: one
-- everybody gets right is not discriminating, and one nobody gets right is
-- probably broken rather than hard.
CREATE TABLE IF NOT EXISTS question_bank (
    id             INTEGER PRIMARY KEY,
    skill_id       TEXT    NOT NULL,
    question       TEXT    NOT NULL,
    correct_option TEXT    NOT NULL,
    distractors    TEXT    NOT NULL,
    model          TEXT,
    created_at     TEXT    NOT NULL,
    times_asked    INTEGER NOT NULL DEFAULT 0,
    times_correct  INTEGER NOT NULL DEFAULT 0,
    -- Kept rather than deleted, so the answers already given still make sense.
    retired        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS answers_by_skill ON answers(skill_id);
CREATE INDEX IF NOT EXISTS bank_by_skill ON question_bank(skill_id, retired);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the store, creating the file and tables the first time.

    The default is read now rather than bound when this file was imported, so
    the store can be pointed somewhere else - a test's temporary file, or a
    different database later - without threading a path through every caller.
    """
    path = Path(path or DEFAULT_PATH)
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
    # is no answer to record for it. Nor is anything carried over from an
    # earlier part of the same question: it was answered once, so it is stored
    # once.
    asked = [r for r in diagnosis.results if r.asked and not r.reused]
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


def record_unplaced(
    connection: sqlite3.Connection,
    question: str,
    match,
    *,
    from_image: bool = False,
    role: str | None = None,
    student_ref: str | None = None,
) -> int:
    """Note a question we could not diagnose, and why.

    This is the cheapest contribution anyone can make, because it is not a
    contribution - it is a side effect of trying to use the tool. A question
    nobody could place is a doorway that does not exist yet.
    """
    cursor = connection.execute(
        """INSERT INTO unplaced (created_at, question, from_image, role,
               student_ref, guessed, confidence, looks_incomplete,
               recognised_as, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            _now(),
            question,
            int(from_image),
            role,
            student_ref,
            getattr(match, "skill_id", None),
            getattr(match, "confidence", None),
            int(bool(getattr(match, "looks_incomplete", False))),
            getattr(match, "recognised_as", None),
            getattr(match, "reason", None),
        ),
    )
    connection.commit()
    return cursor.lastrowid


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


def unplaced_questions(
    connection: sqlite3.Connection, limit: int = 50
) -> list[sqlite3.Row]:
    """Questions nobody could diagnose, newest first.

    Read this as the coverage backlog. A question here is one a real person
    brought and the graph had no doorway for.
    """
    return connection.execute(
        """SELECT created_at, question, from_image, looks_incomplete,
                  recognised_as, reason
           FROM unplaced
           ORDER BY id DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()


def gaps_by_topic(connection: sqlite3.Connection) -> list[tuple[str, int]]:
    """Which topics people keep asking for that we cannot answer, commonest first.

    This is the coverage backlog in priority order. Nobody filled in a form to
    produce it - they pasted questions and we could not help.
    """
    rows = connection.execute(
        """SELECT recognised_as AS topic, COUNT(*) AS times
           FROM unplaced
           WHERE recognised_as IS NOT NULL AND recognised_as != ''
           GROUP BY recognised_as
           ORDER BY times DESC, topic"""
    ).fetchall()
    return [(row["topic"], row["times"]) for row in rows]


# ---- The question bank ----------------------------------------------------


def bank_question(
    connection: sqlite3.Connection,
    skill_id: str,
    question,
    *,
    model: str | None = None,
) -> int:
    """Put a written question on the shelf for the next student."""
    connection.execute(
        """INSERT INTO question_bank (skill_id, question, correct_option,
               distractors, model, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            skill_id,
            question.question,
            question.correct_option,
            json.dumps([d.model_dump() for d in question.distractors]),
            model,
            _now(),
        ),
    )
    connection.commit()
    return connection.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def take_question(connection: sqlite3.Connection, skill_id: str):
    """A banked question for this skill, or None if the shelf is empty.

    Least-used first, ties broken at random. Least-used spreads students across
    the variants instead of hammering one, which both slows memorisation and
    gathers evidence about every question rather than one.
    """
    row = connection.execute(
        """SELECT * FROM question_bank
           WHERE skill_id = ? AND retired = 0
           ORDER BY times_asked ASC, RANDOM()
           LIMIT 1""",
        (skill_id,),
    ).fetchone()

    if row is None:
        return None

    from questions import Distractor, MultipleChoiceQuestion

    question = MultipleChoiceQuestion(
        question=row["question"],
        correct_option=row["correct_option"],
        distractors=[Distractor(**d) for d in json.loads(row["distractors"])],
    )
    return row["id"], question


def mark_asked(connection: sqlite3.Connection, question_id: int, correct: bool) -> None:
    """Record that a question was put to someone, and how it went."""
    connection.execute(
        """UPDATE question_bank
           SET times_asked = times_asked + 1,
               times_correct = times_correct + ?
           WHERE id = ?""",
        (1 if correct else 0, question_id),
    )
    connection.commit()


def retire_question(connection: sqlite3.Connection, question_id: int) -> None:
    """Stop asking a question without deleting the answers already given."""
    connection.execute(
        "UPDATE question_bank SET retired = 1 WHERE id = ?", (question_id,)
    )
    connection.commit()


def bank_counts(connection: sqlite3.Connection) -> dict[str, int]:
    """How many live questions each skill has on the shelf."""
    rows = connection.execute(
        """SELECT skill_id, COUNT(*) AS held
           FROM question_bank WHERE retired = 0
           GROUP BY skill_id"""
    ).fetchall()
    return {row["skill_id"]: row["held"] for row in rows}


def weak_questions(
    connection: sqlite3.Connection, min_asked: int = 10
) -> list[sqlite3.Row]:
    """Questions that are not telling us anything, once enough students have
    seen them.

    One everybody gets right does not discriminate; one nobody gets right is
    more likely broken than hard. Both are worth a human eye.
    """
    return connection.execute(
        """SELECT id, skill_id, question, times_asked, times_correct,
                  CAST(times_correct AS REAL) / times_asked AS pass_rate
           FROM question_bank
           WHERE retired = 0 AND times_asked >= ?
             AND (pass_rate > 0.95 OR pass_rate < 0.05)
           ORDER BY times_asked DESC""",
        (min_asked,),
    ).fetchall()
