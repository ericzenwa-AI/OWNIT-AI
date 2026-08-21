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
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Where the answers live. Overridable because a hosting platform wipes its
# filesystem on every deploy - there the path has to point at a mounted disk
# that survives, or every session ever recorded disappears with the next push.
DEFAULT_PATH = Path(
    os.environ.get(
        "OWNIT_DB", Path(__file__).resolve().parent.parent / "data" / "ownit.db"
    )
)

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
    stopped_early     TEXT,
    -- What reading the attempt established. Decided once, then carried, so a
    -- walk resumed tomorrow narrows the same way it did today.
    reading           TEXT,
    finished          INTEGER NOT NULL DEFAULT 0
);

-- The question currently on a student's screen. One per session, so answering
-- can be scored against exactly what they were shown rather than against a
-- fresh shuffle. Cleared as soon as it is answered.
CREATE TABLE IF NOT EXISTS pending (
    session_id INTEGER PRIMARY KEY REFERENCES sessions(id),
    skill_id   TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    options    TEXT    NOT NULL,
    banked_id  INTEGER
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

-- Anything a tutor wanted to say that was not a verdict on one diagnosis.
--
-- Kept apart from `feedback` rather than folded into it. That table requires a
-- session and a verdict, both of which are exactly right for judging a
-- diagnosis and neither of which a general comment has. Relaxing them in place
-- would mean a create-copy-drop-rename on a live database, and since the
-- schema runs as CREATE TABLE IF NOT EXISTS, a deployed file would quietly
-- keep the old shape while a fresh one got the new. The admin page reads both.
CREATE TABLE IF NOT EXISTS comments (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT    NOT NULL,
    -- Set when the comment came from someone who had just run a diagnosis.
    -- Absent otherwise, which is the whole point of this table existing.
    session_id  INTEGER REFERENCES sessions(id),
    comment     TEXT    NOT NULL,
    -- What they named as missing, put through the same matching the verdict
    -- form uses: a skill id when we recognise one, their own words when not.
    about_skill TEXT,
    matched     INTEGER NOT NULL DEFAULT 0,
    -- Optional, and only so a question can be answered. Never used for
    -- anything else.
    contact     TEXT
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

-- People who asked to be told when it opens.
--
-- These are children's email addresses. They stay in this file, which never
-- goes to GitHub, they are never handed to anyone else, and the only thing
-- they are for is one message saying it is ready. UNIQUE on email so someone
-- signing up twice is one person, not two.
CREATE TABLE IF NOT EXISTS waitlist (
    id         INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    email      TEXT NOT NULL UNIQUE,
    -- Free text: which exam board, which year, what they are stuck on. Optional,
    -- and the most useful thing on the form for deciding what to build next.
    studying   TEXT
);

-- Every time a question had to be written live because nothing was banked for
-- that skill. Each row is a call to the model that the shelf was supposed to
-- have made unnecessary, so a skill appearing here repeatedly is either new or
-- has had everything under it retired.
CREATE TABLE IF NOT EXISTS live_generation (
    id         INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    skill_id   TEXT NOT NULL
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


# ---- A walk in progress ---------------------------------------------------
#
# The terminal keeps a walk alive inside one running function. A server cannot,
# so these let a walk live entirely in the database: opened, answered a question
# at a time, and closed. Nothing is held between requests, which is what lets a
# student shut the tab and come back.


def open_walk(
    connection: sqlite3.Connection,
    *,
    entry_skill_id: str,
    reading,
    question: str | None = None,
    attempt: str | None = None,
    student_ref: str | None = None,
    role: str | None = None,
    match=None,
) -> int:
    """Start a walk and return the id everything else refers to."""
    from dataclasses import asdict

    cursor = connection.execute(
        """INSERT INTO sessions (created_at, student_ref, role, question, attempt,
               entry_skill_id, entry_confidence, entry_confirmed, reading)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            _now(),
            student_ref,
            role,
            question,
            attempt,
            entry_skill_id,
            getattr(match, "confidence", None),
            1 if match is not None else None,
            json.dumps(asdict(reading)),
        ),
    )
    connection.commit()
    return cursor.lastrowid


def leave_comment(
    connection: sqlite3.Connection,
    comment: str,
    *,
    session_id: int | None = None,
    about_skill: str | None = None,
    matched: bool = False,
    contact: str | None = None,
) -> int:
    """Keep something a tutor wanted to say, with or without a diagnosis."""
    cursor = connection.execute(
        """INSERT INTO comments
               (created_at, session_id, comment, about_skill, matched, contact)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            _now(),
            session_id,
            comment.strip(),
            (about_skill or "").strip() or None,
            1 if matched else 0,
            (contact or "").strip() or None,
        ),
    )
    connection.commit()
    return cursor.lastrowid


def everything_said(connection: sqlite3.Connection, limit: int = 300) -> list[sqlite3.Row]:
    """Both kinds of feedback in one list, newest first.

    Verdicts and comments are stored apart because they are shaped differently,
    but there is no reason to read them apart - what matters is what people
    said, in the order they said it.
    """
    return connection.execute(
        """SELECT created_at, 'verdict' AS kind, session_id,
                  verdict AS headline, actual_gap AS about, note AS body,
                  NULL AS contact, NULL AS matched
             FROM feedback
           UNION ALL
           SELECT created_at, 'comment' AS kind, session_id,
                  NULL AS headline, about_skill AS about, comment AS body,
                  contact, matched
             FROM comments
           ORDER BY created_at DESC, kind
           LIMIT ?""",
        (limit,),
    ).fetchall()


def record_live_generation(connection: sqlite3.Connection, skill_id: str) -> None:
    """Note that a question had to be written rather than taken off the shelf."""
    connection.execute(
        "INSERT INTO live_generation (created_at, skill_id) VALUES (?, ?)",
        (_now(), skill_id),
    )
    connection.commit()


def ran_dry(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Which skills have needed a live question written, and how often."""
    return connection.execute(
        """SELECT skill_id, COUNT(*) AS times, MAX(created_at) AS last_time
           FROM live_generation
           GROUP BY skill_id
           ORDER BY times DESC, skill_id"""
    ).fetchall()


def join_waitlist(
    connection: sqlite3.Connection, email: str, studying: str | None = None
) -> bool:
    """Add someone to the waitlist. True if they were not already on it.

    Signing up twice is not an error - people forget - so the second time
    quietly updates what they told us and reports that they were already here.
    """
    email = email.strip().lower()
    studying = (studying or "").strip() or None

    already = connection.execute(
        "SELECT id FROM waitlist WHERE email = ?", (email,)
    ).fetchone()

    if already:
        if studying:
            connection.execute(
                "UPDATE waitlist SET studying = ? WHERE id = ?",
                (studying, already["id"]),
            )
            connection.commit()
        return False

    connection.execute(
        "INSERT INTO waitlist (created_at, email, studying) VALUES (?, ?, ?)",
        (_now(), email, studying),
    )
    connection.commit()
    return True


def waitlist_size(connection: sqlite3.Connection) -> int:
    return connection.execute("SELECT COUNT(*) AS n FROM waitlist").fetchone()["n"]


def starts_today(connection: sqlite3.Connection) -> int:
    """How many questions have been read since midnight.

    Counts the ones we could not place as well as the ones we could. Both spend
    a call on the best model, and a run of unplaceable questions is exactly the
    shape an abusive one takes - so counting only successful walks would leave
    the hole open.
    """
    since = _now()[:10]
    walks = connection.execute(
        "SELECT COUNT(*) AS n FROM sessions WHERE created_at >= ?", (since,)
    ).fetchone()["n"]
    refused = connection.execute(
        "SELECT COUNT(*) AS n FROM unplaced WHERE created_at >= ?", (since,)
    ).fetchone()["n"]
    return walks + refused


def walk_state(connection: sqlite3.Connection, session_id: int):
    return connection.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()


def answers_so_far(connection: sqlite3.Connection, session_id: int) -> list:
    """Rebuild what the student has established, in the order they said it."""
    from walk import SkillResult

    rows = connection.execute(
        """SELECT * FROM answers WHERE session_id = ? ORDER BY position""",
        (session_id,),
    ).fetchall()

    return [
        SkillResult(
            row["skill_id"],
            held=row["outcome"] == "correct",
            mistake=row["misconception"],
            dont_know=row["outcome"] == "dont_know",
            question=row["question"],
            chosen=row["chosen"],
            seconds=row["seconds"],
        )
        for row in rows
    ]


def set_pending(
    connection: sqlite3.Connection,
    session_id: int,
    *,
    skill_id: str,
    question: str,
    options: list[dict],
    banked_id: int | None = None,
) -> None:
    """Remember exactly what we put on screen, so the answer can be scored."""
    connection.execute(
        """INSERT INTO pending (session_id, skill_id, question, options, banked_id)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               skill_id = excluded.skill_id,
               question = excluded.question,
               options  = excluded.options,
               banked_id = excluded.banked_id""",
        (session_id, skill_id, question, json.dumps(options), banked_id),
    )
    connection.commit()


def pending_question(connection: sqlite3.Connection, session_id: int):
    return connection.execute(
        "SELECT * FROM pending WHERE session_id = ?", (session_id,)
    ).fetchone()


def record_answer(
    connection: sqlite3.Connection,
    session_id: int,
    *,
    skill_id: str,
    question: str,
    chosen: str,
    held: bool,
    mistake: str | None,
    dont_know: bool,
    banked_id: int | None = None,
) -> None:
    """File one answer and take the question off the screen."""
    position = connection.execute(
        "SELECT COUNT(*) AS n FROM answers WHERE session_id = ?", (session_id,)
    ).fetchone()["n"] + 1

    outcome = "correct" if held else ("dont_know" if dont_know else "wrong")
    connection.execute(
        """INSERT INTO answers (session_id, position, skill_id, question,
               chosen, outcome, misconception)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session_id, position, skill_id, question, chosen, outcome, mistake),
    )
    connection.execute("DELETE FROM pending WHERE session_id = ?", (session_id,))
    connection.commit()

    if banked_id is not None:
        mark_asked(connection, banked_id, held)


def close_walk(connection: sqlite3.Connection, session_id: int, diagnosis) -> None:
    """Write the conclusion onto the session."""
    connection.execute(
        """UPDATE sessions
           SET root_gaps = ?, chain = ?, unchecked = ?, stopped_early = ?,
               finished = 1
           WHERE id = ?""",
        (
            ",".join(diagnosis.root_gaps),
            " -> ".join(diagnosis.chains[0]) if diagnosis.chains else "",
            ",".join(diagnosis.unchecked),
            diagnosis.stopped_early,
            session_id,
        ),
    )
    connection.execute("DELETE FROM pending WHERE session_id = ?", (session_id,))
    connection.commit()


# ---- Carrying the shelf between machines -----------------------------------
#
# The database holds two kinds of thing that deserve opposite treatment.
#
# Student answers are theirs. They stay on the machine that collected them and
# never go near GitHub, which is why *.db is ignored.
#
# Banked questions are not theirs. Nobody's work is in them - we paid a model
# to write them, and that money buys the same questions every time. Keeping
# them only inside an ignored file means one deletion costs it again, and a
# deployed copy starts with an empty shelf.
#
# So the shelf travels as a text file that can be committed, and the answers
# never move. What travels is the question itself; how often it has been asked
# and how often it was right stay behind, because those are facts about the
# students who saw it, not about the question.


def export_bank(connection: sqlite3.Connection) -> list[dict]:
    """Every live question on the shelf, as plain records.

    Retired questions come too. Retiring is a judgement about the question -
    it was ambiguous, or wrong - and that judgement is worth more than the
    question, because losing it means someone re-reads the same bad question.
    """
    rows = connection.execute(
        """SELECT skill_id, question, correct_option, distractors, model, retired
           FROM question_bank
           ORDER BY skill_id, id"""
    ).fetchall()
    return [
        {
            "skill_id": row["skill_id"],
            "question": row["question"],
            "correct_option": row["correct_option"],
            "distractors": json.loads(row["distractors"]),
            "model": row["model"],
            "retired": row["retired"],
        }
        for row in rows
    ]


def restore_bank(connection: sqlite3.Connection, records: list[dict]) -> int:
    """Put exported questions back, skipping any already here.

    Safe to run twice, and safe to run against a shelf that already has things
    on it - a question is the same question if it asks the same thing about the
    same skill. Returns how many were actually new.
    """
    added = 0
    for record in records:
        # What makes a question the same question: the skill, the wording, the
        # answer, and whether it is retired. All four are needed.
        #
        # Without retired, a retired question and a live replacement that read
        # the same collapse into one, and since the retired one comes first in
        # the file the live one is dropped - a skill silently losing a question.
        #
        # Without the answer, two attempts at the same wording that disagree
        # about what is correct collapse too, which is exactly what a question
        # and its rewrite look like.
        already = connection.execute(
            """SELECT 1 FROM question_bank
               WHERE skill_id = ? AND question = ? AND correct_option = ?
                 AND retired = ? LIMIT 1""",
            (
                record["skill_id"],
                record["question"],
                record["correct_option"],
                record.get("retired", 0),
            ),
        ).fetchone()
        if already:
            continue

        connection.execute(
            """INSERT INTO question_bank (skill_id, question, correct_option,
                   distractors, model, created_at, retired)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                record["skill_id"],
                record["question"],
                record["correct_option"],
                json.dumps(record["distractors"]),
                record.get("model"),
                _now(),
                record.get("retired", 0),
            ),
        )
        added += 1

    connection.commit()
    return added
