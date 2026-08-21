"""The diagnostic over HTTP, one question per request.

A terminal program can sit in a loop waiting for someone to type. A server
cannot: between showing a question and getting an answer, a student reads,
thinks, gets interrupted, and sometimes closes the tab and comes back tomorrow.
So nothing about a walk is held in memory here.

Every request rebuilds where a student had got to from the answers already
stored, asks walk.step() what comes next, and lets go. That means a restarted
server loses nothing, two servers can serve the same student, and an abandoned
walk costs nothing to leave lying around.

    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import json
import secrets
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile

from anthropic import APIError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from pydantic import BaseModel, Field

import bank
import store
import walk
from entry import EntryMatch, MEDIA_TYPES, identify_entry, is_usable, out_of_scope
from graph import SKILLS
from questions import DONT_KNOW_LABEL, DONT_KNOW_OPTION, shuffled_options

log = logging.getLogger("ownit.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fill the shelf before the first student rather than during them.

    Restocking already happens on the first question that finds nothing, so
    this is not what makes the app work - it is what makes it honest. A freshly
    deployed machine otherwise reports an empty bank until somebody uses it,
    which reads identically to the restock having failed, and the first student
    through wears the restore inside their own request.
    """
    try:
        added = bank.restock()
        if added:
            log.info("shelf restocked: %s questions", added)
    except Exception as error:  # noqa: BLE001 - never keep the app from starting
        log.warning("could not restock the shelf at startup: %s", error)
    yield


app = FastAPI(title="OwnIt diagnostic", lifespan=lifespan)

# The page is served from the same place, so this is only for running the
# frontend separately while developing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB = Path(__file__).resolve().parent.parent / "web"
PAGE = WEB / "index.html"
LANDING = WEB / "landing.html"

# Every start is a call to the best model, so every start costs money, and the
# page is public with no sign-in. Without a ceiling, one script - or one class
# sharing a link - empties the account overnight, and the first anyone knows is
# that the app has stopped working for everyone.
#
# A whole day is refused rather than degraded, because a diagnosis built on a
# cheaper model would be a worse diagnosis and nobody would be told.
DAILY_STARTS = int(os.environ.get("OWNIT_DAILY_STARTS", "200"))

# Base64 of a photo. A phone picture is comfortably under this; anything much
# larger is not a question, and it is read into memory before anything checks
# what it is.
MAX_ATTACHMENT_CHARS = 8_000_000

# Long enough for any exam question, short enough that nobody is paying to have
# a novel read to them.
MAX_QUESTION_CHARS = 4_000


@app.exception_handler(APIError)
def model_unavailable(request: Request, error: APIError) -> JSONResponse:
    """The model did not answer, so say so in words a student can act on.

    Seen for real on a dropped connection: without this it is a stack trace and
    a 500. Nothing is lost when it happens - the answers already given are in
    the database, so reloading picks the walk up where it stopped.
    """
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "I could not reach the marking service just then. Nothing you "
                "have answered is lost - try again in a moment."
            )
        },
    )


# ---- What goes over the wire ----------------------------------------------


class StartRequest(BaseModel):
    question: str = Field("", max_length=MAX_QUESTION_CHARS)
    # A photo or PDF of the question, base64 encoded. Text loses powers and
    # fractions on the way out of a PDF, so this is the better route.
    attachment: str | None = Field(None, max_length=MAX_ATTACHMENT_CHARS)
    attachment_type: str | None = Field(None, max_length=100)
    attempt: str | None = Field(None, max_length=MAX_QUESTION_CHARS)
    student_ref: str | None = Field(None, max_length=200)
    role: str = "student"
    # Switching to another part of a question already read. The skill is one we
    # handed over ourselves a moment ago, so reading it is free: identifying
    # the question again would be a second call to the best model for an answer
    # we already have. The page keeps the list of parts itself rather than
    # sending it back, so nothing here has to be trusted beyond a skill id that
    # is checked against the graph anyway.
    start_at: str | None = Field(None, max_length=100)
    start_summary: str | None = Field(None, max_length=500)


class AnswerRequest(BaseModel):
    session_id: int
    # Which option they picked, as shown to them.
    label: str
    # How many questions had been answered when this one was put on screen. A
    # double-tapped button would otherwise answer the *next* question with a
    # letter the student never read.
    answered_before: int


class WaitlistRequest(BaseModel):
    email: str = Field(max_length=254)
    # What they are studying, in their words. Optional, and the most useful
    # thing on the form for deciding which topic to build next.
    studying: str | None = Field(None, max_length=300)


class CommentRequest(BaseModel):
    comment: str = Field(max_length=4000)
    # Set when it came from someone who had just run a diagnosis, absent when
    # it did not. Optional is the point of this endpoint.
    session_id: int | None = None
    # A skill or topic they think is missing or wrong, put through the same
    # matching the verdict form uses.
    about: str | None = Field(None, max_length=300)
    contact: str | None = Field(None, max_length=254)


class FeedbackRequest(BaseModel):
    session_id: int
    # Whether the diagnosis was right. Nothing else is worth recording if this
    # is not answered, so it has no default.
    verdict: str
    # What the gap really was, when we got it wrong. A skill id or name is
    # matched to the graph; anything else is kept as written, because "it was
    # something you do not have a skill for" is the most useful answer there is.
    actual_gap: str | None = None
    note: str | None = None


class PartOut(BaseModel):
    """One lettered part of a question, as offered to the student."""

    label: str
    summary: str
    # The doorway it would start at. None when the part is about something the
    # map does not cover, and then it cannot be chosen.
    skill_id: str | None = None
    covered: bool = True
    # Which part the walk on screen is actually about.
    current: bool = False


class Option(BaseModel):
    label: str
    text: str


class QuestionOut(BaseModel):
    skill_name: str
    question: str
    options: list[Option]


class StateOut(BaseModel):
    """Everything the page needs to draw itself."""

    session_id: int | None = None
    # Waiting for the student to confirm what the question is asking.
    matched: str | None = None
    asking: QuestionOut | None = None
    finished: bool = False
    report: dict | None = None
    message: str | None = None
    asked_so_far: int = 0
    # Set when the question has lettered parts. The walk is on one of them and
    # the others are offered, because the part someone is stuck on is very
    # often not the one the question opens with.
    parts: list[PartOut] | None = None


# ---- Starting a walk ------------------------------------------------------


def _save_attachment(encoded: str, media_type: str | None) -> Path:
    """Write an uploaded photo somewhere the matcher can read it."""
    suffix = next(
        (ext for ext, mime in MEDIA_TYPES.items() if mime == media_type), ".png"
    )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(400, f"That attachment could not be read: {error}")

    handle = NamedTemporaryFile(suffix=suffix, delete=False)
    handle.write(raw)
    handle.close()
    return Path(handle.name)


@app.post("/api/start", response_model=StateOut)
def start(request: StartRequest) -> StateOut:
    """Work out what the question is asking, and ask the first thing."""
    if not (request.question or "").strip() and not request.attachment:
        raise HTTPException(400, "Send the question, or a photo of it.")

    connection = store.connect()
    try:
        started_today = store.starts_today(connection)
    finally:
        connection.close()

    if started_today >= DAILY_STARTS:
        raise HTTPException(
            429,
            "OwnIt has taken as many questions as it can today. It will start "
            "again tomorrow - and if you were part way through, your answers "
            "are saved.",
        )

    # Moving to another part of a question already read. We handed this skill
    # over a moment ago, so identifying the whole thing again would be paying
    # the best model a second time for an answer we already have.
    if request.start_at:
        if request.start_at not in SKILLS or not SKILLS[request.start_at].topic:
            raise HTTPException(400, "That is not somewhere a question can start.")
        match = EntryMatch(
            skill_id=request.start_at,
            confidence="high",
            plain_summary=(request.start_summary or "").strip(),
            reason="",
            recognised_as=SKILLS[request.start_at].topic or "",
        )
        return _begin(request, match)

    attachment = (
        _save_attachment(request.attachment, request.attachment_type)
        if request.attachment
        else None
    )

    try:
        match = identify_entry(request.question, attachment)
    except ValueError as error:
        raise HTTPException(400, str(error))
    finally:
        if attachment:
            attachment.unlink(missing_ok=True)

    _file_uncovered_parts(request, match)
    return _begin(request, match)


class _MissingPart:
    """A single part we cannot place, shaped the way record_unplaced reads."""

    def __init__(self, part):
        self.skill_id = None
        self.confidence = "high"
        self.looks_incomplete = False
        self.reason = part.plain_summary


def _file_uncovered_parts(request: StartRequest, match: EntryMatch) -> None:
    """File the parts of a question the map cannot reach.

    A question where every part is off the map already gets filed. One where
    only part (d) is - graph transformations, say - was being placed happily on
    part (a) and the gap went nowhere. That is a coverage gap somebody actually
    hit, which is the only kind worth ranking a backlog by.
    """
    missing = [part for part in match.other_parts if not part.skill_id]
    if not missing:
        return

    connection = store.connect()
    try:
        for part in missing:
            store.record_unplaced(
                connection,
                f"({part.label}) {part.plain_summary}",
                _MissingPart(part),
                from_image=bool(request.attachment),
                role=request.role,
                student_ref=request.student_ref,
            )
    except Exception as error:  # noqa: BLE001 - bookkeeping must not stop a walk
        log.warning("could not file uncovered parts: %s", error)
    finally:
        connection.close()


def _parts_of(match: EntryMatch, starting_at: str | None) -> list[PartOut] | None:
    """Every lettered part, with the one being walked marked.

    Someone who sends a whole question is rarely stuck on part (a) - it is
    usually the one they could do. Showing the others is most of the value of
    having read them.
    """
    if not match.other_parts:
        return None

    first = PartOut(
        label="a",
        summary=match.plain_summary,
        skill_id=match.skill_id,
        covered=bool(match.skill_id),
        current=match.skill_id == starting_at,
    )
    rest = [
        PartOut(
            label=part.label,
            summary=part.plain_summary,
            skill_id=part.skill_id,
            covered=bool(part.skill_id),
            current=bool(part.skill_id) and part.skill_id == starting_at,
        )
        for part in match.other_parts
    ]
    return [first] + rest


def _begin(request: StartRequest, match: EntryMatch) -> StateOut:
    """Open a walk on a matched question and ask the first thing."""
    connection = store.connect()
    try:
        if not is_usable(match):
            # A question we cannot place is a missing doorway, and filing it is
            # the whole coverage backlog.
            store.record_unplaced(
                connection,
                request.question,
                match,
                from_image=bool(request.attachment),
                role=request.role,
                student_ref=request.student_ref,
            )
            return StateOut(message=out_of_scope(match), finished=True)

        # Reading the attempt costs two model calls and cannot change part
        # way through, so it is done once here and carried in the session.
        reading = (
            walk.read_attempt(SKILLS[match.skill_id], request.attempt)
            if request.attempt
            else walk.Reading()
        )
        session_id = store.open_walk(
            connection,
            entry_skill_id=match.skill_id,
            reading=reading,
            question=request.question,
            attempt=request.attempt,
            student_ref=request.student_ref,
            role=request.role,
            match=match,
        )
    finally:
        connection.close()

    state = _advance(session_id)
    state.matched = match.plain_summary
    state.parts = _parts_of(match, match.skill_id)
    return state


@app.post("/api/answer", response_model=StateOut)
def answer(request: AnswerRequest) -> StateOut:
    """Take one answer and hand back whatever comes next."""
    connection = store.connect()
    try:
        pending = store.pending_question(connection, request.session_id)
        if pending is None:
            raise HTTPException(400, "That session is not waiting for an answer.")

        already = len(store.answers_so_far(connection, request.session_id))
        if request.answered_before != already:
            raise HTTPException(
                409, "That question has already been answered - reload to catch up."
            )

        options = json.loads(pending["options"])
        chosen = next((o for o in options if o["label"] == request.label), None)
        if chosen is None:
            raise HTTPException(400, f"'{request.label}' was not one of the options.")

        store.record_answer(
            connection,
            request.session_id,
            skill_id=pending["skill_id"],
            question=pending["question"],
            chosen=chosen["text"],
            # The correct option carries no mistake - that is what scores it.
            held=chosen["mistake"] is None and not chosen["dont_know"],
            mistake=chosen["mistake"],
            dont_know=chosen["dont_know"],
            banked_id=pending["banked_id"],
        )
    finally:
        connection.close()

    return _advance(request.session_id)


def _advance(session_id: int) -> StateOut:
    """Work out where this walk is up to, and either ask or finish.

    Nothing is remembered between requests - the answers already stored are
    replayed to find the next question. A restarted server loses nothing.
    """
    connection = store.connect()
    try:
        walk_row = store.walk_state(connection, session_id)
        if walk_row is None:
            raise HTTPException(404, "No such session.")

        answers = store.answers_so_far(connection, session_id)
        reading = walk.Reading(**json.loads(walk_row["reading"]))

        current = walk.step(walk_row["entry_skill_id"], answers, reading=reading)

        if current.finished:
            store.close_walk(connection, session_id, current.diagnosis)
            return StateOut(
                session_id=session_id,
                finished=True,
                report=_report(current.diagnosis),
                asked_so_far=len(answers),
            )

        skill = SKILLS[current.ask]
        banked_id, question = bank.question_for(skill.id, connection=connection)

        # Shuffle once, here, and keep it - the page has to show the same order
        # the answer is scored against.
        options = [
            {"label": label, "text": text, "mistake": mistake, "dont_know": False}
            for label, (text, mistake) in zip("ABCD", shuffled_options(question))
        ]
        options.append(
            {
                "label": DONT_KNOW_LABEL,
                "text": DONT_KNOW_OPTION,
                "mistake": None,
                "dont_know": True,
            }
        )

        store.set_pending(
            connection,
            session_id,
            skill_id=skill.id,
            question=question.question,
            options=options,
            banked_id=banked_id,
        )

        return StateOut(
            session_id=session_id,
            asking=QuestionOut(
                skill_name=skill.name,
                question=question.question,
                options=[Option(label=o["label"], text=o["text"]) for o in options],
            ),
            asked_so_far=len(answers),
        )
    finally:
        connection.close()


@app.get("/api/session/{session_id}", response_model=StateOut)
def resume(session_id: int) -> StateOut:
    """Pick a walk back up from its id.

    What makes a link worth sending: the question already on screen is handed
    back exactly as it was, in the order it was shown, rather than a fresh one.
    A student who closes the tab and follows the link later sees the question
    they left, not a different one.
    """
    connection = store.connect()
    try:
        walk_row = store.walk_state(connection, session_id)
        if walk_row is None:
            raise HTTPException(404, "No such session.")

        pending = store.pending_question(connection, session_id)
        answered = len(store.answers_so_far(connection, session_id))

        if pending is not None:
            options = json.loads(pending["options"])
            return StateOut(
                session_id=session_id,
                asking=QuestionOut(
                    skill_name=SKILLS[pending["skill_id"]].name,
                    question=pending["question"],
                    options=[
                        Option(label=o["label"], text=o["text"]) for o in options
                    ],
                ),
                asked_so_far=answered,
            )
    finally:
        connection.close()

    # Nothing waiting: either finished, or interrupted between two questions.
    # Replaying the answers settles which, and writes the next question.
    return _advance(session_id)


# ---- Saying whether it was right ------------------------------------------


def _as_known_skill(text: str) -> str | None:
    """Match what a tutor typed to a skill in the graph, if it is one."""
    wanted = text.strip().casefold()
    for skill in SKILLS.values():
        if wanted in (skill.id.casefold(), skill.name.casefold()):
            return skill.id
    return None


@app.post("/api/feedback")
def feedback(request: FeedbackRequest) -> dict:
    """Record whether a diagnosis was right, and what the real gap was.

    This is the only signal that can correct the graph. Everything else we
    measure says how the walk behaved, not whether it was correct.
    """
    if request.verdict not in ("right", "wrong"):
        raise HTTPException(400, "Verdict must be 'right' or 'wrong'.")

    connection = store.connect()
    try:
        walk_row = store.walk_state(connection, session_id=request.session_id)
        if walk_row is None:
            raise HTTPException(404, "No such session.")
        if not walk_row["finished"]:
            raise HTTPException(
                400, "That walk has not finished, so there is no diagnosis to judge."
            )

        recognised = None
        gap = (request.actual_gap or "").strip()
        if gap:
            recognised = _as_known_skill(gap)

        store.record_feedback(
            connection,
            request.session_id,
            request.verdict,
            actual_gap=recognised or (gap or None),
            note=(request.note or "").strip() or None,
        )
    finally:
        connection.close()

    return {
        "recorded": True,
        # Said plainly, so a typo is visible rather than silently filed as a
        # skill nobody has.
        "actual_gap": recognised or (gap or None),
        "matched_a_known_skill": bool(recognised),
    }


def _report(diagnosis) -> dict:
    """The diagnosis, in the words a person reads rather than skill ids."""
    gaps = []
    for gap, chain in zip(diagnosis.root_gaps, diagnosis.chains):
        result = diagnosis.result_for(gap)

        # Why this skill and not something lower: its own prerequisites were
        # asked about and held. Anything else that held is a sibling somewhere
        # else in the tree and says nothing about this gap - claiming otherwise
        # would be a false reason on the one screen that has to be trusted.
        beneath = [
            SKILLS[need].name
            for need in SKILLS[gap].needs
            if (below := diagnosis.result_for(need)) is not None and below.held
        ]

        gaps.append(
            {
                "skill": SKILLS[gap].name,
                "chain": [SKILLS[step].name for step in chain],
                "nothing_there": bool(result and result.dont_know),
                "mistake": result.mistake if result else None,
                "held_beneath": beneath,
                # Nothing sits below it in the subject at all, so there is
                # nowhere further to look.
                "is_bedrock": not SKILLS[gap].needs,
            }
        )

    return {
        "stuck_on": SKILLS[diagnosis.entry_skill_id].name,
        "gaps": gaps,
        "unchecked": [SKILLS[s].name for s in diagnosis.unchecked],
        "stopped_early": diagnosis.stopped_early,
        "presentation_note": diagnosis.presentation_note,
        "narrowed_because": diagnosis.narrowed_because,
        "asked": [
            {
                "skill": SKILLS[r.skill_id].name,
                "held": r.held,
                "dont_know": r.dont_know,
                "mistake": r.mistake,
            }
            for r in diagnosis.results
            if r.asked
        ],
    }


# ---- Odds and ends --------------------------------------------------------


# Deliberately loose. The only thing that matters is that a typo is caught
# before someone waits months for an email that was never going to arrive;
# anything stricter starts rejecting addresses that are perfectly real.
EMAIL = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")


@app.post("/api/waitlist")
def waitlist(request: WaitlistRequest) -> dict:
    """Take an email address and say we will be in touch."""
    email = request.email.strip()
    if not EMAIL.match(email):
        raise HTTPException(400, "That does not look like an email address.")

    connection = store.connect()
    try:
        new = store.join_waitlist(connection, email, request.studying)
    finally:
        connection.close()

    return {
        "joined": True,
        "already_here": not new,
        "message": (
            "You are on the list. I will email you when it opens."
            if new
            else "You are already on the list - nothing more to do."
        ),
    }


@app.post("/api/comment")
def comment(request: CommentRequest) -> dict:
    """Take a comment from someone who has not just run a diagnosis.

    The verdict endpoint needs a finished walk to judge. Most of what a tutor
    wants to say is not that: a topic that is missing, a question that felt
    wrong for the level, something that simply reads badly. None of it had
    anywhere to go.
    """
    text = (request.comment or "").strip()
    if not text:
        raise HTTPException(400, "There is nothing in the comment.")

    if request.session_id is not None:
        connection = store.connect()
        try:
            if store.walk_state(connection, session_id=request.session_id) is None:
                raise HTTPException(404, "No such session.")
        finally:
            connection.close()

    about = (request.about or "").strip()
    recognised = _as_known_skill(about) if about else None

    connection = store.connect()
    try:
        store.leave_comment(
            connection,
            text,
            session_id=request.session_id,
            about_skill=recognised or (about or None),
            matched=bool(recognised),
            contact=request.contact,
        )
    finally:
        connection.close()

    return {
        "saved": True,
        "about": recognised or (about or None),
        "matched_a_known_skill": bool(recognised),
        "message": (
            "Thank you - that is noted against "
            f"{SKILLS[recognised].name}."
            if recognised
            else "Thank you. That is worth knowing."
        ),
    }


@app.get("/api/health")
def health() -> dict:
    """Enough to tell, from outside, whether a deploy went as intended.

    The two that matter after a deploy are `storage` and `questions`. A disk
    that failed to mount looks exactly like a working app until the day
    somebody notices the waitlist is empty, so it is worth being able to ask
    rather than find out. Counts only - nothing here says who anyone is.
    """
    from graph import topics

    on_a_disk = "OWNIT_DB" in os.environ

    try:
        connection = store.connect()
        try:
            banked = connection.execute(
                "SELECT COUNT(*) AS n FROM question_bank WHERE retired = 0"
            ).fetchone()["n"]
        finally:
            connection.close()
    except Exception:  # noqa: BLE001 - health must answer even when the store cannot
        banked = None

    return {
        "status": "ok",
        "skills": len(SKILLS),
        "topics": topics(),
        # "kept" means the database was put somewhere chosen. "default" means
        # it is sitting next to the code, which on most hosts is wiped.
        "storage": "kept" if on_a_disk else "default",
        "questions": banked,
    }


# ---- Reading what came in --------------------------------------------------


def _admin_ok(request: Request) -> None:
    """Basic auth against one password held in the environment.

    There is no default and no fallback: with nothing set the page refuses to
    open at all, because the failure mode of a default is a page listing what
    tutors said sitting open on the internet.
    """
    expected = os.environ.get("OWNIT_ADMIN_PASSWORD", "")
    if not expected:
        raise HTTPException(503, "No admin password is set, so this is closed.")

    header = request.headers.get("authorization", "")
    given = ""
    if header.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8", "replace")
            given = decoded.partition(":")[2]
        except (binascii.Error, ValueError):
            given = ""

    # Constant time, so the response cannot be timed to guess the password.
    if not secrets.compare_digest(given, expected):
        raise HTTPException(
            401,
            "Not authorised.",
            headers={"WWW-Authenticate": 'Basic realm="OwnIt"'},
        )


@app.get("/admin/feedback", response_class=HTMLResponse)
def admin_feedback(request: Request) -> str:
    """Everything anyone has said, newest first.

    Deliberately plain. This exists so the feedback can be read without opening
    the database by hand, not to be a dashboard.
    """
    _admin_ok(request)

    connection = store.connect()
    try:
        rows = store.everything_said(connection)
        waiting = store.waitlist_size(connection)
    finally:
        connection.close()

    def out(text) -> str:
        """Everything here was typed by someone else, including into a form
        that anyone on the internet can reach."""
        return escape("" if text is None else str(text))

    items = []
    for row in rows:
        when = out(row["created_at"][:16].replace("T", " "))
        bits = [f'<div class="when">{when}</div>']

        if row["kind"] == "verdict":
            right = row["headline"] == "right"
            bits.append(
                f'<div class="head {"right" if right else "wrong"}">'
                f'Diagnosis marked {out(row["headline"])}'
                f' &middot; session {out(row["session_id"])}</div>'
            )
            if row["about"]:
                bits.append(f'<div class="about">Real gap: {out(row["about"])}</div>')
        else:
            where = (
                f' &middot; session {out(row["session_id"])}'
                if row["session_id"] is not None
                else " &middot; not from a diagnosis"
            )
            bits.append(f'<div class="head comment">Comment{where}</div>')
            if row["about"]:
                mark = "" if row["matched"] else " (not a skill on the map)"
                bits.append(
                    f'<div class="about">About: {out(row["about"])}{mark}</div>'
                )
            if row["contact"]:
                bits.append(f'<div class="about">Reply to: {out(row["contact"])}</div>')

        if row["body"]:
            bits.append(f'<div class="body">{out(row["body"])}</div>')

        items.append('<li>' + "".join(bits) + '</li>')

    listing = "".join(items) or "<li><em>Nothing yet.</em></li>"

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Feedback</title>
<style>
  body {{ font: 15px/1.6 ui-monospace, Consolas, monospace; max-width: 44rem;
         margin: 2rem auto; padding: 0 1rem; background: #fff; color: #111; }}
  h1 {{ font-size: 1.1rem; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ border-top: 1px solid #ddd; padding: 1rem 0; }}
  .when {{ color: #777; font-size: 0.8rem; }}
  .head {{ font-weight: 700; margin: 0.2rem 0; }}
  .right {{ color: #2F5D50; }}
  .wrong {{ color: #8C3B2E; }}
  .comment {{ color: #333; }}
  .about {{ color: #555; font-size: 0.9rem; }}
  .body {{ white-space: pre-wrap; margin-top: 0.4rem; }}
  .count {{ color: #777; font-size: 0.85rem; }}
</style>
<h1>Feedback</h1>
<p class="count">{len(rows)} shown &middot; {waiting} on the waitlist</p>
<ul>{listing}</ul>
"""


@app.get("/start")
def diagnostic():
    """The diagnostic itself."""
    if not PAGE.exists():
        raise HTTPException(404, "The page has not been built yet.")
    return FileResponse(PAGE)


@app.get("/")
def front(s: str | None = None):
    """The front door.

    A session id here is from a link made before the diagnostic moved to
    /start. Sending it on rather than showing the landing page means an old
    bookmark still lands on the question it was saved at.
    """
    if s:
        return RedirectResponse(f"/start?s={s}")
    if not LANDING.exists():
        return FileResponse(PAGE) if PAGE.exists() else JSONResponse(
            status_code=404, content={"detail": "The page has not been built yet."}
        )
    return FileResponse(LANDING)
