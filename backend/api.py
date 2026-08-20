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
import os
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from anthropic import APIError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import bank
import store
import walk
from entry import EntryMatch, MEDIA_TYPES, identify_entry, is_usable, out_of_scope
from graph import SKILLS
from questions import DONT_KNOW_LABEL, DONT_KNOW_OPTION, shuffled_options

app = FastAPI(title="OwnIt diagnostic")

# The page is served from the same place, so this is only for running the
# frontend separately while developing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"

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


class AnswerRequest(BaseModel):
    session_id: int
    # Which option they picked, as shown to them.
    label: str
    # How many questions had been answered when this one was put on screen. A
    # double-tapped button would otherwise answer the *next* question with a
    # letter the student never read.
    answered_before: int


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


@app.get("/api/health")
def health() -> dict:
    from graph import topics

    return {"status": "ok", "skills": len(SKILLS), "topics": topics()}


@app.get("/")
def page():
    if not PAGE.exists():
        raise HTTPException(404, "The page has not been built yet.")
    return FileResponse(PAGE)
