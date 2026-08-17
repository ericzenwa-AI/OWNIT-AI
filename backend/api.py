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
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from anthropic import APIError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

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
    question: str = ""
    # A photo or PDF of the question, base64 encoded. Text loses powers and
    # fractions on the way out of a PDF, so this is the better route.
    attachment: str | None = None
    attachment_type: str | None = None
    attempt: str | None = None
    student_ref: str | None = None
    role: str = "student"


class AnswerRequest(BaseModel):
    session_id: int
    # Which option they picked, as shown to them.
    label: str
    # How many questions had been answered when this one was put on screen. A
    # double-tapped button would otherwise answer the *next* question with a
    # letter the student never read.
    answered_before: int


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
