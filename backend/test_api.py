"""Tests for the diagnostic over HTTP.

No API calls: the matcher and the question bank are replaced with canned
answers, so these test the request handling and, mostly, that nothing about a
walk is held between requests.

Run from the repo root with:  pytest backend/
"""

import pytest
from fastapi.testclient import TestClient

import api
import bank
import store
from entry import EntryMatch
from questions import Distractor, MultipleChoiceQuestion


@pytest.fixture(autouse=True)
def never_the_committed_shelf(tmp_path, monkeypatch):
    """No test may read data/question_bank.jsonl.

    Startup restocks the shelf, so without this every test database would be
    filled with the 548 real questions and the canned ones below would never
    be reached - the tests would still pass, while testing something else.
    """
    monkeypatch.setattr(bank, "SHELF_FILE", tmp_path / "question_bank.jsonl")
    monkeypatch.setattr(bank, "_looked_in_the_file", False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "api.db")

    monkeypatch.setattr(
        api,
        "identify_entry",
        lambda question, attachment=None, **kw: EntryMatch(
            skill_id="simplify_index_expression",
            confidence="high",
            plain_summary="Leave the answer as a surd rather than a decimal.",
            reason="",
            recognised_as="surds",
        ),
    )
    monkeypatch.setattr(bank, "question_for", lambda skill_id, connection=None: (None, _question(skill_id)))

    return TestClient(api.app)


def _question(skill_id):
    return MultipleChoiceQuestion(
        question=f"A question about {skill_id}",
        correct_option="right",
        distractors=[
            Distractor(option="wrong one", mistake="first mistake"),
            Distractor(option="wrong two", mistake="second mistake"),
            Distractor(option="wrong three", mistake="third mistake"),
        ],
    )


def label_of(state, text):
    """Which letter is sitting against a given option this time round."""
    return next(o["label"] for o in state["asking"]["options"] if o["text"] == text)


# ---- Starting --------------------------------------------------------------


def test_starting_asks_the_first_question(client):
    state = client.post("/api/start", json={"question": "leave it exact"}).json()

    assert state["session_id"] is not None
    assert state["asking"]["skill_name"]
    assert state["finished"] is False


def test_the_student_is_told_what_we_think_they_asked(client):
    state = client.post("/api/start", json={"question": "leave it exact"}).json()
    assert "surd" in state["matched"]


def test_five_options_are_offered(client):
    """Four real ones and the way out."""
    state = client.post("/api/start", json={"question": "q"}).json()
    labels = [o["label"] for o in state["asking"]["options"]]

    assert labels == ["A", "B", "C", "D", "E"]
    assert state["asking"]["options"][-1]["text"] == "I don't know"


def test_the_answer_is_never_sent_to_the_page(client):
    """Which option is right must not travel to the browser."""
    state = client.post("/api/start", json={"question": "q"}).json()
    for option in state["asking"]["options"]:
        assert set(option) == {"label", "text"}


def test_a_question_we_cannot_place_is_refused(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "identify_entry",
        lambda question, attachment=None, **kw: EntryMatch(
            skill_id=None,
            confidence="high",
            plain_summary="",
            reason="binomial expansion",
            recognised_as="binomial expansion",
        ),
    )
    state = client.post("/api/start", json={"question": "expand it"}).json()

    assert state["session_id"] is None
    assert state["finished"] is True
    assert "binomial expansion" in state["message"]


def test_an_unplaceable_question_is_filed_as_a_gap(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        api,
        "identify_entry",
        lambda question, attachment=None, **kw: EntryMatch(
            skill_id=None,
            confidence="high",
            plain_summary="",
            reason="vectors",
            recognised_as="vectors",
        ),
    )
    client.post("/api/start", json={"question": "find the magnitude"})

    connection = store.connect()
    assert store.gaps_by_topic(connection) == [("vectors", 1)]
    connection.close()


# ---- Answering -------------------------------------------------------------


def test_answering_moves_on(client):
    started = client.post("/api/start", json={"question": "q"}).json()
    first_skill = started["asking"]["skill_name"]

    state = client.post(
        "/api/answer",
        json={"session_id": started["session_id"], "label": label_of(started, "right"), "answered_before": 0},
    ).json()

    assert state["asked_so_far"] == 1
    assert state["asking"] is None or state["asking"]["skill_name"] != first_skill


def test_a_wrong_answer_is_scored_as_wrong(client):
    started = client.post("/api/start", json={"question": "q"}).json()
    client.post(
        "/api/answer",
        json={
            "session_id": started["session_id"],
            "label": label_of(started, "wrong one"),
            "answered_before": 0,
        },
    )

    connection = store.connect()
    row = connection.execute("SELECT * FROM answers").fetchone()
    connection.close()

    assert row["outcome"] == "wrong"
    assert row["misconception"] == "first mistake"


def test_saying_you_do_not_know_is_kept_apart(client):
    started = client.post("/api/start", json={"question": "q"}).json()
    client.post(
        "/api/answer", json={"session_id": started["session_id"], "label": "E", "answered_before": 0}
    )

    connection = store.connect()
    row = connection.execute("SELECT * FROM answers").fetchone()
    connection.close()

    assert row["outcome"] == "dont_know"
    assert row["misconception"] is None


def test_an_option_that_was_not_offered_is_refused(client):
    started = client.post("/api/start", json={"question": "q"}).json()
    response = client.post(
        "/api/answer", json={"session_id": started["session_id"], "label": "Z", "answered_before": 0}
    )
    assert response.status_code == 400


def test_answering_twice_is_refused(client):
    """The question was taken off the screen, so there is nothing to answer."""
    started = client.post("/api/start", json={"question": "q"}).json()
    payload = {"session_id": started["session_id"], "label": "A", "answered_before": 0}

    assert client.post("/api/answer", json=payload).status_code == 200
    # The same submission again would otherwise answer the *next* question with
    # a letter the student never saw.
    assert client.post("/api/answer", json=payload).status_code == 409


def test_an_unknown_session_is_not_found(client):
    response = client.post("/api/answer", json={"session_id": 9999, "label": "A", "answered_before": 0})
    assert response.status_code in (400, 404)


# ---- Holding nothing between requests -------------------------------------


def test_the_server_keeps_nothing_in_memory(client, tmp_path, monkeypatch):
    """The whole point. Restart the app mid-walk and it carries on.

    A student reads, thinks, gets interrupted, closes the tab. Nothing about
    their walk can depend on a process staying alive.
    """
    started = client.post("/api/start", json={"question": "q"}).json()
    session_id = started["session_id"]
    client.post(
        "/api/answer",
        json={
            "session_id": session_id,
            "label": label_of(started, "wrong one"),
            "answered_before": 0,
        },
    )

    # A completely fresh application, as if the server had been restarted.
    fresh = TestClient(api.app)
    state = fresh.post(
        "/api/answer",
        json={"session_id": session_id, "label": "E", "answered_before": 1},
    ).json()

    assert state["asked_so_far"] == 2


def test_a_walk_runs_to_a_diagnosis(client):
    """Answer everything wrong and it should reach the floor and report."""
    state = client.post("/api/start", json={"question": "q"}).json()
    session_id = state["session_id"]

    for _ in range(20):
        if state.get("finished"):
            break
        state = client.post(
            "/api/answer",
            json={
                "session_id": session_id,
                "label": label_of(state, "wrong one"),
                "answered_before": state["asked_so_far"],
            },
        ).json()

    assert state["finished"] is True
    assert state["report"]["stuck_on"]
    assert state["report"]["gaps"] or state["report"]["stopped_early"]


def test_the_report_names_skills_not_ids(client):
    state = client.post("/api/start", json={"question": "q"}).json()
    session_id = state["session_id"]

    while not state.get("finished"):
        state = client.post(
            "/api/answer",
            json={
                "session_id": session_id,
                "label": label_of(state, "right"),
                "answered_before": state["asked_so_far"],
            },
        ).json()

    # "Exact form", not "exact_form" - this is read by a person.
    assert "_" not in state["report"]["stuck_on"]
    for asked in state["report"]["asked"]:
        assert "_" not in asked["skill"]


def test_a_finished_walk_is_marked_finished(client):
    state = client.post("/api/start", json={"question": "q"}).json()
    session_id = state["session_id"]

    while not state.get("finished"):
        state = client.post(
            "/api/answer",
            json={
                "session_id": session_id,
                "label": label_of(state, "right"),
                "answered_before": state["asked_so_far"],
            },
        ).json()

    connection = store.connect()
    row = connection.execute(
        "SELECT finished FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    pending = connection.execute("SELECT COUNT(*) c FROM pending").fetchone()["c"]
    connection.close()

    assert row["finished"] == 1
    assert pending == 0


# ---- Odds and ends ---------------------------------------------------------


def test_health_reports_what_is_covered(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["skills"] > 0
    assert "differentiation" in body["topics"]


def test_a_broken_attachment_is_refused(client):
    response = client.post(
        "/api/start",
        json={"question": "", "attachment": "not base64!!", "attachment_type": "image/png"},
    )
    assert response.status_code == 400


def test_a_model_failure_is_explained_not_dumped(client, monkeypatch):
    """Seen for real on a dropped connection. A student needs words, not a 500."""
    from anthropic import APITimeoutError

    def times_out(*args, **kwargs):
        raise APITimeoutError(request=None)

    monkeypatch.setattr(api, "identify_entry", times_out)
    response = client.post("/api/start", json={"question": "q"})

    assert response.status_code == 503
    assert "lost" in response.json()["detail"]


def test_nothing_is_lost_when_the_model_fails_mid_walk(client, monkeypatch):
    """The answers are already in the database, so the walk resumes."""
    from anthropic import APITimeoutError

    started = client.post("/api/start", json={"question": "q"}).json()
    client.post(
        "/api/answer",
        json={
            "session_id": started["session_id"],
            "label": label_of(started, "right"),
            "answered_before": 0,
        },
    )

    def times_out(*args, **kwargs):
        raise APITimeoutError(request=None)

    monkeypatch.setattr(bank, "question_for", times_out)
    failed = client.post(
        "/api/answer",
        json={"session_id": started["session_id"], "label": "A", "answered_before": 1},
    )
    assert failed.status_code == 503

    connection = store.connect()
    kept = len(store.answers_so_far(connection, started["session_id"]))
    connection.close()
    assert kept >= 1

# ---- Picking a walk back up ------------------------------------------------


def _finish(client, question="q"):
    """Answer everything wrong until a diagnosis comes out."""
    state = client.post("/api/start", json={"question": question}).json()
    session_id = state["session_id"]
    for _ in range(20):
        if state.get("finished"):
            break
        state = client.post(
            "/api/answer",
            json={
                "session_id": session_id,
                "label": label_of(state, "wrong one"),
                "answered_before": state["asked_so_far"],
            },
        ).json()
    return session_id, state


def test_a_session_can_be_picked_up_from_its_id(client):
    """What a bookmarked link has to do."""
    started = client.post("/api/start", json={"question": "q"}).json()
    session_id = started["session_id"]

    resumed = client.get(f"/api/session/{session_id}").json()

    assert resumed["session_id"] == session_id
    assert resumed["asking"]["question"] == started["asking"]["question"]


def test_resuming_hands_back_the_same_question_in_the_same_order(client):
    """The options are shuffled once and scored against that order. Handing
    back a fresh shuffle would mark the answer against letters the student
    never saw."""
    started = client.post("/api/start", json={"question": "q"}).json()
    session_id = started["session_id"]

    resumed = client.get(f"/api/session/{session_id}").json()

    assert resumed["asking"]["options"] == started["asking"]["options"]


def test_resuming_does_not_move_the_walk_on(client):
    """Following the link twice must not skip a question."""
    started = client.post("/api/start", json={"question": "q"}).json()
    session_id = started["session_id"]

    client.get(f"/api/session/{session_id}")
    again = client.get(f"/api/session/{session_id}").json()

    assert again["asked_so_far"] == started["asked_so_far"]
    assert again["asking"]["question"] == started["asking"]["question"]


def test_resuming_mid_walk_keeps_what_was_already_answered(client):
    started = client.post("/api/start", json={"question": "q"}).json()
    session_id = started["session_id"]
    client.post(
        "/api/answer",
        json={
            "session_id": session_id,
            "label": label_of(started, "wrong one"),
            "answered_before": 0,
        },
    )

    resumed = client.get(f"/api/session/{session_id}").json()

    assert resumed["asked_so_far"] == 1
    assert resumed["asking"] is not None


def test_resuming_a_finished_walk_gives_the_report_back(client):
    session_id, finished = _finish(client)

    resumed = client.get(f"/api/session/{session_id}").json()

    assert resumed["finished"] is True
    assert resumed["report"]["stuck_on"] == finished["report"]["stuck_on"]


def test_resuming_a_session_that_does_not_exist(client):
    assert client.get("/api/session/9999").status_code == 404


# ---- Saying whether the diagnosis was right --------------------------------


def test_a_diagnosis_can_be_marked_right(client):
    session_id, _ = _finish(client)

    response = client.post(
        "/api/feedback", json={"session_id": session_id, "verdict": "right"}
    )

    assert response.status_code == 200
    assert response.json()["recorded"] is True


def test_a_wrong_diagnosis_records_the_real_gap(client):
    session_id, _ = _finish(client)

    response = client.post(
        "/api/feedback",
        json={
            "session_id": session_id,
            "verdict": "wrong",
            "actual_gap": "Expanding brackets",
            "note": "They could not multiply out the second bracket.",
        },
    )

    body = response.json()
    assert body["matched_a_known_skill"] is True
    # Written as a name, stored as the id, so it can be counted later.
    assert body["actual_gap"] == "expand_brackets"


def test_a_gap_we_have_no_skill_for_is_kept_as_written(client):
    """The most useful answer a tutor can give is that the real gap is not in
    the graph at all. Insisting on a known skill would throw that away."""
    session_id, _ = _finish(client)

    body = client.post(
        "/api/feedback",
        json={
            "session_id": session_id,
            "verdict": "wrong",
            "actual_gap": "reading the question properly",
        },
    ).json()

    assert body["matched_a_known_skill"] is False
    assert body["actual_gap"] == "reading the question properly"


def test_feedback_is_stored_against_the_session(client):
    session_id, _ = _finish(client)
    client.post(
        "/api/feedback",
        json={"session_id": session_id, "verdict": "wrong", "actual_gap": "surds"},
    )

    connection = store.connect()
    try:
        rows = connection.execute(
            "SELECT verdict, actual_gap FROM feedback WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    finally:
        connection.close()

    assert [(r["verdict"], r["actual_gap"]) for r in rows] == [("wrong", "surds")]


def test_a_verdict_has_to_be_right_or_wrong(client):
    session_id, _ = _finish(client)

    response = client.post(
        "/api/feedback", json={"session_id": session_id, "verdict": "maybe"}
    )

    assert response.status_code == 400


def test_an_unfinished_walk_has_no_diagnosis_to_judge(client):
    started = client.post("/api/start", json={"question": "q"}).json()

    response = client.post(
        "/api/feedback",
        json={"session_id": started["session_id"], "verdict": "right"},
    )

    assert response.status_code == 400


def test_feedback_on_a_session_that_does_not_exist(client):
    response = client.post(
        "/api/feedback", json={"session_id": 9999, "verdict": "right"}
    )

    assert response.status_code == 404


# ---- Not being emptied overnight -------------------------------------------


def test_a_question_with_nothing_in_it_costs_nothing(client):
    """Refused before any model is called, not after."""
    response = client.post("/api/start", json={"question": "   "})

    assert response.status_code == 400


def test_an_attachment_too_large_to_be_a_question_is_refused(client):
    """Read into memory before anything checks what it is, so the check has to
    happen at the edge."""
    too_big = "A" * (api.MAX_ATTACHMENT_CHARS + 1)

    response = client.post(
        "/api/start",
        json={"question": "q", "attachment": too_big, "attachment_type": "image/png"},
    )

    assert response.status_code == 422


def test_a_question_longer_than_any_exam_question_is_refused(client):
    response = client.post(
        "/api/start", json={"question": "x" * (api.MAX_QUESTION_CHARS + 1)}
    )

    assert response.status_code == 422


def test_the_day_stops_when_the_ceiling_is_reached(client, monkeypatch):
    """The page is public and every start costs a call on the best model."""
    monkeypatch.setattr(api, "DAILY_STARTS", 2)

    assert client.post("/api/start", json={"question": "q"}).status_code == 200
    assert client.post("/api/start", json={"question": "q"}).status_code == 200

    refused = client.post("/api/start", json={"question": "q"})
    assert refused.status_code == 429
    assert "tomorrow" in refused.json()["detail"]


def test_a_walk_already_going_is_not_stopped_by_the_ceiling(client, monkeypatch):
    """Turning someone away part way through would lose them the session they
    already answered, which is worse than the spend it saves."""
    started = client.post("/api/start", json={"question": "q"}).json()
    monkeypatch.setattr(api, "DAILY_STARTS", 0)

    answered = client.post(
        "/api/answer",
        json={
            "session_id": started["session_id"],
            "label": label_of(started, "wrong one"),
            "answered_before": 0,
        },
    )

    assert answered.status_code == 200


# ---- The waitlist ----------------------------------------------------------


def test_joining_the_waitlist(client):
    response = client.post("/api/waitlist", json={"email": "ada@example.com"})

    assert response.status_code == 200
    body = response.json()
    assert body["joined"] is True
    assert body["already_here"] is False


def test_what_they_are_stuck_on_is_kept(client):
    """The most useful thing on the form for deciding what to build next."""
    client.post(
        "/api/waitlist",
        json={"email": "ada@example.com", "studying": "Edexcel Y13, integration"},
    )

    connection = store.connect()
    try:
        row = connection.execute(
            "SELECT email, studying FROM waitlist"
        ).fetchone()
    finally:
        connection.close()

    assert row["email"] == "ada@example.com"
    assert row["studying"] == "Edexcel Y13, integration"


def test_signing_up_twice_is_not_an_error(client):
    """People forget. Two sign-ups is one person, not two, and telling them so
    is friendlier than an error."""
    client.post("/api/waitlist", json={"email": "ada@example.com"})
    second = client.post("/api/waitlist", json={"email": "ada@example.com"})

    assert second.status_code == 200
    assert second.json()["already_here"] is True

    connection = store.connect()
    try:
        assert store.waitlist_size(connection) == 1
    finally:
        connection.close()


def test_the_same_address_in_different_case_is_the_same_person(client):
    client.post("/api/waitlist", json={"email": "Ada@Example.com"})
    client.post("/api/waitlist", json={"email": "ada@example.com "})

    connection = store.connect()
    try:
        assert store.waitlist_size(connection) == 1
    finally:
        connection.close()


def test_a_typo_is_caught_before_they_wait_months_for_nothing(client):
    response = client.post("/api/waitlist", json={"email": "ada.example.com"})

    assert response.status_code == 400


def test_an_absurd_email_is_refused(client):
    response = client.post("/api/waitlist", json={"email": "x" * 300})

    assert response.status_code == 422


# ---- The front door --------------------------------------------------------


def test_the_landing_page_is_the_front_door(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "waitlist" in response.text.lower()


def test_the_diagnostic_has_its_own_address(client):
    response = client.get("/start")

    assert response.status_code == 200
    assert "id=\"screen\"" in response.text


def test_an_old_session_link_still_finds_its_question(client):
    """Links made before the diagnostic moved to /start point at /?s=42."""
    response = client.get("/?s=42", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/start?s=42"


def test_health_says_whether_the_data_is_being_kept(client, monkeypatch):
    """A disk that failed to mount looks exactly like a working app until the
    day somebody notices the waitlist is empty."""
    monkeypatch.delenv("OWNIT_DB", raising=False)
    assert client.get("/api/health").json()["storage"] == "default"

    monkeypatch.setenv("OWNIT_DB", "/var/data/ownit.db")
    assert client.get("/api/health").json()["storage"] == "kept"


def test_health_reports_how_many_questions_are_banked(client):
    """Zero after a deploy means the shelf did not restock, which is the
    difference between a free session and a bill for every question."""
    body = client.get("/api/health").json()

    assert "questions" in body
    assert isinstance(body["questions"], int)


def test_a_note_can_be_left_when_the_diagnosis_was_right(client):
    """"Right skill, but they guessed it" is worth as much as a correction, and
    there was nowhere to put it while the box only opened for disagreement."""
    session_id, _ = _finish(client)

    response = client.post(
        "/api/feedback",
        json={
            "session_id": session_id,
            "verdict": "right",
            "note": "Right skill, but they got there by guessing.",
        },
    )

    assert response.status_code == 200

    connection = store.connect()
    try:
        row = connection.execute(
            "SELECT verdict, note FROM feedback WHERE session_id = ?", (session_id,)
        ).fetchone()
    finally:
        connection.close()

    assert row["verdict"] == "right"
    assert row["note"] == "Right skill, but they got there by guessing."


def test_the_same_session_can_be_commented_on_more_than_once(client):
    """A tutor who says more later should not overwrite what they said first."""
    session_id, _ = _finish(client)
    client.post("/api/feedback", json={"session_id": session_id, "verdict": "right"})
    client.post(
        "/api/feedback",
        json={"session_id": session_id, "verdict": "wrong", "actual_gap": "surds"},
    )

    connection = store.connect()
    try:
        rows = connection.execute(
            "SELECT verdict FROM feedback WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    finally:
        connection.close()

    assert [r["verdict"] for r in rows] == ["right", "wrong"]


# ---- Saying something without having run a diagnosis -----------------------


def test_a_comment_needs_no_session(client):
    """The whole reason this exists. Most of what a tutor wants to say is not a
    verdict on one walk."""
    response = client.post(
        "/api/comment",
        json={"comment": "Nothing here covers trigonometry, which is half my week."},
    )

    assert response.status_code == 200
    assert response.json()["saved"] is True


def test_a_comment_can_name_a_skill_we_know(client):
    body = client.post(
        "/api/comment",
        json={"comment": "The questions here are too easy", "about": "Expanding brackets"},
    ).json()

    assert body["matched_a_known_skill"] is True
    assert body["about"] == "expand_brackets"


def test_a_comment_about_something_not_on_the_map_is_kept_as_written(client):
    body = client.post(
        "/api/comment",
        json={"comment": "no trig at all", "about": "trigonometry"},
    ).json()

    assert body["matched_a_known_skill"] is False
    assert body["about"] == "trigonometry"


def test_an_empty_comment_is_refused(client):
    assert client.post("/api/comment", json={"comment": "   "}).status_code == 400


def test_a_comment_can_be_tied_to_a_session_when_there_is_one(client):
    session_id, _ = _finish(client)

    body = client.post(
        "/api/comment",
        json={"comment": "took too many questions", "session_id": session_id},
    ).json()

    assert body["saved"] is True

    connection = store.connect()
    try:
        row = connection.execute("SELECT session_id FROM comments").fetchone()
    finally:
        connection.close()
    assert row["session_id"] == session_id


def test_a_comment_against_a_session_that_does_not_exist(client):
    response = client.post(
        "/api/comment", json={"comment": "hello", "session_id": 9999}
    )
    assert response.status_code == 404


# ---- Reading it back -------------------------------------------------------


def test_the_admin_page_will_not_open_without_a_password_set(client, monkeypatch):
    """No default and no fallback. The failure mode of a default is a page of
    what tutors said, open on the internet."""
    monkeypatch.delenv("OWNIT_ADMIN_PASSWORD", raising=False)

    assert client.get("/admin/feedback").status_code == 503


def test_the_admin_page_refuses_a_wrong_password(client, monkeypatch):
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "the-real-one")

    assert client.get("/admin/feedback").status_code == 401
    assert client.get(
        "/admin/feedback", auth=("me", "guessing")
    ).status_code == 401


def test_the_admin_page_shows_both_kinds(client, monkeypatch):
    session_id, _ = _finish(client)
    client.post(
        "/api/feedback",
        json={"session_id": session_id, "verdict": "wrong", "actual_gap": "surds"},
    )
    client.post("/api/comment", json={"comment": "trigonometry is missing"})

    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")
    page = client.get("/admin/feedback", auth=("me", "letmein"))

    assert page.status_code == 200
    assert "Diagnosis marked wrong" in page.text
    assert "surds" in page.text
    assert "trigonometry is missing" in page.text


def test_a_comment_cannot_put_a_tag_in_the_admin_page(client, monkeypatch):
    """Anyone on the internet can reach the comment form, and this page is
    rendered for one person who would have no reason to suspect it."""
    client.post(
        "/api/comment",
        json={"comment": "<script>alert(1)</script>", "about": "<b>x</b>"},
    )

    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")
    page = client.get("/admin/feedback", auth=("me", "letmein")).text

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_the_admin_page_is_not_indexed(client, monkeypatch):
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")
    page = client.get("/admin/feedback", auth=("me", "letmein")).text

    assert "noindex" in page
