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


# ---- Questions with lettered parts -----------------------------------------


def _multipart(monkeypatch, covered_last=False):
    """A question read as four parts, the last one off the map by default."""
    from entry import QuestionPart

    monkeypatch.setattr(
        api,
        "identify_entry",
        lambda question, attachment=None, **kw: EntryMatch(
            skill_id="simplify_index_expression",
            confidence="high",
            plain_summary="Part a, the easy one.",
            reason="",
            recognised_as="indices",
            other_parts=[
                QuestionPart(label="b", skill_id="use_discriminant",
                             plain_summary="Show it has no stationary points."),
                QuestionPart(label="c", skill_id="polynomial_division",
                             plain_summary="Divide by (x - 4)."),
                QuestionPart(label="d",
                             skill_id="solve_quadratic" if covered_last else None,
                             plain_summary="Graph transformations."),
            ],
        ),
    )


def test_all_the_parts_come_back(client, monkeypatch):
    """Someone who sends a whole question is rarely stuck on part (a)."""
    _multipart(monkeypatch)

    state = client.post("/api/start", json={"question": "q"}).json()

    assert [p["label"] for p in state["parts"]] == ["a", "b", "c", "d"]


def test_a_question_with_parts_asks_which_one_first(client, monkeypatch):
    """Part (a) is usually the one they could do, so starting there is
    starting on the part they are least likely to be stuck on."""
    _multipart(monkeypatch)

    state = client.post("/api/start", json={"question": "q"}).json()

    assert state["session_id"] is None
    assert state["asking"] is None
    assert not any(p["current"] for p in state["parts"])


def test_nothing_is_walked_until_a_part_is_chosen(client, monkeypatch):
    _multipart(monkeypatch)
    client.post("/api/start", json={"question": "q"})

    connection = store.connect()
    try:
        assert connection.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"] == 0
    finally:
        connection.close()


def test_choosing_a_part_is_what_starts_the_walk(client, monkeypatch):
    _multipart(monkeypatch)
    client.post("/api/start", json={"question": "q"})

    state = client.post(
        "/api/start", json={"question": "q", "start_at": "polynomial_division"}
    ).json()

    assert state["session_id"] is not None
    assert state["asking"] is not None


def test_a_part_we_cannot_reach_is_marked_as_such(client, monkeypatch):
    _multipart(monkeypatch)

    parts = client.post("/api/start", json={"question": "q"}).json()["parts"]

    assert [p["covered"] for p in parts] == [True, True, True, False]


def test_a_question_with_one_part_offers_nothing(client):
    """The bar would be noise on a question that has no other parts."""
    assert client.post("/api/start", json={"question": "q"}).json()["parts"] is None


def test_switching_part_does_not_read_the_question_again(client, monkeypatch):
    """The skill was worked out when the question was first read. Identifying
    it again would be paying the best model twice for the same answer."""
    _multipart(monkeypatch)
    state = client.post("/api/start", json={"question": "q"}).json()

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("read the question again just to change part")

    monkeypatch.setattr(api, "identify_entry", must_not_be_called)

    switched = client.post(
        "/api/start",
        json={"question": "q", "start_at": "polynomial_division",
              "start_summary": "Divide by (x - 4)."},
    ).json()

    assert switched["session_id"] != state["session_id"]
    assert switched["matched"] == "Divide by (x - 4)."


def test_switching_starts_a_walk_on_that_part(client, monkeypatch):
    _multipart(monkeypatch)
    client.post("/api/start", json={"question": "q"})

    switched = client.post(
        "/api/start", json={"question": "q", "start_at": "polynomial_division"}
    ).json()

    connection = store.connect()
    try:
        row = store.walk_state(connection, session_id=switched["session_id"])
    finally:
        connection.close()
    assert row["entry_skill_id"] == "polynomial_division"


def test_a_part_that_is_not_a_doorway_is_refused(client):
    """A skill can be real and still not be somewhere a question starts."""
    response = client.post(
        "/api/start", json={"question": "q", "start_at": "negatives"}
    )
    assert response.status_code == 400


def test_a_skill_that_does_not_exist_is_refused(client):
    response = client.post(
        "/api/start", json={"question": "q", "start_at": "not_a_skill_at_all"}
    )
    assert response.status_code == 400


def test_an_uncovered_part_is_filed_as_a_coverage_gap(client, monkeypatch):
    """The whole question places fine on part (a), so nothing was being filed -
    and a gap somebody actually hit is the only kind worth ranking by."""
    _multipart(monkeypatch)

    client.post("/api/start", json={"question": "q"})

    connection = store.connect()
    try:
        rows = connection.execute("SELECT question FROM unplaced").fetchall()
    finally:
        connection.close()

    assert len(rows) == 1
    assert "(d)" in rows[0]["question"]
    assert "Graph transformations" in rows[0]["question"]


def test_nothing_is_filed_when_every_part_is_covered(client, monkeypatch):
    _multipart(monkeypatch, covered_last=True)

    client.post("/api/start", json={"question": "q"})

    connection = store.connect()
    try:
        assert connection.execute("SELECT COUNT(*) c FROM unplaced").fetchone()["c"] == 0
    finally:
        connection.close()


# ---- The descent as it happens ---------------------------------------------


def test_nothing_answered_yet_is_an_empty_descent(client):
    state = client.post("/api/start", json={"question": "q"}).json()

    assert state["so_far"] == []


def test_each_answer_joins_the_list(client):
    started = client.post("/api/start", json={"question": "q"}).json()
    first = started["asking"]["skill_name"]

    state = client.post(
        "/api/answer",
        json={
            "session_id": started["session_id"],
            "label": label_of(started, "right"),
            "answered_before": 0,
        },
    ).json()

    assert [a["skill_name"] for a in state["so_far"]] == [first]
    assert state["so_far"][0]["held"] is True


def test_the_list_says_what_happened_to_each_one(client):
    started = client.post("/api/start", json={"question": "q"}).json()
    state = client.post(
        "/api/answer",
        json={
            "session_id": started["session_id"],
            "label": label_of(started, "wrong one"),
            "answered_before": 0,
        },
    ).json()

    assert state["so_far"][0]["held"] is False
    assert state["so_far"][0]["dont_know"] is False


def test_not_knowing_is_kept_apart_in_the_list(client):
    """It is not a mistake and it is not a pass, and a tutor watching wants to
    see which of the two it was."""
    started = client.post("/api/start", json={"question": "q"}).json()
    state = client.post(
        "/api/answer",
        json={"session_id": started["session_id"], "label": "E", "answered_before": 0},
    ).json()

    assert state["so_far"][0]["held"] is False
    assert state["so_far"][0]["dont_know"] is True


def test_the_list_is_in_the_order_they_were_answered(client):
    state = client.post("/api/start", json={"question": "q"}).json()
    session_id = state["session_id"]
    asked = []

    for _ in range(3):
        if not state.get("asking"):
            break
        asked.append(state["asking"]["skill_name"])
        state = client.post(
            "/api/answer",
            json={
                "session_id": session_id,
                "label": label_of(state, "wrong one"),
                "answered_before": state["asked_so_far"],
            },
        ).json()

    assert [a["skill_name"] for a in state["so_far"]] == asked


def test_the_list_survives_being_picked_up_again(client):
    """A tutor who follows the link back should see the working, not a blank."""
    started = client.post("/api/start", json={"question": "q"}).json()
    client.post(
        "/api/answer",
        json={
            "session_id": started["session_id"],
            "label": label_of(started, "wrong one"),
            "answered_before": 0,
        },
    )

    resumed = client.get(f"/api/session/{started['session_id']}").json()

    assert len(resumed["so_far"]) == 1


def test_the_report_still_carries_the_list(client):
    session_id, finished = _finish(client)

    assert len(finished["so_far"]) == finished["asked_so_far"]


def test_switching_part_works_when_the_question_came_as_a_photo(client, monkeypatch):
    """A photo carries the maths, so the text is empty. Switching sends no
    photo back - there is nothing to read - and asking for one anyway told
    somebody who had just sent a photo to send a photo."""
    _multipart(monkeypatch)
    client.post(
        "/api/start",
        json={"question": "", "attachment": "aGVsbG8=", "attachment_type": "image/png"},
    )

    switched = client.post(
        "/api/start",
        json={"question": "", "start_at": "polynomial_division",
              "start_summary": "Divide by (x - 4)."},
    )

    assert switched.status_code == 200
    assert switched.json()["asking"] is not None


def test_switching_part_is_not_stopped_by_the_days_ceiling(client, monkeypatch):
    """Nothing is being read, so nothing is being paid for - and someone part
    way through a question should not be stranded on one part of it."""
    _multipart(monkeypatch)
    client.post("/api/start", json={"question": "q"})
    monkeypatch.setattr(api, "DAILY_STARTS", 0)

    switched = client.post(
        "/api/start", json={"question": "q", "start_at": "polynomial_division"}
    )

    assert switched.status_code == 200


# ---- Working sent as a photo -----------------------------------------------


def test_a_photo_of_the_working_is_read(client, monkeypatch):
    """Working is done on paper. Typing it out is where the notation goes."""
    seen = {}

    def fake_read(entry, attempt, attachment=None, client=None):
        seen["attempt"] = attempt
        seen["had_photo"] = attachment is not None
        return walk_module.Reading()

    import walk as walk_module
    monkeypatch.setattr(api.walk, "read_attempt", fake_read)

    client.post(
        "/api/start",
        json={
            "question": "q",
            "attempt_attachment": "aGVsbG8=",
            "attempt_attachment_type": "image/png",
        },
    )

    assert seen["had_photo"] is True
    assert seen["attempt"] == ""


def test_typing_and_a_photo_both_arrive(client, monkeypatch):
    seen = {}

    def fake_read(entry, attempt, attachment=None, client=None):
        seen["attempt"] = attempt
        seen["had_photo"] = attachment is not None
        import walk as w
        return w.Reading()

    monkeypatch.setattr(api.walk, "read_attempt", fake_read)

    client.post(
        "/api/start",
        json={
            "question": "q",
            "attempt": "I got 15x^4",
            "attempt_attachment": "aGVsbG8=",
            "attempt_attachment_type": "image/png",
        },
    )

    assert seen["attempt"] == "I got 15x^4"
    assert seen["had_photo"] is True


def test_no_attempt_at_all_reads_nothing(client, monkeypatch):
    """Two model calls that must not happen when there is nothing to read."""
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("read an attempt that was never sent")

    monkeypatch.setattr(api.walk, "read_attempt", must_not_be_called)

    assert client.post("/api/start", json={"question": "q"}).status_code == 200


def test_the_working_photo_is_deleted_afterwards(client, monkeypatch):
    """It is a child's handwriting on a server disk. It goes as soon as it has
    been read."""
    kept = {}

    def fake_read(entry, attempt, attachment=None, client=None):
        kept["path"] = attachment
        import walk as w
        return w.Reading()

    monkeypatch.setattr(api.walk, "read_attempt", fake_read)
    client.post(
        "/api/start",
        json={"question": "q", "attempt_attachment": "aGVsbG8=",
              "attempt_attachment_type": "image/png"},
    )

    assert kept["path"] is not None
    assert not kept["path"].exists()


def test_the_pages_are_never_served_from_cache_without_asking(client):
    """The whole app is one file with the JavaScript inside it, so a cached
    copy is a cached version of the product. A browser left to guess how long
    to keep it will happily run last week's build."""
    for path in ("/", "/start"):
        headers = client.get(path).headers
        assert "no-cache" in headers.get("cache-control", ""), path
        assert headers.get("etag"), path


def test_asking_costs_a_round_trip_and_not_the_whole_file(client):
    """Always revalidating is only affordable if revalidating is cheap.
    FileResponse sets an ETag and ignores the one that comes back, so every
    visit was re-sending the whole page to a browser that already had it."""
    for path in ("/", "/start"):
        etag = client.get(path).headers["etag"]

        again = client.get(path, headers={"If-None-Match": etag})

        assert again.status_code == 304, path
        assert again.content == b"", path


def test_a_changed_page_is_sent_again(client):
    """The other half: a browser holding an old copy must not be told to keep
    it, or this becomes the bug it was meant to fix."""
    response = client.get("/start", headers={"If-None-Match": 'W/"something-else"'})

    assert response.status_code == 200
    assert b"askScreen" in response.content


# ---- Knowing whether anyone is using it ------------------------------------


def test_opening_a_page_is_counted(client):
    """The top of the funnel, and the only part of it that was missing. A
    hundred people opening and three starting is a different problem from a
    hundred opening and ninety starting, and both look the same without this."""
    client.get("/")
    client.get("/start")
    client.get("/start")

    connection = store.connect()
    try:
        rows = connection.execute(
            "SELECT page, COUNT(*) c FROM page_view GROUP BY page ORDER BY page"
        ).fetchall()
    finally:
        connection.close()

    assert [(r["page"], r["c"]) for r in rows] == [("/", 1), ("/start", 2)]


def test_a_visit_is_counted_even_when_the_page_is_not_re_sent(client):
    """A 304 still means somebody is looking at it."""
    etag = client.get("/start").headers["etag"]
    client.get("/start", headers={"If-None-Match": etag})

    connection = store.connect()
    try:
        n = connection.execute(
            "SELECT COUNT(*) c FROM page_view WHERE page = '/start'"
        ).fetchone()["c"]
    finally:
        connection.close()

    assert n == 2


def test_nothing_about_who_is_kept(client):
    """Counts, not people. No address, no cookie, no identifier."""
    client.get("/")

    connection = store.connect()
    try:
        columns = [r[1] for r in connection.execute("PRAGMA table_info(page_view)")]
    finally:
        connection.close()

    assert columns == ["id", "created_at", "page"]


def test_anyone_can_say_whether_it_helped(client):
    """Not gated on being a tutor - it is the only question everybody can
    answer, and the one that says whether this is worth anything."""
    session_id, _ = _finish(client)

    response = client.post(
        "/api/rating", json={"session_id": session_id, "useful": True}
    )

    assert response.status_code == 200
    connection = store.connect()
    try:
        row = connection.execute("SELECT useful FROM rating").fetchone()
    finally:
        connection.close()
    assert row["useful"] == 1


def test_saying_it_did_not_help_is_kept_too(client):
    session_id, _ = _finish(client)
    client.post("/api/rating", json={"session_id": session_id, "useful": False})

    connection = store.connect()
    try:
        assert connection.execute("SELECT useful FROM rating").fetchone()["useful"] == 0
    finally:
        connection.close()


def test_rating_a_session_that_does_not_exist(client):
    response = client.post("/api/rating", json={"session_id": 9999, "useful": True})
    assert response.status_code == 404


def test_the_numbers_page_needs_the_password(client, monkeypatch):
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")
    assert client.get("/admin/numbers").status_code == 401


def test_the_numbers_page_counts_the_funnel(client, monkeypatch):
    client.get("/")
    session_id, _ = _finish(client)
    client.post("/api/rating", json={"session_id": session_id, "useful": True})

    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")
    page = client.get("/admin/numbers", auth=("me", "letmein"))

    assert page.status_code == 200
    assert "walks started" in page.text
    assert "Where people stop" in page.text


def test_the_numbers_page_is_fine_with_nothing_to_show(client, monkeypatch):
    """It will be empty on the day the link goes out, and that must not be an
    error page."""
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")

    page = client.get("/admin/numbers", auth=("me", "letmein"))

    assert page.status_code == 200
    assert "Nobody has answered anything yet" in page.text


# ---- Being told about a signup ---------------------------------------------


def test_a_signup_tells_you_about_itself(client, monkeypatch):
    told = []
    monkeypatch.setattr(api.notify, "someone_joined",
                        lambda email, studying, total: told.append((email, studying, total)))

    client.post("/api/waitlist",
                json={"email": "ada@example.com", "studying": "Edexcel A-level"})

    assert told == [("ada@example.com", "Edexcel A-level", 1)]


def test_signing_up_twice_does_not_tell_you_twice(client, monkeypatch):
    """People forget they signed up. You should not be told they arrived
    again, because they did not."""
    told = []
    monkeypatch.setattr(api.notify, "someone_joined",
                        lambda *a: told.append(a))

    client.post("/api/waitlist", json={"email": "ada@example.com"})
    client.post("/api/waitlist", json={"email": "ada@example.com"})

    assert len(told) == 1


def test_a_broken_mail_server_does_not_cost_you_the_signup(client, monkeypatch):
    """The row is saved before anything is sent. Losing a tutor who asked to
    hear from you, because a mail server was down, is far the worse failure."""
    def explode(*args, **kwargs):
        raise RuntimeError("smtp is on fire")

    monkeypatch.setattr(api.notify, "someone_joined", explode)

    response = client.post("/api/waitlist", json={"email": "ada@example.com"})

    assert response.status_code == 200

    connection = store.connect()
    try:
        assert store.waitlist_size(connection) == 1
    finally:
        connection.close()


# ---- Being able to ask whether email works ---------------------------------


def _mail_settings(monkeypatch, **extra):
    monkeypatch.setenv("OWNIT_SMTP_HOST", "smtp.resend.com")
    monkeypatch.setenv("OWNIT_SMTP_USER", "resend")
    monkeypatch.setenv("OWNIT_SMTP_PASSWORD", "re_a_secret_key")
    monkeypatch.setenv("OWNIT_SMTP_FROM", "onboarding@resend.dev")
    monkeypatch.setenv("OWNIT_NOTIFY_TO", "me@gmail.com")
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


def test_the_email_page_needs_the_password(client, monkeypatch):
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")
    assert client.get("/admin/email").status_code == 401


def test_it_says_when_email_is_not_switched_on(client, monkeypatch):
    for name in ("OWNIT_SMTP_HOST", "OWNIT_SMTP_USER", "OWNIT_SMTP_PASSWORD",
                 "OWNIT_SMTP_FROM", "OWNIT_NOTIFY_TO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")

    page = client.get("/admin/email", auth=("me", "letmein")).text

    assert "Not switched on" in page
    assert "OWNIT_SMTP_FROM" in page


def test_a_send_that_works_says_so_and_shows_the_addresses(client, monkeypatch):
    _mail_settings(monkeypatch)
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")

    class Fine:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, *a): pass

    monkeypatch.setattr(api.notify.smtplib, "SMTP", lambda *a, **k: Fine())

    page = client.get("/admin/email", auth=("me", "letmein")).text

    assert "Sent." in page
    assert "onboarding@resend.dev" in page
    assert "me@gmail.com" in page


def test_a_send_that_fails_gives_the_reason(client, monkeypatch):
    """The reason is the whole point of the page. Everywhere else it is
    swallowed, which is what made this impossible to diagnose."""
    _mail_settings(monkeypatch)
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")

    def refuse(*a, **k):
        raise OSError("Bad sender address syntax")

    monkeypatch.setattr(api.notify.smtplib, "SMTP", refuse)

    page = client.get("/admin/email", auth=("me", "letmein")).text

    assert "Not sent." in page
    assert "Bad sender address syntax" in page


def test_the_password_is_never_on_the_page(client, monkeypatch):
    """It is an API key, and this page shows every other setting beside it."""
    _mail_settings(monkeypatch)
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "letmein")

    def refuse(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(api.notify.smtplib, "SMTP", refuse)

    page = client.get("/admin/email", auth=("me", "letmein")).text

    assert "re_a_secret_key" not in page
