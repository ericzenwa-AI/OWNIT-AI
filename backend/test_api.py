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
