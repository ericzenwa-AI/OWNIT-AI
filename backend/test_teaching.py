"""The lesson that runs after the diagnosis.

The diagnostic was never the hard part. A student is told what they are
missing, reads a correct paragraph about it, and is no better at it than
before - particularly the student this exists for, who loses the thread in a
minute. So after the diagnosis there is a lesson, and these tests are about
the three things that make it a lesson rather than a chatbot with a prompt:

  the server owns the stage, so a confident-sounding student cannot be hurried
  to the end and the model cannot wander;
  moving on takes evidence - two understood replies, not one right answer;
  the student's own question is never worked for them, at any stage.

And the one promise that outranks all of it: none of this may touch the
diagnosis. A student who never opens the lesson has exactly the session they
had before.
"""

import pytest
from fastapi.testclient import TestClient

import api
import store
import teach
from questions import Distractor, MultipleChoiceQuestion
from walk import Reading


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "teach.db")
    connection = store.connect()
    yield connection
    connection.close()


@pytest.fixture
def client(db):
    return TestClient(api.app)


class _Move:
    """A turn from a tutor that is not a model."""

    def __init__(self, read="understood", say="here is the idea",
                 ask="what do you get?", finished=False, why="they said so",
                 demo=None):
        self.read, self.say, self.ask = read, say, ask
        self.finished, self.why = finished, why
        self.demo = demo or []


def _diagnosed(connection, question="Solve 3x + 2y = 12 and x - y = 1"):
    """A finished walk, which is what a lesson starts from.

    root_gaps is comma-separated, not JSON. Writing it the convenient way here
    is what let the endpoint read it the wrong way and still pass every test -
    the first real question put through it crashed.
    """
    session_id = store.open_walk(
        connection, entry_skill_id="solve_simultaneous_elimination",
        reading=Reading(), question=question)
    connection.execute(
        """UPDATE sessions SET finished = 1, root_gaps = ?, question = ?
            WHERE id = ?""",
        ("eliminate_by_adding_or_subtracting", question, session_id))
    connection.commit()
    return session_id


def test_the_gap_is_read_the_way_the_walk_writes_it(db):
    """Guards the fixture itself: if these two ever disagree again, the
    endpoint breaks on real data while the tests stay green."""
    from walk import Diagnosis, SkillResult

    diagnosis = Diagnosis(entry_skill_id="solve_simultaneous_elimination",
                          had_attempt=False)
    diagnosis.root_gaps = ["eliminate_by_adding_or_subtracting"]
    session_id = _diagnosed(db)
    store.close_walk(db, session_id, diagnosis)

    stored = db.execute(
        "SELECT root_gaps FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert [g for g in stored["root_gaps"].split(",") if g] == diagnosis.root_gaps


# ---- the diagnosis is not touched ------------------------------------------


def test_a_lesson_writes_nothing_to_the_walk(client, db, monkeypatch):
    session_id = _diagnosed(db)
    before = dict(db.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())

    monkeypatch.setattr(teach, "next_move", lambda **kw: _Move())
    client.post("/api/teach", json={"session_id": session_id, "said": ""})

    after = dict(db.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())
    assert after == before
    assert db.execute("SELECT COUNT(*) n FROM answers").fetchone()["n"] == 0


def test_a_walk_that_has_not_finished_cannot_be_taught(client, db):
    """Teaching a gap we have not found yet would be teaching a guess."""
    session_id = store.open_walk(
        connection=db, entry_skill_id="solve_linear", reading=Reading(), question="q")

    response = client.post("/api/teach", json={"session_id": session_id, "said": ""})

    assert response.status_code == 409


# ---- the server owns the stage ---------------------------------------------


def test_one_right_answer_is_not_enough_to_move_on():
    """The whole complaint about the old report was that it moved on from one.
    Two, and it is a counter rather than an instruction in a prompt."""
    stage, carried = teach.advance("show", "understood", 0)
    assert stage == "show", "still teaching"

    stage, carried = teach.advance("show", "understood", carried)
    assert stage == "try", "now it has seen it twice"
    assert carried == 0, "and the evidence resets for the next stage"


def test_a_muddled_answer_costs_ground_without_undoing_everything():
    assert teach.advance("try", "muddled", 1) == ("try", 0)
    assert teach.advance("try", "muddled", 0) == ("try", 0)


def test_a_partly_right_answer_earns_nothing():
    """A bare right number after a hint is not understanding."""
    assert teach.advance("try", "partly", 0) == ("try", 0)


def test_the_lesson_only_ever_moves_one_stage_at_a_time():
    stage = "connect"
    seen = [stage]
    for _ in range(12):
        stage, _ = teach.advance(stage, "understood", teach.TO_ADVANCE - 1)
        seen.append(stage)
    for before, after in zip(seen, seen[1:]):
        gap = teach.STAGES.index(after) - teach.STAGES.index(before)
        assert gap in (0, 1), f"jumped from {before} to {after}"
    assert seen[-1] == "done", "and it does get there"


def test_it_never_goes_backwards():
    for stage in teach.STAGES:
        for read in ("understood", "partly", "muddled", "no_answer", "first"):
            went, _ = teach.advance(stage, read, 0)
            assert teach.STAGES.index(went) >= teach.STAGES.index(stage)


def test_the_model_cannot_choose_the_stage_itself(client, db, monkeypatch):
    """It reports how the answer went. Where that leaves the lesson is ours."""
    session_id = _diagnosed(db)
    monkeypatch.setattr(teach, "next_move", lambda **kw: _Move(read="understood"))

    opening = client.post(
        "/api/teach", json={"session_id": session_id, "said": ""}).json()
    assert opening["stage"] == "connect", "nothing said yet"

    first = client.post(
        "/api/teach", json={"session_id": session_id, "said": "yes that is it"}).json()
    assert first["stage"] == "show", "connect is agreement, so one reply leaves it"

    second = client.post(
        "/api/teach", json={"session_id": session_id, "said": "x = 4"}).json()
    assert second["stage"] == "show", "but show takes two"


# ---- it responds differently to right and wrong ----------------------------


def test_the_same_question_answered_well_and_badly_goes_to_different_places(
        client, db, monkeypatch):
    """The point of the whole thing: what they say decides what happens next."""

    def lesson_with(read):
        session_id = _diagnosed(db)
        monkeypatch.setattr(teach, "next_move", lambda **kw: _Move(read=read))
        client.post("/api/teach", json={"session_id": session_id, "said": ""})
        for _ in range(2):
            client.post("/api/teach", json={"session_id": session_id, "said": "x = 4"})
        return client.post(
            "/api/teach", json={"session_id": session_id, "said": "x = 4"}).json()

    understood = lesson_with("understood")
    muddled = lesson_with("muddled")

    assert understood["stage"] != muddled["stage"]
    assert understood["stage"] == "try", "they showed it, so they move on"
    assert muddled["stage"] == "show", "they did not, so it is taught again"


def test_what_the_student_said_is_given_to_the_tutor(client, db, monkeypatch):
    session_id = _diagnosed(db)
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return _Move()

    monkeypatch.setattr(teach, "next_move", capture)
    client.post("/api/teach", json={"session_id": session_id, "said": ""})
    client.post("/api/teach", json={"session_id": session_id, "said": "I added them"})

    assert seen["said"] == "I added them"
    assert seen["skill_id"] == "eliminate_by_adding_or_subtracting"
    assert "3x + 2y = 12" in seen["question"], "their own question goes with it"
    assert seen["transcript"], "and everything said so far"


# ---- the student's own question is never worked for them -------------------


def test_the_tutor_is_told_not_to_answer_their_question():
    assert "never work their own question" in teach.SYSTEM.lower()
    assert "never state" in teach.SYSTEM.lower()
    assert "never work the whole question" in teach.THE_JOB["apply"].lower()


def test_the_tutor_never_answers_its_own_question_either():
    assert "never give the answer to a question you have just asked" in teach.SYSTEM.lower()


def test_the_prompt_is_built_on_the_checked_explanation():
    """Rather than inventing the maths again every turn. That text was written
    once and confirmed by a model that did not write it."""
    prompt = teach.build_prompt(
        skill_id="eliminate_by_adding_or_subtracting",
        question="Solve 3x + 2y = 12 and x - y = 1",
        stage="show", transcript=[], said="")

    assert "WHAT IS ALREADY WRITTEN ABOUT IT" in prompt
    assert "WHAT USUALLY GOES WRONG" in prompt
    assert "3x + 2y = 12" in prompt


# ---- when it goes wrong ----------------------------------------------------


def test_a_failed_turn_stores_nothing(client, db, monkeypatch):
    """So pressing again picks up exactly where they were, rather than half a
    turn further on with a gap in the conversation."""
    session_id = _diagnosed(db)

    def boom(**kwargs):
        raise teach.CannotTeach("the tutor is down")

    monkeypatch.setattr(teach, "next_move", boom)
    response = client.post("/api/teach", json={"session_id": session_id, "said": "hi"})

    assert response.status_code == 503
    assert "again" in response.json()["detail"].lower()
    assert store.lesson_so_far(db, session_id) == []


def test_a_lesson_going_nowhere_stops_and_says_so(client, db, monkeypatch):
    """Rather than going round for ever. Some things need a person."""
    session_id = _diagnosed(db)
    monkeypatch.setattr(teach, "next_move", lambda **kw: _Move())
    for position in range(teach.MAX_TURNS):
        store.record_turn(db, session_id, stage="try", said="x",
                          move=_Move(), understood=0)

    got = client.post("/api/teach", json={"session_id": session_id, "said": "x"}).json()

    assert got["finished"] is True
    assert "tutor" in got["say"] or "teacher" in got["say"]


def test_an_unknown_skill_is_refused_rather_than_improvised():
    with pytest.raises(teach.CannotTeach):
        teach.next_move(skill_id="not_a_skill", question="q", stage="show",
                        transcript=[], said="")


# ---- coming back to it -----------------------------------------------------


def test_the_lesson_survives_the_tab_being_closed(client, db, monkeypatch):
    session_id = _diagnosed(db)
    monkeypatch.setattr(teach, "next_move", lambda **kw: _Move(say="step one"))
    client.post("/api/teach", json={"session_id": session_id, "said": ""})
    client.post("/api/teach", json={"session_id": session_id, "said": "got it"})

    came_back = client.get(f"/api/teach/{session_id}").json()

    assert len(came_back["turns"]) == 2
    assert came_back["turns"][0]["say"] == "step one"
    assert came_back["stage"]


def test_the_judgement_of_their_answer_is_not_sent_to_them(client, db, monkeypatch):
    """Same reason as everywhere else here: a label for how somebody is
    thinking is for a teacher, not for the student mid-lesson."""
    session_id = _diagnosed(db)
    monkeypatch.setattr(teach, "next_move",
                        lambda **kw: _Move(read="muddled", why="they added instead"))

    got = client.post("/api/teach", json={"session_id": session_id, "said": "?"}).json()

    assert "read" not in got and "why" not in got
    assert "muddled" not in str(got)
    kept = store.lesson_so_far(db, session_id)[-1]
    assert kept["read"] == "muddled" and kept["why"] == "they added instead"


# ---- the page --------------------------------------------------------------


def _page():
    import pathlib

    return (pathlib.Path(__file__).resolve().parent.parent
            / "web" / "index.html").read_text(encoding="utf-8")


def test_the_finding_offers_the_lesson_before_the_reading():
    """It was below the explanation, the worked example, the usual mistake and
    the practice line - four things to scroll past before being offered the
    one that actually teaches."""
    page = _page()
    assert "Take me through it" in page
    assert "function startLesson(" in page

    report = page[page.index("function report()"):page.index("function usefulCard(")]
    assert report.index("startLesson()") < report.index("explanation.plain")
    assert report.index("startLesson()") < report.index("named.practice")


def test_their_question_stays_on_screen_through_it():
    """And it is the same pinned panel the diagnostic uses, not a second one
    that looks nearly the same - it is the one thing on screen that should not
    move between the two halves of a session."""
    page = _page()
    assert "function askedCard(" in page

    lesson = page[page.index("function drawLesson("):page.index("function report(")]
    assert "askedCard()" in lesson

    opened = page.index("function question() {")
    asking = page[opened:page.index(chr(10) + "function ", opened + 10)]
    assert "askedCard()" in asking


def test_the_lesson_owns_the_screen_while_it_runs():
    assert "if (lesson) return;" in _page(), "draw() must stand aside"


def test_there_is_a_way_out_and_a_way_to_say_you_are_stuck():
    page = _page()
    assert "Back to what we found" in page
    assert "I am stuck" in page


# ---- not asking the same thing over and over -------------------------------


def _turns(stage, reads):
    return [{"stage": stage, "read": read} for read in reads]


def test_failing_twice_running_is_counted():
    assert teach.stuck_for(_turns("show", ["muddled", "partly"]), "show") == 2
    assert teach.stuck_for(_turns("show", ["muddled", "understood"]), "show") == 0, (
        "getting it resets the count")
    assert teach.stuck_for(_turns("try", ["muddled"]), "show") == 0, (
        "a different stage is a different question")


def test_after_two_failures_it_is_told_to_change_approach():
    """The model does not notice it is repeating itself. Put a student in
    front of it who answers something else and it will ask the same question
    five times, which is the one thing a tutor must not do."""
    patient = teach.build_prompt(
        skill_id="substitute_expr", question="q", stage="show",
        transcript=_turns("show", ["muddled"]), said="?")
    assert "NOT GOT THIS" not in patient

    changing = teach.build_prompt(
        skill_id="substitute_expr", question="q", stage="show",
        transcript=_turns("show", ["muddled", "muddled"]), said="?")
    assert "NOT GOT THIS IN 2 TRIES" in changing
    assert "Do not repeat your last question" in changing
    assert "DIFFERENT and easier question" in changing


def test_it_is_never_told_to_work_their_own_question_even_when_stuck():
    """The one place that instruction would be dangerous is the stage that is
    already looking at their question."""
    changing = teach.build_prompt(
        skill_id="substitute_expr", question="q", stage="apply",
        transcript=_turns("apply", ["muddled", "muddled"]), said="?")

    assert "never work any part of their own question" in changing
    assert "Work this one step out for them" not in changing


def test_the_tutor_cannot_declare_the_lesson_over_early(client, db, monkeypatch):
    """Left to itself it finishes three turns in because one answer sounded
    confident. A lesson ends on their own question or not at all."""
    session_id = _diagnosed(db)
    monkeypatch.setattr(teach, "next_move",
                        lambda **kw: _Move(read="understood", finished=True))

    first = client.post(
        "/api/teach", json={"session_id": session_id, "said": "I think I get it"}).json()

    assert first["finished"] is False
    assert first["stage"] == "show", "the earliest it can possibly be"


def test_it_can_finish_once_it_is_on_their_own_question(client, db, monkeypatch):
    session_id = _diagnosed(db)
    store.record_turn(db, session_id, stage="apply", said="x",
                      move=_Move(), understood=0)
    monkeypatch.setattr(teach, "next_move",
                        lambda **kw: _Move(read="understood", finished=True))

    got = client.post("/api/teach", json={"session_id": session_id, "said": "x = 3"}).json()

    assert got["finished"] is True
    assert got["stage"] == "done"


# ---- showing, not just telling ---------------------------------------------


class _Step:
    def __init__(self, show, changed="", note=""):
        self.show, self.changed, self.note = show, changed, note

    def model_dump(self):
        return {"show": self.show, "changed": self.changed, "note": self.note}


def test_a_good_demonstration_passes():
    assert teach.unusable_demo([
        _Step("sqrt(x) * sqrt(x) = x"),
        _Step("x^(1/2) * x^(1/2) = x^1", changed="x^(1/2)",
              note="adding the powers gives 1"),
    ]) is None


def test_no_demonstration_at_all_is_fine():
    """Most turns are a question or a nudge, and should not force a picture."""
    assert teach.unusable_demo([]) is None


def test_one_line_is_not_a_demonstration():
    assert "not a demonstration" in teach.unusable_demo([_Step("x = 2")])


def test_a_demonstration_that_is_really_a_worked_solution_is_refused():
    many = [_Step(f"line {n}") for n in range(teach.MAX_DEMO + 1)]
    assert "worked solution" in teach.unusable_demo(many)


def test_a_highlight_that_is_not_in_the_line_is_refused():
    """It is how they see WHAT changed. One that is not there underlines
    nothing, or worse, the wrong thing."""
    why = teach.unusable_demo([
        _Step("x^(1/2)"), _Step("sqrt(x)", changed="x^(1/3)")])

    assert "not in the line" in why


def test_a_line_too_long_to_follow_is_refused():
    assert "too long" in teach.unusable_demo([
        _Step("x = 1"), _Step("y = " + "1 + " * 40)])


def test_non_ascii_in_a_demonstration_is_refused():
    """Stray Unicode is one of the four rendering faults that reached real
    students, and it came in exactly this way."""
    assert "ASCII" in teach.unusable_demo([_Step("x = 1"), _Step("x → 2")])


def test_a_broken_demonstration_is_dropped_and_the_turn_survives(
        client, db, monkeypatch):
    """A lesson that dies because a picture was malformed is worse than one
    without a picture."""
    session_id = _diagnosed(db)
    monkeypatch.setattr(teach, "next_move", lambda **kw: _Move(
        say="still worth saying", demo=[_Step("only one line")]))

    got = client.post("/api/teach", json={"session_id": session_id, "said": ""}).json()

    assert got["demo"] == []
    assert got["say"] == "still worth saying"


def test_a_demonstration_reaches_the_student_and_is_kept(client, db, monkeypatch):
    session_id = _diagnosed(db)
    monkeypatch.setattr(teach, "next_move", lambda **kw: _Move(demo=[
        _Step("sqrt(x) * sqrt(x) = x"),
        _Step("x^(1/2) * x^(1/2) = x", changed="x^(1/2)", note="same thing"),
    ]))

    got = client.post("/api/teach", json={"session_id": session_id, "said": ""}).json()
    assert len(got["demo"]) == 2
    assert got["demo"][1]["changed"] == "x^(1/2)"

    # And it comes back the same when they return to it, rather than being
    # reworded into a different demonstration.
    again = client.get(f"/api/teach/{session_id}").json()
    assert again["turns"][0]["demo"][0]["show"] == "sqrt(x) * sqrt(x) = x"


def test_the_tutor_is_told_to_show_and_to_stop_one_step_short():
    assert "SHOWING BEATS TELLING" in teach.SYSTEM
    assert "one step short" in teach.SYSTEM
    assert "Never demonstrate on their own question" in teach.SYSTEM
    assert "`demo`" in teach.THE_JOB["show"]


def test_the_page_uncovers_a_demonstration_a_step_at_a_time():
    page = _page()
    assert "function demoHtml(" in page
    assert "Show the next step" in page
    assert "data-more" in page
    assert 'class="demo-step' in page


def test_a_resumed_lesson_does_not_make_them_click_through_it_again():
    assert "shown: (turn.demo || []).length" in _page()


def test_leaving_the_opening_stage_turns_on_whether_they_replied(client, db,
                                                                 monkeypatch):
    """Not on the model's reading of the reply. Asked to judge "yes, that is
    the bit" it kept answering "first", and the lesson sat on the opening
    stage agreeing with itself for three turns."""
    session_id = _diagnosed(db)
    monkeypatch.setattr(teach, "next_move", lambda **kw: _Move(read="first"))

    opening = client.post(
        "/api/teach", json={"session_id": session_id, "said": ""}).json()
    assert opening["stage"] == "connect", "nothing said yet"

    answered = client.post(
        "/api/teach", json={"session_id": session_id, "said": "yes that is the bit"}).json()
    assert answered["stage"] == "show", "they replied, so the teaching starts"


def test_silence_does_not_move_the_lesson_on():
    assert teach.advance("connect", "understood", 0, replied=False) == ("connect", 0)


def test_a_highlight_covering_the_whole_line_marks_nothing(client, db, monkeypatch):
    """Which is what the first line of a demonstration usually comes back as:
    a chip on screen repeating the line directly above it."""
    session_id = _diagnosed(db)
    monkeypatch.setattr(teach, "next_move", lambda **kw: _Move(demo=[
        _Step("sqrt(x^5)", changed="sqrt(x^5)"),
        _Step("x^(5/2)", changed="5/2", note="the root becomes a half power"),
    ]))

    got = client.post("/api/teach", json={"session_id": session_id, "said": ""}).json()

    assert got["demo"][0]["changed"] == "", "nothing is new on the first line"
    assert got["demo"][1]["changed"] == "5/2", "but a real change still shows"


# ---- the unaided question at the end ---------------------------------------


def _banked(connection, skill_id, question, correct="2"):
    store.bank_question(connection, skill_id, MultipleChoiceQuestion(
        question=question, correct_option=correct,
        distractors=[Distractor(option="9", mistake="added instead"),
                     Distractor(option="7", mistake="dropped a sign"),
                     Distractor(option="0", mistake="stopped too early")]))
    connection.commit()


def test_the_check_never_reuses_a_question_they_were_diagnosed_on(client, db):
    """That would measure whether they remember an answer, which is the one
    thing it must not measure."""
    session_id = _diagnosed(db)
    seen = "Which sign eliminates x?"
    _banked(db, "eliminate_by_adding_or_subtracting", seen)
    _banked(db, "eliminate_by_adding_or_subtracting", "A fresh one")
    db.execute(
        """INSERT INTO answers (session_id, position, skill_id, question, chosen,
               outcome) VALUES (?, 1, ?, ?, 'A', 'wrong')""",
        (session_id, "eliminate_by_adding_or_subtracting", seen))
    db.commit()

    got = client.get(f"/api/check/{session_id}").json()

    assert got["available"] is True
    assert got["question"] == "A fresh one"


def test_the_page_is_never_told_which_option_is_right(client, db):
    session_id = _diagnosed(db)
    _banked(db, "eliminate_by_adding_or_subtracting", "Only one here")

    got = client.get(f"/api/check/{session_id}").json()

    assert set(got["options"][0]) == {"label", "text"}
    assert "correct" not in str(got["options"])
    assert "mistake" not in str(got)


def test_it_is_marked_against_what_was_actually_shown(client, db):
    session_id = _diagnosed(db)
    _banked(db, "eliminate_by_adding_or_subtracting", "Only one here", correct="2")
    shown = client.get(f"/api/check/{session_id}").json()
    right = next(o["label"] for o in shown["options"] if o["text"] == "2")

    got = client.post("/api/check",
                      json={"session_id": session_id, "label": right}).json()

    assert got["correct"] is True
    assert "on your own" in got["say"]


def test_a_miss_is_told_what_they_did_rather_than_just_wrong(client, db):
    session_id = _diagnosed(db)
    _banked(db, "eliminate_by_adding_or_subtracting", "Only one here", correct="2")
    shown = client.get(f"/api/check/{session_id}").json()
    wrong = next(o["label"] for o in shown["options"] if o["text"] == "9")

    got = client.post("/api/check",
                      json={"session_id": session_id, "label": wrong}).json()

    assert got["correct"] is False
    assert got["say"] == "added instead", "the named misconception, not 'wrong'"


def test_how_the_check_went_is_recorded(client, db):
    """So the dashboard can answer the only question that matters about the
    teaching: does anybody come out of it able to do the thing."""
    session_id = _diagnosed(db)
    _banked(db, "eliminate_by_adding_or_subtracting", "Only one here", correct="2")
    shown = client.get(f"/api/check/{session_id}").json()
    right = next(o["label"] for o in shown["options"] if o["text"] == "2")
    client.post("/api/check", json={"session_id": session_id, "label": right})

    names = [row["name"] for row in store.clicks(db)]
    assert "check_passed" in names


def test_nothing_unseen_on_the_shelf_is_said_plainly(client, db):
    """Rather than asking them something they have already answered."""
    session_id = _diagnosed(db)

    got = client.get(f"/api/check/{session_id}").json()

    assert got == {"available": False}


def test_a_check_cannot_be_marked_before_one_is_given(client, db):
    session_id = _diagnosed(db)
    response = client.post("/api/check", json={"session_id": session_id, "label": "A"})
    assert response.status_code == 404


def test_the_lesson_offers_it_only_once_it_has_finished():
    page = _page()
    lesson = page[page.index("function drawLesson("):page.index("function report(")]
    assert "Try one on your own" in lesson
    assert "lesson.finished ?" in lesson
    assert lesson.index("lesson.finished ?") < lesson.index("Try one on your own")
