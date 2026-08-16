"""Tests for the coverage check.

No API calls: check_one is replaced with canned results, so these test the
counting and the reporting rather than the matching.

Run from the repo root with:  pytest backend/
"""

import coverage
from coverage import Checked, Report, find_images, read_questions


def placed(source="q", skill_id="find_stationary_points"):
    return Checked(source, True, skill_id, "differentiation", "high", False, "")


def gap(source="q", recognised_as="binomial expansion"):
    """Understood perfectly, just not covered. A doorway to build."""
    return Checked(source, False, None, recognised_as, "high", False, "not covered")


def unreadable(source="q"):
    """We could not even tell what it was."""
    return Checked(source, False, None, "", "low", True, "unreadable")


# ---- Counting -------------------------------------------------------------


def test_the_rate_is_placed_over_total():
    report = Report([placed(), placed(), gap(), unreadable()])
    assert report.rate == 0.5


def test_a_gap_is_not_the_same_as_unreadable():
    """One is a doorway to build, the other is a bad input. Different jobs."""
    report = Report([gap(), unreadable()])

    assert len(report.gaps) == 1
    assert len(report.unreadable) == 1
    assert report.gaps[0].unreadable is False
    assert report.unreadable[0].unreadable is True


def test_the_backlog_is_ordered_by_how_often_it_comes_up():
    report = Report(
        [
            gap(recognised_as="binomial expansion"),
            gap(recognised_as="vectors"),
            gap(recognised_as="binomial expansion"),
            gap(recognised_as="binomial expansion"),
        ]
    )
    assert report.missing_topics() == [("binomial expansion", 3), ("vectors", 1)]


def test_topics_are_counted_regardless_of_case():
    report = Report([gap(recognised_as="Vectors"), gap(recognised_as="vectors")])
    assert report.missing_topics() == [("vectors", 2)]


def test_a_question_that_errored_is_left_out_of_the_rate():
    """A crash is not evidence about coverage either way."""
    broken = Checked("q", False, None, "", "", False, "", error="boom")
    report = Report([placed(), broken])

    assert report.rate == 1.0
    assert broken not in report.gaps


def test_an_empty_run_does_not_divide_by_zero():
    assert Report([]).rate == 0.0


# ---- Running a batch ------------------------------------------------------


def test_every_question_is_checked(monkeypatch):
    seen = []

    def fake(source, question="", image=None):
        seen.append(question)
        return placed(source)

    monkeypatch.setattr(coverage, "check_one", fake)
    report = coverage.check_questions(["one", "two", "three"], workers=2)

    assert sorted(seen) == ["one", "three", "two"]
    assert len(report.checked) == 3


def test_one_bad_question_does_not_stop_the_run(monkeypatch):
    def exploding(question, image=None, **kw):
        raise RuntimeError("no")

    monkeypatch.setattr(coverage, "identify_entry", exploding)
    result = coverage.check_one("line 1", question="anything")

    assert result.error is not None
    assert result.placed is False


# ---- Reading the inputs ---------------------------------------------------


def test_questions_are_read_one_per_line(tmp_path):
    path = tmp_path / "questions.txt"
    path.write_text(
        "# a comment\n\nFind the stationary points\n  Solve 5^x = 40  \n",
        encoding="utf-8",
    )
    assert read_questions(path) == ["Find the stationary points", "Solve 5^x = 40"]


def test_only_images_are_picked_up_from_a_folder(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    assert [p.name for p in find_images(tmp_path)] == ["a.png", "b.jpg"]
