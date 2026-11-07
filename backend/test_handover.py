"""What the report hands a tutor, now that the app does not teach.

The first tutor to use OwnIt on a real question said he does not believe in
AI teaching students. He is the buyer - students have the problem, tutors pay
- so an app that teaches was competing with the hour he bills for, and the
better it got at teaching the worse that got. The lesson came out of the
product.

What is left has to stand on its own, and these pin the two halves of that.
The app must not offer to teach, and the report must put the thing a tutor
needs - what to start on, and the evidence for it - where it can be read in
the ten seconds before a lesson starts.

Read off the page source rather than a browser, same as the other page tests:
the report is built in a template string, so what matters is what the string
says, and a missing button is exactly the kind of thing a passing suite will
not notice.
"""

from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
PAGE = (WEB / "index.html").read_text(encoding="utf-8")


# ---- the app does not teach ------------------------------------------------


def test_nothing_on_the_page_offers_to_teach_them():
    """The decision, as a test. It is one button away from coming back."""
    assert "startLesson()" not in PAGE.split("// ---- The lesson")[0], (
        "something still starts a lesson from the report")
    # The label, as a button rather than as the comment explaining its going.
    assert ">Take me through it<" not in PAGE, (
        "the report still offers to teach the student")


def test_the_lesson_code_is_still_there():
    """Taken out of the product, not out of the repo.

    Deleting it would throw away a working feature on the word of one tutor,
    and the positioning may yet move again. A student arriving without a
    tutor is still a real person with no one to teach them.
    """
    assert Path(__file__).resolve().parent.joinpath("teach.py").exists()
    assert "drawLesson" in PAGE, "the lesson screen has been deleted, not hidden"


def test_the_reason_is_written_down_where_the_button_was():
    """So the next person to read it does not helpfully add it back."""
    gone = PAGE.split("There was a \"Take me through it\" button here")
    assert len(gone) == 2, "the note explaining why teaching went is missing"
    assert "tutor" in gone[1][:800].lower()


# ---- what a tutor gets instead ---------------------------------------------


def test_the_report_opens_with_the_handover():
    """Before the cards written to the student, not after them."""
    assert "${handover(" in PAGE, "the report does not render a handover"
    body = PAGE.split("screen.innerHTML = `\n    ${partsBar()}")[1]
    assert body.lstrip().startswith("${handover("), (
        "something comes before the handover on the report")


@pytest.mark.parametrize("needed, why", [
    ("Start on", "a tutor needs to be told where to begin"),
    ("held:", "naming a gap is worthless without what held up under it"),
    ("gave way:", "and without what did not"),
])
def test_the_handover_carries_its_evidence(needed, why):
    block = PAGE.split("function handover(")[1].split("\nfunction ")[0]
    assert needed in block, why


def test_the_handover_says_nothing_the_walk_did_not_find():
    """It is the same diagnosis reordered, not a second opinion.

    Everything it prints has to come off the report object. The moment it
    starts generating its own advice it is a lesson again, by another name,
    and it is a lesson nobody checked.
    """
    block = PAGE.split("function handover(")[1].split("\nfunction ")[0]
    for invented in ("fetch(", "post(", "/api/"):
        assert invented not in block, (
            f"the handover calls {invented} - it must only re-say the report")
