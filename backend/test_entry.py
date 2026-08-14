"""Tests for matching a question to an entry skill.

No API calls: identify_entry takes a client, so these hand it a canned reply.

Run from the repo root with:  pytest backend/
"""

import pytest

import entry
from entry import (
    EntryMatch,
    build_prompt,
    confirm_in_terminal,
    describe,
    entry_skills,
    identify_entry,
    is_usable,
)
from graph import SKILLS


def fake_client(parsed):
    """Stands in for the Anthropic client, returning a canned parse result."""

    class _Response:
        parsed_output = parsed

    class _Messages:
        sent = {}

        def parse(self, **kwargs):
            _Messages.sent.update(kwargs)
            return _Response()

    class _Client:
        messages = _Messages()

    return _Client()


def match(**overrides):
    fields = dict(
        skill_id="find_stationary_points",
        confidence="high",
        plain_summary="Find the high and low points of a curve.",
        reason="It asks for the turning points.",
    )
    fields.update(overrides)
    return EntryMatch(**fields)


def typing(*answers):
    replies = iter(answers)
    return lambda prompt: next(replies)


# ---- What gets offered ----------------------------------------------------


def test_only_exam_level_skills_are_entry_points():
    """A prerequisite is something we descend into, not somewhere to start."""
    assert all(skill.level == 0 for skill in entry_skills())
    assert "find_stationary_points" in {s.id for s in entry_skills()}
    assert "index_laws" not in {s.id for s in entry_skills()}


def test_every_entry_skill_is_offered_to_claude():
    prompt = build_prompt("Find the stationary points of y = x^3 - 3x")
    for skill in entry_skills():
        assert skill.id in prompt


def test_the_prompt_allows_no_match():
    prompt = build_prompt("Integrate x^2")
    assert "null" in prompt
    assert "integration" in prompt.lower()


# ---- Deciding whether to trust the match ----------------------------------


def test_a_confident_match_is_usable():
    assert is_usable(match()) is True


def test_no_match_is_not_forced():
    """Out of scope has to stay out of scope."""
    assert is_usable(match(skill_id=None, confidence="high")) is False


def test_a_low_confidence_match_is_rejected():
    assert is_usable(match(confidence="low")) is False


def test_a_skill_that_does_not_exist_is_rejected():
    assert is_usable(match(skill_id="integrate_by_parts")) is False


def test_a_skill_that_is_not_an_entry_point_is_rejected():
    """index_laws is real, but no exam question asks for it directly."""
    assert is_usable(match(skill_id="index_laws")) is False


# ---- Talking to the student -----------------------------------------------


def test_the_student_is_shown_plain_words_not_an_id():
    text = describe(match())
    assert "Find the high and low points of a curve." in text
    assert "find_stationary_points" not in text


def test_an_unsure_match_says_so():
    assert "not certain" in describe(match(confidence="medium"))
    assert "not certain" not in describe(match(confidence="high"))


def test_confirming_returns_the_skill_to_walk_from():
    assert confirm_in_terminal(match(), input_fn=typing("y")) == "find_stationary_points"


def test_rejecting_lets_the_student_choose_instead():
    """A wrong guess must not dead-end them."""
    options = entry_skills()
    chosen = confirm_in_terminal(match(), input_fn=typing("n", "2"))
    assert chosen == options[1].id


def test_rejecting_and_recognising_nothing_stops_the_walk():
    assert confirm_in_terminal(match(), input_fn=typing("n", "0")) is None


def test_unreadable_answers_are_asked_again():
    assert confirm_in_terminal(match(), input_fn=typing("maybe", "y")) is not None


# ---- The call itself ------------------------------------------------------


def test_identify_returns_the_match():
    result = identify_entry("Find the stationary points", client=fake_client(match()))
    assert result.skill_id == "find_stationary_points"


def test_an_unreadable_reply_is_treated_as_no_match():
    result = identify_entry("something", client=fake_client(None))
    assert is_usable(result) is False


def test_a_photo_is_sent_as_an_image_block(tmp_path):
    """Room for a photo now, so adding it later is not a rewrite."""
    photo = tmp_path / "question.png"
    photo.write_bytes(b"not really a png, but the bytes travel the same way")

    client = fake_client(match())
    identify_entry("", photo, client=client)

    content = client.messages.sent["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    # The image comes before the text it relates to.
    assert content[1]["type"] == "text"


def test_an_unsupported_image_type_is_refused(tmp_path):
    photo = tmp_path / "question.bmp"
    photo.write_bytes(b"...")
    with pytest.raises(ValueError, match="Cannot read"):
        identify_entry("", photo, client=fake_client(match()))


def test_nothing_to_go_on_is_an_error():
    with pytest.raises(ValueError, match="question text"):
        identify_entry("", client=fake_client(match()))


# ---- The walk is untouched ------------------------------------------------


def test_the_walk_still_takes_a_skill_id():
    """This module hands off an id; it does not change how the walk works."""
    import inspect

    import walk

    signature = inspect.signature(walk.diagnose)
    assert list(signature.parameters)[0] == "entry_skill_id"
    assert "question" not in signature.parameters
