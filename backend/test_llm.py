"""Tests for which model each job gets.

No API calls: these check the settings we build, not what comes back.

Run from the repo root with:  pytest backend/
"""

import llm


# ---- Building the settings ------------------------------------------------


def test_effort_is_sent_to_models_that_take_it():
    settings = llm.Task(llm.OPUS, effort="low").kwargs()
    assert settings["output_config"] == {"effort": "low"}


def test_effort_is_withheld_from_models_that_reject_it():
    """Haiku 4.5 predates the effort parameter and errors on it."""
    settings = llm.Task(llm.HAIKU, effort="low").kwargs()
    assert "output_config" not in settings
    assert settings["model"] == llm.HAIKU


def test_no_effort_asked_for_means_none_sent():
    assert "output_config" not in llm.Task(llm.OPUS).kwargs()


def test_every_task_carries_a_token_ceiling():
    assert llm.Task(llm.OPUS).kwargs()["max_tokens"] > 0


# ---- The choices themselves -----------------------------------------------


def test_judgement_keeps_the_best_model():
    """These decide where the whole diagnosis goes. A cheap misread here costs
    far more than it saves."""
    for task in (llm.ENTRY_MATCH, llm.NARROW, llm.PRESENTATION):
        assert task.model == llm.OPUS


def test_writing_a_question_does_not():
    """A fixed shape with structured output does not need the top model."""
    assert llm.QUESTION.model != llm.OPUS


def test_reading_long_documents_does_not_either():
    """Nearly all the cost of a PDF is input, so the model choice is the lever."""
    for task in (llm.READ_PAPER, llm.READ_SCHEME):
        assert task.model != llm.OPUS


def test_every_task_names_a_real_model():
    known = {llm.OPUS, llm.SONNET, llm.HAIKU}
    tasks = [
        llm.ENTRY_MATCH,
        llm.NARROW,
        llm.PRESENTATION,
        llm.QUESTION,
        llm.READ_PAPER,
        llm.READ_SCHEME,
        llm.MAP_HINTS,
    ]
    for task in tasks:
        assert task.model in known
        # And anything asking for effort must be on a model that takes it.
        if task.effort:
            assert task.model in llm.TAKES_EFFORT


# ---- Caching --------------------------------------------------------------


def test_a_cached_block_is_marked_for_reuse():
    block = llm.cached("the skill catalogue")
    assert block["cache_control"] == {"type": "ephemeral"}
    assert block["text"] == "the skill catalogue"
    assert block["type"] == "text"
