"""Which model each job gets, and how hard it is allowed to think.

Running everything on the most capable model at full effort is the expensive
default, and most of these jobs do not need it. The split is by how much
judgement the job actually takes:

  judgement       reading a student's working, deciding what a question is
                  asking - get these wrong and every question after them is
                  wrong. Worth the best model.
  transcription   copying a paper or a mark scheme out faithfully. Long input,
                  little reasoning, and input is where the cost is.
  generation      writing one multiple-choice question to a fixed shape. Well
                  specified, structured output, cheapest model that does it.

Effort is not a free dial either. Thinking tokens are billed at the output
rate, so a job at high effort can cost several times the same job at low.

Careful with Haiku 4.5: it predates the effort parameter and rejects it, and
its thinking is configured the old way. So it runs with neither, which is fine
for a job that only has to follow a shape.
"""

from __future__ import annotations

from dataclasses import dataclass

OPUS = "claude-opus-5"
SONNET = "claude-sonnet-5"
HAIKU = "claude-haiku-4-5"

# Models that take output_config.effort. Older ones error on it.
TAKES_EFFORT = (OPUS, SONNET)


@dataclass(frozen=True)
class Task:
    model: str
    # low | medium | high | xhigh | max, or None to leave the model's default.
    effort: str | None = None
    max_tokens: int = 16000

    def kwargs(self) -> dict:
        """The model settings to spread into a messages call."""
        settings: dict = {"model": self.model, "max_tokens": self.max_tokens}
        if self.effort and self.model in TAKES_EFFORT:
            settings["output_config"] = {"effort": self.effort}
        return settings


# Judgement. These decide where the whole diagnosis goes, so they keep the
# best model - a cheaper entry match that misreads a question costs far more
# than it saves.
ENTRY_MATCH = Task(OPUS, effort="medium")
NARROW = Task(OPUS, effort="medium")
PRESENTATION = Task(OPUS, effort="low")

# Generation. This was Haiku, on the reasoning that writing one question to a
# fixed shape is well specified and structured output does the hard part. That
# was wrong, and auditing the bank showed how wrong: 10% of the answers marked
# correct were not correct, and the errors were not spread evenly. Expanding
# brackets and negative numbers came out right nearly every time; areas between
# curves and binomial expansions were wrong about a third of the time. Writing
# the question was never the hard part - working out the answer was.
#
# The cost argument for a cheap model also stopped applying once the bank
# existed. A question is written once and served to many students, so the price
# of writing it is spread across all of them, while a wrong answer key is
# charged to every single one: a student who answers correctly is recorded as
# not holding the skill, and the diagnosis goes looking for a gap that is not
# there.
QUESTION = Task(SONNET, effort="medium")

# Transcription and offline analysis. Long PDFs, so the cost is nearly all
# input, and a cheaper model saves most of it.
READ_PAPER = Task(SONNET, effort="low")
READ_SCHEME = Task(SONNET, effort="medium")
MAP_HINTS = Task(SONNET, effort="medium")


def for_model(model: str, like: Task | None = None) -> Task:
    """The same job on a different model.

    Settings are not portable between models - effort is the one that bites,
    since Haiku rejects it and the others expect it - so swapping a model means
    rebuilding the task and letting kwargs() decide what that model will take.
    Patching the name into settings already built produces a combination no
    model accepts.
    """
    like = like or QUESTION
    return Task(model, effort=like.effort, max_tokens=like.max_tokens)


def cached(text: str) -> dict:
    """A text block the API should keep, so repeat calls do not re-read it.

    Caching is a prefix match, so a cached block only pays off when it sits
    ahead of everything that changes between calls. Put the catalogue of skills
    first and the student's question last, never the other way round.
    """
    return {
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }
