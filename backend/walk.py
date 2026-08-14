"""Walks the skill graph downward to find what a student is actually missing.

The student is stuck on one question. We take the skill that question asks for
as the entry node and treat it as failed - being stuck is the premise, not
something we test. Then we work downward through `needs`, testing prerequisites
and descending only into the ones that come back wrong, until we reach a skill
they cannot do whose own prerequisites they can. That skill is the root gap.

An attempt is optional. "I don't know how to start" is the most common case,
and the students who say it are the ones who need this most. With an attempt we
can narrow the first round to the branch where the working actually broke;
without one the first round tests everything on the level below the entry node.
After that first round the two paths are identical - descend into what failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable

from anthropic import Anthropic
from pydantic import BaseModel

from graph import SKILLS, Skill
from questions import MODEL, MAX_TOKENS, ask_in_terminal, generate_question


# ---- What comes out -------------------------------------------------------


@dataclass(frozen=True)
class SkillResult:
    skill_id: str
    held: bool
    # The misconception behind the option they picked, when they got it wrong.
    mistake: str | None = None
    # False for the entry node, which we take as failed rather than testing.
    asked: bool = True
    # They said they didn't know. Not held either way, so the walk descends the
    # same - but the report needs to tell these apart, because a wrong rule gets
    # corrected and an absent one gets taught from scratch.
    dont_know: bool = False


@dataclass
class Diagnosis:
    entry_skill_id: str
    had_attempt: bool
    # Only ever set when there was an attempt to inspect.
    presentation_note: str | None = None
    # Which prerequisites the attempt pointed at, and why. None without one.
    narrowed_to: list[str] | None = None
    narrowed_because: str | None = None
    # True when that branch held and we had to test the rest of the level after all.
    widened: bool = False
    # Every skill we established something about, in the order we did it.
    results: list[SkillResult] = field(default_factory=list)
    # The skills to actually teach, deepest thing that broke.
    root_gaps: list[str] = field(default_factory=list)
    # Entry node down to each root gap.
    chains: list[list[str]] = field(default_factory=list)

    def result_for(self, skill_id: str) -> SkillResult | None:
        for result in self.results:
            if result.skill_id == skill_id:
                return result
        return None


# ---- Reading the attempt --------------------------------------------------


class Narrowing(BaseModel):
    branch_skill_ids: list[str]
    reason: str


class PresentationCheck(BaseModel):
    presentation_only: bool
    note: str


def narrow_to_branch(
    entry: Skill,
    attempt: str,
    *,
    client: Anthropic | None = None,
    model: str = MODEL,
) -> Narrowing:
    """Pick which of the entry skill's prerequisites the attempt broke in.

    This only saves the student questions - it never decides the diagnosis. If
    the guess is wrong, the branch it picks comes back held and the walk stops
    there rather than going somewhere untrue.
    """
    client = client or Anthropic()

    options = "\n".join(
        f"- {need}: {SKILLS[need].name} - {SKILLS[need].probe}"
        for need in entry.needs
        if need in SKILLS
    )

    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=(
            "You read a student's partial working on an A-level maths question "
            "and say which prerequisite skill their working actually broke in."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"The question asks the student to: {entry.probe}\n\n"
                    f"Their attempt:\n{attempt}\n\n"
                    "These are the prerequisites that skill rests on:\n"
                    f"{options}\n\n"
                    "Find the first step where the working actually goes wrong, "
                    "then say which of the prerequisites above that step "
                    "belongs to. Return the ids in `branch_skill_ids`.\n"
                    "- Use ids from the list and nothing else.\n"
                    "- Usually one id. Return two or three only when the attempt "
                    "genuinely does not distinguish between them.\n"
                    "- If the attempt is too thin to tell, return every id.\n"
                    "- Judge what the working shows, not how it is written out."
                ),
            }
        ],
        output_format=Narrowing,
    )

    narrowing = response.parsed_output
    valid = [] if narrowing is None else [
        skill_id for skill_id in narrowing.branch_skill_ids if skill_id in entry.needs
    ]

    # A bad guess must not shrink the search to nothing - fall back to testing
    # the whole level, which is what we would have done with no attempt at all.
    if not valid:
        return Narrowing(
            branch_skill_ids=list(entry.needs),
            reason="The attempt did not point at any one branch, so all were tested.",
        )

    return Narrowing(branch_skill_ids=valid, reason=narrowing.reason)


def check_presentation(
    entry: Skill,
    attempt: str,
    *,
    client: Anthropic | None = None,
    model: str = MODEL,
) -> PresentationCheck:
    """Is this a student who can do the maths but wrote it out badly?

    Runs only when there is an attempt, because there is nothing to inspect
    otherwise. It reports; it does not stop the walk. A student who presents
    work badly may still be missing a prerequisite, and we would rather ask
    than assume.
    """
    client = client or Anthropic()

    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=(
            "You judge whether a student's maths went wrong in the method or "
            "only in how it was written out."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"The question asks the student to: {entry.probe}\n\n"
                    f"Their attempt:\n{attempt}\n\n"
                    "Set `presentation_only` true only if the method is right "
                    "the whole way through and what let them down is how the "
                    "work is set out: notation, an answer left unsimplified or "
                    "in the wrong form, missing working, a slip in copying, or "
                    "arithmetic they plainly know how to do.\n"
                    "Set it false if any step depends on a rule or idea they "
                    "have got wrong, however tidily it is written.\n"
                    "In `note`, say in one or two sentences what you saw."
                ),
            }
        ],
        output_format=PresentationCheck,
    )

    check = response.parsed_output
    if check is None:
        return PresentationCheck(
            presentation_only=False, note="Could not read the attempt."
        )
    return check


# ---- Testing one skill ----------------------------------------------------


def check_by_asking(
    skill: Skill,
    *,
    client: Anthropic | None = None,
    model: str = MODEL,
) -> SkillResult:
    """Generate a question for one skill and put it to the student."""
    question = generate_question(skill.id, client=client, model=model)
    header = f"\n[{skill.name}]\n"
    answered = ask_in_terminal(question, header=header)
    return SkillResult(
        skill.id,
        held=answered.correct,
        mistake=answered.mistake,
        dont_know=answered.dont_know,
    )


# ---- The walk -------------------------------------------------------------


Check = Callable[[Skill], SkillResult]


def diagnose(
    entry_skill_id: str,
    attempt: str | None = None,
    *,
    check: Check = check_by_asking,
    client: Anthropic | None = None,
) -> Diagnosis:
    """Find the skill underneath the question that the student is missing."""
    entry = SKILLS.get(entry_skill_id)
    if entry is None:
        raise ValueError(f"'{entry_skill_id}' is not a skill in the graph")

    diagnosis = Diagnosis(entry_skill_id=entry.id, had_attempt=attempt is not None)

    # Being stuck on the question is the premise. We never test the entry node -
    # if every prerequisite below it holds, the entry node itself is the gap.
    results: dict[str, SkillResult] = {
        entry.id: SkillResult(entry.id, held=False, asked=False)
    }
    diagnosis.results.append(results[entry.id])

    # Who sent us to each skill, so we can trace a chain back up afterwards.
    parents: dict[str, str] = {}

    # Round one is the only place the two paths differ.
    if attempt is not None:
        presentation = check_presentation(entry, attempt, client=client)
        if presentation.presentation_only:
            diagnosis.presentation_note = presentation.note

        narrowing = narrow_to_branch(entry, attempt, client=client)
        frontier = list(narrowing.branch_skill_ids)
        diagnosis.narrowed_to = list(frontier)
        diagnosis.narrowed_because = narrowing.reason
    else:
        frontier = list(entry.needs)

    for skill_id in frontier:
        parents[skill_id] = entry.id

    # Narrowing is a shortcut, not a decision. If the branch it picked turns out
    # to be held, the guess was wrong, and we owe the student the rest of the
    # level rather than a shrug.
    widen_pending = attempt is not None

    # Everything from here is the same either way: test the frontier, then
    # descend into whatever failed.
    while frontier:
        failed: list[str] = []
        for skill_id in frontier:
            if skill_id in results:
                continue
            result = check(SKILLS[skill_id])
            results[skill_id] = result
            diagnosis.results.append(result)
            if not result.held:
                failed.append(skill_id)

        next_frontier: list[str] = []
        for skill_id in failed:
            for need in SKILLS[skill_id].needs:
                if need not in results and need not in next_frontier:
                    parents.setdefault(need, skill_id)
                    next_frontier.append(need)

        if widen_pending and not next_frontier:
            for need in entry.needs:
                if need not in results:
                    parents.setdefault(need, entry.id)
                    next_frontier.append(need)
            diagnosis.widened = bool(next_frontier)

        # Only ever right after the narrowed first round.
        widen_pending = False
        frontier = next_frontier

    diagnosis.root_gaps = _root_gaps(results)
    diagnosis.chains = [_chain_to(gap, entry.id, parents) for gap in diagnosis.root_gaps]
    return diagnosis


def _root_gaps(results: dict[str, SkillResult]) -> list[str]:
    """The failed skills that aren't explained by something below them.

    A skill they failed whose prerequisites they all hold is the bottom of that
    branch - nothing further down accounts for it, so it is the thing to teach.
    A failed floor node with no prerequisites qualifies the same way.
    """
    gaps = []
    for skill_id, result in results.items():
        if result.held:
            continue
        needs = SKILLS[skill_id].needs
        if all(skill_id_below in results and results[skill_id_below].held
               for skill_id_below in needs):
            gaps.append(skill_id)
    return gaps


def _chain_to(skill_id: str, entry_id: str, parents: dict[str, str]) -> list[str]:
    """The path from the entry node down to one root gap."""
    chain = [skill_id]
    while chain[-1] != entry_id and chain[-1] in parents:
        chain.append(parents[chain[-1]])
    return list(reversed(chain))


# ---- Command line ---------------------------------------------------------


def _print_diagnosis(diagnosis: Diagnosis) -> None:
    print()
    print("=" * 60)
    entry = SKILLS[diagnosis.entry_skill_id]
    print(f"Stuck on: {entry.name}")

    if diagnosis.presentation_note:
        print()
        print("Presentation, not method:")
        print(f"  {diagnosis.presentation_note}")

    if diagnosis.narrowed_because:
        print()
        print("Narrowed from the attempt:")
        print(f"  {diagnosis.narrowed_because}")
        if diagnosis.widened:
            print("  That branch held, so the rest of the level was tested too.")

    print()
    print("Asked:")
    for result in diagnosis.results:
        if not result.asked:
            continue
        if result.held:
            mark = "held"
        elif result.dont_know:
            mark = "not held - said they didn't know"
        else:
            mark = "not held"
        print(f"  {SKILLS[result.skill_id].name}: {mark}")
        if result.mistake:
            print(f"      {result.mistake}")

    print()
    if not diagnosis.root_gaps:
        print("No gap found below this question.")
        return

    print("Teach this:")
    for gap, chain in zip(diagnosis.root_gaps, diagnosis.chains):
        result = diagnosis.result_for(gap)
        if result and result.dont_know:
            note = "  (nothing there yet - teach it from scratch)"
        elif result and result.mistake:
            note = "  (holds a rule, and it is the wrong one - correct it)"
        else:
            note = ""
        print(f"  {SKILLS[gap].name}{note}")
        print("      " + " -> ".join(SKILLS[step].name for step in chain))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trace a stuck question down to the missing prerequisite."
    )
    parser.add_argument("entry_skill_id", help="the skill the question asks for")
    parser.add_argument(
        "--attempt",
        help="what the student tried, if anything. Leave it out for 'I don't know how to start'.",
    )
    parser.add_argument("--attempt-file", help="read the attempt from a file instead")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead")
    args = parser.parse_args(argv)

    attempt = args.attempt
    if args.attempt_file:
        attempt = open(args.attempt_file, encoding="utf-8").read()

    try:
        diagnosis = diagnose(args.entry_skill_id, attempt)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        print("run 'python backend/questions.py --list' to see the ids", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(diagnosis.__dict__, indent=2, default=lambda o: o.__dict__))
    else:
        _print_diagnosis(diagnosis)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
