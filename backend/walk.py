"""Walks the skill graph downward to find what a student is actually missing.

The student is stuck on one question. We take the skill that question asks for
as the entry node and treat it as failed - being stuck is the premise, not
something we test. Then we work downward through `needs`, testing prerequisites
and descending into the first one that comes back wrong, until we reach a skill
they cannot do whose own prerequisites they can.

That skill is a root gap, not the root gap. The graph forks, so the real gap is
every broken skill below the break point, and one descent finds one of them.
Testing the lot costs up to 42 questions from some entry nodes, which no stuck
student will sit through, so we follow one branch and keep a list of what we
skipped. The report offers those branches as a follow-up rather than quietly
dropping them.

An attempt is optional. "I don't know how to start" is the most common case,
and the students who say it are the ones who need this most. With an attempt we
can narrow the first round to the branch where the working actually broke;
without one the first round tests everything on the level below the entry node.
After that first round the two paths are identical - descend into what failed.

The walk stops early in two cases: it has asked its budget of questions, or the
student has said "I don't know" three times running, which means they are far
enough below this question that pinpointing the floor helps nobody.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, replace
from typing import Callable

from anthropic import Anthropic
from pydantic import BaseModel

import llm
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
    # What they were actually asked and what they picked. Kept so the answer can
    # be stored and looked at later, not used by the walk itself.
    question: str | None = None
    chosen: str | None = None
    seconds: float | None = None
    # Carried over from an earlier part of the same question rather than asked
    # again. Stored once, not twice.
    reused: bool = False


@dataclass
class Diagnosis:
    entry_skill_id: str
    had_attempt: bool
    # Only ever set when there was an attempt to inspect.
    presentation_note: str | None = None
    # Which prerequisites the attempt pointed at, and why. None without one.
    narrowed_to: list[str] | None = None
    narrowed_because: str | None = None
    # True when that branch held and we tested the rest of the level after all.
    widened: bool = False
    # Every skill we established something about, in the order we did it.
    results: list[SkillResult] = field(default_factory=list)
    # A confirmed gap: failed, with every one of its prerequisites held. One
    # descent finds at most one of these.
    root_gaps: list[str] = field(default_factory=list)
    # Entry node down to each root gap.
    chains: list[list[str]] = field(default_factory=list)
    # Branches we walked past. These are not cleared skills - they are skills
    # nobody has asked about, and any of them could hold another gap. The report
    # offers them as a follow-up.
    unchecked: list[str] = field(default_factory=list)
    # Why we stopped short, if we did. None means the descent finished.
    stopped_early: str | None = None
    # The lowest skill we saw fail. When we stop early this is the best answer
    # available, but its own prerequisites were never checked.
    deepest_failure: str | None = None

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
        **llm.NARROW.kwargs(),
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
        **llm.PRESENTATION.kwargs(),
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
    # Off the shelf if we have written one for this skill before. Generation is
    # most of what a session costs, and a question belongs to a skill rather
    # than to a student.
    import bank

    banked_id, question = bank.question_for(skill.id)

    header = f"\n[{skill.name}]\n"
    answered = ask_in_terminal(question, header=header)

    if banked_id is not None:
        _record_asked(banked_id, answered.correct)

    return SkillResult(
        skill.id,
        held=answered.correct,
        mistake=answered.mistake,
        dont_know=answered.dont_know,
        question=question.question,
        chosen=answered.chosen,
        seconds=answered.seconds,
    )


# ---- The walk -------------------------------------------------------------


def _record_asked(banked_id: int, correct: bool) -> None:
    """Note that a banked question was used, and how it went.

    Which is how a question that teaches us nothing gets found later. Never
    worth failing a student's session over.
    """
    import store

    try:
        connection = store.connect()
        try:
            store.mark_asked(connection, banked_id, correct)
        finally:
            connection.close()
    except Exception:  # noqa: BLE001 - bookkeeping, not the point of the session
        pass


Check = Callable[[Skill], SkillResult]


def reusing(known: dict[str, SkillResult], check: Check = check_by_asking) -> Check:
    """Wrap a check so a skill settled earlier in the session is not asked again.

    The parts of one exam question sit on the same foundations, so part (b)
    usually walks straight back through what part (a) already established.
    Asking a student the same thing twice is slow, and it reads as not having
    listened the first time.
    """

    def remembering(skill: Skill) -> SkillResult:
        if skill.id in known:
            return replace(known[skill.id], reused=True)
        result = check(skill)
        known[skill.id] = result
        return result

    return remembering

# A straight descent uses 5-7 questions before any sibling gets tested, so the
# budget has to clear that comfortably or it fires on legitimate walks.
QUESTION_CAP = 15

# Three in a row means the student is well below this question. Grinding down to
# the arithmetic floor to prove it is slow, and tells them nothing they can use.
DONT_KNOW_RUN = 3


# How far below the question we are prepared to go.
#
# Two used to feel like enough - deep enough to name something teachable,
# shallow enough not to end every diagnosis at Year 7 arithmetic. Measuring it
# said otherwise. At two levels the walk names the true root cause 40% of the
# time; at five it is 87%, for about three more questions on the students who
# actually have deep gaps. Everyone else still finishes in three or four,
# because a walk only gets longer when it keeps finding things broken.
#
# What settled it: a broken floor skill breaks everything resting on it, so the
# walk gets absorbed into the first failing branch and stalls. At depth two a
# student missing fraction arithmetic is not found even when fraction
# arithmetic is a direct prerequisite of their question. Those are exactly the
# students this is for - the ones who lost the thread years ago and have been
# failing ever since.
#
# The old worry was that a deep answer is not the one to act on. That is about
# what a tutor teaches, not about what we should look for. The report gives the
# whole chain, so they can start where they judge best - but only if we looked.
MAX_DEPTH = 5


def _closest_first(skill_ids) -> list[str]:
    """Nearest the student's question first, then further down.

    Level 0 is what an exam asks for and level 4 is the assumed floor, so a
    lower level number is closer to what they actually sent us.

    The other order - deepest first - sounds more thorough and is worse. It
    starts by asking about something that looks unrelated, and because the walk
    dives into whatever fails, it can name arithmetic as the gap without ever
    checking whether the student knows the method at all. Failing still
    descends, so a genuinely shaky foundation is still found; it is found
    because the evidence led there rather than because we began there.
    """
    return sorted(skill_ids, key=lambda s: (SKILLS[s].level, s))


@dataclass
class Reading:
    """What reading the student's attempt established.

    Decided once at the start and carried from then on, because it costs two
    model calls and the answer cannot change part way through a walk.
    """

    had_attempt: bool = False
    narrowed_to: list[str] | None = None
    narrowed_because: str | None = None
    presentation_note: str | None = None


@dataclass
class Step:
    """Where a walk has got to: one more question, or a finished diagnosis."""

    ask: str | None
    diagnosis: Diagnosis

    @property
    def finished(self) -> bool:
        return self.ask is None


def read_attempt(
    entry: Skill, attempt: str, *, client: Anthropic | None = None
) -> Reading:
    """The two judgements we make about an attempt, before any question."""
    presentation = check_presentation(entry, attempt, client=client)
    narrowing = narrow_to_branch(entry, attempt, client=client)

    return Reading(
        had_attempt=True,
        narrowed_to=list(narrowing.branch_skill_ids),
        narrowed_because=narrowing.reason,
        presentation_note=(
            presentation.note if presentation.presentation_only else None
        ),
    )


def step(
    entry_skill_id: str,
    answers: list[SkillResult] | tuple = (),
    *,
    reading: Reading | None = None,
    cap: int = QUESTION_CAP,
    dont_know_run: int = DONT_KNOW_RUN,
    max_depth: int = MAX_DEPTH,
) -> Step:
    """Given what a student has answered so far, what should we ask next?

    This is the whole walk, and it holds nothing between calls. Hand it the
    answers and it replays the descent from the top, stopping at the first
    skill it has no answer for - that is the next question. When it runs out of
    questions, the diagnosis is finished.

    That is what makes the walk work over a web request. A terminal program can
    sit inside a loop waiting for a keypress; a server cannot hold a half
    finished walk in memory while a student thinks, wanders off, or closes the
    tab. Replaying is cheap, and it means the answer never depends on anything
    being remembered - only on what the student actually said.
    """
    entry = SKILLS.get(entry_skill_id)
    if entry is None:
        raise ValueError(f"'{entry_skill_id}' is not a skill in the graph")

    reading = reading or Reading()
    given = {answer.skill_id: answer for answer in answers}
    needed: str | None = None

    diagnosis = Diagnosis(
        entry_skill_id=entry.id,
        had_attempt=reading.had_attempt,
        presentation_note=reading.presentation_note,
        narrowed_to=list(reading.narrowed_to) if reading.narrowed_to else None,
        narrowed_because=reading.narrowed_because,
    )

    # Being stuck on the question is the premise. We never test the entry node -
    # if every prerequisite below it holds, the entry node itself is the gap.
    results: dict[str, SkillResult] = {
        entry.id: SkillResult(entry.id, held=False, asked=False)
    }
    diagnosis.results.append(results[entry.id])

    # Who sent us to each skill, so we can trace a chain back up afterwards.
    parents: dict[str, str] = {}

    # Round one is the only place the two paths differ. The branch the attempt
    # pointed at goes first, with the rest of the level queued behind it, so a
    # wrong guess costs questions rather than ending the walk with nothing.
    if reading.narrowed_to:
        candidates = list(reading.narrowed_to)
        candidates += _closest_first(
            [need for need in entry.needs if need not in candidates]
        )
    else:
        candidates = _closest_first(entry.needs)

    # Depth first: test siblings until one fails, descend into it, and leave the
    # rest for a follow-up. `parent` is whatever we most recently descended from,
    # so when a whole level holds, it is the gap.
    parent = entry.id
    depth = 1
    asked = 0
    dont_knows_running = 0

    while candidates:
        broke_into = None

        for position, skill_id in enumerate(candidates):
            if skill_id in results:
                continue

            if asked >= cap:
                diagnosis.stopped_early = f"asked the maximum of {cap} questions"
                diagnosis.unchecked += [
                    later for later in candidates[position:] if later not in results
                ]
                break

            # The first skill we have no answer for is the next question. Stop
            # here and hand it back rather than asking for it ourselves.
            result = given.get(skill_id)
            if result is None:
                needed = skill_id
                parents.setdefault(skill_id, parent)
                break

            results[skill_id] = result
            diagnosis.results.append(result)
            parents.setdefault(skill_id, parent)
            asked += 1

            dont_knows_running = dont_knows_running + 1 if result.dont_know else 0

            if not result.held:
                diagnosis.deepest_failure = skill_id
                broke_into = skill_id
                # Siblings we never got to. Not cleared - just unasked.
                diagnosis.unchecked += [
                    later
                    for later in candidates[position + 1 :]
                    if later not in results
                ]
                break

        if needed or diagnosis.stopped_early:
            break

        if dont_knows_running >= dont_know_run:
            diagnosis.stopped_early = (
                f"{dont_know_run} 'I don't know' answers in a row"
            )
            break

        # Every sibling held, so nothing below explains it: `parent` is the gap.
        if broke_into is None:
            break

        # As deep as we go. Whatever just failed is the answer we hand back,
        # and the report has to say we did not look beneath it.
        if depth >= max_depth:
            diagnosis.stopped_early = (
                f"we look at most {max_depth} levels below the question"
            )
            diagnosis.unchecked += [
                need for need in SKILLS[broke_into].needs if need not in results
            ]
            break

        parent = broke_into
        depth += 1
        candidates = _closest_first(
            [need for need in SKILLS[broke_into].needs if need not in results]
        )

    # A skill that failed with every prerequisite held is a confirmed gap. Read
    # off the results rather than the traversal, so this stays true whatever
    # order we walked in. Stopping early leaves the bottom skill unconfirmed,
    # which is why this can come back empty.
    # A sibling queued as skipped can be reached later down another branch.
    # Without pruning, the report both names it as the gap and says we never
    # looked at it. Drop anything that ended up asked, and keep the order.
    seen_unchecked = set()
    diagnosis.unchecked = [
        skill_id
        for skill_id in diagnosis.unchecked
        if skill_id not in results
        and not (skill_id in seen_unchecked or seen_unchecked.add(skill_id))
    ]

    diagnosis.root_gaps = _root_gaps(results)
    diagnosis.chains = [_chain_to(gap, entry.id, parents) for gap in diagnosis.root_gaps]

    if diagnosis.narrowed_to is not None:
        diagnosis.widened = any(
            result.skill_id in entry.needs
            and result.skill_id not in diagnosis.narrowed_to
            for result in diagnosis.results
            if result.asked
        )

    return Step(ask=needed, diagnosis=diagnosis)


def diagnose(
    entry_skill_id: str,
    attempt: str | None = None,
    *,
    check: Check = check_by_asking,
    client: Anthropic | None = None,
    cap: int = QUESTION_CAP,
    dont_know_run: int = DONT_KNOW_RUN,
    max_depth: int = MAX_DEPTH,
) -> Diagnosis:
    """Run a whole walk here and now, asking as it goes.

    The terminal version: it can afford to sit in a loop waiting for someone to
    type. A server takes the same steps one request at a time - the deciding is
    all in step(), so the two cannot drift apart.
    """
    entry = SKILLS.get(entry_skill_id)
    if entry is None:
        raise ValueError(f"'{entry_skill_id}' is not a skill in the graph")

    reading = (
        read_attempt(entry, attempt, client=client) if attempt is not None else Reading()
    )

    answers: list[SkillResult] = []
    while True:
        current = step(
            entry_skill_id,
            answers,
            reading=reading,
            cap=cap,
            dont_know_run=dont_know_run,
            max_depth=max_depth,
        )
        if current.finished:
            return current.diagnosis
        answers.append(check(SKILLS[current.ask]))


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
    if diagnosis.stopped_early:
        print(f"Stopped early: {diagnosis.stopped_early}.")
        lead = diagnosis.deepest_failure
        # Only unconfirmed when something below it went unasked. A floor node
        # has nothing below it, so stopping there costs us nothing.
        if lead and lead not in diagnosis.root_gaps:
            print(
                f"  Lowest skill that failed: {SKILLS[lead].name}. Nothing "
                "underneath it was checked, so treat this as a lead rather "
                "than a diagnosis."
            )
        print()

    if diagnosis.root_gaps:
        print("Start here:")
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
    elif not diagnosis.stopped_early:
        print("No gap found below this question.")

    # One descent finds one gap. Anything the walk stepped past is unexamined,
    # not cleared, and saying so is the difference between a partial answer and
    # a wrong one.
    if diagnosis.unchecked:
        print()
        print("Not checked - any of these could hold another gap:")
        for skill_id in diagnosis.unchecked:
            print(f"  {SKILLS[skill_id].name}")
        print()
        print("Check the other branches? Re-run with one of these as the entry:")
        print(f"  python backend/walk.py {diagnosis.unchecked[0]}")


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
    parser.add_argument(
        "--cap",
        type=int,
        default=QUESTION_CAP,
        help=f"most questions to ask (default: {QUESTION_CAP})",
    )
    parser.add_argument("--student", help="anonymous reference, e.g. student_7")
    parser.add_argument("--no-save", action="store_true", help="do not record this walk")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead")
    args = parser.parse_args(argv)

    attempt = args.attempt
    if args.attempt_file:
        attempt = open(args.attempt_file, encoding="utf-8").read()

    try:
        diagnosis = diagnose(args.entry_skill_id, attempt, cap=args.cap)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        print("run 'python backend/questions.py --list' to see the ids", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(diagnosis.__dict__, indent=2, default=lambda o: o.__dict__))
    else:
        _print_diagnosis(diagnosis)

    if not args.no_save:
        save_walk(diagnosis, attempt=attempt, student_ref=args.student)

    return 0


def save_walk(diagnosis: Diagnosis, **details) -> int | None:
    """Record a finished walk, without letting a storage problem lose it.

    The diagnosis has already been shown by the time this runs. Failing to file
    it is worth a warning, not a crash that hides the answer the student waited
    for.
    """
    import store

    try:
        connection = store.connect()
        try:
            session_id = store.save_session(connection, diagnosis, **details)
        finally:
            connection.close()
        return session_id
    except Exception as error:  # noqa: BLE001 - storage must never lose a result
        print(f"\n(could not save this walk: {error})", file=sys.stderr)
        return None


if __name__ == "__main__":
    raise SystemExit(main())
