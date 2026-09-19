"""What a skill actually means, in words a student uses.

The graph has three things to say about a skill and none of them is an
explanation. `probe` says what a question testing it must require - that is for
writing questions. `practice` says what to go and do about it - that is a task.
Neither says what the thing IS, so a student who does not know is told to go and
practise something they cannot name.

That was tolerable while this was aimed at tutors, who supply the explaining
themselves. It is the whole product when it is not.

THE SHAPE

Five beats, because they do five different jobs:

  plain    the idea said out loud to somebody who does not do maths, with no
           symbols in it at all. This is the beat that has to land - a student
           who reads it should think "oh, THAT'S what that means", and if they
           read nothing else on the page it still has to have been worth
           opening. Everything after it is detail.
  like     the same idea as something that happens in real life. Optional, and
           deliberately so: a forced analogy teaches a wrong model and is worse
           than none, so it is left empty wherever nothing honest fits.
  what     what it is, in the words a person would use out loud. Not a
           definition - a definition is what you read when you already know.
  example  one concrete case, small numbers, worked. The thing itself rather
           than a description of the thing.
  wrong    what students actually do instead, and why it survives. This is the
           beat nothing else in the product has, and it is the one a tutor would
           have supplied: "reading 3/4 as three and four" is worth more than any
           amount of correct exposition, because it names the thing the student
           is probably doing right now.

`plain` and `like` were added after the first 172 were written and checked, and
are written on their own (--beats) rather than by rewriting the three that were
already confirmed accurate. The order above is the order the finding page tells
it in: what it means, what it is like, what it is, one worked case, the trap.

Written per skill rather than per student's question. A skill does not change,
so one explanation can be checked once and served free forever - the same trade
the question bank makes, and that one has held: 831 questions, every answer key
confirmed by a model that did not write it.

    python backend/explain.py                 # what is written, what is missing
    python backend/explain.py --fill          # write the missing ones
    python backend/explain.py --check         # verify with another model
    python backend/explain.py --show surds    # read one
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import llm
from graph import SKILLS, Skill

load_dotenv(Path(__file__).resolve().parent / ".env")

EXPLAIN_FILE = Path(__file__).resolve().parent.parent / "data" / "explanations.json"
# Not committed - it is the record of one check run, not part of the product.
CHECK_FILE = Path(__file__).resolve().parent / "explanations_checked.json"

BATCH = 6
WORKERS = 5

# Same model as the bank and the practice lines, for the same reason: the
# writing is the easy half and being right about the maths is the hard one.
# 16000 is the ceiling for a non-streaming request - the SDK refuses anything
# that could run past ten minutes. Six skills at three short fields each is
# nowhere near it, and streaming for this would be machinery for nothing.
WRITE = llm.Task(llm.SONNET, effort="medium")

# Judgement about someone else's writing, so not the model that wrote it.
CHECK = llm.Task(llm.SONNET, effort="medium")


class Explanation(BaseModel):
    skill_id: str
    what: str = Field(description="What it is, in plain spoken words. One or two sentences.")
    example: str = Field(description="One small worked case. Plain ASCII maths.")
    wrong: str = Field(
        description="What students do instead, and why it survives. One or two sentences.")


class Batch(BaseModel):
    explanations: list[Explanation]


class Beats(BaseModel):
    """The two that have to land before any maths does."""

    skill_id: str
    plain: str = Field(
        description="The idea said out loud to a ten year old. No symbols, no "
                    "equals sign, no powers. One or two short sentences.")
    like: str = Field(
        default="",
        description="The same idea as something in real life. EMPTY if nothing "
                    "honest fits - a forced comparison is worse than none.")


class BeatsBatch(BaseModel):
    beats: list[Beats]


class Judgement(BaseModel):
    problem: str = Field(default="", description="What is wrong with it. Empty if fine.")
    accurate: bool = Field(description="True if the maths, including the example, is right.")
    plain: bool = Field(description="True if a struggling student could read it unaided.")
    real_mistake: bool = Field(
        description="True if `wrong` names something students actually do, not a strawman.")


class BeatsJudgement(BaseModel):
    """Whether the two landing beats land, and whether they are honest.

    An analogy is the dangerous one. A comparison that nearly fits is worse
    than no comparison, because the student keeps it and reasons with it long
    after the lesson, and the places it breaks are exactly the harder
    questions. So it is judged on whether it maps, not on whether it reads
    nicely.
    """

    problem: str = Field(default="", description="What is wrong with it. Empty if fine.")
    ten_year_old: bool = Field(
        description="True if a ten year old who does not do this maths would "
                    "understand `plain` on one read, with no symbols in it.")
    true_to_the_maths: bool = Field(
        description="True if `plain` is actually correct, not just simple. "
                    "Simplified past the point of being true fails this.")
    analogy_holds: bool = Field(
        description="True if `like` genuinely maps onto the maths, or is empty. "
                    "A comparison that breaks where the maths does not fails.")


SYSTEM_WRITE = (
    "You explain one maths skill to the student who has just been told it is "
    "the thing standing between them and the question they were stuck on.\n"
    "They are on their own, probably on a phone, possibly late, and possibly "
    "years behind. They did not ask for a lesson - they asked for help with a "
    "question - so every word has to earn its place.\n"
    "Write the way a good tutor talks, not the way a textbook writes. Short "
    "sentences. No 'simply', no 'just', no 'obviously', no encouragement, and "
    "nothing that implies they should already know this."
)


def _describe(skill: Skill) -> str:
    return (f"id: {skill.id}\nname: {skill.name}\nkind: {skill.kind}\n"
            f"what a question on it must require: {skill.probe}\n")


def build_prompt(skills: list[Skill]) -> str:
    listed = "\n".join(_describe(s) for s in skills)
    return (
        "Explain each of these.\n\n" + listed + "\n"
        "FOR EACH, THREE PARTS\n"
        "- `what`: what it is, in the words someone would say out loud. One or "
        "two sentences. Not a definition - a definition is what you read when "
        "you already know. If there is an idea underneath it, that idea is what "
        "goes here.\n"
        "- `example`: one small worked case with real numbers, short enough to "
        "follow in one read. Show the thing rather than describe it. Check it "
        "comes out right before you write it down.\n"
        "- `wrong`: what students actually DO instead, and why it survives. Be "
        "specific - name the wrong answer they get. This is the most useful "
        "part, so it must be a real mistake a teacher would recognise, not an "
        "invented one. If a wrong method still gets easy questions right, say "
        "so, because that is why nobody has caught it.\n\n"
        "RULES\n"
        "- A 'concept' is understood, a 'procedure' is done, a 'fact' is known. "
        "Explain each accordingly.\n"
        "- Plain ASCII maths: / for division, ^ for powers, sqrt() for roots. "
        "No LaTeX, no characters outside ASCII.\n"
        "- Do not mention the diagnosis, the gap, the other skills, or this "
        "system. They are reading about the thing itself.\n"
        "- Use the id exactly as given."
    )


# Structured output has leaked the model's own working into a field before -
# a fragment of JSON, a note about the schema, once the words "No wait". A
# checker reading for meaning waves those through, because the sentence in
# front of the rubbish is fine.
SCRATCH = re.compile(r"}\]|'\]|':|\\|\bper schema\b|\bLet's write\b|\bNo wait\b", re.I)
WELDED = re.compile(r"(?<![a-z])(?<!\d)\.(?!\d)[a-z]{3,}")


def _tidy(text: str) -> str:
    """Trim junk welded onto the end, without eating the whole field.

    The first version trimmed backwards to the last sentence end. That is safe
    for a practice line, which always finishes with a search phrase in quotes,
    and destroys an example - `d/dx(3x^5) = 15x^4` contains no full stop at all,
    so every character got stripped and the field came back empty. Six of them
    shipped that way before the checker caught it.
    """
    text = " ".join((text or "").split())
    trimmed = text
    while trimmed and trimmed[-1] not in '.!?"':
        trimmed = trimmed[:-1].rstrip()
    return trimmed or text


def looks_wrong(one: dict) -> str | None:
    """Why this must never be shown, or None if it is safe."""
    for field in ("what", "example", "wrong"):
        body = one.get(field) or ""
        if not body.strip():
            return f"{field} is empty"
        if any(ord(c) > 127 for c in body):
            return f"{field} is not plain ASCII"
        if SCRATCH.search(body):
            return f"{field} carries the model's own scratch text"
        if WELDED.search(body):
            return f"{field} has a full stop welded to the next word"
        # 450 rather than a round 400. The cap is about a student on a phone
        # not being handed a wall, and at that size the difference between
        # the two is nothing - whereas the best-written `wrong` in the set,
        # on writing an expression from words, came in at 403 and would have
        # been thrown away to satisfy a number that was a guess.
        if len(body) > 450:
            return f"{field} is too long at {len(body)} characters"

    # The two landing beats, when they are there. Entries written before they
    # existed are not broken, they are unfinished - `beats_missing` is what
    # reports those, so that a skill can never be BOTH waved through here and
    # invisible there.
    for field, cap in (("plain", 300), ("like", 360)):
        body = one.get(field)
        if body is None:
            continue
        if any(ord(c) > 127 for c in body):
            return f"{field} is not plain ASCII"
        if SCRATCH.search(body):
            return f"{field} carries the model's own scratch text"
        if WELDED.search(body):
            return f"{field} has a full stop welded to the next word"
        if len(body) > cap:
            return f"{field} is too long at {len(body)} characters"

    # `plain` earns its place by having no maths on it. A student who could
    # read symbols did not need this beat, and the one who could not is handed
    # the same wall a second time.
    said = one.get("plain")
    if said is not None:
        if not said.strip():
            return "plain is empty"
        found = [c for c in ("^", "=", "<", ">", "*") if c in said]
        if found:
            return f"plain carries maths notation: {' '.join(found)}"
    return None


def beats_missing(one: dict) -> bool:
    """Written before the landing beats existed, so the finding page opens on
    the maths rather than on the meaning."""
    return not (one.get("plain") or "").strip()


def write_batch(skills: list[Skill], client: Anthropic | None = None) -> dict:
    client = client or Anthropic()
    response = client.messages.parse(
        **WRITE.kwargs(), system=SYSTEM_WRITE,
        messages=[{"role": "user", "content": build_prompt(skills)}],
        output_format=Batch)

    parsed = response.parsed_output
    if parsed is None:
        return {}
    wanted = {s.id for s in skills}
    return {
        one.skill_id: {
            "what": _tidy(one.what),
            "example": _tidy(one.example),
            "wrong": _tidy(one.wrong),
        }
        for one in parsed.explanations if one.skill_id in wanted
    }


# The console this is run from is often cp1252, which cannot print a root sign
# or a minus sign - and this file is full of both. A print that raises after 147
# calls have been made throws away work that has already been paid for.
def _say(line: str = "") -> None:
    try:
        print(line)
    except UnicodeEncodeError:
        print(str(line).encode("ascii", "replace").decode("ascii"))


def load() -> dict:
    if not EXPLAIN_FILE.exists():
        return {}
    with EXPLAIN_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def save(written: dict) -> None:
    EXPLAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EXPLAIN_FILE.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(written.items())), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


_loaded: dict | None = None


def for_skill(skill_id: str) -> dict | None:
    """The explanation for one skill, or None. Never raises - a missing
    explanation must not cost a student their diagnosis."""
    global _loaded
    if _loaded is None:
        try:
            _loaded = load()
        except Exception:  # noqa: BLE001
            _loaded = {}
    return _loaded.get(skill_id)


def fill(skill_ids: list[str], workers: int = WORKERS) -> dict:
    batches = [[SKILLS[s] for s in skill_ids[i:i + BATCH]]
               for i in range(0, len(skill_ids), BATCH)]
    written: dict = {}
    client = Anthropic()

    def work(batch):
        try:
            return write_batch(batch, client)
        except Exception as error:  # noqa: BLE001 - one bad batch must not stop the run
            print(f"  batch failed ({error})", file=sys.stderr)
            return {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(work, batches):
            written.update(got)
            print(f"  {len(written)}/{len(skill_ids)}")

    # Every batch failing looks exactly like every batch returning nothing, and
    # the difference matters: one is a bug and the other is a model having a bad
    # day. Saying so beats writing an empty file and reporting success.
    if skill_ids and not written:
        raise RuntimeError(
            f"all {len(batches)} batches came back empty - nothing was written. "
            "The errors above say why.")
    return written


# ---- The two landing beats -------------------------------------------------

SYSTEM_BEATS = (
    "A student has just been told which one thing is standing between them and "
    "the question they were stuck on. You write the first two lines they read "
    "about it.\n"
    "They are sixteen, on a phone, and they have just found out they are "
    "missing something. They are not stupid and they are not a child - but the "
    "words have to be ones a ten year old would follow, because whatever they "
    "were taught in those words did not stick, and saying it again the same "
    "way will not work either.\n"
    "Aim for the moment where somebody goes 'oh - THAT'S what that means'. "
    "That is the entire job.\n"
    "Never say 'simply', 'just', 'obviously', 'of course', or 'all you have to "
    "do is'. Never imply they should already know it. Do not praise them, do "
    "not encourage them, and do not mention this system."
)


def build_beats_prompt(skills: list[Skill], written: dict) -> str:
    listed = []
    for skill in skills:
        one = written.get(skill.id) or {}
        listed.append(
            f"id: {skill.id}\nname: {skill.name}\nkind: {skill.kind}\n"
            f"what it is: {one.get('what', skill.probe)}\n"
            f"worked example: {one.get('example', '')}\n"
            f"what goes wrong: {one.get('wrong', '')}\n")

    return (
        "For each of these, write the two lines that come BEFORE the "
        "explanation already written under it.\n\n" + "\n".join(listed) + "\n"
        "`plain` - the idea said out loud, to somebody who does not do maths.\n"
        "- No symbols at all. No equals sign, no ^, no <, no >, no *. Numbers "
        "in words where you can.\n"
        "- Say what the thing MEANS, not what you do with it. 'Cut into five "
        "equal pieces and take three' means something; 'the numerator over the "
        "denominator' does not.\n"
        "- One or two short sentences. It has to be true as well as simple: "
        "simplified past the point of being correct is the one failure that "
        "matters here, because they will build on it.\n\n"
        "`like` - the same idea as something that actually happens in life.\n"
        "- Money, food, time, distance, music, sport, queues, recipes, phone "
        "storage, splitting a bill. Things a sixteen year old has touched.\n"
        "- It has to MAP. Say which part of the real thing is which part of "
        "the maths. A picture that breaks where the maths does not is worse "
        "than nothing, because they will keep it and use it on a harder "
        "question.\n"
        "- LEAVE IT EMPTY if nothing honest fits. Some things are not like "
        "anything, and reaching for a comparison to fill a box is how students "
        "end up believing something false. An empty `like` is a correct "
        "answer and there is no penalty for it.\n\n"
        "RULES\n"
        "- Plain ASCII only, in both.\n"
        "- British English, and British life: pounds not dollars, metres, "
        "maths not math, travelled not traveled.\n"
        "- Do not mention the diagnosis, the other skills, or this system.\n"
        "- Use the id exactly as given."
    )


def write_beats_batch(skills: list[Skill], written: dict,
                      client: Anthropic | None = None) -> dict:
    client = client or Anthropic()
    response = client.messages.parse(
        **WRITE.kwargs(), system=SYSTEM_BEATS,
        messages=[{"role": "user", "content": build_beats_prompt(skills, written)}],
        output_format=BeatsBatch)

    parsed = response.parsed_output
    if parsed is None:
        return {}
    wanted = {s.id for s in skills}
    return {
        one.skill_id: {"plain": _tidy(one.plain), "like": _tidy(one.like)}
        for one in parsed.beats if one.skill_id in wanted
    }


def add_beats(skill_ids: list[str], workers: int = WORKERS) -> dict:
    """Write plain and like onto explanations that already exist.

    The three maths beats are left exactly as they are. They have been checked
    by a model that did not write them, and rewriting them to add two fields
    would throw that away and pay for it again.
    """
    written = load()
    batches = [[SKILLS[s] for s in skill_ids[i:i + BATCH]]
               for i in range(0, len(skill_ids), BATCH)]
    got: dict = {}
    client = Anthropic()

    def work(batch):
        try:
            return write_beats_batch(batch, written, client)
        except Exception as error:  # noqa: BLE001 - one bad batch must not stop the run
            print(f"  batch failed ({error})", file=sys.stderr)
            return {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done in pool.map(work, batches):
            got.update(done)
            print(f"  {len(got)}/{len(skill_ids)}")

    if skill_ids and not got:
        raise RuntimeError(
            f"all {len(batches)} batches came back empty - nothing was written.")
    return got


SYSTEM_CHECK_BEATS = (
    "You are checking the first two lines a struggling student reads about the "
    "one thing standing between them and their question. You did not write "
    "them. Say plainly what is wrong before deciding."
)


def check_beats_one(skill: Skill, one: dict,
                    client: Anthropic | None = None) -> BeatsJudgement:
    client = client or Anthropic()
    like = one.get("like") or ""
    response = client.messages.parse(
        **CHECK.kwargs(), system=SYSTEM_CHECK_BEATS,
        messages=[{"role": "user", "content": (
            f"THE SKILL\n{_describe(skill)}\n"
            f"THE ACCURATE EXPLANATION OF IT\n{one.get('what', '')}\n\n"
            f"SAID PLAINLY\n{one.get('plain', '')}\n\n"
            f"COMPARED TO REAL LIFE\n{like or '(left empty)'}\n\n"
            "Decide three things.\n"
            "- ten_year_old: would a ten year old who does not do this maths "
            "follow the plain line on one read? Any symbol, any word they "
            "would have to look up, fails it.\n"
            "- true_to_the_maths: is the plain line actually CORRECT, or has "
            "it been simplified past the point of being true? This is the "
            "failure that matters, because they will build on it.\n"
            "- analogy_holds: does the real-life comparison genuinely map onto "
            "the maths, with the right part of the picture standing for the "
            "right part of the maths? A comparison that breaks where the maths "
            "does not is worse than none. An empty one passes - leaving it out "
            "is a correct answer."
        )}],
        output_format=BeatsJudgement)
    return response.parsed_output or BeatsJudgement(
        problem="the checker returned nothing",
        ten_year_old=False, true_to_the_maths=False, analogy_holds=False)


def check_beats_all(written: dict, workers: int = WORKERS) -> list:
    client = Anthropic()
    items = [(s, one) for s, one in sorted(written.items()) if s in SKILLS]

    def work(item):
        skill_id, one = item
        try:
            return skill_id, check_beats_one(SKILLS[skill_id], one, client)
        except Exception as error:  # noqa: BLE001
            return skill_id, BeatsJudgement(
                problem=f"check failed: {error}",
                ten_year_old=False, true_to_the_maths=False, analogy_holds=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, items))


# ---- Checking --------------------------------------------------------------

SYSTEM_CHECK = (
    "You are checking an explanation of one maths skill, written for a student "
    "who has just been told this is what is standing between them and their "
    "question. You did not write it. Say plainly what is wrong before deciding."
)


def check_one(skill: Skill, one: dict, client: Anthropic | None = None) -> Judgement:
    client = client or Anthropic()
    response = client.messages.parse(
        **CHECK.kwargs(), system=SYSTEM_CHECK,
        messages=[{"role": "user", "content": (
            f"THE SKILL\n{_describe(skill)}\n"
            f"WHAT IT SAYS IT IS\n{one['what']}\n\n"
            f"THE EXAMPLE\n{one['example']}\n\n"
            f"WHAT IT SAYS GOES WRONG\n{one['wrong']}\n\n"
            "Decide three things.\n"
            "- accurate: is the maths right, including the example? An example "
            "that does not come out to what it claims is the worst failure "
            "here, because it is the part a student will copy.\n"
            "- plain: could a student who does NOT know this read it unaided? "
            "Jargon they would have to look up fails this.\n"
            "- real_mistake: is the mistake one students actually make, that a "
            "teacher would recognise? An invented or strawman error fails."
        )}],
        output_format=Judgement)
    return response.parsed_output or Judgement(
        problem="the checker returned nothing",
        accurate=False, plain=False, real_mistake=False)


def check_all(written: dict, workers: int = WORKERS) -> list:
    client = Anthropic()
    items = [(s, one) for s, one in sorted(written.items()) if s in SKILLS]

    def work(item):
        skill_id, one = item
        try:
            return skill_id, check_one(SKILLS[skill_id], one, client)
        except Exception as error:  # noqa: BLE001
            return skill_id, Judgement(
                problem=f"check failed: {error}",
                accurate=False, plain=False, real_mistake=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, items))


# ---- Command line ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explanations, one per skill.")
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--all", action="store_true", help="rewrite every one")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--beats", action="store_true",
                        help="write the plain and like beats where missing")
    parser.add_argument("--check-beats", action="store_true",
                        help="verify those two with another model")
    parser.add_argument("--show")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)

    written = load()

    if args.show:
        one = written.get(args.show)
        if not one:
            _say(f"Nothing written for '{args.show}'.")
            return 1
        for field in ("plain", "like", "what", "example", "wrong"):
            if one.get(field) is not None:
                _say(f"{field:>8}: {one[field] or '(empty)'}")
        return 0

    if args.beats:
        todo = sorted(s for s in SKILLS
                      if s in written and (args.all or beats_missing(written[s])))
        if not todo:
            _say("Every explanation already has both. Use --all to rewrite them.")
            return 0

        _say(f"Writing the plain and like beats for {len(todo)}...")
        got = add_beats(todo, args.workers)
        for skill_id, beats in got.items():
            written[skill_id] = {**written[skill_id], **beats}

        broken = {s: why for s in got if (why := looks_wrong(written[s]))}
        for skill_id in broken:
            # Never save something that must not be shown. Dropping the two
            # new fields leaves the explanation exactly as it was.
            written[skill_id].pop("plain", None)
            written[skill_id].pop("like", None)

        save(written)
        empty = [s for s in got if not (written[s].get("like") or "")]
        _say(f"\nWrote {len(got) - len(broken)}. Left {len(empty)} without a "
             "real-life comparison, which is allowed.")
        if broken:
            _say(f"{len(broken)} came back unusable and were left alone:")
            for skill_id, why in broken.items():
                _say(f"  [{why}] {skill_id}")
        return 0

    if args.check_beats:
        have = {s: one for s, one in written.items() if not beats_missing(one)}
        if not have:
            _say("Nothing to check - run --beats first.")
            return 1

        _say(f"Checking {len(have)} on {CHECK.model}...")
        judged = check_beats_all(have, args.workers)

        out = CHECK_FILE.with_name("beats_checked.json")
        out.write_text(
            json.dumps({s: v.model_dump() for s, v in judged}, indent=2),
            encoding="utf-8")
        _say(f"(verdicts saved to {out.name})")

        bad = [(s, v) for s, v in judged
               if not (v.ten_year_old and v.true_to_the_maths and v.analogy_holds)]
        _say()
        if not bad:
            _say(f"All {len(have)} read plainly, stayed true, and none of the "
                 "comparisons breaks where the maths does not.")
            return 0
        for skill_id, verdict in bad:
            flags = ",".join(name for name, ok in (
                ("not plain enough", verdict.ten_year_old),
                ("not true", verdict.true_to_the_maths),
                ("analogy breaks", verdict.analogy_holds)) if not ok)
            _say(f"  [{flags}] {SKILLS[skill_id].name}")
            _say(f"      -> {verdict.problem[:150]}")
        return 1

    if args.check:
        if not written:
            _say("Nothing written yet.")
            return 1

        broken = {s: why for s, one in written.items() if (why := looks_wrong(one))}
        if broken:
            _say(f"{len(broken)} are malformed and must not ship:")
            for skill_id, why in broken.items():
                _say(f"  [{why}] {skill_id}")
            _say()

        _say(f"Checking {len(written)} explanations on {CHECK.model}...")
        judged = check_all(written, args.workers)

        # To disk first. A check that cannot be printed is still a check that
        # was paid for, and the answer is worth more than the screen.
        CHECK_FILE.write_text(
            json.dumps({s: v.model_dump() for s, v in judged}, indent=2),
            encoding="utf-8")
        _say(f"(verdicts saved to {CHECK_FILE.name})")

        bad = [(s, v) for s, v in judged
               if not (v.accurate and v.plain and v.real_mistake)]

        _say()
        if not bad and not broken:
            _say(f"All {len(written)} were confirmed accurate, plain, and about "
                  "a mistake students really make.")
            return 0
        for skill_id, verdict in bad:
            flags = ",".join(name for name, ok in (
                ("inaccurate", verdict.accurate), ("not plain", verdict.plain),
                ("strawman mistake", verdict.real_mistake)) if not ok)
            _say(f"  [{flags}] {SKILLS[skill_id].name}")
            _say(f"      -> {verdict.problem[:150]}")
        return 1

    if not args.fill:
        missing = [s for s in SKILLS if s not in written]
        _say(f"{len(written)} written, {len(missing)} of {len(SKILLS)} missing")
        for skill_id in missing[:12]:
            _say(f"  {SKILLS[skill_id].name}")
        if len(missing) > 12:
            _say(f"  ... and {len(missing) - 12} more")
        return 0

    wanted = list(SKILLS) if args.all else [s for s in SKILLS if s not in written]
    if not wanted:
        _say("Every skill already has one. Use --all to rewrite them.")
        return 0

    _say(f"Writing {len(wanted)} explanations...")
    written.update(fill(wanted, args.workers))
    save(written)
    _say(f"\nWrote {EXPLAIN_FILE.name}. Run --check before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
