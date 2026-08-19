"""Measures whether the walk finds the gap it is supposed to find.

Everything else we measure is about behaviour: how many questions, how fast,
how many placed. None of it says whether a diagnosis is right, and a fluent
wrong answer is worse than no answer - a tutor acts on it and wastes a session.

This is checkable without a single student, because we can invent one. Pick a
skill and declare it the thing they are missing. A student who cannot do that
skill also cannot do anything resting on it, so those fail too. Then run the
walk and see whether it names the skill we hid.

Three numbers come out, and the third matters most:

  found        of gaps within reach, how often we name the right one
  declined     of gaps too deep to reach, how often we correctly refuse to
               confirm anything rather than inventing a nearer answer
  false        how often we name a skill the student can actually do

`false` is the dangerous one. Missing a gap costs a session; naming a skill
they already have sends a tutor to teach something the student knows, and
stops anyone looking further.

    python backend/eval.py
    python backend/eval.py --depth 5      # what a deeper walk would find
    python backend/eval.py --show-misses
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass

from graph import SKILLS, entry_points
from walk import MAX_DEPTH, SkillResult, diagnose


def _depths_below(entry_id: str, limit: int = 6) -> dict[str, int]:
    """Every skill under a doorway, and how many steps down it sits."""
    depth: dict[str, int] = {}
    frontier = [(need, 1) for need in SKILLS[entry_id].needs]
    while frontier:
        skill_id, level = frontier.pop()
        if skill_id in depth and depth[skill_id] <= level:
            continue
        depth[skill_id] = level
        if level < limit:
            frontier += [(need, level + 1) for need in SKILLS[skill_id].needs]
    return depth


def _cannot_do(missing: str) -> set[str]:
    """The missing skill, plus everything that rests on it.

    A student who cannot do index laws cannot do the things built on index
    laws either. Simulating one broken skill in isolation would be a student
    who does not exist, and would make the walk's job artificially easy.
    """
    broken = {missing}
    changed = True
    while changed:
        changed = False
        for skill in SKILLS.values():
            if skill.id not in broken and any(n in broken for n in skill.needs):
                broken.add(skill.id)
                changed = True
    return broken


@dataclass
class Trial:
    entry: str
    hidden: str
    depth: int
    reachable: bool
    named: list[str]
    stopped: str | None
    asked: int

    @property
    def found(self) -> bool:
        return self.hidden in self.named

    @property
    def declined(self) -> bool:
        """Named nothing, which is the honest answer when out of reach."""
        return not self.named

    def false_positive(self, broken: set[str]) -> list[str]:
        """Skills it named that the student can actually do."""
        return [name for name in self.named if name not in broken]


def run_trial(entry: str, hidden: str, depth: int, max_depth: int) -> tuple[Trial, set[str]]:
    broken = _cannot_do(hidden)

    def check(skill):
        if skill.id in broken:
            return SkillResult(skill.id, held=False, mistake=f"cannot do {skill.id}")
        return SkillResult(skill.id, held=True)

    diagnosis = diagnose(entry, check=check, max_depth=max_depth)
    trial = Trial(
        entry=entry,
        hidden=hidden,
        depth=depth,
        reachable=depth <= max_depth,
        named=list(diagnosis.root_gaps),
        stopped=diagnosis.stopped_early,
        asked=len([r for r in diagnosis.results if r.asked]),
    )
    return trial, broken


def run_all(max_depth: int = MAX_DEPTH) -> list[tuple[Trial, set[str]]]:
    """Hide each skill under each doorway in turn, and see what comes back."""
    results = []
    for door in entry_points():
        for skill_id, depth in _depths_below(door.id).items():
            results.append(run_trial(door.id, skill_id, depth, max_depth))
    return results


# ---- Reporting ------------------------------------------------------------


def report(results, show_misses: bool = False) -> None:
    reachable = [(t, b) for t, b in results if t.reachable]
    too_deep = [(t, b) for t, b in results if not t.reachable]

    found = [t for t, _ in reachable if t.found]
    missed = [t for t, _ in reachable if not t.found]
    declined = [t for t, _ in too_deep if t.declined]
    wrong = [(t, t.false_positive(b)) for t, b in results if t.false_positive(b)]

    print()
    print("=" * 66)
    print(f"{len(results)} trials: one per skill under each doorway")
    print()
    if reachable:
        print(
            f"  found      {len(found):>4} / {len(reachable):<4} "
            f"({len(found) / len(reachable):.0%})  gaps within reach, named correctly"
        )
    if too_deep:
        print(
            f"  declined   {len(declined):>4} / {len(too_deep):<4} "
            f"({len(declined) / len(too_deep):.0%})  too deep to reach, correctly unconfirmed"
        )
    print(
        f"  false      {len(wrong):>4} / {len(results):<4} "
        f"({len(wrong) / len(results):.0%})  named a skill the student could do"
    )
    print()
    asked = [t.asked for t, _ in results]
    print(f"  questions  {sum(asked) / len(asked):.1f} on average, {max(asked)} at worst")

    if wrong:
        print()
        print("Named a skill the student could actually do:")
        for trial, named in wrong[:10]:
            print(f"  {trial.entry} hiding {trial.hidden} -> named {named}")

    if missed:
        print()
        by_depth = defaultdict(int)
        for trial in missed:
            by_depth[trial.depth] += 1
        print("Within reach but not found, by how deep the gap was:")
        for depth in sorted(by_depth):
            print(f"  {depth} level(s) down: {by_depth[depth]}")

        if show_misses:
            print()
            for trial in missed[:20]:
                got = ", ".join(trial.named) or "(nothing)"
                print(f"  {trial.entry}: hid {trial.hidden}, got {got}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the walk finds the gap it is meant to find."
    )
    parser.add_argument("--depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args(argv)

    print(f"Running every doorway against every skill beneath it (depth {args.depth})...")
    report(run_all(args.depth), args.show_misses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
