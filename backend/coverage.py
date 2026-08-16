"""Runs a pile of real questions past the matcher to see what we cannot place.

Coverage is guesswork until you measure it. A tutor pastes a question, the tool
says it cannot help, and that is one data point at a time. This does the same
thing to fifty questions at once and hands back the two numbers that matter:
how often we place a real question, and which topics we keep failing on.

The point is not the percentage. It is the list underneath it - a backlog of
doorways to build, ordered by how often real questions actually need them,
rather than by my guesses about the specification.

Feed it a text file, one question per line:

    python backend/coverage.py questions.txt

Or a folder of screenshots, which is the better test because it is how a
student would really send a question:

    python backend/coverage.py --images data/papers
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic

from entry import MEDIA_TYPES, identify_entry, is_usable
from graph import SKILLS

# Questions are independent, so they run side by side. Serially, fifty
# questions is a quarter of an hour of watching a terminal.
WORKERS = 5


@dataclass
class Checked:
    """What happened to one question."""

    source: str
    placed: bool
    skill_id: str | None
    recognised_as: str
    confidence: str
    looks_incomplete: bool
    reason: str
    error: str | None = None

    @property
    def unreadable(self) -> bool:
        """Not a coverage gap - we could not even tell what it was."""
        return not self.placed and not self.recognised_as


@dataclass
class Report:
    checked: list[Checked]

    @property
    def placed(self) -> list[Checked]:
        return [c for c in self.checked if c.placed]

    @property
    def gaps(self) -> list[Checked]:
        """Understood, but not covered. These are doorways to build."""
        return [c for c in self.checked if not c.placed and not c.unreadable]

    @property
    def unreadable(self) -> list[Checked]:
        return [c for c in self.checked if c.unreadable and not c.error]

    @property
    def rate(self) -> float:
        usable = [c for c in self.checked if not c.error]
        return len(self.placed) / len(usable) if usable else 0.0

    def missing_topics(self) -> list[tuple[str, int]]:
        """The backlog: topics we failed to cover, commonest first."""
        counted = Counter(c.recognised_as.lower() for c in self.gaps)
        return counted.most_common()


def check_one(source: str, question: str = "", image: Path | None = None) -> Checked:
    """Put one question to the matcher and record what came back."""
    try:
        match = identify_entry(question, image, client=Anthropic())
    except Exception as error:  # noqa: BLE001 - one bad question must not stop the run
        return Checked(source, False, None, "", "", False, "", error=str(error))

    return Checked(
        source=source,
        placed=is_usable(match),
        skill_id=match.skill_id,
        recognised_as=match.recognised_as,
        confidence=match.confidence,
        looks_incomplete=match.looks_incomplete,
        reason=match.reason,
    )


def check_questions(questions: list[str], workers: int = WORKERS) -> Report:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(
            lambda pair: check_one(f"line {pair[0]}", question=pair[1]),
            list(enumerate(questions, start=1)),
        )
    return Report(list(results))


def check_images(paths: list[Path], workers: int = WORKERS) -> Report:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda p: check_one(p.name, image=p), paths)
    return Report(list(results))


def read_questions(path: Path) -> list[str]:
    """One question per line. Blank lines and # comments ignored."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def find_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in MEDIA_TYPES
    )


# ---- Reporting ------------------------------------------------------------


def print_report(report: Report) -> None:
    total = len(report.checked)
    print()
    print("=" * 64)
    print(f"Placed {len(report.placed)} of {total}   ({report.rate:.0%})")
    print()

    if report.gaps:
        print("Topics we could not cover - the backlog, commonest first:")
        for topic, times in report.missing_topics():
            print(f"  {times:>3}x  {topic}")
        print()

    if report.unreadable:
        print(f"Could not be read at all: {len(report.unreadable)}")
        for item in report.unreadable:
            print(f"  {item.source}")
        print("  (pasted maths often loses powers and fractions - try photos)")
        print()

    errors = [c for c in report.checked if c.error]
    if errors:
        print(f"Failed to run: {len(errors)}")
        for item in errors:
            print(f"  {item.source}: {item.error[:80]}")
        print()

    if report.placed:
        print("Placed, by skill:")
        counted = Counter(c.skill_id for c in report.placed)
        for skill_id, times in counted.most_common():
            print(f"  {times:>3}x  {SKILLS[skill_id].name}")


def save_gaps(report: Report) -> int:
    """File every gap so the backlog accumulates across runs."""
    import store

    connection = store.connect()
    try:
        for item in report.gaps + report.unreadable:
            store.record_unplaced(
                connection,
                item.source,
                item,
                from_image=not item.source.startswith("line "),
                role="coverage check",
            )
    finally:
        connection.close()
    return len(report.gaps) + len(report.unreadable)


# ---- Command line ---------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure how many real questions we can place."
    )
    parser.add_argument("path", help="a text file of questions, or a folder of images")
    parser.add_argument(
        "--images", action="store_true", help="treat the path as a folder of images"
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS, help=f"default: {WORKERS}"
    )
    parser.add_argument("--no-save", action="store_true", help="do not file the gaps")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    if args.images or path.is_dir():
        images = find_images(path)
        if not images:
            print(f"error: no images in {path}", file=sys.stderr)
            return 2
        print(f"Checking {len(images)} images...")
        report = check_images(images, args.workers)
    else:
        questions = read_questions(path)
        if not questions:
            print(f"error: no questions in {path}", file=sys.stderr)
            return 2
        print(f"Checking {len(questions)} questions...")
        report = check_questions(questions, args.workers)

    print_report(report)

    if not args.no_save:
        filed = save_gaps(report)
        if filed:
            print()
            print(f"Filed {filed} gap(s) to the backlog.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
