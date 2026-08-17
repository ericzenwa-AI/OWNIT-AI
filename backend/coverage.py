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

from pydantic import BaseModel

from entry import MEDIA_TYPES, attachment_block, identify_entry, is_usable
from graph import SKILLS
from questions import MAX_TOKENS, MODEL

# Questions are independent, so they run side by side. Serially, fifty
# questions is a quarter of an hour of watching a terminal.
WORKERS = 5


class PaperQuestion(BaseModel):
    """One question lifted off a paper."""

    number: str
    text: str
    recognised_as: str


class Paper(BaseModel):
    questions: list[PaperQuestion]


def read_paper(
    pdf: Path, *, client: Anthropic | None = None, model: str = MODEL
) -> list[PaperQuestion]:
    """Pull every question out of a past paper, maths intact.

    The model reads the PDF itself rather than us scraping its text layer -
    scraping is what collapses a fractional power into a stray number, and a
    question whose maths has been destroyed tells us nothing about coverage.
    """
    client = client or Anthropic()

    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=(
            "You transcribe A-level maths exam papers. You copy questions out "
            "faithfully, including the mathematics, and never solve them."
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    attachment_block(pdf),
                    {
                        "type": "text",
                        "text": (
                            "List every question in this paper.\n"
                            "- One entry per question. Keep lettered parts "
                            "together in the same entry, as they appear.\n"
                            "- Copy the wording, and write the maths in plain "
                            "ASCII: / for division, ^ for powers, sqrt() for "
                            "roots, * for multiplication. Getting the maths "
                            "right matters more than the exact wording.\n"
                            "- Skip the front cover, formula sheets and blank "
                            "pages.\n"
                            "- In `recognised_as`, name the topic in two or "
                            "three words.\n"
                            "- Do not answer anything."
                        ),
                    },
                ],
            }
        ],
        output_format=Paper,
    )

    paper = response.parsed_output
    return paper.questions if paper else []


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
    parser.add_argument(
        "path", help="a text file, a folder of images, or a past paper PDF"
    )
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

    if path.suffix.lower() == ".pdf":
        print(f"Reading {path.name}...")
        questions = read_paper(path)
        if not questions:
            print(f"error: could not read any questions from {path}", file=sys.stderr)
            return 2
        print(f"Found {len(questions)} questions. Checking...")
        report = check_questions([q.text for q in questions], args.workers)
        for item, question in zip(report.checked, questions):
            item.source = f"Q{question.number}"
    elif args.images or path.is_dir():
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
