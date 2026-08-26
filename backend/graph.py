"""Loads the skill graph from data/skills.yaml and checks it is well formed.

The v2 diagnostic walks this graph downward, so a broken graph would send it
somewhere that doesn't exist or spin forever in a loop. Both of those are
caught here, at import time, rather than in the middle of a student session.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

# The YAML lives at the repo root, one level up from backend/.
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.yaml"

# Every node must carry these. `note` is optional.
REQUIRED_FIELDS = ("id", "name", "level", "kind", "probe", "needs")


GCSE = "gcse"
A_LEVEL = "a-level"
STAGES = (GCSE, A_LEVEL)


class SkillGraphError(Exception):
    """Raised when the skill graph file is malformed."""


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    level: int
    kind: str
    probe: str
    needs: tuple[str, ...] = ()
    note: str | None = None
    # The kind of question this is, on the skills a paper can actually ask for.
    # Shared skills like index laws sit underneath several topics and belong to
    # none of them, so they carry no topic at all.
    topic: str | None = None
    # Which qualification asks it. On doorways only, and for the same reason as
    # topic: expanding brackets is asked outright at GCSE and is a prerequisite
    # at A-level, so the question is not what a skill belongs to but what kind
    # of question is allowed to start there.
    #
    # It is also what lets GCSE be watched separately while it is unproven -
    # a session's stage is the stage of the doorway it entered by.
    stage: str = A_LEVEL


def _parse_nodes(raw_nodes: list) -> dict[str, Skill]:
    """Turn the raw YAML list into {id: Skill}, rejecting bad or duplicate rows."""
    skills: dict[str, Skill] = {}

    for position, raw in enumerate(raw_nodes, start=1):
        if not isinstance(raw, dict):
            raise SkillGraphError(f"Node {position} is not a mapping: {raw!r}")

        stage = raw.get("stage", A_LEVEL)
        if stage not in STAGES:
            raise SkillGraphError(
                f"{raw.get('id', f'node {position}')} has stage {stage!r}; "
                f"expected one of {', '.join(STAGES)}"
            )

        missing = [field for field in REQUIRED_FIELDS if field not in raw]
        if missing:
            label = raw.get("id", f"node {position}")
            raise SkillGraphError(f"'{label}' is missing field(s): {', '.join(missing)}")

        skill_id = raw["id"]
        if skill_id in skills:
            raise SkillGraphError(f"Duplicate id: '{skill_id}'")

        needs = raw["needs"] or []
        if not isinstance(needs, list):
            raise SkillGraphError(f"'{skill_id}' has a needs field that is not a list")

        skills[skill_id] = Skill(
            id=skill_id,
            name=raw["name"],
            level=raw["level"],
            kind=raw["kind"],
            probe=raw["probe"],
            needs=tuple(needs),
            note=raw.get("note"),
            topic=raw.get("topic"),
            stage=stage,
        )

    return skills


def entry_points(skills: dict[str, Skill] | None = None) -> list[Skill]:
    """The skills a question can ask for.

    Carrying a topic is what makes a skill an entry point - not its level.
    Level is a hint for ordering, and the graph's own notes warn against
    treating it as a strict hierarchy, so whether a paper can ask for something
    is written down rather than inferred.
    """
    skills = SKILLS if skills is None else skills
    return sorted((s for s in skills.values() if s.topic), key=lambda s: s.id)


def topics(skills: dict[str, Skill] | None = None) -> list[str]:
    """Every kind of question the graph can currently start from."""
    skills = SKILLS if skills is None else skills
    return sorted({s.topic for s in skills.values() if s.topic})


def _find_unknown_needs(skills: dict[str, Skill]) -> list[str]:
    """Every id in a needs list must be a node we actually have."""
    problems = []
    for skill in skills.values():
        for need in skill.needs:
            if need not in skills:
                problems.append(f"'{skill.id}' needs '{need}', which is not a node")
    return problems


def _find_cycle(skills: dict[str, Skill]) -> list[str] | None:
    """Depth-first search for a loop. Returns the cycle as a path, or None.

    Each node is unvisited, on the current path, or fully explored. Meeting a
    node that is already on the current path means we've come back round to it.
    """
    UNVISITED, ON_PATH, DONE = 0, 1, 2
    state = {skill_id: UNVISITED for skill_id in skills}
    path: list[str] = []

    def visit(skill_id: str) -> list[str] | None:
        state[skill_id] = ON_PATH
        path.append(skill_id)

        for need in skills[skill_id].needs:
            if need not in skills:
                continue  # unknown ids are reported separately
            if state[need] == ON_PATH:
                return path[path.index(need):] + [need]
            if state[need] == UNVISITED:
                cycle = visit(need)
                if cycle:
                    return cycle

        path.pop()
        state[skill_id] = DONE
        return None

    for skill_id in skills:
        if state[skill_id] == UNVISITED:
            cycle = visit(skill_id)
            if cycle:
                return cycle

    return None


def validate(skills: dict[str, Skill]) -> dict[str, Skill]:
    """Raise SkillGraphError if the graph has dangling ids or a cycle."""
    unknown = _find_unknown_needs(skills)
    if unknown:
        raise SkillGraphError("Unknown prerequisite ids:\n  " + "\n  ".join(unknown))

    cycle = _find_cycle(skills)
    if cycle:
        raise SkillGraphError("Circular dependency: " + " -> ".join(cycle))

    # With nothing tagged, no question can ever be placed and the whole graph is
    # unreachable, so this is a broken file rather than an empty one.
    if not entry_points(skills):
        raise SkillGraphError(
            "No skill carries a topic, so there is nowhere for a question to "
            "start. Tag the skills a paper can ask for."
        )

    return skills


def load_skills(path: Path = DEFAULT_PATH) -> dict[str, Skill]:
    """Read the YAML, build the Skill objects, and validate the whole graph."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SkillGraphError(f"No skill graph at {path}")

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise SkillGraphError(f"{path} is not valid YAML: {error}") from error

    if not isinstance(document, dict) or "nodes" not in document:
        raise SkillGraphError(f"{path} has no top-level 'nodes' list")

    return validate(_parse_nodes(document["nodes"]))


# Loaded once, when this module is first imported. If the graph is broken the
# import fails loudly instead of the app starting up with a bad graph.
SKILLS: dict[str, Skill] = load_skills()
