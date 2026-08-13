"""Tests for the skill graph loader and its validation.

Run from the repo root with:  pytest backend/
"""

import pytest
import yaml

from graph import (
    SKILLS,
    Skill,
    SkillGraphError,
    load_skills,
    validate,
)


def make_skill(skill_id: str, needs: tuple[str, ...] = ()) -> Skill:
    """A throwaway node, for building deliberately broken graphs in tests."""
    return Skill(
        id=skill_id,
        name=skill_id.replace("_", " "),
        level=0,
        kind="concept",
        probe="probe",
        needs=needs,
    )


# ---- The real file --------------------------------------------------------


def test_real_graph_loads_and_validates():
    assert len(SKILLS) > 0
    assert "differentiate_function" in SKILLS
    validate(SKILLS)  # raises if the shipped graph is broken


def test_real_graph_has_no_dangling_needs():
    for skill in SKILLS.values():
        for need in skill.needs:
            assert need in SKILLS, f"{skill.id} needs missing node {need}"


def test_real_graph_reaches_the_floor():
    """At least one node has no prerequisites, or the walk could never stop."""
    assert any(skill.needs == () for skill in SKILLS.values())


def test_fields_are_populated():
    for skill in SKILLS.values():
        assert skill.name
        assert skill.probe
        assert skill.kind in {"procedure", "fact", "concept"}
        assert isinstance(skill.level, int)


# ---- Unknown prerequisite ids ---------------------------------------------


def test_unknown_need_raises():
    skills = {
        "a": make_skill("a", needs=("b",)),
        "b": make_skill("b", needs=("nonexistent",)),
    }
    with pytest.raises(SkillGraphError) as error:
        validate(skills)
    assert "nonexistent" in str(error.value)


# ---- Circular dependencies ------------------------------------------------


def test_cycle_raises():
    skills = {
        "a": make_skill("a", needs=("b",)),
        "b": make_skill("b", needs=("c",)),
        "c": make_skill("c", needs=("a",)),
    }
    with pytest.raises(SkillGraphError) as error:
        validate(skills)
    assert "Circular dependency" in str(error.value)


def test_self_reference_raises():
    skills = {"a": make_skill("a", needs=("a",))}
    with pytest.raises(SkillGraphError) as error:
        validate(skills)
    assert "Circular dependency" in str(error.value)


def test_diamond_is_not_a_cycle():
    """Two paths to the same node is normal, and must not trip the detector."""
    skills = {
        "top": make_skill("top", needs=("left", "right")),
        "left": make_skill("left", needs=("bottom",)),
        "right": make_skill("right", needs=("bottom",)),
        "bottom": make_skill("bottom"),
    }
    assert validate(skills) is skills


# ---- Loading from a file --------------------------------------------------


def write_yaml(tmp_path, nodes):
    path = tmp_path / "skills.yaml"
    path.write_text(yaml.safe_dump({"nodes": nodes}), encoding="utf-8")
    return path


def valid_node(skill_id, needs=None):
    return {
        "id": skill_id,
        "name": skill_id,
        "level": 0,
        "kind": "concept",
        "probe": "probe",
        "needs": needs or [],
    }


def test_load_valid_file(tmp_path):
    path = write_yaml(tmp_path, [valid_node("a", ["b"]), valid_node("b")])
    skills = load_skills(path)
    assert set(skills) == {"a", "b"}
    assert skills["a"].needs == ("b",)


def test_load_file_with_cycle_raises(tmp_path):
    path = write_yaml(tmp_path, [valid_node("a", ["b"]), valid_node("b", ["a"])])
    with pytest.raises(SkillGraphError, match="Circular dependency"):
        load_skills(path)


def test_duplicate_id_raises(tmp_path):
    path = write_yaml(tmp_path, [valid_node("a"), valid_node("a")])
    with pytest.raises(SkillGraphError, match="Duplicate id"):
        load_skills(path)


def test_missing_field_raises(tmp_path):
    node = valid_node("a")
    del node["probe"]
    path = write_yaml(tmp_path, [node])
    with pytest.raises(SkillGraphError, match="probe"):
        load_skills(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(SkillGraphError, match="No skill graph"):
        load_skills(tmp_path / "does_not_exist.yaml")


def test_file_without_nodes_key_raises(tmp_path):
    path = tmp_path / "skills.yaml"
    path.write_text("meta: {topic: differentiation}\n", encoding="utf-8")
    with pytest.raises(SkillGraphError, match="nodes"):
        load_skills(path)
