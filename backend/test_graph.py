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


def make_skill(
    skill_id: str, needs: tuple[str, ...] = (), topic: str | None = None
) -> Skill:
    """A throwaway node, for building deliberately broken graphs in tests."""
    return Skill(
        id=skill_id,
        name=skill_id.replace("_", " "),
        level=0,
        kind="concept",
        probe="probe",
        needs=needs,
        topic=topic,
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


# ---- Topics ---------------------------------------------------------------


def test_topic_marks_what_a_paper_can_ask_for():
    from graph import entry_points

    assert {s.id for s in entry_points()} == {
        s.id for s in SKILLS.values() if s.topic
    }
    assert "find_stationary_points" in {s.id for s in entry_points()}


def test_shared_skills_belong_to_no_topic():
    """The whole point: these sit underneath several topics, not inside one.

    Shorter than it was. Fraction arithmetic and factorising a common factor
    used to be here and are now GCSE doorways - not because the rule changed
    but because they are asked outright at GCSE while still being underneath
    half of A-level. Which qualification may start there is what `stage`
    settles; these four are asked by neither.
    """
    for shared in ("index_laws", "negatives", "substitute_expr", "bidmas"):
        assert SKILLS[shared].topic is None, shared


def test_a_skill_can_be_a_doorway_at_one_stage_and_a_step_at_another():
    """Expanding brackets is a GCSE question and an A-level prerequisite. If
    that were not allowed the same skill would have to exist twice, and a
    student's answer to one copy would say nothing about the other."""
    brackets = SKILLS["expand_brackets"]

    assert brackets.topic == "expanding and factorising"
    assert brackets.stage == "gcse"
    # And it is still what several A-level skills rest on.
    rests_on_it = [s.id for s in SKILLS.values() if "expand_brackets" in s.needs]
    assert "first_principles" in rests_on_it


def test_topics_lists_what_we_can_start_from():
    from graph import topics

    assert "differentiation" in topics()
    assert "indices and surds" in topics()
    # Every topic named is one some entry point actually carries.
    assert set(topics()) == {s.topic for s in SKILLS.values() if s.topic}


def test_a_graph_with_no_topics_is_rejected():
    """Nothing tagged means no question can ever be placed."""
    skills = {"a": make_skill("a")}
    with pytest.raises(SkillGraphError, match="nowhere for a question to start"):
        validate(skills)


def test_topic_is_optional_on_a_node(tmp_path):
    node = valid_node("a")
    node["topic"] = "differentiation"
    path = write_yaml(tmp_path, [node, valid_node("b")])
    skills = load_skills(path)

    assert skills["a"].topic == "differentiation"
    assert skills["b"].topic is None


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
        "top": make_skill("top", needs=("left", "right"), topic="differentiation"),
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
    entry = valid_node("a", ["b"])
    entry["topic"] = "differentiation"
    path = write_yaml(tmp_path, [entry, valid_node("b")])
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
