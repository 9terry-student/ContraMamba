from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from scripts import build_reason_router_gen4_generator_family_prevalence_cohorts as cohorts
from scripts import validate_reason_router_gen4_generator_family_prevalence_cohorts as validator


def test_design_identity_and_immediate_phase_constants():
    assert cohorts.DESIGN_FREEZE_COMMIT == (
        "4b6f0c831ebc89f0e786e1eb526739c7c9c06413"
    )
    assert cohorts.DESIGN_SHA256 == (
        "ffc1705ef5c68ff0a737bee088d93285f18f4093e0127dc94f99fc60be7b53d5"
    )
    assert cohorts.FAMILY_KEYS == ("xg2", "xg3", "xg4")
    assert cohorts.SOURCE_PAIR_COUNT == 300
    assert cohorts.EXPECTED_ROW_COUNT == 1800


def test_production_builder_has_no_historical_or_xg1_generator_dependency():
    source = inspect.getsource(cohorts)
    for token in (
        "build_controlled_v5",
        "materialize_reason_router_gen4_six_cell_contrast",
        "build_reason_router_gen4_xg1_cross_generator_cohort",
        "FACT_TEMPLATES",
        "fact_templates_for_count",
        "_generated_fact_template",
        "._statement",
        "._paraphrase",
    ):
        assert token not in source


def test_each_family_has_independent_lexicon_minima_and_distinct_renderer_schedule():
    renderers = set()
    schedules = set()
    configured_sets = {}
    for key in cohorts.FAMILY_KEYS:
        spec = cohorts.family_spec(key)
        assert len(spec.names) >= 12
        assert len(spec.titles) >= 6
        assert len(spec.roles) >= 6
        assert len(spec.predicate_pairs) >= 7
        assert len(spec.object_stems) >= 12
        assert len(spec.locations) >= 12
        assert len(spec.times) >= 12
        renderers.add(spec.renderer)
        schedules.add(spec.schedule)
        configured_sets[key] = {
            value.casefold()
            for value in cohorts.configured_inventory_values(spec)
        }

    assert len(renderers) == 3
    assert len(schedules) == 3
    for left, right in (("xg2", "xg3"), ("xg2", "xg4"), ("xg3", "xg4")):
        assert configured_sets[left].isdisjoint(configured_sets[right])


def test_exact_pair_populations_and_axis_inequality():
    for key in cohorts.FAMILY_KEYS:
        spec = cohorts.family_spec(key)
        facts = cohorts.build_source_facts(spec)
        assert len(facts) == 300
        assert [f["pair_id"] for f in facts] == [
            f"{key}_fact_{i:03d}" for i in range(1, 301)
        ]
        for fact in facts:
            assert fact["title"] != fact["alternate_title"]
            assert fact["name"] != fact["alternate_name"]
            assert fact["role"] != fact["alternate_role"]
            assert fact["predicate"] != fact["alternate_predicate"]


def test_three_renderers_are_surface_distinct():
    rendered = []
    for key in cohorts.FAMILY_KEYS:
        spec = cohorts.family_spec(key)
        fact = cohorts.build_source_facts(spec)[0]
        rendered.append(cohorts.render_statement(spec, fact))
    assert len(set(rendered)) == 3
    assert rendered[0].startswith("At ")
    assert rendered[1].startswith("A dispatch dated ")
    assert rendered[2].startswith("From ")


def test_exact_six_cell_masks_and_order():
    assert cohorts.CELL_SPEC == (
        ("C0_SHAM", (0, 0, 0, 0), ()),
        ("C1_TITLE", (1, 0, 0, 0), (("title", "alternate_title"),)),
        ("C2_NAME", (0, 1, 0, 0), (("name", "alternate_name"),)),
        ("C3_ROLE", (0, 0, 1, 0), (("role", "alternate_role"),)),
        (
            "C4_PREDICATE",
            (0, 0, 0, 1),
            (("predicate", "alternate_predicate"),),
        ),
        (
            "C5_TITLE_NAME",
            (1, 1, 0, 0),
            (("title", "alternate_title"), ("name", "alternate_name")),
        ),
    )


def test_materialization_exact_counts_claim_invariance_and_no_outcomes():
    for key in cohorts.FAMILY_KEYS:
        spec = cohorts.family_spec(key)
        rows = cohorts.materialize_facts(
            spec,
            cohorts.build_source_facts(spec),
        )
        assert len(rows) == 1800
        assert len({row["row_id"] for row in rows}) == 1800
        assert not any(cohorts.FORBIDDEN_FIELDS & set(row) for row in rows)
        first = rows[:6]
        assert [row["contrast_cell_id"] for row in first] == list(
            cohorts.CELL_IDS
        )
        assert len({row["claim"] for row in first}) == 1


def test_authorized_cell_substitutions_are_exact_for_every_family():
    for key in cohorts.FAMILY_KEYS:
        spec = cohorts.family_spec(key)
        fact = cohorts.build_source_facts(spec)[0]
        rows = {
            row["contrast_cell_id"]: row
            for row in cohorts.materialize_fact(spec, fact)
        }
        assert rows["C0_SHAM"]["evidence"] == cohorts.render_statement(spec, fact)
        assert rows["C1_TITLE"]["evidence"] == cohorts.render_statement(
            spec,
            fact,
            title=fact["alternate_title"],
        )
        assert rows["C2_NAME"]["evidence"] == cohorts.render_statement(
            spec,
            fact,
            name=fact["alternate_name"],
        )
        assert rows["C3_ROLE"]["evidence"] == cohorts.render_statement(
            spec,
            fact,
            role=fact["alternate_role"],
        )
        assert rows["C4_PREDICATE"]["evidence"] == cohorts.render_statement(
            spec,
            fact,
            predicate=fact["alternate_predicate"],
        )
        assert rows["C5_TITLE_NAME"]["evidence"] == cohorts.render_statement(
            spec,
            fact,
            title=fact["alternate_title"],
            name=fact["alternate_name"],
        )


def test_jsonl_regeneration_is_byte_deterministic_for_all_families():
    for key in cohorts.FAMILY_KEYS:
        spec = cohorts.family_spec(key)
        facts_a = cohorts.build_source_facts(spec)
        rows_a = cohorts.materialize_facts(spec, facts_a)
        facts_b = cohorts.build_source_facts(spec)
        rows_b = cohorts.materialize_facts(spec, facts_b)
        assert cohorts.jsonl_bytes(facts_a) == cohorts.jsonl_bytes(facts_b)
        assert cohorts.jsonl_bytes(rows_a) == cohorts.jsonl_bytes(rows_b)


def test_generated_source_inventory_and_rendered_strings_are_cross_family_disjoint():
    values = {}
    claims = {}
    evidence = {}
    for key in cohorts.FAMILY_KEYS:
        spec = cohorts.family_spec(key)
        facts = cohorts.build_source_facts(spec)
        rows = cohorts.materialize_facts(spec, facts)
        values[key] = {
            str(fact[field]).casefold()
            for fact in facts
            for field in (
                "title",
                "name",
                "role",
                "predicate",
                "object",
                "time",
                "location",
                "alternate_title",
                "alternate_name",
                "alternate_role",
                "alternate_predicate",
            )
        }
        claims[key] = {str(row["claim"]) for row in rows}
        evidence[key] = {str(row["evidence"]) for row in rows}

    for left, right in (("xg2", "xg3"), ("xg2", "xg4"), ("xg3", "xg4")):
        assert values[left].isdisjoint(values[right])
        assert claims[left].isdisjoint(claims[right])
        assert evidence[left].isdisjoint(evidence[right])


def test_builder_writes_only_six_prevalidation_files(tmp_path, monkeypatch):
    design = tmp_path / "design.md"
    design.write_bytes(b"x")
    monkeypatch.setattr(cohorts, "ROOT", tmp_path)
    monkeypatch.setattr(cohorts, "DESIGN_REL", Path("design.md"))
    monkeypatch.setattr(
        cohorts,
        "DESIGN_SHA256",
        hashlib.sha256(b"x").hexdigest(),
    )

    output = tmp_path / "out"
    hashes = cohorts.write_cohorts(output)
    assert set(hashes) == {"xg2", "xg3", "xg4"}
    assert {p.name for p in output.iterdir()} == {"xg2", "xg3", "xg4"}
    for key in cohorts.FAMILY_KEYS:
        assert {p.name for p in (output / key).iterdir()} == {
            cohorts.SOURCE_FILE,
            cohorts.ROW_FILE,
        }


def test_output_collision_fails_closed(tmp_path, monkeypatch):
    design = tmp_path / "design.md"
    design.write_bytes(b"x")
    monkeypatch.setattr(cohorts, "ROOT", tmp_path)
    monkeypatch.setattr(cohorts, "DESIGN_REL", Path("design.md"))
    monkeypatch.setattr(
        cohorts,
        "DESIGN_SHA256",
        hashlib.sha256(b"x").hexdigest(),
    )

    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(
        cohorts.PrevalenceCohortBuildError,
        match="OUTPUT_DIR_COLLISION",
    ):
        cohorts.write_cohorts(output)


def test_validator_is_static_only_by_source_inspection():
    source = inspect.getsource(validator)
    for token in (
        "torch.",
        "torch.cuda",
        "AutoTokenizer",
        "load_representative_model",
        "capture_branch(",
        "run_full(",
        "R_ALIGN_LARGE",
        "alignment_model_forward_count",
    ):
        assert token not in source


def test_validator_reference_identities_are_frozen():
    assert validator.PRIOR_HOLDOUT_SHA256 == (
        "f4289173ba3837fad217728830b8aefb8baa7dce4f45dde7c58441021b436bdc"
    )
    assert validator.XG1_SOURCE_SHA256 == (
        "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0"
    )
    assert validator.XG1_ROW_SHA256 == (
        "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f"
    )
    assert validator.XG1_STRUCTURAL_MANIFEST_SHA256 == (
        "f7c881dd1a4a400e600eea4b03b46e05a48bf0da07d29905462fc3543d8af822"
    )


def test_validator_manifest_schemas_are_structural_only():
    assert validator.FAMILY_MANIFEST_SCHEMA == (
        "GEN4_PREVALENCE_FAMILY_STRUCTURAL_MANIFEST_V1"
    )
    assert validator.CROSS_MANIFEST_SCHEMA == (
        "GEN4_PREVALENCE_CROSS_FAMILY_STRUCTURAL_MANIFEST_V1"
    )


def test_validator_production_builder_guard_accepts_current_source(
    monkeypatch,
    tmp_path,
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    path = (
        scripts
        / "build_reason_router_gen4_generator_family_prevalence_cohorts.py"
    )
    path.write_text(inspect.getsource(cohorts), encoding="utf-8")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    validator._production_builder_independence()


def test_cross_family_validator_rejects_overlap():
    good = {}
    for key in cohorts.FAMILY_KEYS:
        spec = cohorts.family_spec(key)
        facts = cohorts.build_source_facts(spec)
        rows = cohorts.materialize_facts(spec, facts)
        good[key] = {
            "values": {
                str(fact["name"])
                for fact in facts
            },
            "claims": {str(row["claim"]) for row in rows},
            "evidence": {str(row["evidence"]) for row in rows},
        }

    tampered = {
        key: {
            "values": set(item["values"]),
            "claims": set(item["claims"]),
            "evidence": set(item["evidence"]),
        }
        for key, item in good.items()
    }
    tampered["xg3"]["values"].add(next(iter(tampered["xg2"]["values"])))
    with pytest.raises(
        validator.PrevalenceStructuralValidationError,
        match="CROSS_FAMILY_EXACT_INVENTORY:xg2:xg3",
    ):
        validator._validate_cross_family(tampered)
