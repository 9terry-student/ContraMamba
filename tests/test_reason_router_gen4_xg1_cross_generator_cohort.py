from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as xg1
from scripts import validate_reason_router_gen4_xg1_cross_generator_cohort as validator


def test_design_identity_and_immediate_phase_constants():
    assert xg1.DESIGN_FREEZE_COMMIT == "a31d2bc5ab4b939f52e969c89f3783feb9c3b233"
    assert xg1.DESIGN_SHA256 == "21ce2dd133721766a9a18d924774b8f02aa026364e1102809ea839c3985bf403"
    assert xg1.SOURCE_PAIR_COUNT == 300
    assert xg1.EXPECTED_ROW_COUNT == 1800


def test_production_builder_has_no_historical_generator_dependency():
    source = inspect.getsource(xg1)
    for token in (
        "build_controlled_v5",
        "materialize_reason_router_gen4_six_cell_contrast",
        "FACT_TEMPLATES",
        "fact_templates_for_count",
        "_generated_fact_template",
        "._statement",
        "._paraphrase",
    ):
        assert token not in source


def test_independent_lexicon_minima():
    assert len(xg1.NAMES) >= 12
    assert len(xg1.TITLES) >= 6
    assert len(xg1.ROLES) >= 6
    assert len(xg1.PREDICATE_PAIRS) >= 7
    assert len(xg1.OBJECT_STEMS) >= 12
    assert len(xg1.LOCATIONS) >= 12
    assert len(xg1.TIMES) >= 12


def test_exact_pair_population_and_axis_inequality():
    facts = xg1.build_source_facts()
    assert len(facts) == 300
    assert [f["pair_id"] for f in facts] == [
        f"xg1_fact_{i:03d}" for i in range(1, 301)
    ]
    for fact in facts:
        assert fact["title"] != fact["alternate_title"]
        assert fact["name"] != fact["alternate_name"]
        assert fact["role"] != fact["alternate_role"]
        assert fact["predicate"] != fact["alternate_predicate"]


def test_renderer_is_exactly_frozen_surface():
    fact = {
        "time": "the review window",
        "location": "Kestrel Port",
        "title": "Envoy",
        "name": "Talia Voss",
        "role": "archive custodian",
        "predicate": "certified",
        "object": "the Quasar registry unit 001",
    }
    assert xg1.render_statement(fact) == (
        "During the review window, records from Kestrel Port identify Envoy "
        "Talia Voss as archive custodian; this person certified "
        "the Quasar registry unit 001."
    )


def test_exact_six_cell_masks_and_order():
    assert xg1.CELL_SPEC == (
        ("C0_SHAM", (0, 0, 0, 0), ()),
        ("C1_TITLE", (1, 0, 0, 0), (("title", "alternate_title"),)),
        ("C2_NAME", (0, 1, 0, 0), (("name", "alternate_name"),)),
        ("C3_ROLE", (0, 0, 1, 0), (("role", "alternate_role"),)),
        ("C4_PREDICATE", (0, 0, 0, 1), (("predicate", "alternate_predicate"),)),
        (
            "C5_TITLE_NAME",
            (1, 1, 0, 0),
            (("title", "alternate_title"), ("name", "alternate_name")),
        ),
    )


def test_materialization_has_exact_1800_rows_and_claim_invariance():
    rows = xg1.materialize_facts(xg1.build_source_facts())
    assert len(rows) == 1800
    assert len({row["row_id"] for row in rows}) == 1800
    assert not any(xg1.FORBIDDEN_FIELDS & set(row) for row in rows)
    first = rows[:6]
    assert [row["contrast_cell_id"] for row in first] == list(xg1.CELL_IDS)
    assert len({row["claim"] for row in first}) == 1


def test_authorized_cell_substitutions_are_exact():
    fact = xg1.build_source_facts()[0]
    rows = {row["contrast_cell_id"]: row for row in xg1.materialize_fact(fact)}
    assert rows["C0_SHAM"]["evidence"] == xg1.render_statement(fact)
    assert rows["C1_TITLE"]["evidence"] == xg1.render_statement(
        fact, title=fact["alternate_title"]
    )
    assert rows["C2_NAME"]["evidence"] == xg1.render_statement(
        fact, name=fact["alternate_name"]
    )
    assert rows["C3_ROLE"]["evidence"] == xg1.render_statement(
        fact, role=fact["alternate_role"]
    )
    assert rows["C4_PREDICATE"]["evidence"] == xg1.render_statement(
        fact, predicate=fact["alternate_predicate"]
    )
    assert rows["C5_TITLE_NAME"]["evidence"] == xg1.render_statement(
        fact,
        title=fact["alternate_title"],
        name=fact["alternate_name"],
    )


def test_jsonl_regeneration_is_byte_deterministic():
    facts_a = xg1.build_source_facts()
    rows_a = xg1.materialize_facts(facts_a)
    facts_b = xg1.build_source_facts()
    rows_b = xg1.materialize_facts(facts_b)
    assert xg1.jsonl_bytes(facts_a) == xg1.jsonl_bytes(facts_b)
    assert xg1.jsonl_bytes(rows_a) == xg1.jsonl_bytes(rows_b)


def test_builder_writes_only_two_prevalidation_files(tmp_path, monkeypatch):
    design = tmp_path / "design.md"
    design.write_bytes(b"x")
    monkeypatch.setattr(xg1, "ROOT", tmp_path)
    monkeypatch.setattr(xg1, "DESIGN_REL", Path("design.md"))
    monkeypatch.setattr(xg1, "DESIGN_SHA256", hashlib.sha256(b"x").hexdigest())

    output = tmp_path / "out"
    hashes = xg1.write_cohort(output)
    assert {p.name for p in output.iterdir()} == {
        xg1.SOURCE_FILE,
        xg1.ROW_FILE,
    }
    assert set(hashes) == {xg1.SOURCE_FILE, xg1.ROW_FILE}


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
    ):
        assert token not in source


def test_prior_holdout_identity_is_frozen():
    assert validator.PRIOR_HOLDOUT_SHA256 == (
        "f4289173ba3837fad217728830b8aefb8baa7dce4f45dde7c58441021b436bdc"
    )


def test_validator_manifest_schema_is_structural_only():
    assert validator.MANIFEST_SCHEMA == "GEN4_XG1_STRUCTURAL_MANIFEST_V1"


def test_forbidden_outcome_surface_is_absent_from_materialized_rows():
    rows = xg1.materialize_facts(xg1.build_source_facts())
    forbidden = {
        "target_C",
        "reference_C",
        "alignment_shift_abs",
        "regime",
        "delta_baseline",
        "R_ALIGN",
        "logits",
        "prediction",
    }
    assert all(not (forbidden & set(row)) for row in rows)


def test_output_collision_fails_closed(tmp_path, monkeypatch):
    design = tmp_path / "design.md"
    design.write_bytes(b"x")
    monkeypatch.setattr(xg1, "ROOT", tmp_path)
    monkeypatch.setattr(xg1, "DESIGN_REL", Path("design.md"))
    monkeypatch.setattr(xg1, "DESIGN_SHA256", hashlib.sha256(b"x").hexdigest())

    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(xg1.XG1BuildError, match="OUTPUT_DIR_COLLISION"):
        xg1.write_cohort(output)


def test_validator_production_builder_guard_accepts_current_source(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source_path = scripts / "build_reason_router_gen4_xg1_cross_generator_cohort.py"
    source_path.write_text(inspect.getsource(xg1), encoding="utf-8")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    validator._production_builder_independence()

def test_jsonl_round_trip_preserves_schema_semantics_after_sorted_keys():
    facts = xg1.build_source_facts()
    rows = xg1.materialize_facts(facts)

    loaded_facts = [
        json.loads(line)
        for line in xg1.jsonl_bytes(facts).decode("utf-8").splitlines()
        if line
    ]
    loaded_rows = [
        json.loads(line)
        for line in xg1.jsonl_bytes(rows).decode("utf-8").splitlines()
        if line
    ]

    xg1.validate_source_facts(loaded_facts)
    xg1.validate_materialized_rows(
        loaded_rows,
        expected_pairs=xg1.SOURCE_PAIR_COUNT,
    )
