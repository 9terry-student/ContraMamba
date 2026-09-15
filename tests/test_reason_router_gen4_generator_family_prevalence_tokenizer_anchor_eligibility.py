
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_reason_router_gen4_generator_family_prevalence_cohorts as cohorts
from scripts import reason_router_gen4_generator_family_prevalence_tokenizer_anchor_eligibility as gate


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        offsets = []
        cursor = 0
        for token_id, word in enumerate(text.split(" "), 1):
            start = text.index(word, cursor)
            end = start + len(word)
            ids.append(token_id)
            offsets.append((start, end))
            cursor = end
        return SimpleNamespace(ids=ids, offsets=offsets)


def test_frozen_inputs_and_contract_constants():
    assert gate.STRUCTURAL_FREEZE_COMMIT == (
        "341de2398e1c06fffa73ee323502f52972c2d58e"
    )
    assert gate.BUILDER_BLOB == "4505acc99db0627733592694da290a007d281421"
    assert gate.ANCHOR_EXPECTED_COUNTS == {"A_IDENTITY": 1200, "A_NAME": 600}
    assert gate.EXPECTED_ANCHOR_ROWS == 1800
    assert tuple(gate.FROZEN) == ("xg2", "xg3", "xg4")


@pytest.mark.parametrize("family", ["xg2", "xg4"])
def test_contiguous_identity_topology_for_supported_renderers(family):
    spec = cohorts.family_spec(family)
    fact = cohorts.build_source_facts(spec)[0]
    for cell_id in ("C0_SHAM", "C1_TITLE", "C2_NAME", "C5_TITLE_NAME"):
        overrides = gate.overrides_for_cell(fact, cell_id)
        rendered, spans = gate.realized_statement_and_spans(
            family, fact, overrides
        )
        values = {**fact, **overrides}
        identity = rendered[slice(*spans["A_IDENTITY"])]
        name = rendered[slice(*spans["A_NAME"])]
        assert identity == f"{values['title']} {values['name']}"
        assert name == values["name"]
        assert spans["A_IDENTITY"][1] == spans["A_NAME"][1]


def test_xg3_is_structurally_blocked_before_tokenizer():
    spec = cohorts.family_spec("xg3")
    facts = cohorts.build_source_facts(spec)
    failures = gate.topology_failures("xg3", facts)
    assert len(failures) == 1200
    assert {x["contrast_cell_id"] for x in failures} == {
        "C0_SHAM", "C1_TITLE", "C2_NAME", "C5_TITLE_NAME"
    }
    assert all(
        x["error"].startswith("NONCONTIGUOUS_IDENTITY_ANCHOR:xg3:")
        for x in failures
    )


def test_xg2_xg4_have_zero_static_topology_failures():
    for family in ("xg2", "xg4"):
        facts = cohorts.build_source_facts(cohorts.family_spec(family))
        assert gate.topology_failures(family, facts) == []


def test_xg3_compute_never_loads_tokenizer(monkeypatch):
    root = Path(__file__).resolve().parents[1]

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("tokenizer loader must not run for XG3")

    monkeypatch.setattr(
        gate.xg1,
        "load_canonical_analysis_tokenizer",
        forbidden_loader,
    )
    rows, summary = gate.compute_eligibility(family="xg3", root=root)
    assert rows == []
    assert summary["tokenizer_executed"] is False
    assert summary["structural_anchor_topology_failure_count"] == 1200
    assert summary["primary_complete_pair_prefix_feasibility"] == (
        "BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY"
    )
    assert summary["model_forward_count"] == 0
    assert summary["gpu_used"] is False


@pytest.mark.parametrize("family", ["xg2", "xg4"])
def test_fake_tokenizer_preserves_target_identity_name_terminal(family):
    spec = cohorts.family_spec(family)
    fact = cohorts.build_source_facts(spec)[0]
    rows = cohorts.materialize_fact(spec, fact)
    tokenizer = FakeTokenizer()

    for cell_id in ("C0_SHAM", "C2_NAME"):
        row = next(r for r in rows if r["contrast_cell_id"] == cell_id)
        events = gate.analyze_row(family, row, fact, tokenizer)
        lookup = {e["anchor_name"]: e for e in events}
        assert lookup["A_IDENTITY"]["absolute_anchor_token_index"] == (
            lookup["A_NAME"]["absolute_anchor_token_index"]
        )


def test_repository_frozen_inputs_are_exact():
    root = Path(__file__).resolve().parents[1]
    for family in gate.FROZEN:
        facts, rows, manifest = gate.load_family(family, root)
        assert len(facts) == 300
        assert len(rows) == 1800
        assert manifest["family_key"] == family


def test_source_has_no_model_cuda_or_training_dependency():
    source = Path(gate.__file__).read_text(encoding="utf-8")
    for token in (
        "import torch",
        "torch.cuda",
        "selected_checkpoint.pt",
        "load_representative_model",
        "capture_branch(",
        ".backward(",
    ):
        assert token not in source
