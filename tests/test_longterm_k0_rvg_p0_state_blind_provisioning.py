from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "longterm_k0_rvg_p0_state_blind_provisioning.py"

spec = importlib.util.spec_from_file_location("k0_rvg_p0", SCRIPT)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_authority_and_scope_constants():
    assert m.AUTHORITY_COMMIT == "ec4230c735be4355531bd46c50ef0571693cb641"
    assert m.PARENT_PREREG_COMMIT == "cdad87acf664cd61e48406f9d4568b6ab206da24"
    assert m.FRESH_START == 1236
    assert m.FRESH_STOP == 1572
    assert m.FRESH_N == 336
    assert m.PHASE_PERIOD == 168
    assert m.SOURCE_ROW_COUNT == 4368
    assert m.SCRIPT_REL == "scripts/longterm_k0_rvg_p0_state_blind_provisioning.py"
    assert m.TEST_REL == "tests/test_longterm_k0_rvg_p0_state_blind_provisioning.py"


def test_generator_identity_and_explicit_template_count():
    generator = m.load_generator(ROOT)
    assert len(generator.FACT_TEMPLATES) == 30
    assert m.file_sha256(ROOT / m.GENERATOR_REL) == m.GENERATOR_SHA256


def test_exact_fresh_range_source_count_and_pair_ids():
    generator = m.load_generator(ROOT)
    source_rows, candidates, templates = m.build_source_and_candidates(generator)
    assert len(source_rows) == 4368
    assert len(candidates) == 336
    assert len(templates) == 336
    assert candidates[0]["pair_id"] == "generated_fact_1237"
    assert candidates[-1]["pair_id"] == "generated_fact_1572"
    assert [c["local_template_index"] for c in candidates] == list(range(336))


def test_phase_structure_and_correction_balance():
    generator = m.load_generator(ROOT)
    _, candidates, templates = m.build_source_and_candidates(generator)
    counts = m.validate_phase_structure(candidates, templates)
    assert counts == {"polarity_flip": 168, "none": 168}
    for p in range(168):
        expected_phase = (m.FRESH_START + p - m.EXPLICIT_TEMPLATE_COUNT) % m.PHASE_PERIOD
        assert candidates[p]["generator_phase_class"] == expected_phase
        assert candidates[p + 168]["generator_phase_class"] == expected_phase
        assert {
            candidates[p]["correction_source_intervention"],
            candidates[p + 168]["correction_source_intervention"],
        } == {"polarity_flip", "none"}


def test_phase_mapping_exact_indices():
    generator = m.load_generator(ROOT)
    _, candidates, _ = m.build_source_and_candidates(generator)
    mapping = m.phase_mapping(candidates)
    assert len(mapping["blocks"]) == 168
    assert mapping["blocks"][0]["item_a_local_index"] == 0
    assert mapping["blocks"][0]["item_b_local_index"] == 168
    assert mapping["blocks"][0]["phase_class"] == 30
    assert mapping["blocks"][-1]["item_a_local_index"] == 167
    assert mapping["blocks"][-1]["item_b_local_index"] == 335


def test_candidate_stable_id_reproducible():
    generator = m.load_generator(ROOT)
    _, a, _ = m.build_source_and_candidates(generator)
    _, b, _ = m.build_source_and_candidates(generator)
    assert [x["stable_item_id"] for x in a] == [x["stable_item_id"] for x in b]


def test_malformed_source_row_multiplicity_fails_closed(monkeypatch):
    generator = m.load_generator(ROOT)
    original = generator._build_records

    def bad(templates):
        rows = original(templates)
        return rows[:-1]

    monkeypatch.setattr(generator, "_build_records", bad)
    with pytest.raises(m.ContractError, match="SOURCE_ROW_COUNT_MISMATCH"):
        m.build_source_and_candidates(generator)


def test_prior_pool_sha_and_zero_overlap():
    generator = m.load_generator(ROOT)
    _, candidates, _ = m.build_source_and_candidates(generator)
    pools = m.authenticate_prior_pools(ROOT)
    counts = m.overlap_audit(candidates, pools)
    assert len(counts) == 12
    assert set(counts.values()) == {0}


def test_overlap_fails_closed():
    generator = m.load_generator(ROOT)
    _, candidates, _ = m.build_source_and_candidates(generator)
    fake = [{
        "pair_id": candidates[0]["pair_id"],
        "prefix_text": candidates[0]["prefix_text"],
        "base_claim_sha256": candidates[0]["base_claim_sha256"],
    }]
    with pytest.raises(m.ContractError, match="PRIOR_POOL_OVERLAP"):
        m.overlap_audit(candidates, {"fake": fake})


def test_canonical_jsonl_is_lf_only_and_deterministic():
    rows = [{"b": 2, "a": 1}, {"x": "한글"}]
    a = m.canonical_jsonl_bytes(rows)
    b = m.canonical_jsonl_bytes(rows)
    assert a == b
    assert a.endswith(b"\n")
    assert b"\r" not in a
    assert not a.startswith(b"\xef\xbb\xbf")


def test_divergence_anchor():
    assert m.divergence_anchor([1, 2, 3, 9, 5], [1, 2, 3, 8, 5], 3) == 3
    with pytest.raises(m.ContractError, match="DIVERGENCE_NOT_WITHIN_FIRST_8"):
        m.divergence_anchor([1, 2, 3], [1, 2, 3], 3)


class FakeTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        # Deterministic character IDs preserve exact prefix identity.
        return {"input_ids": [ord(ch) for ch in text]}


def test_token_contract_fixture_prefix_divergence_w8():
    corr = "c" + "x" * 20
    ctrl = "d" + "y" * 20
    base = {
        "schema_version": m.CANDIDATE_SCHEMA,
        "generator_sha256": m.GENERATOR_SHA256,
        "global_template_index": 1236,
        "local_template_index": 0,
        "generator_phase_class": 0,
        "cycle_in_slice": 0,
        "pair_id": "a",
        "truncation_source_id": "a__evidence_truncation",
        "correction_source_id": "a__polarity_flip",
        "correction_source_intervention": "polarity_flip",
        "control_source_id": "a__entity_swap",
        "prefix_text": "Claim: A\nEvidence: E\nAdditional evidence:\n",
        "correction_text": corr,
        "control_text": ctrl,
        "base_claim_sha256": "0" * 64,
        "stable_item_id": "sid-a",
    }
    candidates = []
    for i in range(336):
        item = dict(base)
        item["local_template_index"] = i
        item["generator_phase_class"] = i % 168
        item["pair_id"] = f"p{i}"
        item["stable_item_id"] = f"sid-{i}"
        item["correction_text"] = corr
        item["control_text"] = ctrl
        candidates.append(item)
    rows, matched, swapped, minimum = m.token_contracts(candidates, FakeTokenizer())
    assert len(rows) == 336
    assert rows[0]["phase_block_index"] == 0
    assert rows[168]["phase_block_index"] == 0
    assert rows[167]["phase_block_index"] == 167
    assert rows[335]["phase_block_index"] == 167
    assert matched == {0: 336}
    assert swapped == {0: 336}
    assert minimum >= 8
    assert all(row["matched_w8_available"] and row["swapped_w8_available"] for row in rows)


def test_token_contract_w8_failure():
    base = {
        "generator_phase_class": 0,
        "pair_id": "p",
        "stable_item_id": "s",
        "prefix_text": "P\n",
        "correction_text": "a",
        "control_text": "b",
    }
    candidates = []
    for i in range(336):
        item = dict(base)
        item["generator_phase_class"] = i % 168
        item["pair_id"] = f"p{i}"
        item["stable_item_id"] = f"s{i}"
        candidates.append(item)
    with pytest.raises(m.ContractError, match="MATCHED_W8_UNAVAILABLE"):
        m.token_contracts(candidates, FakeTokenizer())


def test_no_model_checkpoint_observer_or_torch_imports():
    source = SCRIPT.read_text("utf-8")
    tree = ast.parse(source)
    imports = []
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not any(x == "torch" or x.startswith("torch.") for x in imports)
    assert not any("longterm_k0_rvg_raw_recurrence_observer" in x for x in imports)
    assert "load_authenticated_checkpoint" not in called
    assert "build_a0_model" not in called
    assert "_full_model_forward" not in called
    assert "_logits" not in called


def test_cli_exposes_state_blind_only():
    parser = m.build_parser()
    options = {s for action in parser._actions for s in action.option_strings}
    assert "--provision" in options
    assert "--output-dir" in options
    assert "--validate-repeat" in options
    for forbidden in (
        "--scientific",
        "--evaluate",
        "--model",
        "--checkpoint",
        "--observer",
        "--intervene",
    ):
        assert forbidden not in options


def test_candidate_claim_hash_matches_prefix_claim():
    generator = m.load_generator(ROOT)
    _, candidates, _ = m.build_source_and_candidates(generator)
    for item in candidates[:10]:
        claim = m.claim_from_prefix(item["prefix_text"])
        assert m.claim_sha256(claim) == item["base_claim_sha256"]
