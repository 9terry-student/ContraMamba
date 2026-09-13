from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path("scripts/reason_router_gen4_native_mamba_state_q1_q3_extraction.py")


def load():
    spec = importlib.util.spec_from_file_location("q_extraction", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def population():
    cells = ("C0_SHAM", "C1_TITLE", "C2_NAME", "C3_ROLE", "C4_PREDICATE", "C5_TITLE_NAME")
    return [
        {"source_pair_id": f"p{i:03d}", "row_id": f"p{i:03d}-{cell}", "contrast_cell_id": cell}
        for i in range(300)
        for cell in cells
    ]


def synthetic_encoded(rows):
    count = len(rows)
    input_ids = np.arange(count * 128, dtype=np.int64).reshape(count, 128) % 1000
    attention_mask = np.ones((count, 128), dtype=np.bool_)
    claim_mask = np.zeros((count, 128), dtype=np.bool_)
    evidence_mask = np.zeros((count, 128), dtype=np.bool_)
    claim_mask[:, :63] = True
    evidence_mask[:, 64:] = True
    return {
        "row_id": [row["row_id"] for row in rows],
        "source_pair_id": [row["source_pair_id"] for row in rows],
        "contrast_cell_id": [row["contrast_cell_id"] for row in rows],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "claim_mask": claim_mask,
        "evidence_mask": evidence_mask,
    }


def events_for(selected):
    return [
        {
            "source_pair_id": row["source_pair_id"],
            "row_id": row["row_id"],
            "contrast_cell_id": row["contrast_cell_id"],
            "anchor_name": "A_NAME",
            "absolute_anchor_token_index": 10,
            "terminal_index": 20,
            "post4_eligible": True,
        }
        for row in selected
    ]


def independent_coordinate_sha(encoded):
    serializable = []
    for index in range(len(encoded["row_id"])):
        serializable.append({
            "row_id": encoded["row_id"][index],
            "source_pair_id": encoded["source_pair_id"][index],
            "contrast_cell_id": encoded["contrast_cell_id"][index],
            "input_ids": encoded["input_ids"][index].tolist(),
            "attention_mask": encoded["attention_mask"][index].tolist(),
            "claim_mask": encoded["claim_mask"][index].tolist(),
            "evidence_mask": encoded["evidence_mask"][index].tolist(),
        })
    raw = json.dumps(serializable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_authority_and_frozen_provenance_binding():
    x = load()
    assert x.IMPLEMENTATION_AUTHORITY_COMMIT == "c395e634448a3b37c30053d89abef37c5a269afe"
    assert x.R2_ENCODED_COORDINATE_SHA256 == "d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a"
    assert x.LAYERS == (5, 17)
    assert x.CELLS == ("C0_SHAM", "C2_NAME")


def test_encoded_coordinate_hash_matches_frozen_r2_semantics():
    x = load()
    rows = population()
    encoded = synthetic_encoded(rows)
    assert x.encoded_coordinate_sha256(encoded) == independent_coordinate_sha(encoded)
    changed = dict(encoded)
    changed["attention_mask"] = encoded["attention_mask"].copy()
    changed["attention_mask"][0, 0] = False
    assert x.encoded_coordinate_sha256(changed) != x.encoded_coordinate_sha256(encoded)


def test_hash_gate_precedes_subset_and_model_constructor(monkeypatch):
    x = load()
    rows = population()
    encoded = synthetic_encoded(rows)
    subset_calls = []
    model_calls = []
    monkeypatch.setattr(x, "validate_subset", lambda value: subset_calls.append(len(value)))

    with pytest.raises(x.ExtractionContractError, match="encoded coordinate identity mismatch"):
        x.prepare_after_coordinate_gate(
            rows,
            lambda: encoded,
            lambda: model_calls.append(1),
            expected_sha256="0" * 64,
        )
    assert subset_calls == []
    assert model_calls == []

    expected = x.encoded_coordinate_sha256(encoded)
    selected, observed, _ = x.prepare_after_coordinate_gate(
        rows,
        lambda: encoded,
        lambda: model_calls.append(1) or object(),
        expected_sha256=expected,
    )
    assert observed is encoded
    assert len(selected) == 600
    assert subset_calls == [600]
    assert model_calls == [1]


def test_encoded_row_identity_mismatch_blocks():
    x = load()
    rows = population()
    encoded = synthetic_encoded(rows)
    encoded["row_id"][0] = "wrong-row"
    expected = x.encoded_coordinate_sha256(encoded)
    with pytest.raises(x.ExtractionContractError, match="encoded row_id ordering"):
        x.reconstruct_then_select(rows, lambda: encoded, expected_sha256=expected)


def test_events_use_absolute_anchor_and_bind_identity():
    x = load()
    selected = [row for row in population() if row["contrast_cell_id"] in x.CELLS]
    events = events_for(selected)
    bound = x.bind_events(selected, events)
    support = x.validate_execution_plan(selected, bound)
    assert len(bound) == 600
    assert len(support) == 7200
    assert {row["layer_index"] for row in support} == {5, 17}
    assert {row["support_token_index"] for row in support[:6]} == {9, 10, 11, 12, 13, 14}

    wrong = list(events)
    wrong[0] = dict(wrong[0])
    wrong[0]["source_pair_id"] = "wrong"
    with pytest.raises(x.ExtractionContractError):
        x.bind_events(selected, wrong)


def test_dual_layer_forward_plan_is_600_calls(monkeypatch):
    x = load()
    rows = population()
    encoded = synthetic_encoded(rows)
    selected = [row for row in rows if row["contrast_cell_id"] in x.CELLS]
    bound = x.bind_events(selected, events_for(selected))

    monkeypatch.setattr(
        x.measurement,
        "flatten_scientific_state",
        lambda value: np.asarray([float(value), float(value) + 1], dtype=np.float32),
    )
    monkeypatch.setattr(
        x.measurement,
        "endpoint_from_same_layer",
        lambda states, anchor, layer: {
            "POST4_SPEED": 1.0,
            "POST4_TURNING": 0.0,
            "POST4_PATH_EFFICIENCY": 1.0,
        },
    )

    calls = []
    def forward_once(_input_ids):
        calls.append(1)
        values = list(range(128))
        return {5: values, 17: values}

    support, states, endpoints, count = x.execute_dual_layer_forward_plan(
        selected,
        bound,
        encoded,
        forward_once,
    )
    assert count == 600
    assert len(calls) == 600
    assert len(support) == 7200
    assert states.shape == (7200, 2)
    assert len(endpoints) == 1200
    assert {row["layer_index"] for row in endpoints} == {5, 17}


def test_manifest_is_fail_closed_and_records_required_provenance():
    x = load()
    peer_hashes = {
        "support_state_rows.jsonl": "1" * 64,
        "support_states.npy": "2" * 64,
        "kinematic_endpoints.jsonl": "3" * 64,
    }
    manifest = x.build_manifest(
        implementation_commit="a" * 40,
        measurement_implementation_sha256="b" * 64,
        extraction_runner_sha256="c" * 64,
        runtime_versions=x.EXPECTED_RUNTIME_VERSIONS,
        peer_artifact_hashes=peer_hashes,
    )
    assert manifest["implementation_authority_commit"] == x.IMPLEMENTATION_AUTHORITY_COMMIT
    assert manifest["layer_set"] == [5, 17]
    assert manifest["semantic_anchor"] == "A_NAME"
    assert manifest["source_pair_count"] == 300
    assert manifest["model_input_row_count"] == 600
    assert manifest["backbone_forward_count"] == 600
    assert manifest["support_state_row_count"] == 7200
    assert manifest["endpoint_row_count"] == 1200
    assert manifest["peer_artifact_hashes"] == peer_hashes

    with pytest.raises(x.ExtractionContractError):
        x.build_manifest(
            implementation_commit="a" * 40,
            measurement_implementation_sha256="b" * 64,
            extraction_runner_sha256="c" * 64,
            runtime_versions={**x.EXPECTED_RUNTIME_VERSIONS, "torch": "wrong"},
            peer_artifact_hashes=peer_hashes,
        )


def test_transactional_output_contract_with_small_synthetic_bundle(monkeypatch, tmp_path):
    x = load()
    monkeypatch.setattr(x, "SUPPORT_ROWS", 12)
    monkeypatch.setattr(x, "ENDPOINT_ROWS", 2)
    monkeypatch.setattr(x, "SUPPORT_VECTOR_SIZE", 3)

    support = []
    for layer in (5, 17):
        for token in range(9, 15):
            support.append({
                "source_pair_id": "p",
                "row_id": "r",
                "contrast_cell_id": "C0_SHAM",
                "semantic_anchor": "A_NAME",
                "anchor_token_index": 10,
                "layer_index": layer,
                "support_token_index": token,
                "tensor_row_index": len(support),
            })
    endpoints = []
    for layer, start in ((5, 0), (17, 6)):
        endpoints.append({
            "source_pair_id": "p",
            "row_id": "r",
            "contrast_cell_id": "C0_SHAM",
            "semantic_anchor": "A_NAME",
            "anchor_token_index": 10,
            "layer_index": layer,
            "support_absolute_token_indices": [9, 10, 11, 12, 13, 14],
            "support_tensor_row_indices": list(range(start, start + 6)),
            "POST4_SPEED": 1.0,
            "POST4_TURNING": 0.0,
            "POST4_PATH_EFFICIENCY": 1.0,
        })

    output = tmp_path / "bundle"
    x.write_future_bundle(
        output,
        support_rows=support,
        states=np.zeros((12, 3), dtype=np.float32),
        endpoints=endpoints,
        implementation_commit="a" * 40,
        measurement_implementation_sha256="b" * 64,
        extraction_runner_sha256="c" * 64,
        runtime_versions=x.EXPECTED_RUNTIME_VERSIONS,
    )
    assert {path.name for path in output.iterdir()} == set(x.REQUIRED_ARTIFACTS)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["peer_artifact_hashes"]) == {
        "support_state_rows.jsonl",
        "support_states.npy",
        "kinematic_endpoints.jsonl",
    }
    with pytest.raises(x.ExtractionContractError):
        x.write_future_bundle(
            output,
            support_rows=support,
            states=np.zeros((12, 3), dtype=np.float32),
            endpoints=endpoints,
            implementation_commit="a" * 40,
            measurement_implementation_sha256="b" * 64,
            extraction_runner_sha256="c" * 64,
            runtime_versions=x.EXPECTED_RUNTIME_VERSIONS,
        )


def test_no_statistical_fields_in_declared_endpoint_schema():
    x = load()
    joined = " ".join(sorted(x.ENDPOINT_ROW_KEYS)).lower()
    for prohibited in ("p_value", "pvalue", "holm", "adjusted", "ttest"):
        assert prohibited not in joined
