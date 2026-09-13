from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "reason_router_gen4_native_mamba_state_extraction.py"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


r = load_module("reason_router_gen4_native_mamba_state_extraction_test_target")


def test_frozen_parent_and_dependency_constants():
    assert r.PHASE_D_RUNNER_IMPLEMENTATION_AUTHORITY == "66f56152e97445429eb5b329e0fc849e7cc81492"
    assert r.SOURCE_ROLE_VALIDATOR_CORRECTION_AUTHORITY == "5ee0826c629978a72240bb88f732aacf6f357c4b"
    assert r.KAGGLE_TORCH_BUILD_TAG_CORRECTION_AUTHORITY == "9608e80efced2f6029bb4265eabf8e46df8d638c"
    assert r.BACKEND_DISPATCH_VALIDATOR_CORRECTION_AUTHORITY == "94c23155b8422d23ae01c43c0b63c5f696bd52d1"
    assert r.EXECUTION_AUTHORITY_PATH == Path(
        "reports/reason_router_gen4_native_mamba_state_bridge_"
        "phase_d_extraction_execution_authority_backend_dispatch_correction_spec_candidate.md"
    )
    assert r.PHASE_C_MEASUREMENT_IMPLEMENTATION_COMMIT == "e3c870e7f24e183b0046b568e1de3b71446c182d"
    assert r.MEASUREMENT_SHA256 == "3699cc7a61d777a2dee3c8bdde9c87933e268579d79622a9c519969584c6fdc5"
    assert r.ADAPTER_SHA256 == "83177c351f82a781586c63bd8d4ef1b40e759b5a94858dc1837d65502cbff6e5"
    assert r.R5_SHA256 == "468a758a7d20d048c75a0ca7e298b73a65f538527df55d3ecad3c7ff1760cf4d"
    assert r.HISTORICAL_MODEL_SHA256 == "8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5"


def test_representative_checkpoint_contract():
    assert r.REPRESENTATIVE_SEED == 180
    assert r.REPRESENTATIVE_ARM == "G3-GROUP-D-HALF"
    assert r.REPRESENTATIVE_CHECKPOINT_SHA256 == "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
    assert r.SCIENTIFIC_FORWARD_BATCH_SIZE == 1
    assert r.EXPECTED_BACKBONE_FORWARD_COUNT == 1800
    assert r.MODEL_REPLICATION_COUNT == 1


def test_actual_frozen_event_manifest_schema_and_counts():
    rows = r.load_event_manifest(ROOT / r.EVENT_MANIFEST)
    assert len(rows) == 3600
    assert min(row["absolute_anchor_token_index"] for row in rows) >= 1
    assert all(row["post4_eligible"] is True for row in rows)


def test_validate_execution_binding_rejects_wrong_head(monkeypatch):
    monkeypatch.setattr(r, "git_head", lambda: "0" * 40)
    with pytest.raises(r.ExtractionContractError, match="execution HEAD mismatch"):
        r.validate_execution_binding(
            expected_execution_head="1" * 40,
            runner_implementation_commit="2" * 40,
            expected_runner_sha256="3" * 64,
            execution_authority_commit="1" * 40,
        )


def test_execution_authority_must_equal_execution_head(monkeypatch):
    monkeypatch.setattr(r, "git_head", lambda: "1" * 40)
    with pytest.raises(r.ExtractionContractError, match=r.BLOCKED_EXECUTION_AUTHORITY):
        r.validate_execution_binding(
            expected_execution_head="1" * 40,
            runner_implementation_commit="2" * 40,
            expected_runner_sha256="3" * 64,
            execution_authority_commit="4" * 40,
        )



def test_runner_implementation_authority_cannot_authorize_scientific_execution(monkeypatch):
    monkeypatch.setattr(r, "git_head", lambda: r.PHASE_D_RUNNER_IMPLEMENTATION_AUTHORITY)
    with pytest.raises(r.ExtractionContractError, match=r.BLOCKED_EXECUTION_AUTHORITY):
        r.validate_execution_binding(
            expected_execution_head=r.PHASE_D_RUNNER_IMPLEMENTATION_AUTHORITY,
            runner_implementation_commit="2" * 40,
            expected_runner_sha256="3" * 64,
            execution_authority_commit=r.PHASE_D_RUNNER_IMPLEMENTATION_AUTHORITY,
        )

def test_runtime_gate_occurs_before_scientific_model_work(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(r, "validate_execution_binding", lambda **kwargs: events.append("binding") or "1" * 40)

    def gate():
        events.append("runtime")
        raise RuntimeError("stop-after-runtime")

    def inputs(_snapshot):
        events.append("inputs")
        raise AssertionError("inputs must not run")

    def model(_snapshot):
        events.append("model")
        raise AssertionError("model must not run")

    with pytest.raises(RuntimeError, match="stop-after-runtime"):
        r.run_scientific_extraction(
            output_dir=tmp_path / "out",
            expected_execution_head="1" * 40,
            runner_implementation_commit="2" * 40,
            expected_runner_sha256="3" * 64,
            execution_authority_commit="1" * 40,
            runtime_gate=gate,
            input_loader=inputs,
            model_loader=model,
        )

    assert events == ["binding", "runtime"]


def test_load_representative_model_uses_only_frozen_seed_arm(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(r.adapter, "expected_checkpoint_sha256", lambda seed, arm: r.REPRESENTATIVE_CHECKPOINT_SHA256 if (seed, arm) == (180, "G3-GROUP-D-HALF") else (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(r.r5, "expected_checkpoint_path", lambda seed, arm: r.ROOT / r.REPRESENTATIVE_CHECKPOINT)

    class FakeModel:
        def __init__(self):
            self.mamba = SimpleNamespace(parameters=lambda: [])

    def fake_loader(**kwargs):
        calls.append((kwargs["seed"], kwargs["arm"], str(kwargs["device"])))
        return FakeModel(), r.REPRESENTATIVE_CHECKPOINT_SHA256

    monkeypatch.setattr(r.r5, "load_evaluator_model", fake_loader)
    model, digest = r.load_representative_model(tmp_path)
    assert digest == r.REPRESENTATIVE_CHECKPOINT_SHA256
    assert calls == [(180, "G3-GROUP-D-HALF", "cpu")]
    assert model is not None



def test_checkpoint_authentication_wrapper_precedes_loader():
    import inspect
    source = inspect.getsource(r.adapter.authenticated_checkpoint_load)
    assert source.index("authenticate_checkpoint") < source.index("loader(path)")


def test_corrected_worktree_dependency_identity():
    measurement_bytes = (
        ROOT / r.MEASUREMENT_PATH
    ).read_bytes().replace(
        b"\r\n",
        b"\n",
    )

    assert (
        r.sha256_bytes(measurement_bytes)
        == r.MEASUREMENT_SHA256
    )

    unchanged = {
        r.ADAPTER_PATH:
            r.ADAPTER_SHA256,
        r.R5_PATH:
            r.R5_SHA256,
        r.HISTORICAL_MODEL_PATH:
            r.HISTORICAL_MODEL_SHA256,
    }

    head = r.git_head()

    for path, digest in unchanged.items():
        assert (
            r.canonical_git_sha256(
                head,
                path,
            )
            == digest
        )


def test_manifest_provenance_records_correction_authorities():
    source = (
        ROOT / r.RUNNER_PATH
    ).read_text(
        encoding="utf-8"
    )

    assert (
        '"source_role_validator_correction_authority": '
        "SOURCE_ROLE_VALIDATOR_CORRECTION_AUTHORITY,"
        in source
    )

    assert (
        '"kaggle_torch_build_tag_correction_authority": '
        "KAGGLE_TORCH_BUILD_TAG_CORRECTION_AUTHORITY,"
        in source
    )

    assert (
        '"backend_dispatch_validator_correction_authority": '
        "BACKEND_DISPATCH_VALIDATOR_CORRECTION_AUTHORITY,"
        in source
    )

def test_support_window_exact():
    event = {
        "absolute_anchor_token_index": 7,
        "terminal_index": 20,
    }
    assert r.support_indices_for_events([event]) == (6, 7, 8, 9, 10, 11)


def test_support_union_sorted_unique():
    events = [
        {"absolute_anchor_token_index": 5, "terminal_index": 20},
        {"absolute_anchor_token_index": 7, "terminal_index": 20},
    ]
    assert r.support_indices_for_events(events) == (4, 5, 6, 7, 8, 9, 10, 11)


def test_active_terminal_prefix_rule_blocks_even_with_128_padding():
    event = {
        "absolute_anchor_token_index": 10,
        "terminal_index": 14,
    }
    with pytest.raises(r.ExtractionContractError, match="active-terminal prefix rule"):
        r.support_indices_for_events([event])


def _mini_rows():
    rows = []
    for pair in ("p1", "p2"):
        for cell in r.CANONICAL_CELLS:
            rows.append({
                "row_id": f"{pair}-{cell}",
                "source_pair_id": pair,
                "contrast_cell_id": cell,
                "claim": "c",
                "evidence": "e",
            })
    return rows


def _mini_events():
    rows = []
    required = r.REQUIRED_ANCHOR_CELLS
    for pair in ("p1", "p2"):
        for anchor_index, anchor in enumerate(r.ANCHOR_ORDER, start=3):
            for cell in required[anchor]:
                rows.append({
                    "source_pair_id": pair,
                    "row_id": f"{pair}-{cell}",
                    "contrast_cell_id": cell,
                    "anchor_name": anchor,
                    "absolute_anchor_token_index": anchor_index,
                    "terminal_index": 20,
                })
    return rows


def test_endpoint_ordering_is_pair_anchor_cell(monkeypatch):
    monkeypatch.setattr(r, "SOURCE_PAIR_COUNT", 2)
    monkeypatch.setattr(r, "REQUIRED_ANCHOR_CELL_ROWS", len(_mini_events()))
    ordered = r.ordered_events(_mini_rows(), _mini_events())
    observed = [(x["source_pair_id"], x["anchor_name"], x["contrast_cell_id"]) for x in ordered]
    expected = []
    for pair in ("p1", "p2"):
        for anchor in r.ANCHOR_ORDER:
            for cell in r.CANONICAL_CELLS:
                if cell in r.REQUIRED_ANCHOR_CELLS[anchor]:
                    expected.append((pair, anchor, cell))
    assert observed == expected


def _linear_states(count=12):
    return [np.array([float(i), 0.0], dtype=np.float32) for i in range(count)]


def test_endpoint_computation_known_linear():
    event = {"absolute_anchor_token_index": 2, "terminal_index": 10}
    result = r.endpoint_from_states(_linear_states(), event)
    assert result["POST4_SPEED"] == pytest.approx(1.0)
    assert result["POST4_TURNING"] == pytest.approx(0.0)
    assert result["POST4_PATH_EFFICIENCY"] == pytest.approx(1.0)


def test_zero_transition_blocker_propagates():
    states = _linear_states()
    states[3] = states[2].copy()
    event = {"absolute_anchor_token_index": 2, "terminal_index": 10}
    with pytest.raises(r.measurement.ContractError, match=r.measurement.BLOCKED_ZERO_TRANSITION):
        r.endpoint_from_states(states, event)


def test_zero_path_blocker_propagates():
    states = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(12)]
    with pytest.raises(
        r.measurement.ContractError,
        match=r.measurement.BLOCKED_ZERO_PATH,
    ):
        r.measurement.post4_path_efficiency(states, anchor=2)


def test_nonfinite_measurement_blocks():
    states = _linear_states()
    states[4] = np.array([np.nan, 0.0], dtype=np.float32)
    event = {"absolute_anchor_token_index": 2, "terminal_index": 10}
    with pytest.raises((r.measurement.ContractError, r.ExtractionContractError)):
        r.endpoint_from_states(states, event)


def test_endpoint_row_support_references_exact_window():
    event = {
        "source_pair_id": "p",
        "row_id": "r",
        "contrast_cell_id": "C0_SHAM",
        "anchor_name": "A_TITLE",
        "absolute_anchor_token_index": 4,
        "terminal_index": 20,
    }
    index_map = {("r", token): i for i, token in enumerate(range(3, 9))}
    result = {"POST4_SPEED": 1.0, "POST4_TURNING": 0.0, "POST4_PATH_EFFICIENCY": 1.0}
    row = r.endpoint_row(event, index_map, result)
    assert row["support_absolute_token_indices"] == [3, 4, 5, 6, 7, 8]
    assert row["support_vector_indices"] == [0, 1, 2, 3, 4, 5]


def test_reconstruction_matches_endpoint_definition():
    vectors = np.asarray([[float(i), 0.0] for i in range(6)], dtype=np.float32)
    endpoint = {
        "support_vector_indices": list(range(6)),
    }
    result = r.reconstruct_endpoint_from_support(endpoint, vectors)
    assert result["POST4_SPEED"] == pytest.approx(1.0)
    assert result["POST4_TURNING"] == pytest.approx(0.0)
    assert result["POST4_PATH_EFFICIENCY"] == pytest.approx(1.0)


def test_statistical_fields_prohibited():
    row = {
        "schema_version": r.ENDPOINT_ROW_SCHEMA,
        "source_pair_id": "p",
        "row_id": "r",
        "contrast_cell_id": "C0_SHAM",
        "anchor_name": "A_TITLE",
        "anchor_token_index": 4,
        "active_terminal_index": 20,
        "layer_index": r.PRIMARY_LAYER,
        "support_absolute_token_indices": [3, 4, 5, 6, 7, 8],
        "support_vector_indices": [0, 1, 2, 3, 4, 5],
        "POST4_SPEED": 1.0,
        "POST4_TURNING": 0.0,
        "POST4_PATH_EFFICIENCY": 1.0,
        "p_value": 0.1,
    }
    with pytest.raises(r.ExtractionContractError, match="statistical field prohibited"):
        r.validate_endpoint_rows([row] * r.REQUIRED_ANCHOR_CELL_ROWS)


def test_npy_dtype_shape_order_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "STATE_VECTOR_SIZE", 4)
    path = tmp_path / "states.npy"
    r.write_support_states(path, [np.arange(4, dtype=np.float32), np.ones(4, dtype=np.float32)])
    array = r.canonical_npy_array(path)
    assert array.shape == (2, 4)
    assert array.dtype == np.dtype("<f4")
    assert array.flags.c_contiguous


def test_npy_nonfinite_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "STATE_VECTOR_SIZE", 2)
    path = tmp_path / "states.npy"
    np.save(path, np.array([[np.nan, 0.0]], dtype="<f4"), allow_pickle=False)
    with pytest.raises(r.ExtractionContractError, match="support_states nonfinite"):
        r.canonical_npy_array(path)


def test_checksum_format_and_validation(tmp_path):
    names = r.REQUIRED_ARTIFACTS[:-1]
    for index, name in enumerate(names):
        (tmp_path / name).write_bytes(f"x{index}".encode())
    checksums = r.artifact_checksums(tmp_path)
    (tmp_path / "SHA256SUMS.txt").write_bytes(r.checksum_bytes(checksums))
    r.validate_checksum_file(tmp_path)
    text = (tmp_path / "SHA256SUMS.txt").read_text()
    assert "SHA256SUMS.txt" not in text


def test_transactional_publish_rejects_output_collision(tmp_path):
    final = tmp_path / "bundle"
    final.mkdir()
    with pytest.raises(r.ExtractionContractError, match="output collision"):
        r.transactional_publish(
            final,
            manifest={},
            support_rows=[],
            support_vectors=[],
            endpoint_rows=[],
        )


def test_transactional_publish_rejects_staging_collision(tmp_path):
    final = tmp_path / "bundle"
    staging = tmp_path / "bundle.staging"
    staging.mkdir()
    with pytest.raises(r.ExtractionContractError, match="staging collision"):
        r.transactional_publish(
            final,
            manifest={},
            support_rows=[],
            support_vectors=[],
            endpoint_rows=[],
        )


def test_cli_has_no_population_layer_checkpoint_endpoint_override():
    forbidden = [
        "--seed",
        "180",
        "--arm",
        "G3-GROUP-Q-HALF",
        "--layer",
        "10",
        "--batch-size",
        "2",
        "--endpoint",
        "other",
    ]
    with pytest.raises(SystemExit):
        r.parse_args([
            "--output-dir", "x",
            "--expected-execution-head", "1" * 40,
            "--runner-implementation-commit", "2" * 40,
            "--expected-runner-sha256", "3" * 64,
            "--execution-authority-commit", "1" * 40,
            *forbidden,
        ])


def test_extract_row_states_requires_batch_one(monkeypatch):
    model = SimpleNamespace(mamba=SimpleNamespace())
    with pytest.raises(r.ExtractionContractError, match="scientific batch/sequence shape"):
        r.extract_row_states(model, torch.zeros((2, r.MAX_MODEL_SEQUENCE_LENGTH), dtype=torch.long))


def test_extract_row_states_calls_only_backbone_once(monkeypatch):
    calls = {"mamba": 0, "downstream": 0}

    class FakeMamba:
        def __init__(self):
            self.layers = [SimpleNamespace(mixer=object()) for _ in range(r.measurement.NATIVE_MAMBA_LAYER_COUNT)]
        def eval(self):
            return self
        def __call__(self, *, input_ids):
            calls["mamba"] += 1
            return SimpleNamespace(last_hidden_state=torch.zeros((1, input_ids.shape[1], 1)))

    class FakeModel:
        def __init__(self):
            self.mamba = FakeMamba()
        def __call__(self, *args, **kwargs):
            calls["downstream"] += 1
            raise AssertionError("downstream must not run")

    class FakeObserver:
        def __init__(self, registered, enabled):
            assert enabled is True
            assert list(registered.values()) == [{"layer_index": r.PRIMARY_LAYER}]
            self.snapshots = {(1, r.PRIMARY_LAYER, i): object() for i in range(r.MAX_MODEL_SEQUENCE_LENGTH)}
        def capture(self):
            class Ctx:
                def __enter__(inner): return self
                def __exit__(inner, *exc): return False
            return Ctx()

    monkeypatch.setattr(r.measurement, "NativeStateObserver", FakeObserver)
    monkeypatch.setattr(
        r.measurement,
        "validate_capture_coordinates",
        lambda snapshots, token_count: [snapshots[(1, r.PRIMARY_LAYER, i)] for i in range(token_count)],
    )

    states = r.extract_row_states(FakeModel(), torch.zeros((1, r.MAX_MODEL_SEQUENCE_LENGTH), dtype=torch.long))
    assert len(states) == r.MAX_MODEL_SEQUENCE_LENGTH
    assert calls == {"mamba": 1, "downstream": 0}


def test_support_plan_deterministic_for_mini_population(monkeypatch):
    monkeypatch.setattr(r, "SOURCE_PAIR_COUNT", 2)
    plan1, map1 = r.build_support_plan(_mini_rows(), _mini_events())
    plan2, map2 = r.build_support_plan(_mini_rows(), _mini_events())
    assert plan1 == plan2
    assert map1 == map2
    assert [row["vector_index"] for row in plan1] == list(range(len(plan1)))


def test_event_schema_rejects_heuristic_field_substitution():
    row = {key: None for key in r.EVENT_KEYS}
    row.pop("absolute_anchor_token_index")
    row["anchor_index"] = 5
    with pytest.raises(r.ExtractionContractError, match="event manifest schema mismatch"):
        r.validate_event_manifest([row] * r.REQUIRED_ANCHOR_CELL_ROWS)
