from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts import reason_router_gen4_native_mamba_state_q1_q3_execution as d


def population() -> list[dict[str, str]]:
    cells = (
        "C0_SHAM",
        "C1_TITLE",
        "C2_NAME",
        "C3_ROLE",
        "C4_PREDICATE",
        "C5_TITLE_NAME",
    )
    return [
        {
            "source_pair_id": f"p{pair}",
            "row_id": f"p{pair}-{cell}",
            "contrast_cell_id": cell,
        }
        for pair in range(300)
        for cell in cells
    ]


def encoded_population(rows):
    return {
        "row_id": [row["row_id"] for row in rows],
        "source_pair_id": [row["source_pair_id"] for row in rows],
        "contrast_cell_id": [row["contrast_cell_id"] for row in rows],
        "input_ids": np.zeros((1800, 128), dtype=np.int64),
        "attention_mask": np.ones((1800, 128), dtype=np.bool_),
        "claim_mask": np.zeros((1800, 128), dtype=np.bool_),
        "evidence_mask": np.zeros((1800, 128), dtype=np.bool_),
    }


class FakeMamba:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def parameters(self):
        return []


class FakeModel:
    def __init__(self):
        self.mamba = FakeMamba()


def test_cli_exposes_only_frozen_execution_controls():
    args = d.parse_args([
        "--output-dir", "out",
        "--expected-execution-head", "a" * 40,
        "--driver-implementation-commit", "b" * 40,
        "--expected-driver-sha256", "c" * 64,
        "--execution-authority-commit", "a" * 40,
    ])
    assert args.output_dir == "out"
    assert not hasattr(args, "layer")
    assert not hasattr(args, "cell")
    assert not hasattr(args, "anchor")
    assert not hasattr(args, "checkpoint_arm")


def test_execution_binding_accepts_exact_synthetic_contract(monkeypatch, tmp_path):
    head = "a" * 40
    driver_commit = "b" * 40
    driver_bytes = b"synthetic-driver\n"
    driver_sha = hashlib.sha256(driver_bytes).hexdigest()
    worktree = tmp_path / "driver.py"
    worktree.write_bytes(driver_bytes)

    monkeypatch.setattr(d, "DRIVER_PATH", worktree)
    monkeypatch.setattr(d, "ROOT", Path("/"))
    monkeypatch.setattr(d, "git_head", lambda: head)
    monkeypatch.setattr(d, "git_commit_exists", lambda _: True)
    monkeypatch.setattr(d, "git_is_ancestor", lambda _a, _b: True)
    monkeypatch.setattr(d, "git_status_porcelain", lambda: "")
    monkeypatch.setattr(
        d,
        "git_show_bytes",
        lambda revision, path: driver_bytes,
    )
    monkeypatch.setattr(
        d,
        "canonical_git_sha256",
        lambda revision, path: driver_sha,
    )
    monkeypatch.setattr(
        d,
        "canonical_worktree_bytes",
        lambda path: driver_bytes,
    )
    monkeypatch.setattr(
        d,
        "validate_execution_authority_document",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        d,
        "validate_frozen_dependency_identities",
        lambda revision: None,
    )

    observed = d.validate_execution_binding(
        expected_execution_head=head,
        driver_implementation_commit=driver_commit,
        expected_driver_sha256=driver_sha,
        execution_authority_commit=head,
    )
    assert observed == head


def test_execution_binding_rejects_wrong_head(monkeypatch):
    monkeypatch.setattr(d, "git_head", lambda: "a" * 40)
    with pytest.raises(d.ExecutionContractError, match="HEAD"):
        d.validate_execution_binding(
            expected_execution_head="b" * 40,
            driver_implementation_commit="c" * 40,
            expected_driver_sha256="d" * 64,
            execution_authority_commit="b" * 40,
        )


def test_execution_binding_rejects_wrong_driver_sha(monkeypatch, tmp_path):
    head = "a" * 40
    driver_commit = "b" * 40
    driver_bytes = b"driver\n"
    worktree = tmp_path / "driver.py"
    worktree.write_bytes(driver_bytes)

    monkeypatch.setattr(d, "DRIVER_PATH", worktree)
    monkeypatch.setattr(d, "ROOT", Path("/"))
    monkeypatch.setattr(d, "git_head", lambda: head)
    monkeypatch.setattr(d, "git_commit_exists", lambda _: True)
    monkeypatch.setattr(d, "git_is_ancestor", lambda _a, _b: True)
    monkeypatch.setattr(d, "git_status_porcelain", lambda: "")
    monkeypatch.setattr(
        d,
        "validate_execution_authority_document",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        d,
        "git_show_bytes",
        lambda revision, path: driver_bytes,
    )

    wrong_sha = "f" * 64
    with pytest.raises(d.ExecutionContractError, match="identity"):
        d.validate_execution_binding(
            expected_execution_head=head,
            driver_implementation_commit=driver_commit,
            expected_driver_sha256=wrong_sha,
            execution_authority_commit=head,
        )


def test_execution_binding_requires_driver_ancestry(monkeypatch):
    head = "a" * 40
    driver_commit = "b" * 40
    monkeypatch.setattr(d, "git_head", lambda: head)
    monkeypatch.setattr(d, "git_commit_exists", lambda _: True)
    monkeypatch.setattr(d, "git_is_ancestor", lambda _a, _b: False)

    with pytest.raises(d.ExecutionContractError):
        d.validate_execution_binding(
            expected_execution_head=head,
            driver_implementation_commit=driver_commit,
            expected_driver_sha256="c" * 64,
            execution_authority_commit=head,
        )


def test_model_loader_is_not_reached_when_coordinate_gate_fails(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        d,
        "validate_execution_binding",
        lambda **kwargs: kwargs["expected_execution_head"],
    )

    called = {"runtime": 0, "input": 0, "model": 0}

    def runtime_gate():
        called["runtime"] += 1

    def input_loader(_snapshot):
        called["input"] += 1
        raise d.ExecutionContractError("coordinate gate")

    def model_loader(_snapshot):
        called["model"] += 1
        raise AssertionError("model loader must not be reached")

    with pytest.raises(d.ExecutionContractError, match="coordinate gate"):
        d.run_scientific_extraction(
            output_dir=tmp_path / "out",
            expected_execution_head="a" * 40,
            driver_implementation_commit="b" * 40,
            expected_driver_sha256="c" * 64,
            execution_authority_commit="a" * 40,
            runtime_gate=runtime_gate,
            input_loader=input_loader,
            model_loader=model_loader,
        )

    assert called == {"runtime": 1, "input": 1, "model": 0}


def test_output_collision_fails_closed_before_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        d,
        "validate_execution_binding",
        lambda **kwargs: kwargs["expected_execution_head"],
    )
    output = tmp_path / "out"
    output.mkdir()

    called = {"runtime": 0}

    def runtime_gate():
        called["runtime"] += 1

    with pytest.raises(d.ExecutionContractError, match="output collision"):
        d.run_scientific_extraction(
            output_dir=output,
            expected_execution_head="a" * 40,
            driver_implementation_commit="b" * 40,
            expected_driver_sha256="c" * 64,
            execution_authority_commit="a" * 40,
            runtime_gate=runtime_gate,
        )

    assert called["runtime"] == 0


def test_execute_scientific_plan_runs_exactly_600_dual_layer_forwards(
    monkeypatch,
):
    rows = population()
    selected = [
        row
        for row in rows
        if row["contrast_cell_id"] in d.extraction.CELLS
    ]
    encoded = encoded_population(rows)
    events = {
        row["row_id"]: {
            "absolute_anchor_token_index": 10,
        }
        for row in selected
    }

    monkeypatch.setattr(
        d.extraction.measurement,
        "flatten_scientific_state",
        lambda _state: np.zeros(1, dtype=np.float32),
    )
    monkeypatch.setattr(
        d.extraction.measurement,
        "endpoint_from_same_layer",
        lambda states, anchor, layer_index: {
            "POST4_SPEED": 0.0,
            "POST4_TURNING": 0.0,
            "POST4_PATH_EFFICIENCY": 0.0,
        },
    )

    calls = []

    def capture_row(_model, input_ids):
        calls.append(tuple(np.asarray(input_ids).shape))
        return {
            5: [object() for _ in range(128)],
            17: [object() for _ in range(128)],
        }

    support, states, endpoints, forward_count = d.execute_scientific_plan(
        selected=selected,
        events=events,
        encoded=encoded,
        model=object(),
        capture_row=capture_row,
    )

    assert forward_count == 600
    assert len(calls) == 600
    assert len(support) == 7200
    assert states.shape == (7200, 1)
    assert len(endpoints) == 1200
    assert {row["layer_index"] for row in endpoints} == {5, 17}


def test_execute_scientific_plan_rejects_missing_secondary_layer(monkeypatch):
    rows = population()
    selected = [
        row
        for row in rows
        if row["contrast_cell_id"] in d.extraction.CELLS
    ]
    encoded = encoded_population(rows)
    events = {
        row["row_id"]: {
            "absolute_anchor_token_index": 10,
        }
        for row in selected
    }

    def capture_row(_model, _input_ids):
        return {
            5: [object() for _ in range(128)],
        }

    with pytest.raises(d.extraction.ExtractionContractError, match="dual-layer"):
        d.execute_scientific_plan(
            selected=selected,
            events=events,
            encoded=encoded,
            model=object(),
            capture_row=capture_row,
        )


def test_run_binds_frozen_primitive_identity_into_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(
        d,
        "validate_execution_binding",
        lambda **kwargs: kwargs["expected_execution_head"],
    )

    model = FakeModel()
    selected = [{"row_id": "x"}] * 600
    encoded = {"sentinel": True}
    events = {"sentinel": True}
    support = [{} for _ in range(7200)]
    endpoints = [{} for _ in range(1200)]
    states = np.zeros((1, 1), dtype=np.float32)

    captured = {}

    def input_loader(_snapshot):
        return selected, encoded, events

    def model_loader(_snapshot):
        return model, d.extraction.REPRESENTATIVE_CHECKPOINT_SHA256

    def plan_executor(**kwargs):
        assert kwargs["model"] is model
        return support, states, endpoints, 600

    def bundle_writer(output_dir, **kwargs):
        captured["output_dir"] = Path(output_dir)
        captured.update(kwargs)

    versions = dict(d.extraction.EXPECTED_RUNTIME_VERSIONS)

    result = d.run_scientific_extraction(
        output_dir=tmp_path / "out",
        expected_execution_head="a" * 40,
        driver_implementation_commit="b" * 40,
        expected_driver_sha256="c" * 64,
        execution_authority_commit="a" * 40,
        runtime_gate=lambda: None,
        input_loader=input_loader,
        model_loader=model_loader,
        plan_executor=plan_executor,
        bundle_writer=bundle_writer,
        runtime_version_loader=lambda: versions,
    )

    assert model.mamba.eval_calls == 1
    assert captured["implementation_commit"] == (
        d.Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT
    )
    assert captured["measurement_implementation_sha256"] == (
        d.Q1_Q3_MEASUREMENT_SHA256
    )
    assert captured["extraction_runner_sha256"] == (
        d.Q1_Q3_EXTRACTION_SHA256
    )
    assert captured["runtime_versions"] == versions
    assert result["backbone_forward_count"] == 600
    assert result["training"] is False
    assert result["backward"] is False
    assert result["statistical_testing"] is False


def test_driver_has_no_training_or_statistical_execution_calls():
    path = Path(
        "scripts/reason_router_gen4_native_mamba_state_q1_q3_execution.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))

    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id.lower())
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr.lower())

    prohibited = {
        "backward",
        "zero_grad",
        "step",
        "ttest_1samp",
        "ttest_rel",
        "multipletests",
    }
    assert not (called & prohibited)


def test_frozen_policy_constants():
    assert d.EXPECTED_LAYERS == (5, 17)
    assert d.EXPECTED_ROWS == 600
    assert d.EXPECTED_FORWARD_COUNT == 600
    assert d.EXPECTED_SUPPORT_ROWS == 7200
    assert d.EXPECTED_ENDPOINT_ROWS == 1200
    assert d.Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT == (
        "d5317b2c5be09464c9196325429479c6bff25efd"
    )
