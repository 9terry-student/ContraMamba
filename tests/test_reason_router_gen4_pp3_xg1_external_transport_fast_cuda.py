from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_pp3_xg1_external_transport_fast_cuda as run,
)


def _signed_probe(orientation: int, f_value: float) -> dict[str, object]:
    return {
        "orientation": orientation,
        "delta_h_l2": 2.0 * run.EPSILON,
        "plus_path_efficiency": f_value,
        "minus_path_efficiency": 0.0,
        "F": f_value,
        "midpoint_max_abs_residual": 0.0,
        "pair_delta_max_abs_residual": 0.0,
        "applied_correction_max_abs_residual": 0.0,
        "runtime_correction_l2": 2.0 * run.EPSILON,
        "model_forward_count": run.FORWARDS_PER_SIGNED_PROBE,
    }


def _direction_probe(
    direction_key: str,
    vector_sha256: str,
    j_value: float,
) -> dict[str, object]:
    f_plus = j_value * run.EPSILON
    f_minus = -j_value * run.EPSILON
    observed_j = (f_plus - f_minus) / (2.0 * run.EPSILON)
    return {
        "schema_version": run.DIRECTION_PROBE_SCHEMA,
        "direction_key": direction_key,
        "vector_sha256": vector_sha256,
        "epsilon": run.EPSILON,
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": observed_j,
        "J_squared": observed_j * observed_j,
        "positive_probe": _signed_probe(1, f_plus),
        "negative_probe": _signed_probe(-1, f_minus),
        "model_forward_count": run.FORWARDS_PER_DIRECTION,
    }


def _seed(index: int = 0) -> dict[str, object]:
    return {
        "schema_version": run.PROBE_SEED_SCHEMA,
        "family_key": "xg1",
        "source_pair_id": run._expected_pairs()[index],
        "pair_index": index,
        "pair_ordinal": index + 1,
        "target_plus_anchor": 10,
        "target_minus_anchor": 11,
        "reference_plus_anchor": 12,
        "reference_minus_anchor": 13,
    }


def _item(index: int, j_plus: float, j_minus: float) -> dict[str, object]:
    seed = _seed(index)
    plus = _direction_probe("pp3_plus", run.PP3_PLUS_SHA256, j_plus)
    minus = _direction_probe("pp3_minus", run.PP3_MINUS_SHA256, j_minus)
    observed_plus = float(plus["J"])
    observed_minus = float(minus["J"])
    c_pp3 = (run.S3 / 5.0) * (
        float(plus["J_squared"]) - float(minus["J_squared"])
    )
    item = dict(seed)
    item["probe_seed_schema_version"] = item["schema_version"]
    item["schema_version"] = run.ITEM_SCHEMA
    item["scope_freeze_commit"] = run.SCOPE_FREEZE_COMMIT
    item["preparation_freeze_commit"] = run.PREPARATION_FREEZE_COMMIT
    item["epsilon"] = run.EPSILON
    item["s3"] = run.S3
    item["pp3_plus_sha256"] = run.PP3_PLUS_SHA256
    item["pp3_minus_sha256"] = run.PP3_MINUS_SHA256
    item["direction_order"] = ["pp3_plus", "pp3_minus"]
    item["pp3_plus_probe"] = plus
    item["pp3_minus_probe"] = minus
    item["J_PP3_PLUS"] = observed_plus
    item["J_PP3_PLUS_squared"] = float(plus["J_squared"])
    item["J_PP3_MINUS"] = observed_minus
    item["J_PP3_MINUS_squared"] = float(minus["J_squared"])
    item["C_PP3"] = c_pp3
    item["baseline_model_forward_count_this_run"] = 0
    item["scientific_model_forward_count_this_run"] = run.FORWARDS_PER_PAIR
    return item


def _summary() -> dict[str, object]:
    return {
        "schema_version": run.SUMMARY_SCHEMA,
        "result": run.RESULT_PASS,
        "execution_head": "f" * 40,
        "scope_freeze_commit": run.SCOPE_FREEZE_COMMIT,
        "preparation_freeze_commit": run.PREPARATION_FREEZE_COMMIT,
        "source_pair_count": run.SOURCE_PAIR_COUNT,
        "pair_id_first": "xg1_fact_001",
        "pair_id_last": "xg1_fact_300",
        "epsilon": run.EPSILON,
        "s3": run.S3,
        "pp3_plus_sha256": run.PP3_PLUS_SHA256,
        "pp3_minus_sha256": run.PP3_MINUS_SHA256,
        "pp3_ambient_dim": run.AMBIENT_DIM,
        "direction_order": ["pp3_plus", "pp3_minus"],
        "model_forwards_per_direction": run.FORWARDS_PER_DIRECTION,
        "model_forwards_per_pair": run.FORWARDS_PER_PAIR,
        "scientific_model_forward_count_this_run": run.SCIENTIFIC_FORWARD_BUDGET,
        "baseline_model_forward_count_this_run": 0,
        "primary_endpoint_definition": "C_PP3=(s3/5)*(J_PP3_PLUS^2-J_PP3_MINUS^2)",
        "primary_endpoint_C_PP3_observed": True,
        "signed_pp3_plus_observed": True,
        "finite_value_audit": "PASS",
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
        "representative_checkpoint_sha256": "1" * 64,
    }


def test_frozen_contract_and_budget() -> None:
    assert run.EXPECTED_BRANCH == "gen4-k-xg2-basis-holdout"
    assert run.SCOPE_FREEZE_COMMIT == "02e4c897d65f8f6e90b855054793594866545bf2"
    assert run.PREPARATION_FREEZE_COMMIT == "30a1dcbb1be7dc9b3a834b0b539b5d29016e55ed"
    assert run.PP3_PLUS_SHA256 == (
        "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff"
    )
    assert run.PP3_MINUS_SHA256 == (
        "ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7"
    )
    assert run.AMBIENT_DIM == 395
    assert run.EPSILON == 0.025
    assert run.S3 == 0.98692852916688512
    assert run.FORWARDS_PER_SIGNED_PROBE == 2
    assert run.FORWARDS_PER_DIRECTION == 4
    assert run.FORWARDS_PER_PAIR == 8
    assert run.SCIENTIFIC_FORWARD_BUDGET == 2400
    assert run.BASELINE_FORWARD_BUDGET_THIS_RUN == 0


def test_exact_xg1_pair_population() -> None:
    pairs = run._expected_pairs()
    assert len(pairs) == 300
    assert pairs[0] == "xg1_fact_001"
    assert pairs[-1] == "xg1_fact_300"
    assert pairs == tuple(f"xg1_fact_{index:03d}" for index in range(1, 301))


def _git_bytes(path: Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        cwd=run.ROOT,
    )


def _jsonl_from_raw(raw: bytes) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in raw.decode("utf-8-sig").splitlines()
        if line.strip()
    ]


def test_frozen_xg1_structural_and_anchor_artifacts() -> None:
    eligibility = run.xg1_eq.eligibility

    source_raw = _git_bytes(eligibility.SOURCE_FACTS_PATH)
    rows_raw = _git_bytes(eligibility.ROWS_PATH)
    manifest_raw = _git_bytes(eligibility.STRUCTURAL_MANIFEST_PATH)

    assert hashlib.sha256(source_raw).hexdigest() == eligibility.EXPECTED_SOURCE_FACTS_SHA256
    assert hashlib.sha256(rows_raw).hexdigest() == eligibility.EXPECTED_ROWS_SHA256
    assert (
        hashlib.sha256(manifest_raw).hexdigest()
        == eligibility.EXPECTED_STRUCTURAL_MANIFEST_SHA256
    )

    facts = _jsonl_from_raw(source_raw)
    rows = _jsonl_from_raw(rows_raw)
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))

    eligibility.validate_structural_manifest(manifest)
    eligibility.xg1.validate_source_facts(facts)
    eligibility.xg1.validate_materialized_rows(rows, expected_pairs=300)

    assert len(facts) == 300
    assert len(rows) == 1800
    assert facts[0]["pair_id"] == "xg1_fact_001"
    assert facts[-1]["pair_id"] == "xg1_fact_300"
    assert manifest["generator_family"] == "xg1_independent_structured_records_v1"

    anchor_raw = _git_bytes(run.xg1_eq.ELIGIBILITY_ANCHOR_MANIFEST)
    summary_raw = _git_bytes(run.xg1_eq.ELIGIBILITY_SUMMARY)
    assert (
        hashlib.sha256(anchor_raw).hexdigest()
        == run.xg1_eq.ELIGIBILITY_ANCHOR_MANIFEST_SHA256
    )
    assert (
        hashlib.sha256(summary_raw).hexdigest()
        == run.xg1_eq.ELIGIBILITY_SUMMARY_SHA256
    )

    summary = json.loads(summary_raw.decode("utf-8-sig"))
    assert summary["primary_complete_pair_prefix_feasibility"] == "PASS_300_OF_300"
    assert summary["complete_source_pair_count"] == 300
    assert summary["required_anchor_row_count"] == eligibility.EXPECTED_ANCHOR_ROWS
    assert summary["model_forward_count"] == 0
    assert summary["checkpoint_load_count"] == 0
    assert summary["gpu_used"] is False
    assert summary["scientific_outcomes_observed"] is False
    assert (
        summary["anchor_manifest_sha256"]
        == run.xg1_eq.ELIGIBILITY_ANCHOR_MANIFEST_SHA256
    )

    anchors = _jsonl_from_raw(anchor_raw)
    assert len(anchors) == 1800
    assert sum(row["anchor_name"] == "A_IDENTITY" for row in anchors) == 1200
    assert sum(row["anchor_name"] == "A_NAME" for row in anchors) == 600
    assert all(row["schema_version"] == eligibility.ANCHOR_MANIFEST_SCHEMA for row in anchors)
    assert all(row["post4_eligible"] is True for row in anchors)
    assert all(row["exclusion_code"] is None for row in anchors)
    assert all(type(row["absolute_anchor_token_index"]) is int for row in anchors)
    assert all(type(row["terminal_index"]) is int for row in anchors)


def test_preparation_vectors_are_exact_frozen_bytes() -> None:
    loaded = run.load_pp3_vectors()
    plus = loaded["pp3_plus"]
    minus = loaded["pp3_minus"]
    assert plus.dtype == torch.float64
    assert minus.dtype == torch.float64
    assert tuple(plus.shape) == (395,)
    assert tuple(minus.shape) == (395,)
    assert float(torch.linalg.vector_norm(plus).item()) == pytest.approx(1.0, abs=1e-12)
    assert float(torch.linalg.vector_norm(minus).item()) == pytest.approx(1.0, abs=1e-12)
    assert float(torch.dot(plus, minus).item()) == pytest.approx(0.0, abs=1e-12)
    assert int(torch.argmax(torch.abs(plus)).item()) == 267
    assert int(torch.argmax(torch.abs(minus)).item()) == 23
    assert float(plus[267].item()) > 0.0
    assert float(minus[23].item()) > 0.0


def test_signed_probe_preserves_frozen_half_delta_semantics(monkeypatch) -> None:
    runtime = run.holdout.phase1.base.prevalence_eq
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run.holdout.phase1,
        "_cells",
        lambda: {"tp": "TP", "tm": "TM", "rp": "RP", "rm": "RM"},
    )
    monkeypatch.setattr(
        run.holdout.phase1,
        "_anchors_for_pair",
        lambda _pair, _events: {"tp": 10, "tm": 11, "rp": 12, "rm": 13},
    )
    monkeypatch.setattr(
        run,
        "_input_row",
        lambda _encoded, _row_index, _pair, cell: torch.tensor([[len(cell)]], dtype=torch.long),
    )

    def fake_capture(
        _model,
        _runtime_ctx,
        *,
        input_ids,
        anchor,
        delta_h,
        plus_branch,
        **_kwargs,
    ):
        calls.append(
            {
                "input_ids": input_ids.clone(),
                "anchor": anchor,
                "delta_h": delta_h.clone(),
                "plus_branch": plus_branch,
            }
        )
        return {
            "intervention_audit": object(),
            "pe": 2.0 if plus_branch else 0.5,
        }

    monkeypatch.setattr(runtime.parent, "capture_branch", fake_capture)
    monkeypatch.setattr(runtime.parent, "path_efficiency", lambda captured: captured["pe"])
    monkeypatch.setattr(
        runtime.transport_runtime,
        "paired_intervention_audit",
        lambda *_args, **_kwargs: {
            "midpoint_max_abs_residual": 0.0,
            "pair_delta_max_abs_residual": 0.0,
            "applied_correction_max_abs_residual": 0.0,
            "runtime_correction_l2": 2.0 * run.EPSILON,
        },
    )

    direction = torch.zeros(run.AMBIENT_DIM, dtype=torch.float64)
    direction[0] = 1.0
    result = run._run_signed_probe(
        _seed(),
        direction,
        orientation=-1,
        model=object(),
        runtime_ctx={},
        trace_code=object(),
        trace_line=1,
        encoded={},
        row_index={},
        events={},
        budget=object(),
    )

    assert len(calls) == 2
    assert calls[0]["plus_branch"] is True
    assert calls[1]["plus_branch"] is False
    expected_delta = -2.0 * run.EPSILON * direction
    assert torch.equal(calls[0]["delta_h"], expected_delta)
    assert torch.equal(calls[1]["delta_h"], expected_delta)
    assert result["F"] == pytest.approx(1.5)
    assert result["model_forward_count"] == 2


def test_direction_j_uses_positive_then_negative_orientation(monkeypatch) -> None:
    calls: list[int] = []

    def fake_signed(_seed, _direction, *, orientation, **_kwargs):
        calls.append(orientation)
        return _signed_probe(orientation, 0.25 if orientation == 1 else -0.25)

    monkeypatch.setattr(run, "_run_signed_probe", fake_signed)
    direction = torch.zeros(run.AMBIENT_DIM, dtype=torch.float64)
    direction[0] = 1.0
    result = run._run_pp3_direction_j(
        _seed(),
        direction,
        direction_key="pp3_plus",
        vector_sha256=run.PP3_PLUS_SHA256,
        model=object(),
        runtime_ctx={},
        trace_code=object(),
        trace_line=1,
        encoded={},
        row_index={},
        events={},
        budget=object(),
    )
    assert calls == [1, -1]
    assert result["J"] == pytest.approx((0.25 - (-0.25)) / (2.0 * run.EPSILON))
    assert result["model_forward_count"] == 4


def test_pair_uses_pp3_plus_then_pp3_minus_and_exact_endpoint(monkeypatch) -> None:
    calls: list[str] = []

    def fake_direction(
        _seed,
        _direction,
        *,
        direction_key,
        vector_sha256,
        **_kwargs,
    ):
        calls.append(direction_key)
        j_value = 3.0 if direction_key == "pp3_plus" else 2.0
        return _direction_probe(direction_key, vector_sha256, j_value)

    monkeypatch.setattr(run, "_run_pp3_direction_j", fake_direction)
    direction = torch.zeros(run.AMBIENT_DIM, dtype=torch.float64)
    direction[0] = 1.0
    result = run._run_pair(
        _seed(),
        pp3_plus=direction,
        pp3_minus=direction,
        model=object(),
        runtime_ctx={},
        trace_code=object(),
        trace_line=1,
        encoded={},
        row_index={},
        events={},
        budget=object(),
    )
    assert calls == ["pp3_plus", "pp3_minus"]
    assert result["C_PP3"] == pytest.approx((run.S3 / 5.0) * (9.0 - 4.0))
    assert result["scientific_model_forward_count_this_run"] == 8
    assert result["baseline_model_forward_count_this_run"] == 0


def test_negative_c_pp3_is_preserved_without_rescue(monkeypatch) -> None:
    def fake_direction(
        _seed,
        _direction,
        *,
        direction_key,
        vector_sha256,
        **_kwargs,
    ):
        j_value = 1.0 if direction_key == "pp3_plus" else 4.0
        return _direction_probe(direction_key, vector_sha256, j_value)

    monkeypatch.setattr(run, "_run_pp3_direction_j", fake_direction)
    direction = torch.zeros(run.AMBIENT_DIM, dtype=torch.float64)
    direction[0] = 1.0
    result = run._run_pair(
        _seed(),
        pp3_plus=direction,
        pp3_minus=direction,
        model=object(),
        runtime_ctx={},
        trace_code=object(),
        trace_line=1,
        encoded={},
        row_index={},
        events={},
        budget=object(),
    )
    assert result["C_PP3"] < 0.0
    assert result["C_PP3"] == pytest.approx((run.S3 / 5.0) * (1.0 - 16.0))


def test_nonfinite_pair_endpoint_fails_closed(monkeypatch) -> None:
    def fake_direction(
        _seed,
        _direction,
        *,
        direction_key,
        vector_sha256,
        **_kwargs,
    ):
        probe = _direction_probe(direction_key, vector_sha256, 1.0)
        if direction_key == "pp3_plus":
            probe["J"] = float("nan")
            probe["J_squared"] = float("nan")
        return probe

    monkeypatch.setattr(run, "_run_pp3_direction_j", fake_direction)
    direction = torch.zeros(run.AMBIENT_DIM, dtype=torch.float64)
    direction[0] = 1.0
    with pytest.raises(run.PP3XG1ExternalTransportError, match="NONFINITE_PAIR_ENDPOINT"):
        run._run_pair(
            _seed(),
            pp3_plus=direction,
            pp3_minus=direction,
            model=object(),
            runtime_ctx={},
            trace_code=object(),
            trace_line=1,
            encoded={},
            row_index={},
            events={},
            budget=object(),
        )


def test_item_validator_accepts_full_population_and_negative_values() -> None:
    items = [
        _item(index, 1.0 + index / 1000.0, 2.0 + index / 1000.0)
        for index in range(run.SOURCE_PAIR_COUNT)
    ]
    assert any(float(item["C_PP3"]) < 0.0 for item in items)
    run._validate_items(items)


def test_output_manifest_and_checksum_round_trip(tmp_path: Path) -> None:
    items = [
        _item(index, 2.0 + index / 1000.0, 1.0 + index / 1000.0)
        for index in range(run.SOURCE_PAIR_COUNT)
    ]
    output = tmp_path / "artifact"
    run._write_outputs(output, items=items, summary=_summary())
    validated = run.validate_artifact(output)
    assert validated["summary"]["result"] == run.RESULT_PASS
    assert len(validated["items"]) == 300
    assert (output / run.MANIFEST_FILE).is_file()
    assert (output / run.CHECKSUM_FILE).is_file()


def test_output_hash_tamper_fails_closed(tmp_path: Path) -> None:
    items = [_item(index, 2.0, 1.0) for index in range(run.SOURCE_PAIR_COUNT)]
    output = tmp_path / "artifact"
    run._write_outputs(output, items=items, summary=_summary())
    with (output / run.SUMMARY_FILE).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(run.PP3XG1ExternalTransportError, match="ARTIFACT_SHA256"):
        run.validate_artifact(output)


def test_authentication_rejects_wrong_branch(monkeypatch) -> None:
    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return "wrong-branch"
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(run, "git", fake_git)
    with pytest.raises(run.PP3XG1ExternalTransportError, match="BRANCH_MISMATCH"):
        run.authenticate_repo("a" * 40)


def test_authentication_rejects_dirty_worktree(monkeypatch) -> None:
    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return run.EXPECTED_BRANCH
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("status", "--porcelain"):
            return "?? stray.txt"
        raise AssertionError(args)

    monkeypatch.setattr(run, "git", fake_git)
    with pytest.raises(run.PP3XG1ExternalTransportError, match="WORKTREE_NOT_CLEAN"):
        run.authenticate_repo("a" * 40)


def test_authentication_rejects_frozen_blob_drift(monkeypatch) -> None:
    head = "a" * 40

    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return run.EXPECTED_BRANCH
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("status", "--porcelain"):
            return ""
        if len(args) == 2 and args[0] == "rev-parse" and args[1].startswith("HEAD:"):
            path = args[1][len("HEAD:") :]
            expected = run.FROZEN_GIT_BLOBS[path]
            if path == run.SCOPE_PATH:
                return "0" * 40
            return expected
        raise AssertionError(args)

    monkeypatch.setattr(run, "git", fake_git)
    monkeypatch.setattr(run, "_git_is_ancestor", lambda _ancestor, _head: True)
    monkeypatch.setattr(run, "_paths_unchanged", lambda _base, _head, _paths: True)
    with pytest.raises(run.PP3XG1ExternalTransportError, match="FROZEN_GIT_BLOB_DRIFT"):
        run.authenticate_repo(head)


def test_authentication_rejects_runtime_dependency_drift(monkeypatch) -> None:
    head = "a" * 40

    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return run.EXPECTED_BRANCH
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("status", "--porcelain"):
            return ""
        if len(args) == 2 and args[0] == "rev-parse" and args[1].startswith("HEAD:"):
            path = args[1][len("HEAD:") :]
            return run.FROZEN_GIT_BLOBS[path]
        raise AssertionError(args)

    monkeypatch.setattr(run, "git", fake_git)
    monkeypatch.setattr(run, "_git_is_ancestor", lambda _ancestor, _head: True)
    monkeypatch.setattr(run, "_paths_unchanged", lambda _base, _head, _paths: False)
    with pytest.raises(run.PP3XG1ExternalTransportError, match="RUNTIME_REUSED_PATH_DRIFT"):
        run.authenticate_repo(head)


def test_source_contains_no_confirmatory_or_training_execution() -> None:
    source = inspect.getsource(run).lower()
    for token in (
        "scipy",
        "ttest",
        "holm",
        "multipletests",
        ".backward(",
        ".train(",
    ):
        assert token not in source

    assert '"primary_inference_executed": false' in source
    assert '"scientific_conclusion": none' in source
    assert "pp1" not in source
    assert "pp2" not in source
    assert "pp4" not in source
    assert "pp5" not in source


def test_parse_args_requires_execution_identity() -> None:
    args = run.parse_args(
        [
            "--expected-head",
            "abc",
            "--model-snapshot",
            "m",
            "--tokenizer-snapshot",
            "t",
            "--checkpoint",
            "c",
            "--output-dir",
            "o",
        ]
    )
    assert args.expected_head == "abc"
    assert args.model_snapshot == Path("m")
    assert args.tokenizer_snapshot == Path("t")
    assert args.checkpoint == Path("c")
    assert args.output_dir == Path("o")
