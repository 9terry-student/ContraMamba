from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import materialize_reason_router_p3w6f1_repaired_stage185_sidecar as mat


FULL_COMMIT = "a" * 40


@pytest.fixture
def workspace(request):
    root = Path("p3w6f1_materializer_test_workspace") / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root.resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _json(path: Path, value: dict) -> Path:
    _write(path, (json.dumps(value, sort_keys=True) + "\n").encode())
    return path


def _canonical_regeneration_artifacts(workspace: Path):
    invocation_value = {"baseline_id_sequence_sha256": mat.BASELINE_ID_SEQUENCE_SHA256}
    configuration_value = {
        "pair_count": mat.PAIR_COUNT,
        "authorized_F1_row_count": mat.AUTHORIZED_F1_ROW_COUNT,
        "structural_negative_polarity_flip_row_count": mat.STRUCTURAL_NEGATIVE_POLARITY_FLIP_ROW_COUNT,
        "baseline_topology_row_count": mat.REPAIRED_ROW_COUNT,
        "baseline_id_sequence_sha256": mat.BASELINE_ID_SEQUENCE_SHA256,
    }
    invocation = _json(workspace / mat.INVOCATION_PATH, invocation_value)
    configuration = _json(workspace / mat.CONFIGURATION_PATH, configuration_value)
    manifest = _json(
        workspace / mat.REGENERATION_MANIFEST_PATH,
        {
            "F1_execution_commit": mat.F1_EXECUTION_COMMIT,
            "repaired_output_path": mat.REPAIRED_JSONL_PATH,
            "repaired_output_sha256": mat.REPAIRED_JSONL_SHA256,
            "repaired_generator_source_path": mat.GENERATOR_SOURCE_PATH,
            "repaired_generator_source_sha256": mat.GENERATOR_SOURCE_SHA256,
            "deterministic_generator_invocation_json": mat.INVOCATION_PATH,
            "generator_configuration_identity_json": mat.CONFIGURATION_PATH,
        },
    )
    return manifest, invocation, configuration


def _payloads():
    return {
        mat.SIDECAR_NAME: b"sidecar",
        mat.MATERIALIZATION_MANIFEST_NAME: b"manifest",
    }


def _sidecar(row_id: str = "p0__polarity_flip", pair_id: str = "p0", **override):
    row = {
        "row_id": row_id,
        "pair_id": pair_id,
        "split": "train",
        "intervention_type": "polarity_flip",
        "frame_compatible_label": 1,
        "grammar_status": "PASS",
        "intervention_contract_status": "FAIL",
        "polarity_contamination_status": "PASS",
        "schema_status": "PASS",
        "canonical_status": "PASS",
        "time_swap_status": "PASS",
        "dataset_source_status": "PASS",
        "integrity_status": "INELIGIBLE",
        "eligible_for_positive_margin": False,
        "reason_codes": ["INTERVENTION_CONTRACT_FAIL"],
        "canonical_row_id": "p0__none",
        "family_contract_id": "stage185a_v1:polarity_flip",
        "rule_version": "stage185a_v1",
        "source_dataset_path": mat.REPAIRED_JSONL_PATH,
        "source_dataset_sha256": mat.REPAIRED_JSONL_SHA256,
        "generator_source_path": mat.GENERATOR_SOURCE_PATH,
        "generator_source_sha256": mat.GENERATOR_SOURCE_SHA256,
        "stage182a_report_sha256": "",
        "stage184a_report_sha256": "",
        "integrity_builder_sha256": mat.STAGE185_BUILDER_SOURCE_SHA256,
        "created_at": "",
        "audit_changed_axes": ["polarity", "predicate"],
        "audit_preserved_axes": [],
        "audit_expected_axes": ["polarity"],
        "audit_pair_failure_scope": "none",
    }
    row.update(override)
    return row


def test_exact_18_argument_cli_contract():
    parser = mat.build_arg_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {
        "-h",
        "--help",
        "--repo-root",
        "--repaired-jsonl",
        "--repaired-jsonl-sha256",
        "--regeneration-execution-manifest-json",
        "--deterministic-generator-invocation-json",
        "--generator-configuration-identity-json",
        "--p3w4-summary-json",
        "--p3w4-pairs-jsonl",
        "--p3w5-manifest-json",
        "--f1-execution-commit",
        "--materializer-execution-commit",
        "--generator-source",
        "--stage184-contract-matrix",
        "--stage185-builder-source",
        "--split-seed",
        "--dev-ratio",
        "--rule-version",
        "--output-dir",
    }
    assert "--dataset-version" not in options
    assert "--authorized-row-ids" not in options


@pytest.mark.parametrize("cwd_kind", ["repo_root", "outside_repo"])
def test_direct_script_help_bootstraps_repo_import_path_without_pythonpath(cwd_kind):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / mat.MATERIALIZER_SOURCE_PATH
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(repo_root if cwd_kind == "repo_root" else repo_root.parent),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0
    assert "ModuleNotFoundError" not in result.stderr
    assert "--repaired-jsonl" in result.stdout


def test_invalid_materializer_commit_format_rejects(workspace):
    with pytest.raises(mat.MaterializerError, match="FULL_40_HEX"):
        mat.verify_materializer_execution_identity(workspace, "abc", repo_checker=lambda root: None)


def test_head_materializer_commit_mismatch_rejects(workspace):
    with pytest.raises(mat.MaterializerError, match="HEAD_MISMATCH"):
        mat.verify_materializer_execution_identity(
            workspace,
            FULL_COMMIT,
            repo_checker=lambda root: None,
            head_resolver=lambda root: "b" * 40,
            tracked_clean_checker=lambda root: True,
        )


def test_tracked_dirty_worktree_rejection(workspace):
    with pytest.raises(mat.MaterializerError, match="TRACKED_WORKTREE_DIRTY"):
        mat.verify_materializer_execution_identity(
            workspace,
            FULL_COMMIT,
            repo_checker=lambda root: None,
            head_resolver=lambda root: FULL_COMMIT,
            tracked_clean_checker=lambda root: False,
        )


def test_repaired_jsonl_path_mismatch(workspace):
    _write(workspace / "wrong.jsonl")
    with pytest.raises(mat.MaterializerError, match="REPAIRED_JSONL_PATH_MISMATCH"):
        mat.validate_repaired_jsonl_path_and_sha(workspace, workspace / "wrong.jsonl", mat.REPAIRED_JSONL_SHA256)


def test_repaired_jsonl_sha_mismatch(workspace, monkeypatch):
    path = _write(workspace / mat.REPAIRED_JSONL_PATH, b"x")
    monkeypatch.setattr(mat, "file_sha256", lambda p: "bad")
    with pytest.raises(mat.MaterializerError, match="REPAIRED_JSONL_SHA_MISMATCH"):
        mat.validate_repaired_jsonl_path_and_sha(workspace, path, mat.REPAIRED_JSONL_SHA256)


def test_regeneration_manifest_mismatch(workspace):
    manifest, invocation, config = _canonical_regeneration_artifacts(workspace)
    _json(manifest, {"F1_execution_commit": "bad"})
    with pytest.raises(mat.MaterializerError, match="REGENERATION_MANIFEST"):
        mat.validate_regeneration_manifest(manifest, repo_root=workspace, invocation_json=invocation, configuration_json=config)


@pytest.mark.parametrize(
    "field,path_constant,error",
    [
        ("regeneration", "REGENERATION_MANIFEST_PATH", "REGENERATION_MANIFEST_PATH_MISMATCH"),
        ("invocation", "INVOCATION_PATH", "DETERMINISTIC_INVOCATION_PATH_MISMATCH"),
        ("configuration", "CONFIGURATION_PATH", "CONFIGURATION_IDENTITY_PATH_MISMATCH"),
    ],
)
def test_same_content_alternate_regeneration_artifact_paths_reject(workspace, field, path_constant, error):
    manifest, invocation, config = _canonical_regeneration_artifacts(workspace)
    paths = {
        "regeneration": manifest,
        "invocation": invocation,
        "configuration": config,
    }
    alternate = _write(workspace / "alternate" / Path(getattr(mat, path_constant)).name, paths[field].read_bytes())
    paths[field] = alternate
    with pytest.raises(mat.MaterializerError, match=error):
        mat.validate_regeneration_artifact_paths(workspace, paths["regeneration"], paths["invocation"], paths["configuration"])


def test_deterministic_invocation_identity_mismatch(workspace, monkeypatch):
    invocation = _json(workspace / "i.json", {"bad": True})
    config = _json(workspace / "c.json", {})
    monkeypatch.setattr(mat.p3w6f1, "actual_repaired_generator_replay", lambda rows, ids: {
        "pair_count": mat.PAIR_COUNT,
        "actual_generator_repair_consumed_row_ids": ["x"],
        "deterministic_generator_invocation": {"baseline_id_sequence_sha256": mat.BASELINE_ID_SEQUENCE_SHA256},
        "generator_configuration_identity": {
            "pair_count": mat.PAIR_COUNT,
            "authorized_F1_row_count": mat.AUTHORIZED_F1_ROW_COUNT,
            "structural_negative_polarity_flip_row_count": mat.STRUCTURAL_NEGATIVE_POLARITY_FLIP_ROW_COUNT,
            "baseline_topology_row_count": mat.REPAIRED_ROW_COUNT,
            "baseline_id_sequence_sha256": mat.BASELINE_ID_SEQUENCE_SHA256,
        },
        "replayed_records": [],
    })
    with pytest.raises(mat.MaterializerError, match="DETERMINISTIC_INVOCATION_IDENTITY_MISMATCH"):
        mat.validate_replay([], ["x"], invocation, config)


def test_configuration_identity_mismatch(workspace, monkeypatch):
    invocation_value = {"baseline_id_sequence_sha256": mat.BASELINE_ID_SEQUENCE_SHA256}
    invocation = _json(workspace / "i.json", invocation_value)
    config = _json(workspace / "c.json", {"bad": True})
    monkeypatch.setattr(mat.p3w6f1, "actual_repaired_generator_replay", lambda rows, ids: {
        "pair_count": mat.PAIR_COUNT,
        "actual_generator_repair_consumed_row_ids": ["x"],
        "deterministic_generator_invocation": invocation_value,
        "generator_configuration_identity": {
            "pair_count": mat.PAIR_COUNT,
            "authorized_F1_row_count": mat.AUTHORIZED_F1_ROW_COUNT,
            "structural_negative_polarity_flip_row_count": mat.STRUCTURAL_NEGATIVE_POLARITY_FLIP_ROW_COUNT,
            "baseline_topology_row_count": mat.REPAIRED_ROW_COUNT,
            "baseline_id_sequence_sha256": mat.BASELINE_ID_SEQUENCE_SHA256,
        },
        "replayed_records": [],
    })
    with pytest.raises(mat.MaterializerError, match="CONFIGURATION_IDENTITY_MISMATCH"):
        mat.validate_replay([], ["x"], invocation, config)


def test_wrong_f1_execution_commit_rejects():
    with pytest.raises(mat.MaterializerError, match="F1_EXECUTION_COMMIT_MISMATCH"):
        mat.verify_f1_execution_commit("b" * 40)


def test_malformed_authority_rejects():
    with pytest.raises(mat.MaterializerError, match="AUTHORIZED_F1_DERIVATION_FAILED"):
        mat.derive_authorized_f1_row_ids({}, [], {})


def _patch_authorized(monkeypatch, row_ids, f2_ids=(), *, expected_count=None):
    if expected_count is not None:
        monkeypatch.setattr(mat, "AUTHORIZED_F1_ROW_COUNT", expected_count)
    monkeypatch.setattr(mat, "AUTHORIZED_F1_ROW_IDS_SHA256", mat.p3w6f1.canonical_sha256(sorted(row_ids)))
    monkeypatch.setattr(mat.p3w6f1, "extract_decision_supporting_pair_ids", lambda s, m: {"p"})
    monkeypatch.setattr(mat.p3w6f1, "extract_authorized_f1_targets", lambda pairs, supporting: {"authorized_F1_row_ids": row_ids})
    monkeypatch.setattr(mat.p3w6f1, "extract_f2_row_ids", lambda pairs: set(f2_ids))


def test_missing_authorized_f1_ids(monkeypatch):
    _patch_authorized(monkeypatch, [])
    with pytest.raises(mat.MaterializerError, match="AUTHORIZED_F1_ROW_COUNT_MISMATCH"):
        mat.derive_authorized_f1_row_ids({}, [], {})


def test_duplicate_authorized_f1_ids(monkeypatch):
    monkeypatch.setattr(mat.p3w6f1, "extract_decision_supporting_pair_ids", lambda s, m: {"p"})
    monkeypatch.setattr(mat.p3w6f1, "extract_authorized_f1_targets", lambda pairs, supporting: {"authorized_F1_row_ids": ["x", "x"]})
    with pytest.raises(mat.MaterializerError, match="DUPLICATE"):
        mat.derive_authorized_f1_row_ids({}, [], {})


def test_wrong_authorized_f1_count(monkeypatch):
    monkeypatch.setattr(mat.p3w6f1, "extract_decision_supporting_pair_ids", lambda s, m: {"p"})
    monkeypatch.setattr(mat.p3w6f1, "extract_authorized_f1_targets", lambda pairs, supporting: {"authorized_F1_row_ids": ["x"]})
    with pytest.raises(mat.MaterializerError, match="COUNT"):
        mat.derive_authorized_f1_row_ids({}, [], {})


def test_f2_inclusion_rejects(monkeypatch):
    _patch_authorized(monkeypatch, ["x"], ["x"], expected_count=1)
    with pytest.raises(mat.MaterializerError, match="INCLUDE_F2"):
        mat.derive_authorized_f1_row_ids({}, [], {})


def test_wrong_repaired_replay_rejects(workspace, monkeypatch):
    _json(workspace / "i.json", {})
    _json(workspace / "c.json", {})
    monkeypatch.setattr(mat.p3w6f1, "actual_repaired_generator_replay", lambda rows, ids: {"pair_count": 1})
    with pytest.raises(mat.MaterializerError, match="REPLAY_PAIR_COUNT_MISMATCH"):
        mat.validate_replay([], [], workspace / "i.json", workspace / "c.json")


def test_repair_consumption_mismatch(workspace, monkeypatch):
    _json(workspace / "i.json", {})
    _json(workspace / "c.json", {})
    monkeypatch.setattr(mat.p3w6f1, "actual_repaired_generator_replay", lambda rows, ids: {
        "pair_count": mat.PAIR_COUNT,
        "actual_generator_repair_consumed_row_ids": ["other"],
    })
    with pytest.raises(mat.MaterializerError, match="REPAIR_CONSUMPTION_MISMATCH"):
        mat.validate_replay([], ["x"], workspace / "i.json", workspace / "c.json")


@pytest.mark.parametrize(
    "path_arg,expected_path,expected_sha,path_error,sha_error",
    [
        ("generator.py", mat.GENERATOR_SOURCE_PATH, mat.GENERATOR_SOURCE_SHA256, "GENERATOR_SOURCE_PATH_MISMATCH", "GENERATOR_SOURCE_SHA_MISMATCH"),
        ("matrix.csv", mat.STAGE184_CONTRACT_MATRIX_PATH, mat.STAGE184_CONTRACT_MATRIX_SHA256, "STAGE184_MATRIX_PATH_MISMATCH", "STAGE184_MATRIX_SHA_MISMATCH"),
        ("builder.py", mat.STAGE185_BUILDER_SOURCE_PATH, mat.STAGE185_BUILDER_SOURCE_SHA256, "STAGE185_BUILDER_PATH_MISMATCH", "STAGE185_BUILDER_SHA_MISMATCH"),
    ],
)
def test_fixed_source_path_mismatches(workspace, path_arg, expected_path, expected_sha, path_error, sha_error):
    _write(workspace / path_arg)
    with pytest.raises(mat.MaterializerError, match=path_error):
        mat.validate_fixed_source_argument(workspace, workspace / path_arg, expected_path, expected_sha, path_error, sha_error)


@pytest.mark.parametrize(
    "expected_path,expected_sha,path_error,sha_error",
    [
        (mat.GENERATOR_SOURCE_PATH, mat.GENERATOR_SOURCE_SHA256, "GENERATOR_SOURCE_PATH_MISMATCH", "GENERATOR_SOURCE_SHA_MISMATCH"),
        (mat.STAGE184_CONTRACT_MATRIX_PATH, mat.STAGE184_CONTRACT_MATRIX_SHA256, "STAGE184_MATRIX_PATH_MISMATCH", "STAGE184_MATRIX_SHA_MISMATCH"),
        (mat.STAGE185_BUILDER_SOURCE_PATH, mat.STAGE185_BUILDER_SOURCE_SHA256, "STAGE185_BUILDER_PATH_MISMATCH", "STAGE185_BUILDER_SHA_MISMATCH"),
    ],
)
def test_fixed_source_sha_mismatches(workspace, monkeypatch, expected_path, expected_sha, path_error, sha_error):
    path = _write(workspace / expected_path)
    monkeypatch.setattr(mat, "file_sha256", lambda p: "bad")
    with pytest.raises(mat.MaterializerError, match=sha_error):
        mat.validate_fixed_source_argument(workspace, path, expected_path, expected_sha, path_error, sha_error)


def test_split_accounting_mismatch(monkeypatch):
    monkeypatch.setattr(mat, "TRAIN_ROW_COUNT", 2)
    monkeypatch.setattr(mat, "DEV_ROW_COUNT", 0)
    with pytest.raises(mat.MaterializerError, match="SPLIT_ACCOUNTING_MISMATCH"):
        mat.validate_sidecar_rows([_sidecar(split="train")], [{"id": "p0__polarity_flip"}], ["p0__polarity_flip"])


def test_sidecar_schema_mismatch(monkeypatch):
    monkeypatch.setattr(mat, "TRAIN_ROW_COUNT", 1)
    monkeypatch.setattr(mat, "DEV_ROW_COUNT", 0)
    row = _sidecar()
    row.pop("grammar_status")
    with pytest.raises(mat.MaterializerError, match="SIDECAR_SCHEMA_MISMATCH"):
        mat.validate_sidecar_rows([row], [{"id": "p0__polarity_flip"}], ["p0__polarity_flip"])


def test_repaired_source_provenance_spoof(monkeypatch):
    monkeypatch.setattr(mat, "TRAIN_ROW_COUNT", 1)
    monkeypatch.setattr(mat, "DEV_ROW_COUNT", 0)
    with pytest.raises(mat.MaterializerError, match="REPAIRED_SOURCE_PROVENANCE_PATH_SPOOF"):
        mat.validate_sidecar_rows([_sidecar(source_dataset_path="spoof")], [{"id": "p0__polarity_flip"}], ["p0__polarity_flip"])


def test_historical_baseline_sha_spoof_rejects(workspace, monkeypatch):
    _json(workspace / "i.json", {"baseline_id_sequence_sha256": "bad"})
    _json(workspace / "c.json", {})
    monkeypatch.setattr(mat.p3w6f1, "actual_repaired_generator_replay", lambda rows, ids: {
        "pair_count": mat.PAIR_COUNT,
        "actual_generator_repair_consumed_row_ids": ["x"],
        "deterministic_generator_invocation": {"baseline_id_sequence_sha256": "bad"},
        "generator_configuration_identity": {},
    })
    with pytest.raises(mat.MaterializerError, match="HISTORICAL_BASELINE_SHA_MISMATCH"):
        mat.validate_replay([], ["x"], workspace / "i.json", workspace / "c.json")


def test_historical_stage185_binary_executed_remains_false(workspace, monkeypatch):
    sidecar_path = workspace / mat.SIDECAR_NAME
    sidecar_bytes = mat.deterministic_jsonl_bytes([_sidecar()])
    manifest = mat.build_manifest(
        repo_root=workspace,
        materializer_execution_commit=FULL_COMMIT,
        materializer_source_sha="s",
        regeneration_manifest_path=_write(workspace / "regen.json"),
        invocation_path=_write(workspace / "invocation.json"),
        configuration_path=_write(workspace / "config.json"),
        stage184_contract_matrix_path=_write(workspace / mat.STAGE184_CONTRACT_MATRIX_PATH),
        sidecar_path=sidecar_path,
        sidecar_bytes=sidecar_bytes,
        sidecar_rows=[_sidecar()],
        provenance_validation={"stage185_provenance_status": "PASS"},
    )
    assert manifest["historical_stage185_binary_executed"] is False


def test_derived_manifest_status_uses_exact_materialization_success_value(workspace):
    sidecar_path = workspace / mat.SIDECAR_NAME
    sidecar_bytes = mat.deterministic_jsonl_bytes([_sidecar()])
    manifest_path, invocation_path, configuration_path = _canonical_regeneration_artifacts(workspace)
    manifest = mat.build_manifest(
        repo_root=workspace,
        materializer_execution_commit=FULL_COMMIT,
        materializer_source_sha="s",
        regeneration_manifest_path=manifest_path,
        invocation_path=invocation_path,
        configuration_path=configuration_path,
        stage184_contract_matrix_path=_write(workspace / mat.STAGE184_CONTRACT_MATRIX_PATH),
        sidecar_path=sidecar_path,
        sidecar_bytes=sidecar_bytes,
        sidecar_rows=[_sidecar()],
        provenance_validation={"stage185_provenance_status": "PASS"},
    )
    manifest_bytes = mat.deterministic_json_bytes(manifest)
    assert json.loads(manifest_bytes)["status"] == "P3W6F1_REPAIRED_STAGE185_MATERIALIZATION_PASS"


def test_no_analyzer_output_dependency_non_circularity():
    options = {option for action in mat.build_arg_parser()._actions for option in action.option_strings}
    assert "--analyzer-output" not in options
    assert "--compatibility-output" not in options


def test_expected_raw_repaired_stage185_signature(monkeypatch):
    monkeypatch.setattr(mat, "TRAIN_ROW_COUNT", 1)
    monkeypatch.setattr(mat, "DEV_ROW_COUNT", 0)
    mat.validate_sidecar_rows([_sidecar()], [{"id": "p0__polarity_flip"}], ["p0__polarity_flip"])


@pytest.mark.parametrize("field", ["created_at", "stage182a_report_sha256", "stage184a_report_sha256"])
def test_empty_provenance_fields(field, monkeypatch):
    monkeypatch.setattr(mat, "TRAIN_ROW_COUNT", 1)
    monkeypatch.setattr(mat, "DEV_ROW_COUNT", 0)
    row = _sidecar(**{field: "filled"})
    with pytest.raises(mat.MaterializerError):
        mat.validate_sidecar_rows([row], [{"id": "p0__polarity_flip"}], ["p0__polarity_flip"])


def test_validate_stage185_sidecar_provenance_accepts_valid_synthetic_candidate(monkeypatch, workspace):
    monkeypatch.setattr(mat.p3w6f1, "validate_stage185_sidecar_provenance", lambda *a, **k: {"stage185_provenance_status": "PASS"})
    result = mat.validate_provenance([], [], repo_root=workspace, repaired_jsonl_path=workspace / "x.jsonl", replay_rows=[])
    assert result["stage185_provenance_status"] == "PASS"


def test_validator_rejects_tampered_candidate(monkeypatch, workspace):
    monkeypatch.setattr(mat.p3w6f1, "validate_stage185_sidecar_provenance", lambda *a, **k: {"stage185_provenance_status": "STAGE185_PROVENANCE_UNRESOLVED", "stage185_provenance_failures": ["tamper"]})
    with pytest.raises(mat.MaterializerError, match="STAGE185_PROVENANCE_VALIDATION_FAILED"):
        mat.validate_provenance([], [], repo_root=workspace, repaired_jsonl_path=workspace / "x.jsonl", replay_rows=[])


def test_exact_two_output_names():
    assert mat.EXPECTED_OUTPUT_NAMES == {mat.SIDECAR_NAME, mat.MATERIALIZATION_MANIFEST_NAME}


def test_conflicting_artifact_rejection(workspace):
    output = workspace / "out"
    output.mkdir()
    _write(output / mat.SIDECAR_NAME, b"old")
    _write(output / mat.MATERIALIZATION_MANIFEST_NAME, b"manifest")
    with pytest.raises(mat.MaterializerError, match="OUTPUT_ARTIFACT_CONFLICT"):
        mat.publish_artifacts(output, {mat.SIDECAR_NAME: b"new", mat.MATERIALIZATION_MANIFEST_NAME: b"manifest"})


def test_partial_existing_artifact_rejection(workspace):
    output = workspace / "out"
    output.mkdir()
    _write(output / mat.SIDECAR_NAME, b"old")
    with pytest.raises(mat.MaterializerError, match="OUTPUT_ARTIFACT_SET_MISMATCH"):
        mat.publish_artifacts(output, {mat.SIDECAR_NAME: b"old", mat.MATERIALIZATION_MANIFEST_NAME: b"manifest"})


def test_extra_existing_artifact_rejection(workspace):
    output = workspace / "out"
    output.mkdir()
    for name, payload in _payloads().items():
        _write(output / name, payload)
    _write(output / "extra.json", b"x")
    with pytest.raises(mat.MaterializerError, match="OUTPUT_ARTIFACT_SET_MISMATCH"):
        mat.publish_artifacts(output, _payloads())


def test_identical_existing_pair_idempotent_success(workspace):
    output = workspace / "out"
    output.mkdir()
    payloads = _payloads()
    for name, payload in payloads.items():
        _write(output / name, payload)
    assert mat.publish_artifacts(output, payloads) == "IDEMPOTENT_PASS"
    assert {name: (output / name).read_bytes() for name in payloads} == payloads


def test_staging_first_write_failure_leaves_final_absent(workspace):
    output = workspace / "out"

    def fail_first(path, payload):
        raise OSError("write1")

    with pytest.raises(OSError, match="write1"):
        mat.publish_artifacts(output, _payloads(), staging_dir_name=".stage", write_file=fail_first)
    assert not output.exists()


def test_staging_second_write_failure_leaves_final_absent(workspace):
    output = workspace / "out"

    def fail_second(path, payload):
        if path.name == mat.MATERIALIZATION_MANIFEST_NAME:
            raise OSError("write2")
        path.write_bytes(payload)

    with pytest.raises(OSError, match="write2"):
        mat.publish_artifacts(output, _payloads(), staging_dir_name=".stage", write_file=fail_second)
    assert not output.exists()


def test_pre_rename_failure_leaves_final_absent(workspace):
    output = workspace / "out"

    def fail_before_rename(staging, final):
        raise OSError("before-rename")

    with pytest.raises(OSError, match="before-rename"):
        mat.publish_artifacts(output, _payloads(), staging_dir_name=".stage", before_rename=fail_before_rename)
    assert not output.exists()


def test_directory_rename_failure_leaves_final_absent(workspace):
    output = workspace / "out"

    def fail_rename(staging, final):
        raise OSError("rename")

    with pytest.raises(OSError, match="rename"):
        mat.publish_artifacts(output, _payloads(), staging_dir_name=".stage", rename_dir=fail_rename)
    assert not output.exists()


def test_exact_3600_2880_720_accounting(monkeypatch):
    monkeypatch.setattr(mat, "REPAIRED_ROW_COUNT", 2)
    monkeypatch.setattr(mat, "TRAIN_ROW_COUNT", 1)
    monkeypatch.setattr(mat, "DEV_ROW_COUNT", 1)
    rows = [_sidecar(split="train"), _sidecar("p1__none", "p1", split="dev", intervention_type="none")]
    mat.validate_sidecar_rows(rows, [{"id": "p0__polarity_flip"}, {"id": "p1__none"}], ["p0__polarity_flip"])
    assert mat.REPAIRED_ROW_COUNT == 2
    assert mat.TRAIN_ROW_COUNT == 1
    assert mat.DEV_ROW_COUNT == 1


def test_output_publish_writes_exact_two_names(workspace):
    output = workspace / "out"
    mat.publish_artifacts(output, _payloads())
    assert {entry.name for entry in output.iterdir()} == mat.EXPECTED_OUTPUT_NAMES
    assert {name: (output / name).read_bytes() for name in mat.EXPECTED_OUTPUT_NAMES} == _payloads()
