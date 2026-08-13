from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import regenerate_reason_router_p3w6f1_deterministic_polarity as wrapper


FULL_COMMIT = "a" * 40


@pytest.fixture
def workspace(request):
    root = Path("p3w6f1_wrapper_test_workspace") / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root.resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _args(repo_root: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=repo_root,
        baseline_jsonl=Path("baseline.jsonl"),
        baseline_jsonl_sha256="baseline-sha",
        p3w4_summary_json=Path("p3w4_summary.json"),
        p3w4_pairs_jsonl=Path("p3w4_pairs.jsonl"),
        p3w5_manifest_json=Path("p3w5_manifest.json"),
        baseline_sidecar_jsonl=Path("baseline_sidecar.jsonl"),
        base_form_authority_commit=wrapper.BASE_FORM_AUTHORITY_COMMIT,
        base_form_authority_path=wrapper.BASE_FORM_AUTHORITY_PATH,
        base_form_authority_symbol=wrapper.BASE_FORM_AUTHORITY_SYMBOL,
        base_form_authority_sha256=wrapper.BASE_FORM_AUTHORITY_SHA256,
        f1_execution_commit=FULL_COMMIT,
        output_dir=output_dir,
    )


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return wrapper.deterministic_jsonl_bytes(rows)


def _f1_record(pair_id: str, row_id: str) -> dict:
    return {
        "pair_id": pair_id,
        "family": "F1",
        "automatic_root_cause_class": "F1_TRUE_POLARITY_GENERATION_DEFECT",
        "remediation_state": "REGENERATION_REQUIRED",
        "members": {
            "polarity_flip": {
                "source_row": {
                    "id": row_id,
                    "intervention_type": "polarity_flip",
                }
            }
        },
    }


def _f2_record(row_id: str) -> dict:
    return {
        "pair_id": "f2",
        "family": "F2",
        "members": {
            "polarity_flip": {
                "source_row": {
                    "id": row_id,
                    "intervention_type": "polarity_flip",
                }
            }
        },
    }


def _authority(count: int = 2) -> tuple[dict, list[dict], dict, list[str]]:
    pair_ids = [f"p{i}" for i in range(count)]
    row_ids = [f"{pair_id}__polarity_flip" for pair_id in pair_ids]
    summary = {"decision_supporting_pair_ids": pair_ids}
    pairs = [_f1_record(pair_id, row_id) for pair_id, row_id in zip(pair_ids, row_ids)]
    manifest = {}
    return summary, pairs, manifest, row_ids


def _base_rows() -> tuple[list[dict], str, str]:
    structural_id = sorted(
        wrapper.p3w6f1.structural_negative_polarity_flip_row_ids_for_pair_count(5)
    )[0]
    pair_id = structural_id.split("__", 1)[0]
    f2_id = "f2_pair__polarity_flip"
    rows = [
        {
            "id": structural_id,
            "pair_id": pair_id,
            "intervention_type": "polarity_flip",
            "evidence": "The report said did not acquired the archive.",
            "claim": "The title acquired the archive.",
        },
        {
            "id": f2_id,
            "pair_id": "f2_pair",
            "intervention_type": "polarity_flip",
            "evidence": "F2 evidence stays fixed.",
            "claim": "F2 claim.",
        },
        {
            "id": "canon_pair__none",
            "pair_id": "canon_pair",
            "intervention_type": "none",
            "evidence": "Canonical evidence stays fixed.",
            "claim": "Canonical claim.",
        },
        {
            "id": "para_pair__paraphrase",
            "pair_id": "para_pair",
            "intervention_type": "paraphrase",
            "evidence": "Paraphrase evidence stays fixed.",
            "claim": "Paraphrase claim.",
        },
        {
            "id": "other_pair__irrelevant_evidence",
            "pair_id": "other_pair",
            "intervention_type": "irrelevant_evidence",
            "evidence": "Other evidence stays fixed.",
            "claim": "Other claim.",
        },
    ]
    return rows, structural_id, f2_id


def _valid_repaired_rows() -> tuple[list[dict], list[dict], str, str]:
    baseline, authorized_id, f2_id = _base_rows()
    repaired = [dict(row) for row in baseline]
    repaired[0]["evidence"] = "The report said did not acquire the archive."
    return baseline, repaired, authorized_id, f2_id


def _valid_invocation_config(baseline: list[dict], authorized_id: str) -> tuple[dict, dict]:
    baseline_ids = [row["id"] for row in baseline]
    invocation = {
        "pair_count": len({row["pair_id"] for row in baseline}),
        "authorized_F1_row_ids_sha256": wrapper.p3w6f1.canonical_sha256([authorized_id]),
        "repair_api": "build_controlled_records_with_f1_polarity_repair_audit",
        "baseline_id_sequence_sha256": wrapper.p3w6f1.id_sequence_sha256(baseline_ids),
        "projection_policy": "baseline_id_sequence",
        "repair_mode": "f1_authorized_polarity_negative_only",
    }
    config = {
        "generator_source_path": wrapper.REPAIRED_GENERATOR_SOURCE_PATH,
        "pair_count": len({row["pair_id"] for row in baseline}),
        "authorized_F1_row_count": 1,
        "structural_negative_polarity_flip_row_count": len(
            wrapper.p3w6f1.structural_negative_polarity_flip_row_ids_for_pair_count(5)
        ),
        "baseline_topology_row_count": len(baseline),
        "baseline_id_sequence_sha256": invocation["baseline_id_sequence_sha256"],
    }
    return invocation, config


def test_cli_contract_contains_only_approved_semantic_arguments():
    parser = wrapper.build_arg_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    expected = {
        "--repo-root",
        "--baseline-jsonl",
        "--baseline-jsonl-sha256",
        "--p3w4-summary-json",
        "--p3w4-pairs-jsonl",
        "--p3w5-manifest-json",
        "--baseline-sidecar-jsonl",
        "--base-form-authority-commit",
        "--base-form-authority-path",
        "--base-form-authority-symbol",
        "--base-form-authority-sha256",
        "--f1-execution-commit",
        "--output-dir",
    }
    assert expected <= options
    assert "--dataset-version" not in options


@pytest.mark.parametrize("cwd_kind", ["repo_root", "outside_repo"])
def test_direct_script_help_bootstraps_repo_import_path_without_pythonpath(cwd_kind: str):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "regenerate_reason_router_p3w6f1_deterministic_polarity.py"
    cwd = repo_root if cwd_kind == "repo_root" else repo_root.parent
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "ModuleNotFoundError" not in result.stderr
    assert "--repo-root" in result.stdout
    assert "--baseline-jsonl" in result.stdout
    assert "--dataset-version" not in result.stdout


def test_invalid_non_40_hex_execution_commit_rejects(workspace: Path):
    with pytest.raises(wrapper.RegenerationWrapperError, match="FULL_40_HEX"):
        wrapper.verify_execution_identity(
            workspace,
            "abc",
            repo_checker=lambda root: None,
            head_resolver=lambda root: "abc",
            tracked_clean_checker=lambda root: True,
        )


def test_execution_commit_not_equal_to_head_rejects(workspace: Path):
    with pytest.raises(wrapper.RegenerationWrapperError, match="HEAD_MISMATCH"):
        wrapper.verify_execution_identity(
            workspace,
            FULL_COMMIT,
            repo_checker=lambda root: None,
            head_resolver=lambda root: "b" * 40,
            tracked_clean_checker=lambda root: True,
        )


def test_tracked_dirty_worktree_rejects(workspace: Path):
    with pytest.raises(wrapper.RegenerationWrapperError, match="TRACKED_WORKTREE_DIRTY"):
        wrapper.verify_execution_identity(
            workspace,
            FULL_COMMIT,
            repo_checker=lambda root: None,
            head_resolver=lambda root: FULL_COMMIT,
            tracked_clean_checker=lambda root: False,
        )


def test_base_form_git_object_sha_mismatch_rejects(workspace: Path):
    with pytest.raises(wrapper.RegenerationWrapperError, match="GIT_OBJECT_SHA"):
        wrapper.verify_base_form_authority(
            workspace,
            commit=wrapper.BASE_FORM_AUTHORITY_COMMIT,
            source_path=wrapper.BASE_FORM_AUTHORITY_PATH,
            symbol=wrapper.BASE_FORM_AUTHORITY_SYMBOL,
            source_sha256=wrapper.BASE_FORM_AUTHORITY_SHA256,
            git_object_reader=lambda root, commit, path: b"not the authority",
        )


def test_baseline_sha_mismatch_rejects(workspace: Path):
    path = workspace / "baseline.jsonl"
    path.write_bytes(b"{}\n")
    with pytest.raises(wrapper.RegenerationWrapperError, match="BASELINE_JSONL_SHA256"):
        wrapper.verify_baseline_sha(path, "0" * 64)


def test_incorrect_deterministic_output_dir_contract_rejects(workspace: Path):
    with pytest.raises(wrapper.RegenerationWrapperError, match="OUTPUT_DIR_CONTRACT"):
        wrapper.verify_output_dir(workspace, workspace / "reports" / "wrong", FULL_COMMIT)


def test_malformed_missing_frozen_authority_input_rejects():
    with pytest.raises(wrapper.RegenerationWrapperError, match="AUTHORIZED_F1_DERIVATION"):
        wrapper.derive_authorized_f1_row_ids_from_authority({}, [], {})


def test_authorized_f1_derivation_accepts_exact_authority_and_rejects_wrong_count():
    summary, pairs, manifest, row_ids = _authority(2)
    assert wrapper.derive_authorized_f1_row_ids_from_authority(
        summary,
        pairs,
        manifest,
        expected_count=2,
    ) == sorted(row_ids)
    with pytest.raises(wrapper.RegenerationWrapperError, match="CARDINALITY"):
        wrapper.derive_authorized_f1_row_ids_from_authority(summary, pairs, manifest)


def test_authorized_f1_derivation_rejects_duplicate_and_f2_inclusion():
    summary, pairs, manifest, row_ids = _authority(2)
    with pytest.raises(wrapper.RegenerationWrapperError, match="duplicate authorized F1"):
        wrapper.derive_authorized_f1_row_ids_from_authority(
            summary,
            [*pairs, pairs[0]],
            manifest,
            expected_count=2,
        )
    with pytest.raises(wrapper.RegenerationWrapperError, match="INCLUDE_F2"):
        wrapper.derive_authorized_f1_row_ids_from_authority(
            summary,
            [*pairs, _f2_record(row_ids[0])],
            manifest,
            expected_count=2,
        )


def test_projection_preserves_exact_baseline_order_and_id_topology():
    baseline = [{"id": "b", "pair_id": "p"}, {"id": "a", "pair_id": "p"}]
    replay = [{"id": "a", "pair_id": "p"}, {"id": "b", "pair_id": "p"}]
    projected = wrapper.p3w6f1.project_replay_to_baseline_topology(replay, baseline)
    assert [row["id"] for row in projected] == ["b", "a"]


def test_wrapper_isolation_accepts_only_authorized_evidence_change():
    baseline, repaired, authorized_id, f2_id = _valid_repaired_rows()
    invocation, config = _valid_invocation_config(baseline, authorized_id)
    result = wrapper.validate_wrapper_isolation(
        baseline,
        repaired,
        authorized_f1_row_ids=[authorized_id],
        f2_row_ids={f2_id},
        repair_consumed_row_ids=[authorized_id],
        deterministic_generator_invocation=invocation,
        generator_configuration_identity=config,
        f1_execution_commit=FULL_COMMIT,
        repaired_generator_source_sha256="sha",
    )
    assert result["validation"]["full_output_isolation_pass"] is True


@pytest.mark.parametrize(
    ("row_id", "field", "expected"),
    [
        ("other_pair__irrelevant_evidence", "evidence", "unauthorized_changed_row_ids"),
        ("f2_pair__polarity_flip", "evidence", "F2_changed_row_ids"),
        ("canon_pair__none", "evidence", "canonical_changed_row_ids"),
        ("para_pair__paraphrase", "evidence", "paraphrase_changed_row_ids"),
    ],
)
def test_wrapper_isolation_rejects_non_target_f2_canonical_and_paraphrase_mutation(
    row_id: str,
    field: str,
    expected: str,
):
    baseline, repaired, authorized_id, f2_id = _valid_repaired_rows()
    invocation, config = _valid_invocation_config(baseline, authorized_id)
    for row in repaired:
        if row["id"] == row_id:
            row[field] = f"mutated {field}"
    with pytest.raises(wrapper.RegenerationWrapperError, match=expected):
        wrapper.validate_wrapper_isolation(
            baseline,
            repaired,
            authorized_f1_row_ids=[authorized_id],
            f2_row_ids={f2_id},
            repair_consumed_row_ids=[authorized_id],
            deterministic_generator_invocation=invocation,
            generator_configuration_identity=config,
            f1_execution_commit=FULL_COMMIT,
            repaired_generator_source_sha256="sha",
        )


def test_invocation_and_configuration_schema_values_are_deterministic():
    baseline, _repaired, authorized_id, _f2_id = _valid_repaired_rows()
    invocation, config = _valid_invocation_config(baseline, authorized_id)
    assert set(invocation) == {
        "pair_count",
        "authorized_F1_row_ids_sha256",
        "repair_api",
        "baseline_id_sequence_sha256",
        "projection_policy",
        "repair_mode",
    }
    assert set(config) == {
        "generator_source_path",
        "pair_count",
        "authorized_F1_row_count",
        "structural_negative_polarity_flip_row_count",
        "baseline_topology_row_count",
        "baseline_id_sequence_sha256",
    }
    assert wrapper.deterministic_json_bytes(invocation) == wrapper.deterministic_json_bytes(invocation)
    assert config["generator_source_path"] == wrapper.REPAIRED_GENERATOR_SOURCE_PATH


def test_execution_manifest_records_repaired_jsonl_sha_and_generator_identity(workspace: Path):
    base_form = {
        "commit": wrapper.BASE_FORM_AUTHORITY_COMMIT,
        "path": wrapper.BASE_FORM_AUTHORITY_PATH,
        "symbol": wrapper.BASE_FORM_AUTHORITY_SYMBOL,
        "raw_source_sha256": wrapper.BASE_FORM_AUTHORITY_SHA256,
    }
    output_dir = workspace / f"{wrapper.EXPECTED_OUTPUT_DIR_PREFIX}{FULL_COMMIT}"
    repaired_bytes = b'{"id":"a"}\n'
    manifest = wrapper.build_execution_manifest(
        repo_root=workspace,
        f1_execution_commit=FULL_COMMIT,
        baseline_jsonl_path=workspace / "baseline.jsonl",
        baseline_jsonl_sha256="baseline-sha",
        output_dir=output_dir,
        repaired_output_sha256=hashlib.sha256(repaired_bytes).hexdigest(),
        repaired_generator_source_sha256="generator-sha",
        base_form_authority=base_form,
    )
    assert manifest["repaired_output_sha256"] == hashlib.sha256(repaired_bytes).hexdigest()
    assert manifest["repaired_generator_source_path"] == wrapper.REPAIRED_GENERATOR_SOURCE_PATH
    assert manifest["repaired_generator_source_path"] != wrapper.WRAPPER_SOURCE_PATH


def test_existing_conflicting_output_artifacts_fail_closed(workspace: Path):
    output_dir = workspace / "out"
    output_dir.mkdir()
    (output_dir / wrapper.REPAIRED_JSONL_NAME).write_bytes(b"old")
    payloads = {name: b"new" for name in wrapper.EXPECTED_ARTIFACT_NAMES}
    with pytest.raises(wrapper.RegenerationWrapperError, match="OUTPUT_ARTIFACT_CONFLICT"):
        wrapper.ensure_no_conflicting_outputs(output_dir, payloads)


def test_successful_synthetic_execution_writes_exact_four_artifacts(workspace: Path, monkeypatch):
    output_dir = workspace / f"{wrapper.EXPECTED_OUTPUT_DIR_PREFIX}{FULL_COMMIT}"
    baseline = [{"id": "auth", "pair_id": "p", "intervention_type": "polarity_flip", "evidence": "old"}]
    (workspace / "baseline.jsonl").write_bytes(_jsonl_bytes(baseline))
    args = _args(workspace, output_dir)
    base_form = {
        "commit": wrapper.BASE_FORM_AUTHORITY_COMMIT,
        "path": wrapper.BASE_FORM_AUTHORITY_PATH,
        "symbol": wrapper.BASE_FORM_AUTHORITY_SYMBOL,
        "raw_source_sha256": wrapper.BASE_FORM_AUTHORITY_SHA256,
    }
    repaired = [dict(baseline[0], evidence="new")]
    invocation = {
        "pair_count": 1,
        "authorized_F1_row_ids_sha256": wrapper.p3w6f1.canonical_sha256(["auth"]),
        "repair_api": "build_controlled_records_with_f1_polarity_repair_audit",
        "baseline_id_sequence_sha256": wrapper.p3w6f1.id_sequence_sha256(["auth"]),
        "projection_policy": "baseline_id_sequence",
        "repair_mode": "f1_authorized_polarity_negative_only",
    }
    config = {
        "generator_source_path": wrapper.REPAIRED_GENERATOR_SOURCE_PATH,
        "pair_count": 1,
        "authorized_F1_row_count": 1,
        "structural_negative_polarity_flip_row_count": 1,
        "baseline_topology_row_count": 1,
        "baseline_id_sequence_sha256": invocation["baseline_id_sequence_sha256"],
    }

    monkeypatch.setattr(wrapper, "verify_execution_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapper, "verify_base_form_authority", lambda *args, **kwargs: base_form)
    monkeypatch.setattr(wrapper, "verify_baseline_sha", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapper, "verify_input_authorities", lambda *args, **kwargs: {})
    monkeypatch.setattr(wrapper, "derive_authorized_f1_row_ids", lambda *args: ["auth"])
    monkeypatch.setattr(wrapper, "extract_f2_row_ids_from_authority", lambda *args: set())
    monkeypatch.setattr(
        wrapper,
        "build_repaired_payload",
        lambda rows, authorized: {
            "pair_count": 1,
            "repaired_rows": repaired,
            "repair_consumed_row_ids": ["auth"],
            "deterministic_generator_invocation": invocation,
            "generator_configuration_identity": config,
        },
    )
    monkeypatch.setattr(wrapper, "git_object_bytes", lambda *args: b"generator source")
    monkeypatch.setattr(wrapper, "validate_wrapper_isolation", lambda *args, **kwargs: {})

    manifest = wrapper.run(args)
    assert manifest["repaired_generator_source_path"] == wrapper.REPAIRED_GENERATOR_SOURCE_PATH
    assert {path.name for path in output_dir.iterdir()} == wrapper.EXPECTED_ARTIFACT_NAMES
    assert json.loads((output_dir / wrapper.INVOCATION_JSON_NAME).read_text(encoding="utf-8")) == invocation
    recorded_sha = manifest["repaired_output_sha256"]
    assert recorded_sha == hashlib.sha256((output_dir / wrapper.REPAIRED_JSONL_NAME).read_bytes()).hexdigest()
