from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import materialize_reason_router_p3w6f2_p4b_r1_stage185_compatibility as compat
from scripts import regenerate_reason_router_p3w6f2_p4b_r1_structured as regen


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def authority_pair_ids() -> list[str]:
    return regen.derive_authorized_f2_pair_ids(regen.load_json(repo_root() / regen.LEVEL1_SUMMARY_PATH))


def payload_case() -> dict[str, bytes]:
    return regen.build_regenerated_payload(
        repo_root=repo_root(),
        historical_rows=regen.load_jsonl(repo_root() / regen.HISTORICAL_DATASET_PATH),
        authorized_pair_ids=authority_pair_ids(),
        execution_commit="c" * 40,
        authority_hashes=regen.verify_authority_files(repo_root()),
    )


def artifacts_from_payload(payloads: dict[str, bytes]) -> dict[str, object]:
    return {
        regen.MEMBERS_NAME: [json.loads(line) for line in payloads[regen.MEMBERS_NAME].decode("utf-8").splitlines()],
        regen.AUDIT_NAME: [json.loads(line) for line in payloads[regen.AUDIT_NAME].decode("utf-8").splitlines()],
    }


def write_regeneration_execution_dir(tmp_path: Path, commit: str = "c") -> Path:
    payloads = payload_case()
    execution_dir = tmp_path / ("reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_" + commit * 40)
    execution_dir.mkdir()
    for name in regen.EXPECTED_ARTIFACT_NAMES:
        (execution_dir / name).write_bytes(payloads[name])
    return execution_dir


def compatibility_payloads(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    execution_dir = write_regeneration_execution_dir(tmp_path)
    artifacts = artifacts_from_payload(payload_case())
    rows = compat.build_compatibility_rows(
        members=artifacts[regen.MEMBERS_NAME],
        audit_rows=artifacts[regen.AUDIT_NAME],
        authorized_member_ids=set(regen.authenticated_frozen_authority(repo_root())["authorized_member_ids"]),
    )
    summary = compat.build_summary(rows)
    rows_bytes = regen.deterministic_jsonl_bytes(rows)
    summary_bytes = regen.deterministic_json_bytes(summary)
    provenance = compat.build_provenance(
        repo_root=repo_root(),
        execution_dir=execution_dir,
        output_dir=execution_dir,
        rows_sha256=regen.sha256_bytes(rows_bytes),
        summary_sha256=regen.sha256_bytes(summary_bytes),
        coverage_path=execution_dir / regen.COVERAGE_NAME,
    )
    return execution_dir, {
        compat.ROWS_NAME: rows_bytes,
        compat.SUMMARY_NAME: summary_bytes,
        compat.PROVENANCE_NAME: regen.deterministic_json_bytes(provenance),
    }


def test_raw_stage185_predicate_observation_retained_and_effective_compatibility_scoped():
    artifacts = artifacts_from_payload(payload_case())
    rows = compat.build_compatibility_rows(
        members=artifacts[regen.MEMBERS_NAME],
        audit_rows=artifacts[regen.AUDIT_NAME],
    )
    assert len(rows) == 357
    negatives = [row for row in rows if row["intervention_type"] in {"none", "paraphrase"}]
    assert negatives
    assert all("predicate" in row["raw_stage185_changed_axes"] for row in negatives)
    assert all(row["raw_stage185_statuses"]["raw_observation_preserved"] is True for row in rows)
    assert all(row["effective_compatibility_status"] == "PASS" for row in rows)


def test_raw_stage185_observation_is_computed_not_hardcoded(monkeypatch):
    calls = []

    def fake_changed_axes(row, canonical, fact, intended):
        calls.append((row["id"], canonical["id"], fact["pair_id"], tuple(sorted(intended))))
        return {"predicate", "sentinel_axis"}

    monkeypatch.setattr(compat.stage185, "changed_axes", fake_changed_axes)
    artifacts = artifacts_from_payload(payload_case())
    rows = compat.build_compatibility_rows(
        members=artifacts[regen.MEMBERS_NAME][:1],
        audit_rows=artifacts[regen.AUDIT_NAME],
    )
    assert calls
    assert "sentinel_axis" in rows[0]["raw_stage185_changed_axes"]


def test_level3_and_training_admission_remain_false():
    artifacts = artifacts_from_payload(payload_case())
    rows = compat.build_compatibility_rows(
        members=artifacts[regen.MEMBERS_NAME],
        audit_rows=artifacts[regen.AUDIT_NAME],
    )
    summary = compat.build_summary(rows)
    assert summary["compatibility_gate_status"] == "PASS"
    assert summary["training_admission_released"] is False
    assert all(row["training_admission_effect"]["training_admission_released"] is False for row in rows)
    assert all(row["training_admission_effect"]["level3_admission_released"] is False for row in rows)


def test_genuine_predicate_mutation_is_not_masked():
    artifacts = artifacts_from_payload(payload_case())
    audit_rows = copy.deepcopy(artifacts[regen.AUDIT_NAME])
    target = next(row for row in audit_rows if row["intervention_type"] == "none")
    target["regenerated_row"]["evidence"] = target["regenerated_row"]["evidence"].replace("did not ", "did not wrongly_")
    rows = compat.build_compatibility_rows(
        members=artifacts[regen.MEMBERS_NAME],
        audit_rows=audit_rows,
    )
    mutated = next(row for row in rows if row["member_id"] == target["member_id"])
    assert mutated["effective_compatibility_status"] == "FAIL"
    assert "regenerated_base_span_missing" in mutated["effective_reason_codes"]
    assert compat.build_summary(rows)["compatibility_gate_status"] == "FAIL"


def test_non_predicate_field_delta_is_not_masked():
    artifacts = artifacts_from_payload(payload_case())
    audit_rows = copy.deepcopy(artifacts[regen.AUDIT_NAME])
    target = next(row for row in audit_rows if row["intervention_type"] == "paraphrase")
    target["regenerated_row"]["claim"] = target["regenerated_row"]["claim"] + " changed"
    rows = compat.build_compatibility_rows(
        members=artifacts[regen.MEMBERS_NAME],
        audit_rows=audit_rows,
    )
    mutated = next(row for row in rows if row["member_id"] == target["member_id"])
    assert mutated["effective_compatibility_status"] == "FAIL"
    assert "unauthorized_field_delta:claim" in mutated["effective_reason_codes"]


def test_polarity_flip_must_remain_byte_identical():
    artifacts = artifacts_from_payload(payload_case())
    audit_rows = copy.deepcopy(artifacts[regen.AUDIT_NAME])
    target = next(row for row in audit_rows if row["intervention_type"] == "polarity_flip")
    target["regenerated_row"]["evidence"] = target["regenerated_row"]["evidence"] + " extra"
    rows = compat.build_compatibility_rows(
        members=artifacts[regen.MEMBERS_NAME],
        audit_rows=audit_rows,
    )
    mutated = next(row for row in rows if row["member_id"] == target["member_id"])
    assert mutated["effective_compatibility_status"] == "FAIL"
    assert "polarity_flip_row_delta" in mutated["effective_reason_codes"]


def test_non_f2_member_cannot_receive_compatibility():
    artifacts = artifacts_from_payload(payload_case())
    member = copy.deepcopy(artifacts[regen.MEMBERS_NAME][0])
    member["pair_id"] = "generated_fact_1"
    member["member_id"] = "generated_fact_1__none"
    with pytest.raises(compat.Stage185CompatibilityError, match="COMPATIBILITY_UNAUTHORIZED_MEMBER"):
        compat.build_compatibility_rows(
            members=[member],
            audit_rows=artifacts[regen.AUDIT_NAME],
            authorized_member_ids=set(regen.authenticated_frozen_authority(repo_root())["authorized_member_ids"]),
        )


def test_compatibility_artifact_names_schema_and_provenance(tmp_path: Path):
    execution_dir = write_regeneration_execution_dir(tmp_path)
    output_dir = execution_dir
    result = compat.materialize(repo_root(), execution_dir, output_dir)
    assert result["status"] == "PASS"
    assert compat.EXPECTED_ARTIFACT_NAMES <= {entry.name for entry in output_dir.iterdir()}
    rows = regen.load_jsonl(output_dir / compat.ROWS_NAME)
    summary = regen.load_json(output_dir / compat.SUMMARY_NAME)
    provenance = regen.load_json(output_dir / compat.PROVENANCE_NAME)
    assert rows[0]["schema_version"] == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_ROW_V1"
    assert summary["schema_version"] == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1"
    assert provenance["schema_version"] == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1"
    assert provenance["stage185_source_script"] == compat.STAGE185_SOURCE_SCRIPT
    assert provenance["compatibility_rows_sha256"] == regen.file_sha256(output_dir / compat.ROWS_NAME)
    assert provenance["compatibility_summary_sha256"] == regen.file_sha256(output_dir / compat.SUMMARY_NAME)


def test_conflicting_compatibility_output_rejected(tmp_path: Path):
    output_dir, payloads = compatibility_payloads(tmp_path)
    (output_dir / compat.ROWS_NAME).write_text("conflict\n", encoding="utf-8")
    (output_dir / compat.SUMMARY_NAME).write_text("{}\n", encoding="utf-8")
    (output_dir / compat.PROVENANCE_NAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(compat.Stage185CompatibilityError, match="COMPATIBILITY_OUTPUT_CONFLICT"):
        compat.publish_artifacts(output_dir, payloads)


def test_partial_compatibility_output_rejected(tmp_path: Path):
    output_dir, payloads = compatibility_payloads(tmp_path)
    (output_dir / compat.ROWS_NAME).write_text("partial\n", encoding="utf-8")
    with pytest.raises(compat.Stage185CompatibilityError, match="COMPATIBILITY_OUTPUT_PARTIAL_PREEXISTING"):
        compat.publish_artifacts(output_dir, payloads)


def test_mid_publication_failure_leaves_no_partial_compatibility_set(tmp_path: Path):
    output_dir, payloads = compatibility_payloads(tmp_path)
    calls = []

    def failing_promote(source: Path, target: Path) -> None:
        calls.append((source, target))
        raise RuntimeError("simulated promotion failure")

    with pytest.raises(RuntimeError, match="simulated promotion failure"):
        compat.publish_artifacts(output_dir, payloads, promote_directory=failing_promote)
    assert calls
    assert not any((output_dir / name).exists() for name in compat.EXPECTED_ARTIFACT_NAMES)
    assert regen.EXPECTED_ARTIFACT_NAMES <= {entry.name for entry in output_dir.iterdir()}


def test_compatibility_publish_success_and_idempotent_replay(tmp_path: Path):
    output_dir, payloads = compatibility_payloads(tmp_path)
    assert compat.publish_artifacts(output_dir, payloads) == "PUBLISHED"
    assert all((output_dir / name).is_file() for name in compat.EXPECTED_ARTIFACT_NAMES)
    before = {name: (output_dir / name).read_bytes() for name in compat.EXPECTED_ARTIFACT_NAMES}
    assert compat.publish_artifacts(output_dir, payloads) == "IDEMPOTENT_PASS"
    after = {name: (output_dir / name).read_bytes() for name in compat.EXPECTED_ARTIFACT_NAMES}
    assert after == before


def test_stage185_source_hash_spoof_rejected(monkeypatch):
    monkeypatch.setattr(regen, "EXPECTED_STAGE185_SOURCE_SHA256", "0" * 64)
    with pytest.raises(compat.Stage185CompatibilityError, match="STAGE185_SOURCE_SHA_MISMATCH"):
        compat.verify_stage185_source(repo_root())


def test_missing_member_audit_rejected():
    artifacts = artifacts_from_payload(payload_case())
    audit_rows = artifacts[regen.AUDIT_NAME][1:]
    with pytest.raises(compat.Stage185CompatibilityError, match="AUDIT_ROW_MISSING"):
        compat.build_compatibility_rows(
            members=artifacts[regen.MEMBERS_NAME],
            audit_rows=audit_rows,
        )
