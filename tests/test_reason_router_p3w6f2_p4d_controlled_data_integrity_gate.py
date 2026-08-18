from __future__ import annotations

from dataclasses import replace
import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import validate_reason_router_p3w6f2_p4d_controlled_data_integrity_gate as gate


FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifact_dir() -> Path:
    return repo_root() / gate.P4B_ARTIFACT_DIR


def historical_rows() -> list[dict]:
    return gate.load_jsonl(repo_root() / gate.HISTORICAL_DATASET_PATH)


def regenerated_rows() -> list[dict]:
    return gate.load_jsonl(artifact_dir() / gate.REGENERATED_DATASET_NAME)


def git_sha(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root()), *args],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def frozen_for(*, art: Path | None = None) -> gate.FrozenInputs:
    hashes = dict(gate.P4B_ARTIFACT_HASHES)
    if art is not None:
        for name in gate.EXPECTED_ARTIFACT_NAMES:
            hashes[name] = gate.file_sha256(art / name)
    return replace(
        gate.FrozenInputs(),
        p4b_artifact_dir=str(art) if art else gate.P4B_ARTIFACT_DIR,
        regenerated_dataset_sha256=hashes[gate.REGENERATED_DATASET_NAME],
        p4b_artifact_hashes=hashes,
    )


def copy_fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    hist = tmp_path / "controlled_v5_v3_without_time_swap.jsonl"
    shutil.copy2(repo_root() / gate.HISTORICAL_DATASET_PATH, hist)
    art = tmp_path / (
        "reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
        + gate.P4B_EXECUTION_COMMIT
    )
    shutil.copytree(artifact_dir(), art)
    return hist, art


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def mutate_artifact_hashes(frozen: gate.FrozenInputs, art: Path, *names: str) -> gate.FrozenInputs:
    hashes = dict(frozen.artifact_hashes())
    for name in names:
        hashes[name] = gate.file_sha256(art / name)
    return replace(frozen, p4b_artifact_hashes=hashes, regenerated_dataset_sha256=hashes.get(gate.REGENERATED_DATASET_NAME, frozen.regenerated_dataset_sha256))


def assert_blocked(report: dict, token: str) -> None:
    assert report["decision_token"] == gate.BLOCKED_TOKEN
    assert any(token in reason for reason in report["failure_reasons"])


def assert_full_sha(value: str) -> None:
    assert FULL_SHA_RE.fullmatch(value) is not None


def test_positive_valid_300x12_topology_seed174_and_deterministic_report():
    report1 = gate.validate_gate(
        repo_root(),
        historical_dataset=gate.HISTORICAL_DATASET_PATH,
        p4b_artifact_dir=gate.P4B_ARTIFACT_DIR,
        stage185_split_seed=174,
    )
    report2 = gate.validate_gate(
        repo_root(),
        historical_dataset=gate.HISTORICAL_DATASET_PATH,
        p4b_artifact_dir=gate.P4B_ARTIFACT_DIR,
        stage185_split_seed=174,
    )
    assert report1 == report2
    assert report1["decision_token"] == gate.PASS_TOKEN
    assert_full_sha(report1["validator_commit"])
    assert_full_sha(report1["current_head"])
    assert report1["validator_commit"] == git_sha("log", "-1", "--format=%H", "--", gate.VALIDATOR_SOURCE_PATH)
    assert report1["current_head"] == git_sha("rev-parse", "HEAD")
    assert report1["validator_commit"] != "UNCOMMITTED_IMPLEMENTATION_STATIC_ONLY"
    assert report1["phase"] == "P4-D GATE 5 CONTROLLED-DATA INTEGRITY VALIDATION - GATE 6 ONLY ON PASS"
    assert report1["training_admission_released"] is False
    assert report1["row_count_historical"] == 3600
    assert report1["row_count_regenerated"] == 3600
    rows = regenerated_rows()
    assert len({row["pair_id"] for row in rows}) == 300
    assert {row["intervention_type"] for row in rows} == gate.ADMITTED_FAMILIES


def test_git_provenance_helpers_resolve_actual_read_only_state():
    assert gate.resolve_current_head(repo_root()) == git_sha("rev-parse", "HEAD")
    assert gate.resolve_validator_source_commit(repo_root()) == git_sha("log", "-1", "--format=%H", "--", gate.VALIDATOR_SOURCE_PATH)
    assert_full_sha(gate.resolve_current_head(repo_root()))
    assert_full_sha(gate.resolve_validator_source_commit(repo_root()))


@pytest.mark.parametrize(
    ("git_output", "expected_failure"),
    [
        ("not-a-sha\n", "CURRENT_HEAD:MALFORMED"),
        ("", "CURRENT_HEAD:AMBIGUOUS"),
    ],
)
def test_unresolved_or_malformed_current_head_fails_closed(monkeypatch, git_output: str, expected_failure: str):
    def fake_check_output(args, **kwargs):
        assert args[3:5] == ["rev-parse", "HEAD"]
        return git_output

    monkeypatch.setattr(gate.subprocess, "check_output", fake_check_output)
    report = gate.validate_gate(repo_root(), historical_dataset=gate.HISTORICAL_DATASET_PATH, p4b_artifact_dir=gate.P4B_ARTIFACT_DIR)
    assert_blocked(report, expected_failure)
    assert report["provenance_status"] == "BLOCKED"
    assert report["current_head"] == gate.UNRESOLVED_GIT_PROVENANCE
    assert report["decision_token"] != gate.PASS_TOKEN


def test_current_head_subprocess_failure_fails_closed(monkeypatch):
    def fake_check_output(args, **kwargs):
        raise subprocess.CalledProcessError(128, args)

    monkeypatch.setattr(gate.subprocess, "check_output", fake_check_output)
    report = gate.validate_gate(repo_root(), historical_dataset=gate.HISTORICAL_DATASET_PATH, p4b_artifact_dir=gate.P4B_ARTIFACT_DIR)
    assert_blocked(report, "CURRENT_HEAD:UNRESOLVED")
    assert report["provenance_status"] == "BLOCKED"
    assert report["decision_token"] != gate.PASS_TOKEN


@pytest.mark.parametrize(
    ("git_output", "expected_failure"),
    [
        ("not-a-sha\n", "VALIDATOR_SOURCE_COMMIT:MALFORMED"),
        ("1" * 40 + "\n" + "2" * 40 + "\n", "VALIDATOR_SOURCE_COMMIT:AMBIGUOUS"),
    ],
)
def test_unresolved_or_malformed_validator_source_commit_fails_closed(monkeypatch, git_output: str, expected_failure: str):
    actual_head = git_sha("rev-parse", "HEAD")

    def fake_check_output(args, **kwargs):
        if args[3:5] == ["rev-parse", "HEAD"]:
            return actual_head + "\n"
        assert args[3:6] == ["log", "-1", "--format=%H"]
        assert args[-2:] == ["--", gate.VALIDATOR_SOURCE_PATH]
        return git_output

    monkeypatch.setattr(gate.subprocess, "check_output", fake_check_output)
    report = gate.validate_gate(repo_root(), historical_dataset=gate.HISTORICAL_DATASET_PATH, p4b_artifact_dir=gate.P4B_ARTIFACT_DIR)
    assert_blocked(report, expected_failure)
    assert report["provenance_status"] == "BLOCKED"
    assert report["current_head"] == actual_head
    assert report["validator_commit"] == gate.UNRESOLVED_GIT_PROVENANCE
    assert report["decision_token"] != gate.PASS_TOKEN


def test_empty_validator_source_commit_output_fails_closed(monkeypatch):
    actual_head = git_sha("rev-parse", "HEAD")

    def fake_check_output(args, **kwargs):
        if args[3:5] == ["rev-parse", "HEAD"]:
            return actual_head + "\n"
        assert args[3:6] == ["log", "-1", "--format=%H"]
        assert args[-2:] == ["--", gate.VALIDATOR_SOURCE_PATH]
        return ""

    monkeypatch.setattr(gate.subprocess, "check_output", fake_check_output)
    with pytest.raises(gate.GateBlocked, match="VALIDATOR_SOURCE_COMMIT:AMBIGUOUS"):
        gate.resolve_validator_source_commit(repo_root())
    report = gate.validate_gate(repo_root(), historical_dataset=gate.HISTORICAL_DATASET_PATH, p4b_artifact_dir=gate.P4B_ARTIFACT_DIR)
    assert_blocked(report, "VALIDATOR_SOURCE_COMMIT:AMBIGUOUS")
    assert report["current_head"] == actual_head
    assert report["validator_commit"] == gate.UNRESOLVED_GIT_PROVENANCE
    assert report["provenance_status"] == "BLOCKED"
    assert report["decision_token"] != gate.PASS_TOKEN


def test_validator_source_commit_subprocess_failure_fails_closed(monkeypatch):
    actual_head = git_sha("rev-parse", "HEAD")

    def fake_check_output(args, **kwargs):
        if args[3:5] == ["rev-parse", "HEAD"]:
            return actual_head + "\n"
        raise subprocess.CalledProcessError(128, args)

    monkeypatch.setattr(gate.subprocess, "check_output", fake_check_output)
    report = gate.validate_gate(repo_root(), historical_dataset=gate.HISTORICAL_DATASET_PATH, p4b_artifact_dir=gate.P4B_ARTIFACT_DIR)
    assert_blocked(report, "VALIDATOR_SOURCE_COMMIT:UNRESOLVED")
    assert report["provenance_status"] == "BLOCKED"
    assert report["decision_token"] != gate.PASS_TOKEN


def test_validator_commit_and_current_head_may_differ(monkeypatch):
    current = "a" * 40
    validator = "b" * 40

    def fake_check_output(args, **kwargs):
        if args[3:5] == ["rev-parse", "HEAD"]:
            return current + "\n"
        if args[3:6] == ["log", "-1", "--format=%H"]:
            return validator + "\n"
        raise AssertionError(args)

    monkeypatch.setattr(gate.subprocess, "check_output", fake_check_output)
    report = gate.validate_gate(repo_root(), historical_dataset=gate.HISTORICAL_DATASET_PATH, p4b_artifact_dir=gate.P4B_ARTIFACT_DIR)
    assert report["decision_token"] == gate.PASS_TOKEN
    assert report["provenance_status"] == "PASS"
    assert report["current_head"] == current
    assert report["validator_commit"] == validator
    assert report["validator_commit"] != report["current_head"]


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (lambda rows: rows[:-12], "ROW_COUNT_MISMATCH"),
        (lambda rows: [dict(row, intervention_type="time_swap") if i == 0 else row for i, row in enumerate(rows)], "INTERVENTION_ENUM_MISMATCH"),
        (lambda rows: [dict(row, intervention_type="extra_family") if i == 0 else row for i, row in enumerate(rows)], "INTERVENTION_ENUM_MISMATCH"),
        (lambda rows: [dict(row, intervention_type="entity_swap") if row["id"].endswith("__none") else row for row in rows], "DUPLICATE_PAIR_INTERVENTION"),
        (lambda rows: rows[:1] + rows, "ROW_COUNT_MISMATCH"),
    ],
)
def test_topology_adversaries_block(mutator, code):
    with pytest.raises(gate.GateBlocked, match=code):
        gate.validate_dataset_structure(mutator(copy.deepcopy(regenerated_rows())), label="MUTATED")


def test_missing_family_and_nonrectangular_topology_block():
    rows = copy.deepcopy(regenerated_rows())
    target = next(row for row in rows if row["pair_id"] == rows[0]["pair_id"] and row["intervention_type"] == "role_swap")
    target["intervention_type"] = "entity_swap"
    with pytest.raises(gate.GateBlocked, match="DUPLICATE_PAIR_INTERVENTION"):
        gate.validate_dataset_structure(rows, label="MUTATED")


def test_pair_count_drift_blocks():
    rows = copy.deepcopy(regenerated_rows())
    fresh_pair_id = "fresh_pair_count_drift_pair"
    assert fresh_pair_id not in {row["pair_id"] for row in rows}
    rows[0]["pair_id"] = fresh_pair_id
    assert len(rows) == gate.ROW_COUNT
    assert len({row["pair_id"] for row in rows}) == gate.PAIR_COUNT + 1
    with pytest.raises(gate.GateBlocked, match="PAIR_COUNT_MISMATCH"):
        gate.validate_dataset_structure(rows, label="MUTATED")


def test_schema_drift_blocks():
    rows = copy.deepcopy(regenerated_rows())
    rows[0]["extra"] = "field"
    with pytest.raises(gate.GateBlocked, match="DATASET_SCHEMA_FIELD_ORDER_MISMATCH"):
        gate.validate_dataset_structure(rows, label="MUTATED")


def test_row_order_label_and_canonical_linkage_drift_block():
    hist = historical_rows()
    regen = copy.deepcopy(regenerated_rows())
    auth = gate._authorized_pair_ids_from_artifacts(artifact_dir())
    swapped = copy.deepcopy(regen)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(gate.GateBlocked, match="ROW_ORDER_DRIFT"):
        gate.validate_identity_label_linkage_and_deltas(hist, swapped, authorized_pair_ids=auth)
    label_drift = copy.deepcopy(regen)
    label_drift[0]["final_label"] = "SUPPORT" if label_drift[0]["final_label"] != "SUPPORT" else "REFUTE"
    with pytest.raises(gate.GateBlocked, match="LABEL_DRIFT"):
        gate.validate_identity_label_linkage_and_deltas(hist, label_drift, authorized_pair_ids=auth)
    claim_drift = copy.deepcopy(regen)
    target = next(row for row in claim_drift if row["intervention_type"] == "paraphrase")
    target["claim"] += " drift"
    with pytest.raises(gate.GateBlocked, match="CLAIM_DRIFT"):
        gate.validate_identity_label_linkage_and_deltas(hist, claim_drift, authorized_pair_ids=auth)


def test_exact_238_authorized_deltas_and_delta_adversaries():
    hist = historical_rows()
    regen = copy.deepcopy(regenerated_rows())
    auth = gate._authorized_pair_ids_from_artifacts(artifact_dir())
    hist_by_id = by_id(hist)
    result = gate.validate_identity_label_linkage_and_deltas(hist, regen, authorized_pair_ids=auth)
    assert result["changed_row_count"] == 238
    assert result["changed_pair_count"] == 119
    f2_field = copy.deepcopy(regen)
    target = next(row for row in f2_field if row["pair_id"] in auth and row["intervention_type"] == "none")
    target["evidence"] += " extra"
    target["primary_failure_type"] = "frame"
    with pytest.raises(gate.GateBlocked, match="LABEL_DRIFT"):
        gate.validate_identity_label_linkage_and_deltas(hist, f2_field, authorized_pair_ids=auth)
    non_f2 = copy.deepcopy(regen)
    target = next(row for row in non_f2 if row["pair_id"] in auth and row["intervention_type"] == "entity_swap")
    target["evidence"] += " extra"
    restored = next(
        row
        for row in non_f2
        if row["pair_id"] == target["pair_id"]
        and row["intervention_type"] in gate.AUTHORIZED_CHANGED_INTERVENTIONS
        and row["evidence"] != hist_by_id[row["id"]]["evidence"]
    )
    restored["evidence"] = hist_by_id[restored["id"]]["evidence"]
    with pytest.raises(gate.GateBlocked, match="NON_F2_MUTATION"):
        gate.validate_identity_label_linkage_and_deltas(hist, non_f2, authorized_pair_ids=auth)
    polarity = copy.deepcopy(regen)
    target = next(row for row in polarity if row["pair_id"] in auth and row["intervention_type"] == "polarity_flip")
    target["evidence"] += " extra"
    restored = next(
        row
        for row in polarity
        if row["pair_id"] == target["pair_id"]
        and row["intervention_type"] in gate.AUTHORIZED_CHANGED_INTERVENTIONS
        and row["evidence"] != hist_by_id[row["id"]]["evidence"]
    )
    restored["evidence"] = hist_by_id[restored["id"]]["evidence"]
    with pytest.raises(gate.GateBlocked, match="F2_POLARITY_FLIP_MUTATION"):
        gate.validate_identity_label_linkage_and_deltas(hist, polarity, authorized_pair_ids=auth)


def test_seed17_and_any_non174_seed_rejected():
    rows = regenerated_rows()
    assert gate.replay_stage185_split(rows, seed=174, ratio=0.2)
    with pytest.raises(gate.GateBlocked, match="STAGE185_SPLIT_SEED_REJECTED:17"):
        gate.replay_stage185_split(rows, seed=17, ratio=0.2)
    with pytest.raises(gate.GateBlocked, match="STAGE185_SPLIT_SEED_REJECTED:175"):
        gate.replay_stage185_split(rows, seed=175, ratio=0.2)


def test_split_drift_blocks():
    hist = historical_rows()
    regen = copy.deepcopy(regenerated_rows())
    regen[0]["pair_id"] = "generated_fact_299"
    with pytest.raises(gate.GateBlocked, match="STAGE185_SPLIT_DRIFT"):
        gate.validate_split_identity(hist, regen, seed=174, ratio=0.2)


def test_mandatory_artifacts_8_9_10_and_hash_spoof_block(tmp_path: Path):
    hist, art = copy_fixture(tmp_path)
    (art / gate.COMPAT_ROWS_NAME).unlink()
    with pytest.raises(gate.GateBlocked, match="P4B_ARTIFACT_SET_MISSING_OR_EXTRA"):
        gate.validate_artifact_set(art, gate.P4B_ARTIFACT_HASHES)
    _hist, art = copy_fixture(tmp_path / "spoof")
    hashes = dict(frozen_for(art=art).artifact_hashes())
    hashes[gate.COMPAT_SUMMARY_NAME] = "0" * 64
    with pytest.raises(gate.GateBlocked, match="P4B_ARTIFACT_HASH_MISMATCH"):
        gate.validate_artifact_set(art, hashes)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("compatibility_pass_count", 356, "COMPATIBILITY_SUMMARY_VALUE_MISMATCH:compatibility_pass_count"),
        ("compatibility_fail_count", 1, "COMPATIBILITY_SUMMARY_VALUE_MISMATCH:compatibility_fail_count"),
        ("compatibility_unresolved_count", 1, "COMPATIBILITY_SUMMARY_VALUE_MISMATCH:compatibility_unresolved_count"),
        ("training_admission_released", True, "COMPATIBILITY_SUMMARY_VALUE_MISMATCH:training_admission_released"),
    ],
)
def test_artifact9_exact_summary_contract(field, value, code, tmp_path: Path):
    _hist, art = copy_fixture(tmp_path)
    summary = gate.load_json(art / gate.COMPAT_SUMMARY_NAME)
    summary[field] = value
    summary["compatibility_gate_status"] = "PASS"
    write_json(art / gate.COMPAT_SUMMARY_NAME, summary)
    frozen = mutate_artifact_hashes(frozen_for(art=art), art, gate.COMPAT_SUMMARY_NAME)
    with pytest.raises(gate.GateBlocked, match=code):
        gate.validate_compatibility_artifacts(art, frozen)


def test_artifact_8_9_10_schema_counts_and_provenance_bindings():
    gate.validate_compatibility_artifacts(artifact_dir(), gate.FrozenInputs())
    rows = gate.load_jsonl(artifact_dir() / gate.COMPAT_ROWS_NAME)
    assert len(rows) == 357
    assert sum(1 for row in rows if "predicate" in row["raw_stage185_changed_axes"]) == 238


def test_stage185_source_hash_spoof_blocks():
    frozen = replace(gate.FrozenInputs(), stage185_source_script_sha256="0" * 64)
    with pytest.raises(gate.GateBlocked, match="STAGE185_SOURCE_SHA_MISMATCH"):
        gate.validate_authority_inputs(repo_root(), frozen)


def test_training_admission_released_in_row_rejected(tmp_path: Path):
    _hist, art = copy_fixture(tmp_path)
    rows = gate.load_jsonl(art / gate.COMPAT_ROWS_NAME)
    rows[0]["training_admission_effect"]["training_admission_released"] = True
    write_jsonl(art / gate.COMPAT_ROWS_NAME, rows)
    frozen = mutate_artifact_hashes(frozen_for(art=art), art, gate.COMPAT_ROWS_NAME)
    with pytest.raises(gate.GateBlocked, match="COMPATIBILITY_ROW_TRAINING_RELEASED"):
        gate.validate_compatibility_artifacts(art, frozen)


def test_malformed_json_jsonl_line_endings_and_duplicate_keys_rejected(tmp_path: Path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"a": 1, "a": 2}\n', encoding="utf-8", newline="\n")
    with pytest.raises(gate.GateBlocked, match="DUPLICATE_JSON_KEY"):
        gate.load_json(bad_json)
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"a": 1}\r\n', encoding="utf-8", newline="")
    with pytest.raises(gate.GateBlocked, match="NON_LF_LINE_ENDING"):
        gate.load_jsonl(bad_jsonl)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_non_standard_numeric_constants_rejected(tmp_path: Path, constant: str):
    bad_json = tmp_path / f"bad_{constant.replace('-', 'neg_')}.json"
    bad_json.write_text(f'{{"value": {constant}}}\n', encoding="utf-8", newline="\n")
    with pytest.raises(gate.GateBlocked, match=f"NON_STANDARD_JSON_CONSTANT:{constant}"):
        gate.load_json(bad_json)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_jsonl_non_standard_numeric_constants_rejected(tmp_path: Path, constant: str):
    bad_jsonl = tmp_path / f"bad_{constant.replace('-', 'neg_')}.jsonl"
    bad_jsonl.write_text(f'{{"value": {constant}}}\n', encoding="utf-8", newline="\n")
    with pytest.raises(gate.GateBlocked, match=f"NON_STANDARD_JSON_CONSTANT:{constant}"):
        gate.load_jsonl(bad_jsonl)


def test_syntactically_malformed_jsonl_rejected(tmp_path: Path):
    bad_jsonl = tmp_path / "truncated.jsonl"
    bad_jsonl.write_text('{"value": 1\n', encoding="utf-8", newline="\n")
    with pytest.raises(gate.GateBlocked, match="MALFORMED_JSONL"):
        gate.load_jsonl(bad_jsonl)


def test_report_tokens_are_pass_or_blocked_only_for_validator_scope():
    report = gate.validate_gate(repo_root(), historical_dataset=gate.HISTORICAL_DATASET_PATH, p4b_artifact_dir=gate.P4B_ARTIFACT_DIR)
    assert report["decision_token"] in {gate.PASS_TOKEN, gate.BLOCKED_TOKEN}
    assert report["training_admission_released"] is False
    assert gate.FAIL_TOKEN == "P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_FAIL"
