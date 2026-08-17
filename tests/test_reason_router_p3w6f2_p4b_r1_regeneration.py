from __future__ import annotations

from contextlib import contextmanager
import json
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from scripts import analyze_reason_router_p3w6f2_p4b_r1_regeneration as analyzer
from scripts import build_controlled_v5 as generator
from scripts import regenerate_reason_router_p3w6f2_p4b_r1_structured as regen


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@contextmanager
def repo_scoped_reports_tmp() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix=".p3w6f2_p4c_pytest_",
        dir=repo_root() / "reports",
    ) as temp_dir:
        yield Path(temp_dir)


def authority_pair_ids() -> list[str]:
    summary = regen.load_json(repo_root() / regen.LEVEL1_SUMMARY_PATH)
    return regen.derive_authorized_f2_pair_ids(summary)


def historical_rows() -> list[dict]:
    return regen.load_jsonl(repo_root() / regen.HISTORICAL_DATASET_PATH)


def regenerated_case() -> tuple[list[dict], list[dict], dict]:
    baseline = historical_rows()
    replay, audit = generator.build_controlled_records_with_f2_p4b_r1_regeneration_audit(
        regen.pair_count(baseline),
        set(authority_pair_ids()),
    )
    projected = regen.project_to_historical_topology(replay, baseline)
    return baseline, projected, audit


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def write_payloads(execution_dir: Path, payloads: dict[str, bytes]) -> None:
    execution_dir.mkdir()
    for name in regen.EXPECTED_ARTIFACT_NAMES:
        (execution_dir / name).write_bytes(payloads[name])


def payload_case(commit: str = "f") -> dict[str, bytes]:
    return regen.build_regenerated_payload(
        repo_root=repo_root(),
        historical_rows=historical_rows(),
        authorized_pair_ids=authority_pair_ids(),
        execution_commit=commit * 40,
        authority_hashes=regen.verify_authority_files(repo_root()),
    )


def test_exact_119_pair_authority_and_357_member_triples():
    pair_ids = authority_pair_ids()
    assert len(pair_ids) == 119
    assert len(set(pair_ids)) == 119
    baseline, regenerated, audit = regenerated_case()
    assert len(audit["regenerated_members"]) == 357
    for pair_id in pair_ids:
        assert {f"{pair_id}__{name}" for name in ("none", "paraphrase", "polarity_flip")} <= {
            row["id"] for row in regenerated
        }


def test_all_seven_predicate_base_mappings_are_exact():
    assert generator.F2_P4B_R1_REQUIRED_PREDICATE_BASES == {
        "approved": "approve",
        "delivered": "deliver",
        "launched": "launch",
        "opened": "open",
        "published": "publish",
        "restored": "restore",
        "selected": "select",
    }
    observed = {
        member["semantic_predicate"]: member["base_predicate"]
        for member in regenerated_case()[2]["regenerated_members"]
    }
    assert observed == generator.F2_P4B_R1_REQUIRED_PREDICATE_BASES


def test_authority_cardinality_missing_mapping_and_unexpected_predicate_rejections(monkeypatch):
    pair_ids = authority_pair_ids()
    with pytest.raises(ValueError, match="F2_P4B_R1_AUTHORITY_CARDINALITY_MISMATCH"):
        generator.build_controlled_records_with_f2_p4b_r1_regeneration(300, set(pair_ids[:-1]))
    monkeypatch.setitem(generator._BASE_PREDICATE_BY_INFLECTED, "approved", "approv")
    with pytest.raises(ValueError, match="F2_P4B_R1_BASE_FORM_COVERAGE_UNRESOLVED"):
        generator.build_controlled_records_with_f2_p4b_r1_regeneration(300, set(pair_ids))
    monkeypatch.setattr(generator, "F2_P4B_R1_REQUIRED_PREDICATE_BASES", {"approved": "approve"})
    with pytest.raises(ValueError, match="F2_P4B_R1_AUTHORIZED_PREDICATE_SET_MISMATCH"):
        generator.build_controlled_records_with_f2_p4b_r1_regeneration(300, set(pair_ids))


def test_official_rejects_substituted_119_pair_universe_with_same_count():
    pair_ids = authority_pair_ids()
    substituted = pair_ids[1:] + ["generated_fact_151"]
    assert len(substituted) == 119
    assert set(substituted) != set(pair_ids)
    with pytest.raises(regen.P4BRegenerationError, match="AUTHORIZED_PAIR_UNIVERSE_NOT_EXACT_FROZEN_AUTHORITY"):
        regen.build_regenerated_payload(
            repo_root=repo_root(),
            historical_rows=historical_rows(),
            authorized_pair_ids=substituted,
            execution_commit="a" * 40,
            authority_hashes=regen.verify_authority_files(repo_root()),
        )


def test_analyzer_rejects_substituted_exact_member_universe(tmp_path: Path):
    payloads = regen.build_regenerated_payload(
        repo_root=repo_root(),
        historical_rows=historical_rows(),
        authorized_pair_ids=authority_pair_ids(),
        execution_commit="b" * 40,
        authority_hashes=regen.verify_authority_files(repo_root()),
    )
    execution_dir = tmp_path / ("reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_" + "b" * 40)
    write_payloads(execution_dir, payloads)
    members = regen.load_jsonl(execution_dir / regen.MEMBERS_NAME)
    members[0]["pair_id"] = "generated_fact_151"
    members[0]["member_id"] = "generated_fact_151__none"
    (execution_dir / regen.MEMBERS_NAME).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in members) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(analyzer.P4BAnalysisError, match="AUTHORIZED_PAIR_UNIVERSE_NOT_EXACT_FROZEN_AUTHORITY|AUTHORIZED_MEMBER_UNIVERSE_NOT_EXACT_FROZEN_AUTHORITY"):
        analyzer.analyze_execution_dir(repo_root(), execution_dir)


def test_frozen_authority_and_dataset_hash_spoofs_rejected(monkeypatch):
    original = dict(regen.EXPECTED_FROZEN_AUTHORITY_SHA256)
    wrong = dict(regen.EXPECTED_FROZEN_AUTHORITY_SHA256)
    wrong[regen.LEVEL1_SUMMARY_PATH] = "0" * 64
    monkeypatch.setattr(regen, "EXPECTED_FROZEN_AUTHORITY_SHA256", wrong)
    with pytest.raises(regen.P4BRegenerationError, match="AUTHORITY_ARTIFACT_SHA_MISMATCH"):
        regen.verify_authority_files(repo_root())

    wrong_spec = dict(original)
    wrong_spec[regen.SPEC_PATH] = "0" * 64
    monkeypatch.setattr(regen, "EXPECTED_FROZEN_AUTHORITY_SHA256", wrong_spec)
    with pytest.raises(regen.P4BRegenerationError, match="AUTHORITY_ARTIFACT_SHA_MISMATCH"):
        regen.verify_authority_files(repo_root())

    monkeypatch.setattr(regen, "EXPECTED_FROZEN_AUTHORITY_SHA256", original)
    wrong_dataset = dict(original)
    wrong_dataset[regen.HISTORICAL_DATASET_PATH] = "0" * 64
    monkeypatch.setattr(regen, "EXPECTED_FROZEN_AUTHORITY_SHA256", wrong_dataset)
    monkeypatch.setattr(regen, "EXPECTED_HISTORICAL_DATASET_SHA256", "0" * 64)
    with pytest.raises(regen.P4BRegenerationError, match="AUTHORITY_ARTIFACT_SHA_MISMATCH|HISTORICAL_DATASET_SHA_MISMATCH"):
        regen.authenticated_frozen_authority(repo_root())


def test_spec_authority_identity_is_separate_from_implementation_execution_identity():
    assert regen.P4B_SPEC_AUTHORITY_COMMIT == "fcc3b9ccaf2bbee33ac18dcef10d50acff54aab4"
    assert not hasattr(regen, "IMPLEMENTATION_AUTHORITY_COMMIT")
    observed = regen.verify_execution_identity(
        repo_root(),
        "1" * 40,
        head_resolver=lambda _root: "1" * 40,
        tracked_clean_checker=lambda _root: True,
    )
    assert observed == "1" * 40
    payloads = regen.build_regenerated_payload(
        repo_root=repo_root(),
        historical_rows=historical_rows(),
        authorized_pair_ids=authority_pair_ids(),
        execution_commit="1" * 40,
        implementation_commit="1" * 40,
        authority_hashes=regen.verify_authority_files(repo_root()),
    )
    summary = json.loads(payloads[regen.SUMMARY_NAME].decode("utf-8"))
    assert summary["p4b_spec_authority_commit"] == regen.P4B_SPEC_AUTHORITY_COMMIT
    assert summary["implementation_commit"] == "1" * 40
    assert summary["execution_commit"] == "1" * 40


def test_dirty_official_execution_is_rejected():
    with pytest.raises(regen.P4BRegenerationError, match="TRACKED_WORKTREE_DIRTY"):
        regen.verify_execution_identity(
            repo_root(),
            "2" * 40,
            head_resolver=lambda _root: "2" * 40,
            tracked_clean_checker=lambda _root: False,
        )


def test_dirty_tracked_worktree_unstaged_diff_rejected(monkeypatch):
    calls = []

    class Result:
        def __init__(self, returncode: int):
            self.returncode = returncode

    def fake_run(args, stderr=None):
        calls.append(tuple(args))
        return Result(1 if "--cached" not in args else 0)

    monkeypatch.setattr(regen.subprocess, "run", fake_run)
    with pytest.raises(regen.P4BRegenerationError, match="TRACKED_WORKTREE_DIRTY"):
        regen.verify_execution_identity(
            repo_root(),
            "3" * 40,
            head_resolver=lambda _root: "3" * 40,
        )
    assert any("--cached" not in call for call in calls)
    assert any("--cached" in call for call in calls)


def test_dirty_index_cached_diff_rejected_separately(monkeypatch):
    calls = []

    class Result:
        def __init__(self, returncode: int):
            self.returncode = returncode

    def fake_run(args, stderr=None):
        calls.append(tuple(args))
        return Result(1 if "--cached" in args else 0)

    monkeypatch.setattr(regen.subprocess, "run", fake_run)
    with pytest.raises(regen.P4BRegenerationError, match="TRACKED_WORKTREE_DIRTY"):
        regen.verify_execution_identity(
            repo_root(),
            "4" * 40,
            head_resolver=lambda _root: "4" * 40,
        )
    assert any("--cached" not in call for call in calls)
    assert any("--cached" in call for call in calls)


def test_structured_generation_and_negative_grammar_for_canonical_and_paraphrase():
    _baseline, regenerated, audit = regenerated_case()
    rows = by_id(regenerated)
    for member in audit["regenerated_members"]:
        row = rows[member["member_id"]]
        if member["intervention_type"] in {"none", "paraphrase"}:
            assert member["generation_root"] == "structured_fact"
            assert member["old_text_used_for_generation"] is False
            assert f"did not {member['base_predicate']}" in row["evidence"]
            assert f"did not {member['semantic_predicate']}" not in row["evidence"]


def test_paraphrase_and_polarity_flip_independent_from_canonical_text():
    baseline, regenerated, _audit = regenerated_case()
    rows = by_id(regenerated)
    for pair_id in authority_pair_ids():
        canonical = rows[f"{pair_id}__none"]["evidence"]
        paraphrase = rows[f"{pair_id}__paraphrase"]["evidence"]
        polarity = rows[f"{pair_id}__polarity_flip"]["evidence"]
        assert paraphrase != canonical
        assert polarity != canonical
        assert "During " in paraphrase


def test_only_authorized_none_and_paraphrase_evidence_changes_and_polarity_bytes_unchanged():
    baseline, regenerated, _audit = regenerated_case()
    old = by_id(baseline)
    new = by_id(regenerated)
    authorized = set(authority_pair_ids())
    for row_id, before in old.items():
        after = new[row_id]
        deltas = regen.changed_fields(before, after)
        intervention = before["intervention_type"]
        if before["pair_id"] in authorized and intervention in {"none", "paraphrase"}:
            assert deltas == ["evidence"]
        else:
            assert deltas == []


def test_semantic_labels_ids_order_split_and_non_f2_preserved():
    baseline, regenerated, _audit = regenerated_case()
    assert [row["id"] for row in baseline] == [row["id"] for row in regenerated]
    assert regen.split_identity(baseline) == regen.split_identity(regenerated)
    old = by_id(baseline)
    for after in regenerated:
        before = old[after["id"]]
        for field in (
            "id",
            "pair_id",
            "claim",
            "final_label",
            "frame_compatible_label",
            "predicate_covered_label",
            "sufficiency_label",
            "polarity_label",
            "primary_failure_type",
            "intervention_type",
        ):
            assert before[field] == after[field]


def test_authorized_label_contracts():
    _baseline, regenerated, _audit = regenerated_case()
    rows = by_id(regenerated)
    for pair_id in authority_pair_ids():
        assert rows[f"{pair_id}__none"]["final_label"] == "REFUTE"
        assert rows[f"{pair_id}__none"]["polarity_label"] == "REFUTE"
        assert rows[f"{pair_id}__none"]["primary_failure_type"] == "none"
        assert rows[f"{pair_id}__paraphrase"]["final_label"] == "REFUTE"
        assert rows[f"{pair_id}__paraphrase"]["polarity_label"] == "REFUTE"
        assert rows[f"{pair_id}__paraphrase"]["primary_failure_type"] == "none"
        assert rows[f"{pair_id}__polarity_flip"]["final_label"] == "SUPPORT"
        assert rows[f"{pair_id}__polarity_flip"]["polarity_label"] == "SUPPORT"
        assert rows[f"{pair_id}__polarity_flip"]["primary_failure_type"] == "polarity"


def test_deterministic_replay_and_old_malformed_text_not_generation_input():
    first = regenerated_case()
    second = regenerated_case()
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert all(member["old_text_used_for_generation"] is False for member in first[2]["regenerated_members"])


def test_analyzer_rejects_forged_pass_flags_and_historical_text_patch(tmp_path: Path):
    payloads = regen.build_regenerated_payload(
        repo_root=repo_root(),
        historical_rows=historical_rows(),
        authorized_pair_ids=authority_pair_ids(),
        execution_commit="9" * 40,
        authority_hashes=regen.verify_authority_files(repo_root()),
    )
    execution_dir = tmp_path / ("reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_" + "9" * 40)
    write_payloads(execution_dir, payloads)
    baseline = by_id(historical_rows())
    dataset = regen.load_jsonl(execution_dir / regen.FULL_DATASET_NAME)
    target = next(row for row in dataset if row["intervention_type"] == "none" and row["pair_id"] in set(authority_pair_ids()))
    target["evidence"] = baseline[target["id"]]["evidence"]
    (execution_dir / regen.FULL_DATASET_NAME).write_bytes(regen.deterministic_jsonl_bytes(dataset, dataset_rows=True))
    members = regen.load_jsonl(execution_dir / regen.MEMBERS_NAME)
    for member in members:
        if member["member_id"] == target["id"]:
            member["old_text_used_for_generation"] = False
    (execution_dir / regen.MEMBERS_NAME).write_bytes(regen.deterministic_jsonl_bytes(members))
    audit_rows = regen.load_jsonl(execution_dir / regen.AUDIT_NAME)
    for row in audit_rows:
        row["old_text_isolation_status"] = "PASS"
        row["member_audit_status"] = "PASS"
    (execution_dir / regen.AUDIT_NAME).write_bytes(regen.deterministic_jsonl_bytes(audit_rows))
    with pytest.raises(analyzer.P4BAnalysisError, match="STRUCTURED_SOURCE_REPLAY_MISMATCH|REGENERATED_DATASET_SHA_MISMATCH"):
        analyzer.analyze_execution_dir(repo_root(), execution_dir)


def test_materializer_payload_contract_and_analyzer_accepts_repo_scoped_execution_dir():
    baseline = historical_rows()
    payloads = regen.build_regenerated_payload(
        repo_root=repo_root(),
        historical_rows=baseline,
        authorized_pair_ids=authority_pair_ids(),
        execution_commit="f" * 40,
        authority_hashes=regen.verify_authority_files(repo_root()),
    )
    with repo_scoped_reports_tmp() as temp_root:
        execution_dir = temp_root / ("reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_" + "f" * 40)
        write_payloads(execution_dir, payloads)
        result = analyzer.analyze_execution_dir(repo_root(), execution_dir)
        assert result["analysis_status"] == "PASS"
        assert result["authorized_pair_count"] == 119
        assert result["authorized_member_count"] == 357


def rewrite_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("final_label", "SUPPORT"),
        ("frame_compatible_label", 0),
        ("predicate_covered_label", 0),
        ("sufficiency_label", 0),
        ("polarity_label", "SUPPORT"),
        ("primary_failure_type", "frame"),
        ("claim", "forged claim"),
        ("intervention_type", "predicate_swap"),
        ("regenerated_row_sha256", "0" * 64),
    ],
)
def test_analyzer_rejects_member_artifact_semantic_forgery(tmp_path: Path, field: str, forged_value):
    payloads = payload_case("8")
    execution_dir = tmp_path / ("reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_" + "8" * 40)
    write_payloads(execution_dir, payloads)
    members = regen.load_jsonl(execution_dir / regen.MEMBERS_NAME)
    target = next(row for row in members if row["intervention_type"] == "none")
    target[field] = forged_value
    (execution_dir / regen.MEMBERS_NAME).write_bytes(regen.deterministic_jsonl_bytes(members))
    with pytest.raises(analyzer.P4BAnalysisError, match=f"MEMBER_ARTIFACT_FIELD_MISMATCH:.*:{field}|AUTHORIZED_MEMBER_UNIVERSE_NOT_EXACT_FROZEN_AUTHORITY|INTERVENTION_TRIPLE_MISMATCH"):
        analyzer.analyze_execution_dir(repo_root(), execution_dir)


def test_analyzer_rejects_row_reorder_even_with_hashes_updated(tmp_path: Path):
    payloads = payload_case("7")
    execution_dir = tmp_path / ("reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_" + "7" * 40)
    write_payloads(execution_dir, payloads)
    rows = regen.load_jsonl(execution_dir / regen.FULL_DATASET_NAME)
    rows[0], rows[1] = rows[1], rows[0]
    (execution_dir / regen.FULL_DATASET_NAME).write_bytes(regen.deterministic_jsonl_bytes(rows, dataset_rows=True))
    physical_sha = regen.file_sha256(execution_dir / regen.FULL_DATASET_NAME)
    semantic_sha = regen.semantic_dataset_hash(rows)
    summary = regen.load_json(execution_dir / regen.SUMMARY_NAME)
    summary["regenerated_dataset_sha256"] = physical_sha
    summary["regenerated_dataset_semantic_sha256"] = semantic_sha
    rewrite_json(execution_dir / regen.SUMMARY_NAME, summary)
    isolation = regen.load_json(execution_dir / regen.ISOLATION_NAME)
    isolation["regenerated_dataset_sha256"] = physical_sha
    rewrite_json(execution_dir / regen.ISOLATION_NAME, isolation)
    with pytest.raises(analyzer.P4BAnalysisError, match="ROW_ORDER_MISMATCH"):
        analyzer.analyze_execution_dir(repo_root(), execution_dir)


def test_source_provenance_spoof_conflict_and_partial_output_rejected(tmp_path: Path):
    baseline = historical_rows()
    payloads = regen.build_regenerated_payload(
        repo_root=repo_root(),
        historical_rows=baseline,
        authorized_pair_ids=authority_pair_ids(),
        execution_commit="e" * 40,
        authority_hashes=regen.verify_authority_files(repo_root()),
    )
    output_dir = tmp_path / ("reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_" + "e" * 40)
    output_dir.mkdir()
    (output_dir / regen.FULL_DATASET_NAME).write_bytes(payloads[regen.FULL_DATASET_NAME] + b"\n")
    with pytest.raises(regen.P4BRegenerationError, match="OUTPUT_ARTIFACT_SET_MISMATCH|OUTPUT_ARTIFACT_CONFLICT"):
        regen.publish_artifacts(output_dir, payloads)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / regen.FULL_DATASET_NAME).write_bytes(payloads[regen.FULL_DATASET_NAME])
    with pytest.raises(analyzer.P4BAnalysisError, match="ARTIFACT_SET_INCOMPLETE_OR_UNEXPECTED"):
        analyzer.load_required_artifacts(partial)


def test_historical_authority_immutability_and_existing_f1_api_regression():
    before = regen.file_sha256(repo_root() / regen.HISTORICAL_DATASET_PATH)
    f1_ids = sorted(generator._negative_polarity_flip_row_ids(generator.fact_templates_for_count(300)))[:121]
    repaired, audit = generator.build_controlled_records_with_f1_polarity_repair_audit(300, set(f1_ids))
    assert len(repaired) == len(generator.build_controlled_records(300))
    assert audit["repair_consumed_row_ids"] == sorted(f1_ids)
    assert regen.file_sha256(repo_root() / regen.HISTORICAL_DATASET_PATH) == before


def test_analyzer_rejects_spoofed_source_hash_and_unauthorized_delta(tmp_path: Path):
    payloads = regen.build_regenerated_payload(
        repo_root=repo_root(),
        historical_rows=historical_rows(),
        authorized_pair_ids=authority_pair_ids(),
        execution_commit="d" * 40,
        authority_hashes=regen.verify_authority_files(repo_root()),
    )
    execution_dir = tmp_path / ("reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_" + "d" * 40)
    write_payloads(execution_dir, payloads)
    summary = json.loads((execution_dir / regen.SUMMARY_NAME).read_text(encoding="utf-8"))
    summary["regenerated_dataset_sha256"] = "0" * 64
    (execution_dir / regen.SUMMARY_NAME).write_text(json.dumps(summary, sort_keys=True), encoding="utf-8", newline="\n")
    with pytest.raises(analyzer.P4BAnalysisError, match="REGENERATED_DATASET_SHA_MISMATCH"):
        analyzer.analyze_execution_dir(repo_root(), execution_dir)


def test_projected_materializer_does_not_create_time_swap_rows():
    _baseline, regenerated, _audit = regenerated_case()
    assert "time_swap" not in {row["intervention_type"] for row in regenerated}
    assert len(regenerated) == len(historical_rows())
