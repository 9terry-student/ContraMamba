"""Focused design coverage for the standalone P4-X prelaunch static control.

These tests deliberately exercise real temporary Git repositories for canonical
blob, CRLF, staged, unstaged, untracked, and symlink defenses.  They are not
run by the Phase-IV implementation-authoring authority.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import hashlib
import ast
import random
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "validate_reason_router_p4x_prelaunch_static_control.py"
SPEC = importlib.util.spec_from_file_location("p4x_static", MODULE_PATH)
assert SPEC and SPEC.loader
p4x = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p4x)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, check=True)
    return result.stdout.strip()


def commit_blob_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"; root.mkdir(); git(root, "init"); git(root, "config", "user.email", "p4x@test.invalid"); git(root, "config", "user.name", "P4X")
    (root / "input.txt").write_text("one\ntwo\n", encoding="utf-8", newline="\n")
    git(root, "add", "input.txt"); git(root, "commit", "-m", "input")
    return root


def identity_repo(tmp_path: Path) -> tuple[Path, str]:
    root = commit_blob_repo(tmp_path)
    git(root, "branch", "-M", p4x.EXPECTED_BRANCH)
    head = git(root, "rev-parse", "HEAD")
    git(root, "remote", "add", "origin", ".")
    git(root, "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")
    remote_ref = f"refs/remotes/origin/{p4x.EXPECTED_BRANCH}"
    git(root, "update-ref", remote_ref, head)
    git(root, "config", f"branch.{p4x.EXPECTED_BRANCH}.remote", "origin")
    git(root, "config", f"branch.{p4x.EXPECTED_BRANCH}.merge", f"refs/heads/{p4x.EXPECTED_BRANCH}")
    assert git(root, "rev-parse", "--abbrev-ref", "@{upstream}") == p4x.EXPECTED_UPSTREAM
    assert git(root, "rev-parse", "@{upstream}") == head
    return root, head


def phase2_lineage_repo(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    root = commit_blob_repo(tmp_path)
    root_commit = git(root, "rev-parse", "HEAD")
    (root / "activation.txt").write_text("activation\n", encoding="utf-8")
    git(root, "add", "activation.txt"); git(root, "commit", "-m", "activation")
    activation = git(root, "rev-parse", "HEAD")
    (root / "freeze.txt").write_text("freeze\n", encoding="utf-8")
    git(root, "add", "freeze.txt"); git(root, "commit", "-m", "freeze")
    freeze = git(root, "rev-parse", "HEAD")
    (root / "future.txt").write_text("future\n", encoding="utf-8")
    git(root, "add", "future.txt"); git(root, "commit", "-m", "future")
    return root, root_commit, activation, freeze, git(root, "rev-parse", "HEAD")


def _bind_temp_phase2(monkeypatch: pytest.MonkeyPatch, root_commit: str, activation: str, freeze: str) -> None:
    monkeypatch.setattr(p4x, "PHASE_II_ACTIVATION_COMMIT", activation)
    monkeypatch.setattr(p4x, "PHASE_II_ACTIVATION_PARENT", root_commit)
    monkeypatch.setattr(p4x, "PHASE_II_EVIDENCE_FREEZE_COMMIT", freeze)
    monkeypatch.setattr(p4x, "PHASE_II_EVIDENCE_FREEZE_PARENT", activation)


def test_phase2_activation_and_freeze_lineage_succeeds_in_real_git_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, parent, activation, freeze, expected_head = phase2_lineage_repo(tmp_path)
    _bind_temp_phase2(monkeypatch, parent, activation, freeze)
    p4x._validate_phase2_lineage(root, expected_head)


def test_phase2_missing_and_wrong_activation_commit_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, parent, activation, freeze, expected_head = phase2_lineage_repo(tmp_path)
    _bind_temp_phase2(monkeypatch, parent, activation, freeze)
    monkeypatch.setattr(p4x, "PHASE_II_ACTIVATION_COMMIT", "0" * 40)
    with pytest.raises(p4x.ContractError, match="P4X_PHASE2_ACTIVATION_COMMIT_UNAVAILABLE"):
        p4x._validate_phase2_lineage(root, expected_head)


def test_phase2_wrong_existing_activation_sha_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, parent, activation, freeze, expected_head = phase2_lineage_repo(tmp_path)
    _bind_temp_phase2(monkeypatch, parent, activation, freeze)
    monkeypatch.setattr(p4x, "PHASE_II_ACTIVATION_COMMIT", parent)
    with pytest.raises(p4x.ContractError, match="P4X_PHASE2_ACTIVATION_PARENT_MISMATCH"):
        p4x._validate_phase2_lineage(root, expected_head)


def test_phase2_wrong_activation_parent_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, parent, activation, freeze, expected_head = phase2_lineage_repo(tmp_path)
    _bind_temp_phase2(monkeypatch, parent, activation, freeze)
    monkeypatch.setattr(p4x, "PHASE_II_ACTIVATION_PARENT", "0" * 40)
    with pytest.raises(p4x.ContractError, match="P4X_PHASE2_ACTIVATION_PARENT_MISMATCH"):
        p4x._validate_phase2_lineage(root, expected_head)


def test_phase2_wrong_freeze_sha_and_parent_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, parent, activation, freeze, expected_head = phase2_lineage_repo(tmp_path)
    _bind_temp_phase2(monkeypatch, parent, activation, freeze)
    monkeypatch.setattr(p4x, "PHASE_II_EVIDENCE_FREEZE_COMMIT", "0" * 40)
    with pytest.raises(p4x.ContractError, match="P4X_PHASE2_FREEZE_COMMIT_UNAVAILABLE"):
        p4x._validate_phase2_lineage(root, expected_head)
    monkeypatch.setattr(p4x, "PHASE_II_EVIDENCE_FREEZE_COMMIT", freeze)
    monkeypatch.setattr(p4x, "PHASE_II_EVIDENCE_FREEZE_PARENT", parent)
    with pytest.raises(p4x.ContractError, match="P4X_PHASE2_FREEZE_PARENT_MISMATCH"):
        p4x._validate_phase2_lineage(root, expected_head)


def test_phase2_freeze_must_ancestor_expected_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, parent, activation, freeze, _ = phase2_lineage_repo(tmp_path)
    _bind_temp_phase2(monkeypatch, parent, activation, freeze)
    git(root, "checkout", "-b", "non-descendant", activation)
    (root / "alternate.txt").write_text("alternate\n", encoding="utf-8")
    git(root, "add", "alternate.txt"); git(root, "commit", "-m", "alternate")
    with pytest.raises(p4x.ContractError, match="P4X_PHASE2_FREEZE_NOT_ANCESTOR_OF_IMPLEMENTATION_ANCHOR"):
        p4x._validate_phase2_lineage(root, git(root, "rev-parse", "HEAD"))


def test_authenticated_head_bytes_uses_git_canonical_bytes_and_survives_crlf(tmp_path: Path) -> None:
    root = commit_blob_repo(tmp_path); blob = git(root, "rev-parse", "HEAD:input.txt")
    (root / "input.txt").write_text("one\ntwo\n", encoding="utf-8", newline="\r\n")
    # A normal CRLF checkout has no content change under text eol normalization.
    git(root, "config", "core.autocrlf", "true")
    expected = hashlib.sha256(b"one\ntwo\n").hexdigest()
    assert p4x.authenticated_head_bytes(root, "input.txt", blob, expected) == b"one\ntwo\n"


def test_authenticated_head_bytes_rejects_wrong_blob_and_canonical_sha(tmp_path: Path) -> None:
    root = commit_blob_repo(tmp_path)
    blob = git(root, "rev-parse", "HEAD:input.txt")
    expected = hashlib.sha256(b"one\ntwo\n").hexdigest()
    with pytest.raises(p4x.ContractError, match="P4X_GIT_BLOB_MISMATCH"):
        p4x.authenticated_head_bytes(root, "input.txt", "0" * 40, expected)
    with pytest.raises(p4x.ContractError, match="P4X_GIT_LF_SHA256_MISMATCH"):
        p4x.authenticated_head_bytes(root, "input.txt", blob, "0" * 64)


@pytest.mark.parametrize("content, label, contract", [(b"{not-json}\n", "SOURCE", "P4X_SOURCE_MALFORMED_JSONL"), (b"[]\n", "SIDECAR", "P4X_SIDECAR_ROW_NOT_OBJECT")])
def test_dataset_and_sidecar_malformed_records_fail(content: bytes, label: str, contract: str) -> None:
    with pytest.raises(p4x.ContractError, match=contract):
        p4x._parse_jsonl(content, label)


@pytest.mark.parametrize("mutation, contract", [("staged", "P4X_STAGED_DIRTY"), ("unstaged", "P4X_UNSTAGED_DIRTY"), ("untracked", "P4X_UNTRACKED_INPUT"), ("symlink", "P4X_SYMLINK_SUBSTITUTION")])
def test_real_git_identity_failure_modes(tmp_path: Path, mutation: str, contract: str) -> None:
    root = commit_blob_repo(tmp_path); blob = git(root, "rev-parse", "HEAD:input.txt"); sha = hashlib.sha256(b"one\ntwo\n").hexdigest()
    if mutation == "staged":
        (root / "input.txt").write_text("changed\n"); git(root, "add", "input.txt")
        with pytest.raises(p4x.ContractError, match=contract): p4x.authenticated_head_bytes(root, "input.txt", blob, sha)
    elif mutation == "unstaged":
        (root / "input.txt").write_text("changed\n")
        with pytest.raises(p4x.ContractError, match=contract): p4x.authenticated_head_bytes(root, "input.txt", blob, sha)
    elif mutation == "untracked":
        (root / "other.txt").write_text("x\n")
        with pytest.raises(p4x.ContractError, match=contract): p4x.authenticated_head_bytes(root, "other.txt", blob, sha)
    else:
        if not hasattr(os, "symlink"): pytest.skip("os.symlink API is unavailable for the symlink fixture")
        (root / "target.txt").write_text("one\ntwo\n"); (root / "input.txt").unlink()
        try:
            os.symlink("target.txt", root / "input.txt")
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows privilege or policy prevents symlink fixture creation (WinError 1314)")
            raise
        with pytest.raises(p4x.ContractError, match=contract): p4x.authenticated_head_bytes(root, "input.txt", blob, sha)


@pytest.mark.parametrize("value", [None, "", "a" * 39, "a" * 41, "g" * 40, "HEAD", "main", "refs/heads/main", "v1.0"])
def test_expected_head_rejects_omission_refs_and_non_full_object_names(value: str | None) -> None:
    with pytest.raises(p4x.ContractError, match="P4X_EXPECTED_HEAD_INVALID"):
        p4x.normalize_expected_head(value)


def test_expected_head_accepts_only_case_normalized_full_sha_and_parser_requires_it() -> None:
    sha = "A1" * 20
    assert p4x.normalize_expected_head(sha) == sha.lower()
    with pytest.raises(SystemExit):
        p4x.main(["--repo-root", "."])


def test_real_git_implementation_anchor_identity_contracts(tmp_path: Path) -> None:
    root, head = identity_repo(tmp_path)
    p4x._validate_repository_identity(root, head.upper())
    (root / "next.txt").write_text("next\n", encoding="utf-8")
    git(root, "add", "next.txt")
    git(root, "commit", "-m", "next")
    current = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", f"refs/remotes/origin/{p4x.EXPECTED_BRANCH}", current)
    assert git(root, "rev-parse", "--abbrev-ref", "@{upstream}") == p4x.EXPECTED_UPSTREAM
    assert git(root, "rev-parse", "@{upstream}") == current
    # A later authority-like execution commit remains valid with its older implementation anchor.
    p4x._validate_repository_identity(root, head)
    git(root, "checkout", "--orphan", "non-ancestor")
    git(root, "rm", "-rf", ".")
    (root / "alternate.txt").write_text("alternate\n", encoding="utf-8")
    git(root, "add", "alternate.txt"); git(root, "commit", "-m", "alternate")
    git(root, "branch", "-f", p4x.EXPECTED_BRANCH, "HEAD")
    git(root, "checkout", p4x.EXPECTED_BRANCH)
    alternate = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", f"refs/remotes/origin/{p4x.EXPECTED_BRANCH}", alternate)
    assert git(root, "rev-parse", "--abbrev-ref", "@{upstream}") == p4x.EXPECTED_UPSTREAM
    assert git(root, "rev-parse", "@{upstream}") == alternate
    with pytest.raises(p4x.ContractError, match="P4X_IMPLEMENTATION_ANCHOR_NOT_ANCESTOR_OF_CURRENT_HEAD"):
        p4x._validate_repository_identity(root, head)
    git(root, "branch", "-M", "wrong-branch")
    with pytest.raises(p4x.ContractError, match="P4X_BRANCH_IDENTITY_MISMATCH"):
        p4x._validate_repository_identity(root, head)


def test_wrong_configured_upstream_ref_fails_even_at_same_sha(tmp_path: Path) -> None:
    root, head = identity_repo(tmp_path)
    git(root, "update-ref", "refs/remotes/origin/wrong", head)
    git(root, "config", f"branch.{p4x.EXPECTED_BRANCH}.merge", "refs/heads/wrong")
    wrong_upstream_tip = git(root, "rev-parse", "@{upstream}")
    configured_symbolic_upstream = git(root, "rev-parse", "--abbrev-ref", "@{upstream}")
    assert wrong_upstream_tip == head
    assert configured_symbolic_upstream == "origin/wrong"
    assert configured_symbolic_upstream != p4x.EXPECTED_UPSTREAM
    with pytest.raises(p4x.ContractError, match="P4X_UPSTREAM_REF_IDENTITY_MISMATCH"):
        p4x._validate_repository_identity(root, head)


def test_current_head_must_equal_configured_upstream_tip(tmp_path: Path) -> None:
    root, head = identity_repo(tmp_path)
    (root / "next.txt").write_text("next\n", encoding="utf-8")
    git(root, "add", "next.txt"); git(root, "commit", "-m", "ahead")
    with pytest.raises(p4x.ContractError, match="P4X_UPSTREAM_TIP_MISMATCH"):
        p4x._validate_repository_identity(root, head)


def test_implementation_anchor_unavailable_and_non_commit_fail(tmp_path: Path) -> None:
    root, head = identity_repo(tmp_path)
    with pytest.raises(p4x.ContractError, match="P4X_IMPLEMENTATION_ANCHOR_COMMIT_UNAVAILABLE"):
        p4x._validate_repository_identity(root, "0" * 40)
    blob = git(root, "rev-parse", "HEAD:input.txt")
    with pytest.raises(p4x.ContractError, match="P4X_IMPLEMENTATION_ANCHOR_OBJECT_TYPE_MISMATCH"):
        p4x._validate_repository_identity(root, blob)


@pytest.mark.parametrize("mutation, contract", [("unstaged", "P4X_UNSTAGED_DIRTY"), ("staged", "P4X_STAGED_DIRTY"), ("untracked", "P4X_UNTRACKED_WORKTREE")])
def test_repository_cleanliness_rejects_all_dirty_states(tmp_path: Path, mutation: str, contract: str) -> None:
    root = commit_blob_repo(tmp_path)
    if mutation == "unstaged":
        (root / "input.txt").write_text("changed\n", encoding="utf-8")
    elif mutation == "staged":
        (root / "input.txt").write_text("changed\n", encoding="utf-8")
        git(root, "add", "input.txt")
    else:
        (root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(p4x.ContractError, match=contract):
        p4x._require_clean_repository(root)


def test_ahead_behind_contract_is_explicit_and_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, head = identity_repo(tmp_path)
    original_git = p4x._git

    def simulated_git(repo: Path, *args: str, **kwargs: object) -> str | bytes:
        if args == ("rev-list", "--left-right", "--count", "HEAD...@{upstream}"):
            return "1\t0"
        return original_git(repo, *args, **kwargs)

    monkeypatch.setattr(p4x, "_git", simulated_git)
    with pytest.raises(p4x.ContractError, match="P4X_AHEAD_BEHIND_MISMATCH"):
        p4x._validate_repository_identity(root, head)


def test_repository_identity_uses_canonical_upstream_selector_only() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    identity = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_validate_repository_identity")
    selectors = [value for _, value in sorted((argument.lineno, argument.value) for call in ast.walk(identity) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_git" for argument in call.args if isinstance(argument, ast.Constant) and argument.value in {"@{up}", "@{upstream}", "HEAD...@{up}", "HEAD...@{upstream}"})]
    assert selectors == ["@{upstream}", "HEAD...@{upstream}"]


def test_semantic_and_split_helpers_fail_closed() -> None:
    row = {field: "x" for field in p4x.SOURCE_FIELDS}; row.update({"frame_compatible_label": 1, "predicate_covered_label": 1, "sufficiency_label": 1})
    assert len(p4x._dataset_semantic([row])) == 64
    row["frame_compatible_label"] = True
    with pytest.raises(p4x.ContractError, match="P4X_SOURCE_BINARY_INVALID"): p4x._dataset_semantic([row])
    with pytest.raises(p4x.ContractError, match="P4X_PAIR_UNIVERSE_INVALID"): p4x.recompute_split([])


def test_identity_hash_uses_lf_terminated_records() -> None:
    authoritative = hashlib.sha256(b"alpha\nbeta\n").hexdigest()
    previous = hashlib.sha256(b"alpha\nbeta").hexdigest()
    assert p4x._identity_hash(["alpha", "beta"]) == authoritative
    assert previous != authoritative


def test_authenticated_dataset_split_identities_use_pair_and_row_serializations() -> None:
    content = p4x.authenticated_head_bytes(MODULE_PATH.parents[1], p4x.DATASET, p4x.DATASET_BLOB, p4x.DATASET_SHA256)
    rows = p4x._parse_jsonl(content, "SOURCE")
    train, dev, audit = p4x.recompute_split(rows)
    assert audit == p4x.SPLIT_IDENTITIES
    assert audit["pair_universe_sha256"] == "41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2"
    assert audit["shuffled_pair_sha256"] == "ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55"
    assert audit["train_pair_sha256"] == "f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049"
    assert audit["dev_pair_sha256"] == "30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4"
    assert audit["ordered_train_row_sha256"] == "478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8"
    assert audit["ordered_dev_row_sha256"] == "7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4"
    assert p4x._identity_hash([str(row["id"]) for row in train]) != audit["ordered_train_row_sha256"]
    assert p4x._identity_hash([str(row["id"]) for row in dev]) != audit["ordered_dev_row_sha256"]


def test_split_identity_mutations_remain_fail_closed() -> None:
    rows = p4x._parse_jsonl((MODULE_PATH.parents[1] / p4x.DATASET).read_bytes(), "SOURCE")
    pair_mutation = [{**row, "pair_id": "pair-universe-mutation" if row["pair_id"] == rows[0]["pair_id"] else row["pair_id"]} for row in rows]
    with pytest.raises(p4x.ContractError, match="P4X_SPLIT_IDENTITY_MISMATCH"):
        p4x.recompute_split(pair_mutation)
    with pytest.raises(p4x.ContractError, match="P4X_SPLIT_IDENTITY_MISMATCH"):
        p4x.recompute_split(list(reversed(rows)))


@pytest.mark.parametrize("n, expected_dev", [(2, 1), (3, 1), (5, 1), (6, 1), (9, 2), (300, 60), (301, 60)])
def test_frozen_split_formula_is_cardinality_dependent(n: int, expected_dev: int) -> None:
    pair_ids = [f"pair-{index:03d}" for index in range(n)]
    shuffled, train_pairs, dev_pairs = p4x.split_pair_ids(pair_ids)
    assert len(dev_pairs) == min(n - 1, max(1, round(n * 0.2))) == expected_dev
    assert len(train_pairs) + len(dev_pairs) == n
    assert not set(train_pairs) & set(dev_pairs)
    expected_shuffled = sorted(pair_ids)
    random.Random(8192).shuffle(expected_shuffled)
    assert shuffled == expected_shuffled
    assert dev_pairs == expected_shuffled[:expected_dev]


def test_current_300_pair_membership_is_seed8192_deterministic_and_not_a_fixed_slice() -> None:
    pair_ids = [f"pair-{index:03d}" for index in range(300)]
    shuffled, train_pairs, dev_pairs = p4x.split_pair_ids(pair_ids)
    assert len(train_pairs) == 240 and len(dev_pairs) == 60
    assert dev_pairs == shuffled[:60]
    source = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    split = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "split_pair_ids")
    assignments = {node.targets[0].id: node.value for node in split.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert "dev_count" in assignments
    assert not any(isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) and isinstance(node.slice.upper, ast.Constant) and node.slice.upper.value == 60 for node in ast.walk(split))


def test_recompute_split_rejects_wrong_frozen_hash_count_and_leakage(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"id": f"row-{index}", "pair_id": f"pair-{index:03d}"} for index in range(300)]
    with pytest.raises(p4x.ContractError, match="P4X_SPLIT_IDENTITY_MISMATCH"):
        p4x.recompute_split(rows)
    monkeypatch.setattr(p4x, "split_pair_ids", lambda ids: (sorted(ids), sorted(ids), sorted(ids)))
    with pytest.raises(p4x.ContractError, match="P4X_PAIR_LEAKAGE"):
        p4x.recompute_split(rows)


def test_frozen_provenance_split_identities_are_accepted() -> None:
    actual = json.loads((MODULE_PATH.parents[1] / p4x.PROVENANCE).read_text(encoding="utf-8"))
    assert set(actual["split_identities"]) == set(p4x.SPLIT_IDENTITIES) | {"historical_seed174_dev_pair_sha256"}
    assert actual["split_identities"]["historical_seed174_dev_pair_sha256"] == "259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d"
    p4x._validate_provenance(actual)


@pytest.mark.parametrize("mutation", ["remove_current", "alter_current", "remove_historical", "alter_historical", "extra"])
def test_provenance_split_identity_contract_rejects_mutations(mutation: str) -> None:
    value = {"schema_version": p4x.PROVENANCE_SCHEMA, "sidecar_schema_version": p4x.SIDECAR_SCHEMA, "lineage_mode": "revised-seed8192", "p4l_authority_commit": "ff181f565cefa0a28280c084246862286daf1f2d", "split_authority_commit": "b4fbb5666d796161f95ae23612ce2448c25063ee", "builder_source_commit": "149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b", "source_dataset_sha256": p4x.DATASET_SHA256, "source_dataset_semantic_sha256": p4x.DATASET_SEMANTIC_SHA256, "sidecar_physical_sha256": p4x.SIDECAR_SHA256, "sidecar_semantic_sha256": p4x.SIDECAR_SEMANTIC_SHA256, "row_count": 3600, "split_identities": dict(p4x.PROVENANCE_SPLIT_IDENTITIES), "implementation_authorized": True, "artifact_materialization_authorized_by_p4l": False, "training_admission_released": False, "a0_execution_authorized": False, "training_authorized": False, "evaluation_authorized": False, "kaggle_authorized": False, "gpu_authorized": False, "provenance_physical_sha256_self_certified": False}
    identities = value["split_identities"]
    assert isinstance(identities, dict)
    current_key = next(iter(p4x.SPLIT_IDENTITIES))
    if mutation == "remove_current":
        del identities[current_key]
    elif mutation == "alter_current":
        identities[current_key] = "altered"
    elif mutation == "remove_historical":
        del identities["historical_seed174_dev_pair_sha256"]
    elif mutation == "alter_historical":
        identities["historical_seed174_dev_pair_sha256"] = "altered"
    else:
        identities["unknown_split_identity"] = "altered"
    with pytest.raises(p4x.ContractError, match="P4X_PROVENANCE_SPLIT_IDENTITY_MISMATCH"):
        p4x._validate_provenance(value)


def test_provenance_split_identity_contract_is_distinct_from_current_split_audit() -> None:
    assert "historical_seed174_dev_pair_sha256" not in p4x.SPLIT_IDENTITIES
    assert p4x.PROVENANCE_SPLIT_IDENTITIES["historical_seed174_dev_pair_sha256"] == "259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d"


@pytest.mark.parametrize("field", ["schema_version", "lineage_mode", "p4l_authority_commit", "split_authority_commit", "builder_source_commit", "source_dataset_sha256", "source_dataset_semantic_sha256", "sidecar_physical_sha256", "sidecar_semantic_sha256", "implementation_authorized", "training_authorized"])
def test_provenance_schema_lineage_authority_and_flags_are_exact(field: str) -> None:
    value = {"schema_version": p4x.PROVENANCE_SCHEMA, "sidecar_schema_version": p4x.SIDECAR_SCHEMA, "lineage_mode": "revised-seed8192", "p4l_authority_commit": "ff181f565cefa0a28280c084246862286daf1f2d", "split_authority_commit": "b4fbb5666d796161f95ae23612ce2448c25063ee", "builder_source_commit": "149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b", "source_dataset_sha256": p4x.DATASET_SHA256, "source_dataset_semantic_sha256": p4x.DATASET_SEMANTIC_SHA256, "sidecar_physical_sha256": p4x.SIDECAR_SHA256, "sidecar_semantic_sha256": p4x.SIDECAR_SEMANTIC_SHA256, "row_count": 3600, "split_identities": p4x.PROVENANCE_SPLIT_IDENTITIES, "implementation_authorized": True, "artifact_materialization_authorized_by_p4l": False, "training_admission_released": False, "a0_execution_authorized": False, "training_authorized": False, "evaluation_authorized": False, "kaggle_authorized": False, "gpu_authorized": False, "provenance_physical_sha256_self_certified": False}
    value[field] = False if value[field] is True else "historical-seed174"
    with pytest.raises(p4x.ContractError): p4x._validate_provenance(value)


@pytest.mark.parametrize("flag_value", [1, 0, "true", "false", None])
def test_provenance_flags_require_literal_booleans(flag_value: object) -> None:
    value = {"schema_version": p4x.PROVENANCE_SCHEMA, "sidecar_schema_version": p4x.SIDECAR_SCHEMA, "lineage_mode": "revised-seed8192", "p4l_authority_commit": "ff181f565cefa0a28280c084246862286daf1f2d", "split_authority_commit": "b4fbb5666d796161f95ae23612ce2448c25063ee", "builder_source_commit": "149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b", "source_dataset_sha256": p4x.DATASET_SHA256, "source_dataset_semantic_sha256": p4x.DATASET_SEMANTIC_SHA256, "sidecar_physical_sha256": p4x.SIDECAR_SHA256, "sidecar_semantic_sha256": p4x.SIDECAR_SEMANTIC_SHA256, "row_count": 3600, "split_identities": p4x.PROVENANCE_SPLIT_IDENTITIES, "implementation_authorized": True, "artifact_materialization_authorized_by_p4l": False, "training_admission_released": False, "a0_execution_authorized": False, "training_authorized": False, "evaluation_authorized": False, "kaggle_authorized": False, "gpu_authorized": False, "provenance_physical_sha256_self_certified": False}
    value["training_authorized"] = flag_value
    with pytest.raises(p4x.ContractError, match="P4X_PROVENANCE_FLAG_MISMATCH"):
        p4x._validate_provenance(value)


def test_frozen_provenance_does_not_require_external_phase2_lineage_fields() -> None:
    value = {"schema_version": p4x.PROVENANCE_SCHEMA, "sidecar_schema_version": p4x.SIDECAR_SCHEMA, "lineage_mode": "revised-seed8192", "p4l_authority_commit": "ff181f565cefa0a28280c084246862286daf1f2d", "split_authority_commit": "b4fbb5666d796161f95ae23612ce2448c25063ee", "builder_source_commit": "149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b", "source_dataset_sha256": p4x.DATASET_SHA256, "source_dataset_semantic_sha256": p4x.DATASET_SEMANTIC_SHA256, "sidecar_physical_sha256": p4x.SIDECAR_SHA256, "sidecar_semantic_sha256": p4x.SIDECAR_SEMANTIC_SHA256, "row_count": 3600, "split_identities": p4x.PROVENANCE_SPLIT_IDENTITIES, "implementation_authorized": True, "artifact_materialization_authorized_by_p4l": False, "training_admission_released": False, "a0_execution_authorized": False, "training_authorized": False, "evaluation_authorized": False, "kaggle_authorized": False, "gpu_authorized": False, "provenance_physical_sha256_self_certified": False}
    assert "p4l_phase2_activation_commit" not in value
    assert "p4l_phase2_evidence_freeze_commit" not in value
    p4x._validate_provenance(value)


def _aggregate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows += [{"p2_reason_supervision_eligible": True, "integrity_status": "ELIGIBLE", "eligible_for_positive_margin": True}] * 695
    rows += [{"p2_reason_supervision_eligible": True, "integrity_status": "ELIGIBLE", "eligible_for_positive_margin": False}] * 1074
    rows += [{"p2_reason_supervision_eligible": False, "integrity_status": "INELIGIBLE", "eligible_for_positive_margin": False}] * 1562
    rows += [{"p2_reason_supervision_eligible": False, "integrity_status": "UNRESOLVED", "eligible_for_positive_margin": False}] * 269
    return rows


@pytest.mark.parametrize("field, value, contract", [("p2_reason_supervision_eligible", None, "P4X_REASON_COUNT_MISMATCH"), ("integrity_status", "BROKEN", "P4X_INTEGRITY_COUNT_MISMATCH"), ("eligible_for_positive_margin", None, "P4X_POSITIVE_MARGIN_COUNT_MISMATCH")])
def test_p4x_aggregate_mismatches_fail(field: str, value: object, contract: str) -> None:
    rows = _aggregate_rows(); rows[0] = {**rows[0], field: value}
    with pytest.raises(p4x.ContractError, match=contract):
        p4x._validate_p4x_aggregates(rows)


def test_historical_seed174_724_2876_and_blob_are_rejected_as_revised_evidence(tmp_path: Path) -> None:
    rows = [{"p2_reason_supervision_eligible": True, "integrity_status": "ELIGIBLE", "eligible_for_positive_margin": True}] * 724
    rows += [{"p2_reason_supervision_eligible": False, "integrity_status": "INELIGIBLE", "eligible_for_positive_margin": False}] * 2876
    with pytest.raises(p4x.ContractError, match="P4X_REASON_COUNT_MISMATCH"):
        p4x._validate_p4x_aggregates(rows)
    root = commit_blob_repo(tmp_path)
    with pytest.raises(p4x.ContractError, match="P4X_GIT_BLOB_MISMATCH"):
        p4x.authenticated_head_bytes(root, "input.txt", p4x.HISTORICAL_SEED174_SIDECAR_BLOB, hashlib.sha256(b"one\ntwo\n").hexdigest())


def test_cohort_join_degeneracy_and_expected_count_checks() -> None:
    source = [{"id": "x", "pair_id": "p", "frame_compatible_label": 1, "predicate_covered_label": 1, "sufficiency_label": 1, "final_label": "SUPPORT", "polarity_label": "SUPPORT", "primary_failure_type": "none", "intervention_type": "none"}]
    sidecar = [{"row_id": "x", "pair_id": "p", "split": "train", "canonical_row_id": "x", "frame_compatible_label": 1, "p2_reason_supervision_eligible": True, **{field: "PASS" for field in ("schema_status", "dataset_source_status", "grammar_status", "canonical_status", "intervention_contract_status", "polarity_contamination_status", "time_swap_status")}}]
    with pytest.raises(p4x.ContractError, match="P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE"): p4x._derive_cohorts(source, sidecar, "train")
    sidecar[0]["row_id"] = "wrong"
    with pytest.raises(p4x.ContractError, match="P4X_SOURCE_SIDECAR_JOIN_MISMATCH"): p4x._derive_cohorts(source, sidecar, "train")


@pytest.mark.parametrize("mutation, contract", [("duplicate-source", "P4X_DUPLICATE_SOURCE_ROW_ID"), ("duplicate-sidecar", "P4X_DUPLICATE_SIDECAR_ROW_ID"), ("missing", "P4X_SOURCE_SIDECAR_JOIN_MISMATCH")])
def test_join_duplicate_and_misaligned_ids_fail(mutation: str, contract: str) -> None:
    source, sidecar = _complete_cohort_fixture()
    if mutation == "duplicate-source":
        source.append({**source[-1]})
    elif mutation == "duplicate-sidecar":
        sidecar.append({**sidecar[-1]})
    else:
        sidecar.pop()
    with pytest.raises(p4x.ContractError, match=contract):
        p4x._derive_cohorts(source, sidecar, "train")


def test_first_blocker_claim_mismatch_fails_before_eligibility_or_counts() -> None:
    source, sidecar = _complete_cohort_fixture()
    source[0]["primary_failure_type"] = "predicate"  # axes say FRAME first.
    sidecar[0]["p2_reason_supervision_eligible"] = False
    with pytest.raises(p4x.ContractError, match="P4X_FIRST_BLOCKER_MISMATCH"):
        p4x._derive_cohorts(source, sidecar, "train")


def test_sidecar_wrong_recomputed_split_membership_fails() -> None:
    source, sidecar = _complete_cohort_fixture()
    sidecar[0]["split"] = "dev"
    with pytest.raises(p4x.ContractError, match="P4X_SOURCE_SIDECAR_JOIN_MISMATCH"):
        p4x._derive_cohorts(source, sidecar, "train")


def _complete_cohort_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    axes = [(0, 1, 1, "frame", "NOT_ENTITLED", "NONE"), (1, 0, 1, "predicate", "NOT_ENTITLED", "NONE"), (1, 1, 0, "sufficiency", "NOT_ENTITLED", "NONE"), (1, 1, 1, "none", "REFUTE", "REFUTE"), (1, 1, 1, "none", "SUPPORT", "SUPPORT")]
    source, sidecar = [], []
    statuses = {field: "PASS" for field in ("schema_status", "dataset_source_status", "grammar_status", "canonical_status", "intervention_contract_status", "polarity_contamination_status", "time_swap_status")}
    for index, (frame, predicate, sufficiency, primary, label, polarity) in enumerate(axes):
        row_id = f"row-{index}"
        source.append({"id": row_id, "pair_id": row_id, "frame_compatible_label": frame, "predicate_covered_label": predicate, "sufficiency_label": sufficiency, "final_label": label, "polarity_label": polarity, "primary_failure_type": primary, "intervention_type": "none"})
        sidecar.append({"row_id": row_id, "pair_id": row_id, "split": "train", "canonical_row_id": row_id, "frame_compatible_label": frame, "p2_reason_supervision_eligible": True, **statuses})
    return source, sidecar


def _non_degenerate_cohort_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source, sidecar = _complete_cohort_fixture()
    copied_source, copied_sidecar = [], []
    for row in source:
        row_id = f"{row['id']}-copy"
        copied_source.append({**row, "id": row_id, "pair_id": row_id})
    for row in sidecar:
        row_id = f"{row['row_id']}-copy"
        copied_sidecar.append({**row, "row_id": row_id, "pair_id": row_id, "canonical_row_id": row_id})
    return source + copied_source, sidecar + copied_sidecar


def _set_fixture_cohorts(monkeypatch: pytest.MonkeyPatch, polarity_one: int = 2) -> None:
    monkeypatch.setitem(p4x.EXPECTED_COHORTS, "train", {"frame": {0: 2, 1: 8 if polarity_one == 2 else 7}, "predicate": {0: 2, 1: 6 if polarity_one == 2 else 5}, "sufficiency": {0: 2, 1: 4 if polarity_one == 2 else 3}, "polarity": {0: 2, 1: polarity_one}})


@pytest.mark.parametrize("status", ["FAIL", "UNRESOLVED", "NOT_APPLICABLE"])
def test_recognized_nonpass_generator_status_is_preserved_exclusion(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    source, sidecar = _non_degenerate_cohort_fixture()
    sidecar[4]["grammar_status"] = status
    sidecar[4]["p2_reason_supervision_eligible"] = False
    _set_fixture_cohorts(monkeypatch, polarity_one=1)
    assert p4x._derive_cohorts(source, sidecar, "train")["polarity"] == {0: 2, 1: 1}


def test_clean_complete_fixture_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    source, sidecar = _non_degenerate_cohort_fixture()
    _set_fixture_cohorts(monkeypatch)
    assert p4x._derive_cohorts(source, sidecar, "train") == p4x.EXPECTED_COHORTS["train"]


def test_reason_eligibility_rejects_clean_false_negative() -> None:
    source, sidecar = _complete_cohort_fixture()
    sidecar[4]["p2_reason_supervision_eligible"] = False
    with pytest.raises(p4x.ContractError, match="P4X_REASON_ELIGIBILITY_DERIVATION_MISMATCH"):
        p4x._derive_cohorts(source, sidecar, "train")


def test_reason_eligibility_rejects_nonclean_false_positive() -> None:
    source, sidecar = _complete_cohort_fixture()
    sidecar[4]["grammar_status"] = "FAIL"
    with pytest.raises(p4x.ContractError, match="P4X_REASON_ELIGIBILITY_DERIVATION_MISMATCH"):
        p4x._derive_cohorts(source, sidecar, "train")


@pytest.mark.parametrize("mutation", ["missing", "non-string", "unknown"])
def test_malformed_generator_status_fails_closed(mutation: str) -> None:
    source, sidecar = _complete_cohort_fixture()
    if mutation == "missing":
        del sidecar[4]["grammar_status"]
    else:
        sidecar[4]["grammar_status"] = 1 if mutation == "non-string" else "OTHER"
    with pytest.raises(p4x.ContractError, match="P4X_GENERATOR_STATUS_MALFORMED"):
        p4x._derive_cohorts(source, sidecar, "train")


def test_frozen_generator_status_population_sets_are_exact() -> None:
    root = MODULE_PATH.parents[1]
    sidecar = p4x._parse_jsonl(p4x.authenticated_head_bytes(root, p4x.SIDECAR, p4x.SIDECAR_BLOB, p4x.SIDECAR_SHA256), "SIDECAR")
    reason_ineligible = {str(row["row_id"]) for row in sidecar if row["p2_reason_supervision_eligible"] is False}
    integrity_noneligible = {str(row["row_id"]) for row in sidecar if row["integrity_status"] != "ELIGIBLE"}
    status_nonpass = {str(row["row_id"]) for row in sidecar if p4x._classify_generator_status(row, str(row["row_id"])) != "GENERATOR_CLEAN"}
    assert len(reason_ineligible) == len(integrity_noneligible) == len(status_nonpass) == 1831
    assert reason_ineligible == integrity_noneligible == status_nonpass


def test_orion_approval_polarity_flip_is_authenticated_preserved_exclusion() -> None:
    root = MODULE_PATH.parents[1]
    source = p4x._parse_jsonl(p4x.authenticated_head_bytes(root, p4x.DATASET, p4x.DATASET_BLOB, p4x.DATASET_SHA256), "SOURCE")
    sidecar = p4x._parse_jsonl(p4x.authenticated_head_bytes(root, p4x.SIDECAR, p4x.SIDECAR_BLOB, p4x.SIDECAR_SHA256), "SIDECAR")
    orion = next(row for row in sidecar if row["row_id"] == "orion_approval__polarity_flip")
    assert orion["split"] == "train"
    assert orion["grammar_status"] == "FAIL"
    assert orion["p2_reason_supervision_eligible"] is False
    assert p4x._classify_generator_status(orion, "orion_approval__polarity_flip") == "GENERATOR_DEFECT"
    train_ids = {str(row["row_id"]) for row in sidecar if row["split"] == "train"}
    assert p4x._derive_cohorts([row for row in source if str(row["id"]) in train_ids], [row for row in sidecar if str(row["row_id"]) in train_ids], "train") == p4x.EXPECTED_COHORTS["train"]


def test_all_four_applicable_cohort_families_require_both_binary_sides(monkeypatch: pytest.MonkeyPatch) -> None:
    source, sidecar = _complete_cohort_fixture()
    monkeypatch.setitem(p4x.EXPECTED_COHORTS, "train", {"frame": {0: 1, 1: 4}, "predicate": {0: 1, 1: 3}, "sufficiency": {0: 1, 1: 2}, "polarity": {0: 1, 1: 1}})
    assert p4x._derive_cohorts(source, sidecar, "train")["polarity"] == {0: 1, 1: 1}
    for family, index in (("frame", 0), ("predicate", 1), ("sufficiency", 2), ("polarity", 3)):
        reduced_source = source[:index] + source[index + 1:]
        reduced_sidecar = sidecar[:index] + sidecar[index + 1:]
        with pytest.raises(p4x.ContractError, match=f"P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE: train:{family}"):
            p4x._derive_cohorts(reduced_source, reduced_sidecar, "train")


@pytest.mark.parametrize("field, value", [("frame_compatible_label", 2), ("predicate_covered_label", True), ("sufficiency_label", "1")])
def test_dataset_binary_axis_malformed_values_fail(field: str, value: object) -> None:
    row = {name: "x" for name in p4x.SOURCE_FIELDS}
    row.update({"frame_compatible_label": 1, "predicate_covered_label": 1, "sufficiency_label": 1})
    row[field] = value
    with pytest.raises(p4x.ContractError, match="P4X_SOURCE_BINARY_INVALID"):
        p4x._dataset_semantic([row])


def test_execution_record_is_static_identity_not_runtime_scientific_input() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(call.func, ast.Name) and call.func.id == "open" for call in calls)
    source = MODULE_PATH.read_text(encoding="utf-8")
    # Both HEAD and freeze-tree checks are identities only; no record bytes are read.
    assert source.count("PHASE_II_EXECUTION_RECORD") >= 3
    assert '"cat-file", "blob", PHASE_II_EXECUTION_RECORD' not in source


def test_machine_readable_clean_success_interface(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI emits PASS only when its complete validator returns successfully."""
    monkeypatch.setattr(p4x, "validate", lambda root, expected: {"status": "PASS", "execution_record_opened": False})
    assert p4x.main(["--repo-root", ".", "--expected-head", "a" * 40]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_module_has_no_trainer_model_cuda_or_checkpoint_dependency() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not {"torch", "transformers", "train_controlled_v6b_minimal"} & imported
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = {"Popen", "runpy", "load", "load_state_dict", "cuda", "main"}
    assert not any(isinstance(call.func, ast.Attribute) and call.func.attr in forbidden for call in calls)
