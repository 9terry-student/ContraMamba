#!/usr/bin/env python3
"""Fail-closed, pre-trainer static control for the revised Seed8192 P4-L inputs.

This module intentionally imports only the Python standard library.  It never
imports the trainer and has no model, checkpoint, torch, CUDA, or subprocess
launch path other than read-only Git inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_BRANCH = "p3w7-a1-a2-a3-factorial-execution-authority-n3-v2"
EXPECTED_UPSTREAM = f"origin/{EXPECTED_BRANCH}"
FROZEN_TRAINER = ("scripts/train_controlled_v6b_minimal.py", "8252f5778944974e20c21acfe203e8f7bb5f3218")
FROZEN_TEST = ("tests/test_reason_router_p4x_trainer_rebind.py", "5699ec92459aebda711fdb94470430cdf9349fce")
DATASET = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
SIDECAR_DIR = "reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b"
SIDECAR = f"{SIDECAR_DIR}/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl"
PROVENANCE = f"{SIDECAR_DIR}/p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json"
DATASET_BLOB = "2b6829bf04a1333446aac6f7c603d9178b339f36"
SIDECAR_BLOB = "83d119e327acacda7cff6b4e24c6502898294e03"
# Historical seed174 evidence is deliberately not an admissible revised input.
HISTORICAL_SEED174_SIDECAR_BLOB = "867ece69f4da680b4bb036530a96f586467d8421"
PROVENANCE_BLOB = "6c970033fae82286452f6d635b94f441d0f3d048"
PHASE_II_ACTIVATION_COMMIT = "cb6f4482b463d5f85331e2a6ddfbbd34499c930a"
PHASE_II_ACTIVATION_PARENT = "149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b"
PHASE_II_EVIDENCE_FREEZE_COMMIT = "ef26310f3532368b9de6cb96a19cb26e7626716d"
PHASE_II_EVIDENCE_FREEZE_PARENT = PHASE_II_ACTIVATION_COMMIT
PHASE_II_EXECUTION_RECORD = "reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase2_execution_record_cb6f4482b463d5f85331e2a6ddfbbd34499c930a.json"
PHASE_II_EXECUTION_RECORD_BLOB = "0d07e52dd4240a84c60805a4495b18a727e289d3"
DATASET_SHA256 = "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
SIDECAR_SHA256 = "9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d"
PROVENANCE_SHA256 = "170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8"
DATASET_SEMANTIC_SHA256 = "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
SIDECAR_SEMANTIC_SHA256 = "2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9"
SIDECAR_SCHEMA = "P3W7_SEED8192_REVISED_P4L_EFFECTIVE_INTEGRITY_SIDECAR_V1"
PROVENANCE_SCHEMA = "P3W7_SEED8192_REVISED_P4L_INTEGRITY_SIDECAR_PROVENANCE_V1"
SOURCE_FIELDS = ("id", "pair_id", "claim", "evidence", "final_label", "frame_compatible_label", "predicate_covered_label", "sufficiency_label", "polarity_label", "primary_failure_type", "intervention_type")
SPLIT_IDENTITIES = {"pair_count": 300, "train_pair_count": 240, "dev_pair_count": 60, "train_row_count": 2880, "dev_row_count": 720, "pair_universe_sha256": "41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2", "shuffled_pair_sha256": "ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55", "train_pair_sha256": "f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049", "dev_pair_sha256": "30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4", "ordered_train_row_sha256": "478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8", "ordered_dev_row_sha256": "7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4"}
PROVENANCE_SPLIT_IDENTITIES = {**SPLIT_IDENTITIES, "historical_seed174_dev_pair_sha256": "259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d"}
EXPECTED_COHORTS = {"train": {"frame": {0: 714, 1: 695}, "predicate": {0: 119, 1: 576}, "sufficiency": {0: 238, 1: 338}, "polarity": {0: 100, 1: 238}}, "dev": {"frame": {0: 186, 1: 174}, "predicate": {0: 31, 1: 143}, "sufficiency": {0: 62, 1: 81}, "polarity": {0: 19, 1: 62}}}
EXPECTED_P4X_AGGREGATES = {"reason": {True: 1769, False: 1831}, "integrity": {"ELIGIBLE": 1769, "INELIGIBLE": 1562, "UNRESOLVED": 269}, "margin": {True: 695, False: 2905}}


class ContractError(RuntimeError):
    """A named immutable prelaunch contract was not satisfied."""


def _git(root: Path, *args: str, text: bool = False) -> str | bytes:
    run = subprocess.run(["git", "-C", str(root), "-c", "core.longpaths=true", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if run.returncode:
        raise ContractError(f"P4X_GIT_COMMAND_FAILED: {' '.join(args)}: {run.stderr.decode('utf-8', 'replace').strip()}")
    return run.stdout.decode("utf-8", "strict").strip() if text else run.stdout


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError(code)


def normalize_expected_head(value: str | None) -> str:
    """Accept only the immutable full-SHA implementation anchor spelling."""
    _require(type(value) is str and len(value) == 40, "P4X_EXPECTED_HEAD_INVALID")
    normalized = value.lower()
    _require(all(character in "0123456789abcdef" for character in normalized), "P4X_EXPECTED_HEAD_INVALID")
    return normalized


def _validate_exact_commit(root: Path, commit: str, parent: str, role: str) -> None:
    """Authenticate a frozen commit object and its one immutable parent."""
    try:
        resolved = str(_git(root, "rev-parse", "--verify", f"{commit}^{{commit}}", text=True)).lower()
        object_type = str(_git(root, "cat-file", "-t", commit, text=True))
        observed_parent = str(_git(root, "show", "-s", "--format=%P", commit, text=True)).lower()
    except ContractError:
        raise ContractError(f"P4X_PHASE2_{role}_COMMIT_UNAVAILABLE") from None
    _require(resolved == commit, f"P4X_PHASE2_{role}_COMMIT_IDENTITY_MISMATCH")
    _require(object_type == "commit", f"P4X_PHASE2_{role}_OBJECT_TYPE_MISMATCH")
    _require(observed_parent == parent, f"P4X_PHASE2_{role}_PARENT_MISMATCH")


def _validate_phase2_lineage(root: Path, implementation_anchor: str) -> None:
    """Keep external Phase-II authorization in Git lineage, never provenance JSON."""
    implementation_anchor = normalize_expected_head(implementation_anchor)
    _validate_exact_commit(root, PHASE_II_ACTIVATION_COMMIT, PHASE_II_ACTIVATION_PARENT, "ACTIVATION")
    _validate_exact_commit(root, PHASE_II_EVIDENCE_FREEZE_COMMIT, PHASE_II_EVIDENCE_FREEZE_PARENT, "FREEZE")
    try:
        _git(root, "merge-base", "--is-ancestor", PHASE_II_EVIDENCE_FREEZE_COMMIT, implementation_anchor)
    except ContractError:
        raise ContractError("P4X_PHASE2_FREEZE_NOT_ANCESTOR_OF_IMPLEMENTATION_ANCHOR") from None


def _validate_phase2_frozen_evidence_binding(root: Path) -> None:
    """Bind Phase-IV inputs to freeze-tree identities; execution record stays unopened."""
    bindings = ((SIDECAR, SIDECAR_BLOB), (PROVENANCE, PROVENANCE_BLOB), (PHASE_II_EXECUTION_RECORD, PHASE_II_EXECUTION_RECORD_BLOB))
    for relative, blob in bindings:
        try:
            observed = str(_git(root, "rev-parse", f"{PHASE_II_EVIDENCE_FREEZE_COMMIT}:{relative}", text=True))
        except ContractError:
            raise ContractError(f"P4X_PHASE2_FROZEN_EVIDENCE_UNAVAILABLE: {relative}") from None
        _require(observed == blob, f"P4X_PHASE2_FROZEN_EVIDENCE_BLOB_MISMATCH: {relative}")


def _reject_symlink(root: Path, relative: str) -> Path:
    path = root / relative
    _require(path.exists() and path.is_file(), f"P4X_PATH_MISSING_OR_NOT_FILE: {relative}")
    current = root
    for component in Path(relative).parts:
        current /= component
        _require(not current.is_symlink(), f"P4X_SYMLINK_SUBSTITUTION: {relative}")
    _require(path.resolve().parent == (root / relative).resolve().parent, f"P4X_PATH_SUBSTITUTION: {relative}")
    return path


def authenticated_head_bytes(root: Path, relative: str, blob: str, sha256: str) -> bytes:
    """Authenticate clean tracked HEAD content and return Git's canonical blob bytes."""
    _reject_symlink(root, relative)
    try:
        tracked_path = _git(root, "ls-files", "--error-unmatch", "--", relative, text=True)
    except ContractError:
        raise ContractError(f"P4X_UNTRACKED_INPUT: {relative}") from None
    _require(tracked_path == relative, f"P4X_UNTRACKED_INPUT: {relative}")
    for state, args in (("UNSTAGED_DIRTY", ("diff", "--quiet", "--", relative)), ("STAGED_DIRTY", ("diff", "--cached", "--quiet", "--", relative))):
        try:
            _git(root, *args)
        except ContractError:
            raise ContractError(f"P4X_{state}: {relative}") from None
    observed_blob = str(_git(root, "rev-parse", f"HEAD:{relative}", text=True))
    _require(observed_blob == blob, f"P4X_GIT_BLOB_MISMATCH: {relative}")
    content = bytes(_git(root, "cat-file", "blob", observed_blob))
    _require(hashlib.sha256(content).hexdigest() == sha256, f"P4X_GIT_LF_SHA256_MISMATCH: {relative}")
    return content


def _parse_jsonl(content: bytes, label: str) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"P4X_{label}_MALFORMED_JSONL") from exc
    _require(all(isinstance(value, dict) for value in values), f"P4X_{label}_ROW_NOT_OBJECT")
    return values


def _dataset_semantic(rows: list[dict[str, Any]]) -> str:
    canonical = []
    for index, row in enumerate(rows):
        _require(all(field in row for field in SOURCE_FIELDS), f"P4X_SOURCE_FIELD_MISSING: index={index}")
        for field in ("frame_compatible_label", "predicate_covered_label", "sufficiency_label"):
            _require(type(row[field]) is int and row[field] in (0, 1), f"P4X_SOURCE_BINARY_INVALID: index={index} field={field}")
        canonical.append({field: row[field] for field in SOURCE_FIELDS})
    return _canonical_json_sha256(canonical)


def _sidecar_semantic(rows: list[dict[str, Any]]) -> str:
    return _canonical_json_sha256([{key: row[key] for key in sorted(row) if key != "created_at"} for row in rows])


def _identity_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def split_pair_ids(pair_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Apply the frozen Seed8192 split algorithm to sorted, unique pair IDs."""
    pairs = sorted(set(pair_ids))
    _require("" not in pairs and len(pairs) >= 2, "P4X_PAIR_UNIVERSE_INVALID")
    shuffled = list(pairs)
    rng = random.Random(8192)
    rng.shuffle(shuffled)
    n = len(pairs)
    dev_count = min(n - 1, max(1, round(n * 0.2)))
    dev_pairs = shuffled[:dev_count]
    train_pairs = shuffled[dev_count:]
    _require(not (set(train_pairs) & set(dev_pairs)), "P4X_PAIR_LEAKAGE")
    _require(len(train_pairs) + len(dev_pairs) == n, "P4X_PAIR_SPLIT_CARDINALITY_MISMATCH")
    return shuffled, train_pairs, dev_pairs


def recompute_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs = sorted({str(row.get("pair_id", "")) for row in rows})
    _require("" not in pairs and len(pairs) == 300, "P4X_PAIR_UNIVERSE_INVALID")
    shuffled, train_pairs, dev_pair_list = split_pair_ids(pairs)
    _require(not (set(train_pairs) & set(dev_pair_list)), "P4X_PAIR_LEAKAGE")
    dev_pairs = set(dev_pair_list)
    train, dev = ([row for row in rows if str(row["pair_id"]) not in dev_pairs], [row for row in rows if str(row["pair_id"]) in dev_pairs])
    audit = {"pair_count": len(pairs), "train_pair_count": len({str(row["pair_id"]) for row in train}), "dev_pair_count": len({str(row["pair_id"]) for row in dev}), "train_row_count": len(train), "dev_row_count": len(dev), "pair_universe_sha256": _identity_hash(pairs), "shuffled_pair_sha256": _identity_hash(shuffled), "train_pair_sha256": _identity_hash(sorted({str(row["pair_id"]) for row in train})), "dev_pair_sha256": _identity_hash(sorted({str(row["pair_id"]) for row in dev})), "ordered_train_row_sha256": _identity_hash([str(row["id"]) for row in train]), "ordered_dev_row_sha256": _identity_hash([str(row["id"]) for row in dev])}
    _require(not ({str(row["pair_id"]) for row in train} & {str(row["pair_id"]) for row in dev}), "P4X_PAIR_LEAKAGE")
    _require(len(train_pairs) == 240 and len(dev_pair_list) == 60, "P4X_FROZEN_PAIR_COUNT_MISMATCH")
    _require(audit == SPLIT_IDENTITIES, "P4X_SPLIT_IDENTITY_MISMATCH")
    return train, dev, audit


def _validate_provenance(value: dict[str, Any]) -> None:
    required = {"schema_version": PROVENANCE_SCHEMA, "sidecar_schema_version": SIDECAR_SCHEMA, "lineage_mode": "revised-seed8192", "p4l_authority_commit": "ff181f565cefa0a28280c084246862286daf1f2d", "split_authority_commit": "b4fbb5666d796161f95ae23612ce2448c25063ee", "builder_source_commit": "149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b", "source_dataset_sha256": DATASET_SHA256, "source_dataset_semantic_sha256": DATASET_SEMANTIC_SHA256, "sidecar_physical_sha256": SIDECAR_SHA256, "sidecar_semantic_sha256": SIDECAR_SEMANTIC_SHA256, "row_count": 3600}
    for field, expected in required.items():
        _require(value.get(field) == expected, f"P4X_PROVENANCE_IDENTITY_MISMATCH: {field}")
    _require(value.get("split_identities") == PROVENANCE_SPLIT_IDENTITIES, "P4X_PROVENANCE_SPLIT_IDENTITY_MISMATCH")
    flags = {"implementation_authorized": True, "artifact_materialization_authorized_by_p4l": False, "training_admission_released": False, "a0_execution_authorized": False, "training_authorized": False, "evaluation_authorized": False, "kaggle_authorized": False, "gpu_authorized": False, "provenance_physical_sha256_self_certified": False}
    for field, expected in flags.items():
        _require(type(value.get(field)) is bool and value[field] is expected, f"P4X_PROVENANCE_FLAG_MISMATCH: {field}")


def _require_clean_repository(root: Path) -> None:
    for code, args in (("P4X_UNSTAGED_DIRTY", ("diff", "--quiet")), ("P4X_STAGED_DIRTY", ("diff", "--cached", "--quiet"))):
        try:
            _git(root, *args)
        except ContractError:
            raise ContractError(code) from None
    _require(not str(_git(root, "ls-files", "--others", "--exclude-standard", text=True)), "P4X_UNTRACKED_WORKTREE")


def _validate_p4x_aggregates(sidecar: list[dict[str, Any]]) -> None:
    observed = {
        "reason": Counter(row.get("p2_reason_supervision_eligible") for row in sidecar),
        "integrity": Counter(row.get("integrity_status") for row in sidecar),
        "margin": Counter(row.get("eligible_for_positive_margin") for row in sidecar),
    }
    _require(observed["reason"] == Counter(EXPECTED_P4X_AGGREGATES["reason"]), "P4X_REASON_COUNT_MISMATCH")
    _require(observed["integrity"] == Counter(EXPECTED_P4X_AGGREGATES["integrity"]), "P4X_INTEGRITY_COUNT_MISMATCH")
    _require(observed["margin"] == Counter(EXPECTED_P4X_AGGREGATES["margin"]), "P4X_POSITIVE_MARGIN_COUNT_MISMATCH")


def _derive_cohorts(source: list[dict[str, Any]], sidecar: list[dict[str, Any]], split: str) -> dict[str, dict[int, int]]:
    by_id = {str(row.get("row_id", "")): row for row in sidecar}
    _require(len(by_id) == len(sidecar), "P4X_DUPLICATE_SIDECAR_ROW_ID")
    source_by_id = {str(row.get("id", "")): row for row in source}
    _require(len(source_by_id) == len(source), "P4X_DUPLICATE_SOURCE_ROW_ID")
    canonical_by_pair: dict[str, str] = {}
    for source_row in source:
        row_id = str(source_row.get("id", "")); pair_id = str(source_row.get("pair_id", "")); side = by_id.get(row_id)
        _require(side is not None, f"P4X_SOURCE_SIDECAR_JOIN_MISMATCH: {row_id}")
        canonical = side.get("canonical_row_id")
        _require(type(canonical) is str and canonical, f"P4X_CANONICAL_ROW_ID_MISSING: {row_id}")
        previous = canonical_by_pair.setdefault(pair_id, canonical)
        _require(previous == canonical, f"P4X_CANONICAL_ROW_ID_CONFLICT: {pair_id}")
    for pair_id, canonical in canonical_by_pair.items():
        canonical_source, canonical_side = source_by_id.get(canonical), by_id.get(canonical)
        _require(canonical_source is not None and canonical_side is not None, f"P4X_CANONICAL_ROW_MISSING: {pair_id}")
        _require(str(canonical_source.get("pair_id", "")) == pair_id and canonical_side.get("canonical_row_id") == canonical, f"P4X_CANONICAL_ROW_NOT_SELF_ANCHORED: {pair_id}")
    cohorts = {name: Counter() for name in ("frame", "predicate", "sufficiency", "polarity")}
    for source_row in source:
        row_id = str(source_row.get("id", "")); side = by_id.get(row_id)
        _require(side is not None and side.get("split") == split and side.get("pair_id") == source_row.get("pair_id"), f"P4X_SOURCE_SIDECAR_JOIN_MISMATCH: {row_id}")
        frame, predicate, sufficiency = (source_row[field] for field in ("frame_compatible_label", "predicate_covered_label", "sufficiency_label"))
        _require(side.get("frame_compatible_label") == frame, f"P4X_SIDECAR_SOURCE_BINARY_MISMATCH: {row_id}")
        statuses = ("schema_status", "dataset_source_status", "grammar_status", "canonical_status", "intervention_contract_status", "polarity_contamination_status", "time_swap_status")
        _require(all(side.get(field) == "PASS" for field in statuses), f"P4X_GENERATOR_STATUS_DEFECT: {row_id}")
        primary = str(source_row.get("primary_failure_type", "")).strip().lower()
        _require(primary in {"none", "frame", "predicate", "sufficiency", "polarity"}, f"P4X_PRIMARY_FAILURE_MALFORMED: {row_id}")
        derived = "frame" if frame == 0 else "predicate" if predicate == 0 else "sufficiency" if sufficiency == 0 else "authorized"
        expected = "authorized" if primary in {"none", "polarity"} else primary
        _require(derived == expected, f"P4X_FIRST_BLOCKER_MISMATCH: {row_id}")
        label = str(source_row.get("final_label", "")).strip().upper()
        directional = label in {"REFUTE", "SUPPORT"}
        raw_polarity = source_row.get("polarity_label", "")
        polarity = ({0: "NONE", 1: "REFUTE", 2: "SUPPORT"}.get(raw_polarity, "UNKNOWN") if type(raw_polarity) is int else str(raw_polarity).strip().upper())
        eligible = (derived == expected and (derived == "authorized" or label == "NOT_ENTITLED") and (primary not in {"none", "polarity"} or directional) and (polarity == label if directional else polarity in {"NONE", "NOT_ENTITLED"}) and not (primary == "polarity" and str(source_row.get("intervention_type", "")).strip().lower() != "polarity_flip"))
        _require(type(side.get("p2_reason_supervision_eligible")) is bool and side["p2_reason_supervision_eligible"] is eligible, f"P4X_REASON_ELIGIBILITY_DERIVATION_MISMATCH: {row_id}")
        if eligible: cohorts["frame"][frame] += 1
        if eligible and frame == 1: cohorts["predicate"][predicate] += 1
        if eligible and frame == 1 and predicate == 1: cohorts["sufficiency"][sufficiency] += 1
        if eligible and frame == predicate == sufficiency == 1:
            if label in {"REFUTE", "SUPPORT"}: cohorts["polarity"][1 if label == "SUPPORT" else 0] += 1
    observed = {name: {0: counts[0], 1: counts[1]} for name, counts in cohorts.items()}
    for name, counts in observed.items(): _require(counts[0] > 0 and counts[1] > 0, f"P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE: {split}:{name}")
    _require(observed == EXPECTED_COHORTS[split], f"P4X_APPLICABLE_COHORT_COUNT_MISMATCH: {split}")
    return observed


def _validate_implementation_anchor(root: Path, implementation_anchor: str, current_head: str) -> None:
    """Authenticate the immutable implementation anchor and its current-HEAD ancestry."""
    try:
        resolved = str(_git(root, "rev-parse", "--verify", implementation_anchor, text=True)).lower()
    except ContractError:
        raise ContractError("P4X_IMPLEMENTATION_ANCHOR_COMMIT_UNAVAILABLE") from None
    _require(resolved == implementation_anchor, "P4X_IMPLEMENTATION_ANCHOR_IDENTITY_MISMATCH")
    try:
        object_type = str(_git(root, "cat-file", "-t", implementation_anchor, text=True))
    except ContractError:
        raise ContractError("P4X_IMPLEMENTATION_ANCHOR_COMMIT_UNAVAILABLE") from None
    _require(object_type == "commit", "P4X_IMPLEMENTATION_ANCHOR_OBJECT_TYPE_MISMATCH")
    try:
        _git(root, "merge-base", "--is-ancestor", implementation_anchor, current_head)
    except ContractError:
        raise ContractError("P4X_IMPLEMENTATION_ANCHOR_NOT_ANCESTOR_OF_CURRENT_HEAD") from None


def _validate_repository_identity(root: Path, expected_head: str) -> str:
    """Require the exact synchronized execution branch and immutable anchor lineage."""
    implementation_anchor = normalize_expected_head(expected_head)
    _require(not any(root.joinpath(part).is_symlink() for part in (".git",)), "P4X_REPOSITORY_SYMLINK_SUBSTITUTION")
    _require(str(_git(root, "rev-parse", "--abbrev-ref", "HEAD", text=True)) == EXPECTED_BRANCH, "P4X_BRANCH_IDENTITY_MISMATCH")
    try:
        remote = str(_git(root, "config", "--get", f"branch.{EXPECTED_BRANCH}.remote", text=True))
        merge = str(_git(root, "config", "--get", f"branch.{EXPECTED_BRANCH}.merge", text=True))
    except ContractError:
        raise ContractError("P4X_UPSTREAM_REF_IDENTITY_MISMATCH") from None
    _require(remote == "origin" and merge == f"refs/heads/{EXPECTED_BRANCH}", "P4X_UPSTREAM_REF_IDENTITY_MISMATCH")
    current_head = str(_git(root, "rev-parse", "HEAD", text=True)).lower()
    try:
        upstream_tip = str(_git(root, "rev-parse", f"refs/remotes/{EXPECTED_UPSTREAM}", text=True)).lower()
        configured_upstream_tip = str(_git(root, "rev-parse", "@{upstream}", text=True)).lower()
    except ContractError:
        raise ContractError("P4X_UPSTREAM_TIP_MISMATCH") from None
    _require(configured_upstream_tip == upstream_tip and current_head == upstream_tip, "P4X_UPSTREAM_TIP_MISMATCH")
    _require(str(_git(root, "rev-list", "--left-right", "--count", "HEAD...@{upstream}", text=True)) == "0\t0", "P4X_AHEAD_BEHIND_MISMATCH")
    _validate_implementation_anchor(root, implementation_anchor, current_head)
    return current_head


def validate(root: Path, expected_head: str) -> dict[str, Any]:
    root = root.resolve()
    _validate_repository_identity(root, expected_head)
    _validate_phase2_lineage(root, expected_head)
    _require_clean_repository(root)
    for relative, blob in (FROZEN_TRAINER, FROZEN_TEST): authenticated_head_bytes(root, relative, blob, hashlib.sha256(bytes(_git(root, "cat-file", "blob", blob))).hexdigest())
    data = _parse_jsonl(authenticated_head_bytes(root, DATASET, DATASET_BLOB, DATASET_SHA256), "SOURCE")
    sidecar = _parse_jsonl(authenticated_head_bytes(root, SIDECAR, SIDECAR_BLOB, SIDECAR_SHA256), "SIDECAR")
    provenance_bytes = authenticated_head_bytes(root, PROVENANCE, PROVENANCE_BLOB, PROVENANCE_SHA256)
    try: provenance = json.loads(provenance_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ContractError("P4X_PROVENANCE_MALFORMED") from exc
    _require(isinstance(provenance, dict), "P4X_PROVENANCE_NOT_OBJECT")
    _require(_dataset_semantic(data) == DATASET_SEMANTIC_SHA256, "P4X_SOURCE_SEMANTIC_SHA_MISMATCH")
    _require(_sidecar_semantic(sidecar) == SIDECAR_SEMANTIC_SHA256, "P4X_SIDECAR_SEMANTIC_SHA_MISMATCH")
    _validate_provenance(provenance)
    _validate_phase2_frozen_evidence_binding(root)
    _require(str(_git(root, "rev-parse", f"HEAD:{PHASE_II_EXECUTION_RECORD}", text=True)) == PHASE_II_EXECUTION_RECORD_BLOB, "P4X_EXECUTION_RECORD_IDENTITY_MISMATCH")
    train, dev, split = recompute_split(data)
    _require([str(row.get("id", "")) for row in data] == [str(row.get("row_id", "")) for row in sidecar], "P4X_STABLE_JOIN_ORDER_MISMATCH")
    _validate_p4x_aggregates(sidecar)
    return {"status": "PASS", "split": split, "cohorts": {"train": _derive_cohorts(train, sidecar, "train"), "dev": _derive_cohorts(dev, sidecar, "dev")}, "execution_record_opened": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-head", required=True, metavar="FULL_40_HEX_SHA", help="Immutable implementation anchor: a full 40-hex commit SHA, not the current execution HEAD.")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate(args.repo_root, args.expected_head), sort_keys=True))
        return 0
    except ContractError as exc:
        print(json.dumps({"status": "FAIL", "contract": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
