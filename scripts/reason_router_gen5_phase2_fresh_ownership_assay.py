from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from scripts import reason_router_gen4_pp3_necessity_fast_cuda as pp3
from scripts import reason_router_gen5_phase1b_q22_cuda_equivalence as cuda_eq
from scripts import reason_router_gen5_phase1b_r22_restoration_confirmation as restoration
from scripts import reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu as restoration_cuda
from contramamba.gen5_phase2_state_update_ownership import (
    CorrectionShape,
    StateWriteCorrection,
    load_frozen_owner_bases,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

IMPLEMENTATION_AUTHORITY_COMMIT = "037922770ef51d5ac825690458790e59ff774d63"
IMPLEMENTATION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_phase2_fresh_ownership_assay_"
    "implementation_authority_spec_candidate.md"
)
IMPLEMENTATION_AUTHORITY_BLOB = "765b4ea71999658c9822bef9e8eb7ed2d0b09748"
PHASE2_DESIGN_COMMIT = "d9b84bca8871464807d2dccf6380a6e911d6dbf8"
TRAINING_EXECUTION_COMMIT = "acdf2ee8940070a6e60190a1c2fcbede8933e419"
PROSPECTIVE_CHECKSUM_FIX_COMMIT = "e1f03e36d83a8e2a1ee6c21d87fbd555eadca427"

EXECUTION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_phase2_fresh_ownership_assay_"
    "execution_authority_spec_candidate.md"
)

FROZEN_SOURCE_BLOBS = {
    "src/contramamba/gen5_phase2_state_update_ownership.py":
        "a5c18f5b5d597d9830c3c44a9299af8233677f37",
    "scripts/reason_router_gen5_phase1b_r22_restoration_confirmation.py":
        "786f186ab89ffc44bb5373b70df7a1041165b7a5",
    "scripts/reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu.py":
        "ed73904fdf9a555e11d30e3b9068ac12941225cd",
    "scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py":
        "421d30f00cf71690ed41c983ccf0540808e1de1c",
    "scripts/reason_router_gen4_pp3_necessity_fast_cuda.py":
        "26ca67ad8603799a849c151a39728368227326df",
}

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset({
    "scripts/reason_router_gen5_phase2_fresh_ownership_assay.py",
    "tests/test_reason_router_gen5_phase2_fresh_ownership_assay.py",
})

DATA_ROOT = Path("data/reason_router_gen5_phase2_xg1_ownership_assay_v1")
SOURCE_FILE = "structured_source_facts.jsonl"
ROWS_FILE = "synthetic_reason_router_six_cell.jsonl"
STRUCTURAL_FILE = "structural_manifest.json"
ANCHOR_FILE = "tokenizer_anchor_manifest.jsonl"
ELIGIBILITY_FILE = "tokenizer_eligibility_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

STATIC_INPUT_SHA256 = {
    STRUCTURAL_FILE: "aef8599da138cbf293b97bd48d20b71e8ae486c9b6cee78caada8754d7788911",
    SOURCE_FILE: "f316d7e2a187e90451ff0743829d6c615c063969b66ca51f91e15158bcc1ee06",
    ROWS_FILE: "529027ba79d3bd9fc4e300fbf317f0d84668152ed4fa0593c55af601e7934976",
    ANCHOR_FILE: "880586e043e50e960c05482f3edc6b953b358e50ed0eba5fc279c7785cc33c00",
    ELIGIBILITY_FILE: "ff64b60e3651cf78fe6cb9e9757446593471781cc5977d427eb4c1a6bffd0229",
}

PAIR_FIRST = 8701
PAIR_LAST = 9000
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR
GPU0_FIRST = 8701
GPU0_LAST = 8850
GPU1_FIRST = 8851
GPU1_LAST = 9000
SHARD_PAIR_COUNT = 150

TRAINING_SEEDS = (5201, 5202, 5203)
TRAINING_ARMS = ("G5-C0", "G5-C1", "G5-M1")
TRAINING_MATRIX = tuple(
    (seed, arm)
    for seed in TRAINING_SEEDS
    for arm in TRAINING_ARMS
)
PRIMARY_ARMS = ("G5-C1", "G5-M1")

PARENT_CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)
NATIVE_BACKBONE_SIGNATURE_SHA256 = (
    "81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415"
)
R22_SHA256 = "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
C22_SHA256 = "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"

TRAINING_MATRIX_RESULT = "PASS_GEN5_PHASE2_TRAINING_MATRIX"
TRAINING_CELL_RESULT = "PASS_GEN5_PHASE2_TRAINING_CELL"
TRAINING_ARTIFACT_STALE_CHECKSUM_SHA256 = (
    "91727eafc46f2dc40e460e1932dbf88694bffd8e126759cbc34f86920a800aab"
)
TRAINING_ARTIFACT_FINAL_PROVENANCE_SHA256 = (
    "6fb216b75e084bbdac4fbcd6143de94f2233c3c0c2a879d6a1143277673dd603"
)

CONDITIONS = ("B", "RR", "RC")
DIRECTIONS = restoration.DIRECTIONS
K = restoration.K
EPSILON = restoration.EPSILON
MATCH_TOL = restoration.MATCH_TOL
STATE_SHAPE = restoration.STATE_SHAPE
STATE_WIDTH = restoration.STATE_WIDTH
LAYER22 = restoration.LAYER22

# Preserve the exact Phase1B restoration forward topology. The same frozen-parent
# captures are shared analytically across all nine correction checkpoints because
# a layer-22 additive write correction cannot affect its own layer-22 mixer input.
FORWARDS_PER_PAIR = restoration.FORWARDS_PER_PAIR
SHARD_MODEL_FORWARD_BUDGET = SHARD_PAIR_COUNT * FORWARDS_PER_PAIR
FULL_MODEL_FORWARD_BUDGET = PAIR_COUNT * FORWARDS_PER_PAIR
CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET = 0

ITEM_FILE = "ownership_items.jsonl"
SUMMARY_FILE = "ownership_summary.json"
WORKER0_FILE = "worker0_manifest.json"
WORKER1_FILE = "worker1_manifest.json"
MANIFEST_FILE = "artifact_manifest.json"
OUTPUT_CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen5-phase2-fresh-ownership-item-v1"
SUMMARY_SCHEMA = "gen5-phase2-fresh-ownership-summary-v1"
MANIFEST_SCHEMA = "gen5-phase2-fresh-ownership-manifest-v1"
RESULT_PASS = "PASS_GEN5_PHASE2_FRESH_OWNERSHIP_ASSAY"
LABEL_SUPPORTED = (
    "GEN5_R22_STATE_UPDATE_OWNERSHIP_PRESERVES_CAUSAL_ROLE_INTEGRITY_"
    "OVER_MATCHED_C22_CONTROL"
)
LABEL_NOT_ESTABLISHED = "GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED"


class OwnershipAssayError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise OwnershipAssayError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OwnershipAssayError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_lf_sha256(path: str | Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return sha256_bytes(raw)


def tensor_sha256(value: torch.Tensor) -> str:
    return sha256_bytes(
        value.detach().cpu().contiguous().numpy().tobytes()
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
            out.append(value)
    return out


def expected_pairs() -> tuple[str, ...]:
    return tuple(f"xg1_fact_{i}" for i in range(PAIR_FIRST, PAIR_LAST + 1))


def shard_pairs(shard_id: int) -> tuple[str, ...]:
    require(shard_id in (0, 1), f"SHARD_ID:{shard_id}")
    lo, hi = (GPU0_FIRST, GPU0_LAST) if shard_id == 0 else (GPU1_FIRST, GPU1_LAST)
    rows = tuple(f"xg1_fact_{i}" for i in range(lo, hi + 1))
    require(len(rows) == SHARD_PAIR_COUNT, f"SHARD_PAIR_COUNT:{shard_id}")
    return rows


def cell_key(seed: int, arm: str) -> str:
    require(seed in TRAINING_SEEDS, f"TRAINING_SEED:{seed}")
    require(arm in TRAINING_ARMS, f"TRAINING_ARM:{arm}")
    return f"{seed}/{arm}"


def _status_paths() -> set[str]:
    # Do not route porcelain output through git(): its `.strip()` removes
    # the leading status-column space from the first row, corrupting that
    # first pathname (e.g. "scripts/..." -> "cripts/...").
    try:
        raw = subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OwnershipAssayError("GIT_STATUS_FAILURE") from exc

    out: set[str] = set()
    for line in raw.splitlines():
        if not line:
            continue
        require(len(line) >= 4, f"STATUS_ROW:{line}")
        require(line[2] == " ", f"STATUS_SEPARATOR:{line}")
        out.add(line[3:].replace("\\", "/"))
    return out


def authenticate_repo(
    expected_head: str,
    *,
    allow_implementation_worktree: bool = False,
) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")

    if allow_implementation_worktree:
        require(
            _status_paths().issubset(AUTHORIZED_IMPLEMENTATION_PATHS),
            f"IMPLEMENTATION_SCOPE_DRIFT:{sorted(_status_paths())}",
        )
    else:
        require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (IMPLEMENTATION_AUTHORITY_COMMIT, "IMPLEMENTATION_AUTHORITY"),
        (PHASE2_DESIGN_COMMIT, "PHASE2_DESIGN"),
        (TRAINING_EXECUTION_COMMIT, "TRAINING_EXECUTION"),
        (PROSPECTIVE_CHECKSUM_FIX_COMMIT, "CHECKSUM_FIX"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, expected_head) == 0,
            f"{label}_NOT_ANCESTOR",
        )

    observed_authority = git("rev-parse", f"HEAD:{IMPLEMENTATION_AUTHORITY_PATH}")
    require(
        observed_authority == IMPLEMENTATION_AUTHORITY_BLOB,
        f"IMPLEMENTATION_AUTHORITY_BLOB_DRIFT:{observed_authority}",
    )

    for path, expected_blob in FROZEN_SOURCE_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected_blob, f"FROZEN_SOURCE_BLOB_DRIFT:{path}:{observed}")


def _validate_execution_authority_text(
    text: str,
    *,
    implementation_freeze_commit: str,
) -> None:
    require(
        "SCIENTIFIC_EXECUTION_ALLOWED_AFTER_FREEZE" in text
        and "YES_EXACTLY_ONE_CONFIRMATORY_OWNERSHIP_RUN" in text,
        "EXECUTION_AUTHORITY_NOT_OPEN",
    )
    require(
        f"`{implementation_freeze_commit}`" in text,
        "EXECUTION_AUTHORITY_IMPLEMENTATION_BINDING",
    )


def validate_execution_authority(
    *,
    expected_head: str,
    implementation_freeze_commit: str,
    execution_authority_commit: str,
) -> None:
    path = ROOT / EXECUTION_AUTHORITY_PATH
    require(path.is_file(), "SCIENTIFIC_EXECUTION_AUTHORITY_MISSING")
    require(
        expected_head == git("rev-parse", "HEAD"),
        "EXECUTION_AUTHORITY_HEAD",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            implementation_freeze_commit,
            execution_authority_commit,
        ) == 0,
        "IMPLEMENTATION_FREEZE_NOT_ANCESTOR_OF_EXECUTION_AUTHORITY",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            execution_authority_commit,
            expected_head,
        ) == 0,
        "EXECUTION_AUTHORITY_NOT_ANCESTOR_OF_HEAD",
    )

    # The scientific implementation must remain byte-identical from its
    # freeze commit through the execution head. The execution-authority
    # commit may therefore add only authority/provenance material.
    for rel in sorted(AUTHORIZED_IMPLEMENTATION_PATHS):
        require(
            git_rc(
                "diff",
                "--quiet",
                implementation_freeze_commit,
                expected_head,
                "--",
                rel,
            ) == 0,
            f"IMPLEMENTATION_DRIFT_AFTER_FREEZE:{rel}",
        )

    authority_rel = EXECUTION_AUTHORITY_PATH
    frozen_blob = git(
        "rev-parse",
        f"{execution_authority_commit}:{authority_rel}",
    )
    current_blob = git(
        "rev-parse",
        f"HEAD:{authority_rel}",
    )
    require(
        frozen_blob == current_blob,
        "EXECUTION_AUTHORITY_BLOB_DRIFT",
    )

    text = git(
        "show",
        f"{execution_authority_commit}:{authority_rel}",
    )
    _validate_execution_authority_text(
        text,
        implementation_freeze_commit=implementation_freeze_commit,
    )


def validate_static_inputs() -> dict[str, Any]:
    root = ROOT / DATA_ROOT
    sums = root / CHECKSUM_FILE
    require(sums.is_file(), "XG1_CHECKSUM_FILE_MISSING")
    listed = _parse_checksum_file(sums)
    require(listed == STATIC_INPUT_SHA256, "XG1_CHECKSUM_FILE_CONTENT")

    for name, expected in STATIC_INPUT_SHA256.items():
        path = root / name
        require(path.is_file(), f"XG1_STATIC_INPUT_MISSING:{name}")
        require(git_lf_sha256(path) == expected, f"XG1_STATIC_INPUT_SHA256:{name}")

    structural = json.loads((root / STRUCTURAL_FILE).read_text(encoding="utf-8-sig"))
    require(
        structural.get("result")
        == "PASS_GEN5_PHASE2_XG1_OWNERSHIP_ASSAY_STRUCTURAL_PREPARATION",
        "XG1_STRUCTURAL_RESULT",
    )
    require(structural.get("role") == "state_update_ownership_primary_assay", "XG1_ROLE")
    require(structural.get("pair_id_first") == "xg1_fact_8701", "XG1_FIRST")
    require(structural.get("pair_id_last") == "xg1_fact_9000", "XG1_LAST")
    require(structural.get("source_pair_count") == PAIR_COUNT, "XG1_PAIR_COUNT")
    require(structural.get("row_count") == ROW_COUNT, "XG1_ROW_COUNT")
    require(structural.get("rows_per_pair") == ROWS_PER_PAIR, "XG1_ROWS_PER_PAIR")
    require(structural.get("pair_id_overlap_with_prior") == 0, "XG1_PAIR_OVERLAP")
    require(structural.get("row_filtering_allowed") is False, "XG1_ROW_FILTERING")
    require(structural.get("cohort_replacement_allowed") is False, "XG1_COHORT_REPLACEMENT")
    for key in (
        "labels_present",
        "response_fields_present",
        "endpoint_values_present",
        "checkpoint_loaded",
        "model_executed",
        "cuda_executed",
        "training_executed",
        "backward_executed",
        "task_evaluation_executed",
    ):
        require(structural.get(key) is False, f"XG1_STATIC_BOUNDARY:{key}")

    eligibility = json.loads((root / ELIGIBILITY_FILE).read_text(encoding="utf-8-sig"))
    require(
        eligibility.get("result") == "PASS_GEN5_PHASE2_TOKENIZER_ANCHOR_ELIGIBILITY",
        "XG1_ELIGIBILITY_RESULT",
    )
    require(eligibility.get("source_pair_count") == PAIR_COUNT, "XG1_ELIGIBILITY_PAIRS")
    require(eligibility.get("eligible_anchor_row_count") == ROW_COUNT, "XG1_ELIGIBILITY_ROWS")
    require(
        eligibility.get("identity_name_coordinate_mismatch_count") == 0,
        "XG1_ELIGIBILITY_COORDINATES",
    )
    tokenizer = eligibility.get("tokenizer") or {}
    require(tokenizer.get("tokenizers_version") == "0.22.2", "XG1_TOKENIZERS_VERSION")
    require(tokenizer.get("max_length") == 128, "XG1_MAX_LENGTH")
    require(tokenizer.get("claim_budget") == 63, "XG1_CLAIM_BUDGET")
    require(tokenizer.get("evidence_budget") == 64, "XG1_EVIDENCE_BUDGET")
    require(tokenizer.get("effective_pad_token_id") == 0, "XG1_PAD_ID")
    require(
        tokenizer.get("serialization") == "claim[:63]+EOS(0)+evidence[:64]",
        "XG1_SERIALIZATION",
    )
    return {
        "pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "anchor_manifest_sha256": STATIC_INPUT_SHA256[ANCHOR_FILE],
        "rows_sha256": STATIC_INPUT_SHA256[ROWS_FILE],
    }


@dataclass(frozen=True)
class CorrectionArtifact:
    seed: int
    arm: str
    checkpoint_path: Path
    checkpoint_sha256: str
    a_sha256: str
    b_sha256: str


def _validate_correction_payload(
    payload: Mapping[str, Any],
    *,
    seed: int,
    arm: str,
    expected_a_sha256: str | None = None,
    expected_b_sha256: str | None = None,
) -> tuple[str, str]:
    require(payload.get("schema_version") == "GEN5_PHASE2_FINAL_CORRECTION_V1", "CORRECTION_SCHEMA")
    require(payload.get("execution_commit") == TRAINING_EXECUTION_COMMIT, "CORRECTION_EXECUTION_COMMIT")
    require(payload.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256, "CORRECTION_PARENT")
    require(payload.get("r22_sha256") == R22_SHA256, "CORRECTION_R22")
    require(payload.get("c22_sha256") == C22_SHA256, "CORRECTION_C22")
    require(int(payload.get("seed", -1)) == seed, "CORRECTION_SEED")
    require(str(payload.get("arm")) == arm, "CORRECTION_ARM")
    state = payload.get("state_dict")
    require(isinstance(state, Mapping), "CORRECTION_STATE_DICT")
    require(set(state) == {"A_theta.weight", "B_theta.weight"}, "CORRECTION_STATE_KEYS")
    a = state["A_theta.weight"]
    b = state["B_theta.weight"]
    require(torch.is_tensor(a) and tuple(a.shape) == (2, 768), "CORRECTION_A_SHAPE")
    require(torch.is_tensor(b) and tuple(b.shape) == (24576, 2), "CORRECTION_B_SHAPE")
    require(bool(torch.isfinite(a).all().item()), "CORRECTION_A_NONFINITE")
    require(bool(torch.isfinite(b).all().item()), "CORRECTION_B_NONFINITE")
    a_sha = tensor_sha256(a)
    b_sha = tensor_sha256(b)
    hashes = payload.get("tensor_sha256") or {}
    require(hashes.get("A_theta.weight") == a_sha, "CORRECTION_A_TENSOR_SHA")
    require(hashes.get("B_theta.weight") == b_sha, "CORRECTION_B_TENSOR_SHA")
    if expected_a_sha256 is not None:
        require(a_sha == expected_a_sha256, "CORRECTION_A_EXPECTED_SHA")
    if expected_b_sha256 is not None:
        require(b_sha == expected_b_sha256, "CORRECTION_B_EXPECTED_SHA")
    return a_sha, b_sha


def _parse_checksum_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        require(rel not in out, f"CHECKSUM_DUPLICATE:{rel}")
        out[rel] = digest
    return out


def validate_training_artifacts(root: str | Path) -> dict[tuple[int, str], CorrectionArtifact]:
    root = Path(root)
    require(root.is_dir(), f"TRAINING_ARTIFACT_ROOT_MISSING:{root}")

    sums_path = root / "SHA256SUMS.txt"
    require(sums_path.is_file(), "TRAINING_SHA256SUMS_MISSING")
    listed = _parse_checksum_file(sums_path)
    mismatches: list[tuple[str, str, str]] = []
    for rel, expected in listed.items():
        path = root / rel
        require(path.is_file(), f"TRAINING_CHECKSUM_FILE_MISSING:{rel}")
        observed = sha256_file(path)
        if observed != expected:
            mismatches.append((rel, expected, observed))
    require(len(mismatches) == 1, f"TRAINING_CHECKSUM_MISMATCH_COUNT:{mismatches}")
    rel, stale, current = mismatches[0]
    require(rel == "matrix_provenance.json", f"TRAINING_UNEXPECTED_STALE_FILE:{rel}")
    require(stale == TRAINING_ARTIFACT_STALE_CHECKSUM_SHA256, "TRAINING_STALE_SHA_IDENTITY")
    require(current == TRAINING_ARTIFACT_FINAL_PROVENANCE_SHA256, "TRAINING_FINAL_PROVENANCE_SHA")

    manifest_path = root / "matrix_manifest.json"
    provenance_path = root / "matrix_provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require(manifest.get("result") == TRAINING_MATRIX_RESULT, "TRAINING_MATRIX_RESULT")
    require(manifest.get("execution_commit") == TRAINING_EXECUTION_COMMIT, "TRAINING_MATRIX_COMMIT")
    require(int(manifest.get("cell_count", -1)) == 9, "TRAINING_MATRIX_CELL_COUNT")
    require(manifest.get("fresh_xg1_loaded") is False, "TRAINING_MATRIX_XG1_FIREWALL")
    require(manifest.get("scientific_p_value_count") == 0, "TRAINING_MATRIX_PVALUE_FIREWALL")
    require(provenance.get("status") == "PASS", "TRAINING_MATRIX_PROVENANCE_STATUS")
    require(provenance.get("execution_commit") == TRAINING_EXECUTION_COMMIT, "TRAINING_PROVENANCE_COMMIT")
    require(sha256_file(provenance_path) == TRAINING_ARTIFACT_FINAL_PROVENANCE_SHA256, "TRAINING_PROVENANCE_FROZEN_SHA")
    require(
        provenance.get("matrix_manifest_sha256") == sha256_file(manifest_path),
        "TRAINING_MANIFEST_SHA_CHAIN",
    )

    cells = {
        (int(row["seed"]), str(row["arm"])): row
        for row in manifest.get("cells", [])
    }
    require(set(cells) == set(TRAINING_MATRIX), "TRAINING_MATRIX_CELL_SET")

    out: dict[tuple[int, str], CorrectionArtifact] = {}
    for seed, arm in TRAINING_MATRIX:
        cell_dir = root / f"seed{seed}" / arm
        report_path = cell_dir / "training_report.json"
        run_path = cell_dir / "run_provenance.json"
        checkpoint_path = cell_dir / "final_correction.pt"
        for path in (report_path, run_path, checkpoint_path):
            require(path.is_file(), f"TRAINING_CELL_FILE_MISSING:{seed}:{arm}:{path.name}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        run = json.loads(run_path.read_text(encoding="utf-8"))
        require(report.get("result") == TRAINING_CELL_RESULT, f"TRAINING_CELL_RESULT:{seed}:{arm}")
        require(report.get("execution_commit") == TRAINING_EXECUTION_COMMIT, "TRAINING_CELL_COMMIT")
        require(report.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256, "TRAINING_CELL_PARENT")
        require(report.get("r22_sha256") == R22_SHA256, "TRAINING_CELL_R22")
        require(report.get("c22_sha256") == C22_SHA256, "TRAINING_CELL_C22")
        require(int(report.get("seed", -1)) == seed and str(report.get("arm")) == arm, "TRAINING_CELL_ID")
        require(int(report.get("optimizer_steps", -1)) == 20, "TRAINING_CELL_STEPS")
        require(report.get("fresh_xg1_loaded") is False, "TRAINING_CELL_XG1_FIREWALL")
        require(report.get("scientific_p_value_count") == 0, "TRAINING_CELL_PVALUE_FIREWALL")
        require(report.get("scientific_conclusion") is None, "TRAINING_CELL_CONCLUSION_FIREWALL")
        require(report.get("training_success") is True, "TRAINING_CELL_SUCCESS")
        require(report.get("parent_signature_before") == report.get("parent_signature_after"), "TRAINING_PARENT_MUTATION")
        require(report.get("r22_runtime_sha256_before") == report.get("r22_runtime_sha256_after"), "TRAINING_R22_MUTATION")
        require(report.get("c22_runtime_sha256_before") == report.get("c22_runtime_sha256_after"), "TRAINING_C22_MUTATION")

        checkpoint_sha = sha256_file(checkpoint_path)
        require(report.get("final_correction_file_sha256") == checkpoint_sha, "TRAINING_CORRECTION_FILE_SHA")
        require(run.get("status") == "PASS", "TRAINING_RUN_STATUS")
        require(run.get("training_report_sha256") == sha256_file(report_path), "TRAINING_REPORT_SHA_CHAIN")
        require(run.get("final_correction_file_sha256") == checkpoint_sha, "TRAINING_RUN_CORRECTION_SHA")
        manifest_cell = cells[(seed, arm)]
        require(manifest_cell.get("training_report_sha256") == sha256_file(report_path), "TRAINING_MANIFEST_REPORT_SHA")
        require(manifest_cell.get("run_provenance_sha256") == sha256_file(run_path), "TRAINING_MANIFEST_RUN_SHA")
        require(manifest_cell.get("final_correction_sha256") == checkpoint_sha, "TRAINING_MANIFEST_CORRECTION_SHA")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(isinstance(payload, Mapping), "CORRECTION_PAYLOAD_MAPPING")
        a_sha, b_sha = _validate_correction_payload(payload, seed=seed, arm=arm)
        out[(seed, arm)] = CorrectionArtifact(
            seed=seed,
            arm=arm,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            a_sha256=a_sha,
            b_sha256=b_sha,
        )

    require(len(out) == 9, "TRAINING_ARTIFACT_CELL_COUNT")
    return out


def load_inputs(tokenizer_snapshot: str | Path | None):
    validate_static_inputs()
    rows = restoration.necessity.adapter.validate_gen4_rows(
        read_jsonl(ROOT / DATA_ROOT / ROWS_FILE),
        require_canonical_shape=True,
    )
    require(len(rows) == ROW_COUNT, "XG1_RUNTIME_ROW_COUNT")
    order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        pair = str(row["source_pair_id"])
        if pair not in seen:
            seen.add(pair)
            order.append(pair)
    require(tuple(order) == expected_pairs(), "XG1_RUNTIME_PAIR_ORDER")

    tokenizer, tokenizer_provenance = (
        restoration.necessity.tokenizer_gate.load_canonical_analysis_tokenizer(tokenizer_snapshot)
    )
    encoded = restoration.necessity.adapter.encode_gen4_rows(rows, tokenizer)
    event_rows = read_jsonl(ROOT / DATA_ROOT / ANCHOR_FILE)
    require(len(event_rows) == ROW_COUNT, "XG1_RUNTIME_ANCHOR_COUNT")
    events = restoration.necessity.parent.event_lookup(event_rows)
    restoration.necessity.parent.validate_transport_event_plan(tuple(order), events)
    row_index = restoration.necessity.parent.build_row_index(rows)
    require(
        list(encoded["source_pair_id"]) == [str(row["source_pair_id"]) for row in rows],
        "XG1_RUNTIME_ENCODED_ORDER",
    )
    return rows, encoded, events, row_index, tokenizer_provenance


def _feature_row(
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = int(row_index[(pair, cell)])
    input_ids = encoded["input_ids"][index].unsqueeze(0)
    attention_mask = encoded["attention_mask"][index].unsqueeze(0)
    require(tuple(input_ids.shape) == (1, 128), "INPUT_IDS_SHAPE")
    require(tuple(attention_mask.shape) == (1, 128), "ATTENTION_MASK_SHAPE")
    return input_ids, attention_mask


def load_corrections(
    artifact_index: Mapping[tuple[int, str], CorrectionArtifact],
    *,
    r22: torch.Tensor,
    c22: torch.Tensor,
    device: torch.device,
) -> dict[tuple[int, str], StateWriteCorrection]:
    out: dict[tuple[int, str], StateWriteCorrection] = {}
    for seed, arm in TRAINING_MATRIX:
        artifact = artifact_index[(seed, arm)]
        payload = torch.load(artifact.checkpoint_path, map_location="cpu", weights_only=True)
        _validate_correction_payload(
            payload,
            seed=seed,
            arm=arm,
            expected_a_sha256=artifact.a_sha256,
            expected_b_sha256=artifact.b_sha256,
        )
        module = StateWriteCorrection(
            arm=arm,
            r22=r22,
            c22=c22,
            seed=seed,
        ).to(device=device)
        with torch.no_grad():
            module.A_theta.weight.copy_(
                payload["state_dict"]["A_theta.weight"].to(
                    device=device,
                    dtype=module.A_theta.weight.dtype,
                )
            )
            module.B_theta.weight.copy_(
                payload["state_dict"]["B_theta.weight"].to(
                    device=device,
                    dtype=module.B_theta.weight.dtype,
                )
            )
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
        require(tensor_sha256(module.A_theta.weight) == artifact.a_sha256, "LOADED_A_SHA")
        require(tensor_sha256(module.B_theta.weight) == artifact.b_sha256, "LOADED_B_SHA")
        out[(seed, arm)] = module
    return out


def correction_signatures(
    corrections: Mapping[tuple[int, str], StateWriteCorrection],
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for seed, arm in TRAINING_MATRIX:
        module = corrections[(seed, arm)]
        out[cell_key(seed, arm)] = {
            "A_theta.weight": tensor_sha256(module.A_theta.weight),
            "B_theta.weight": tensor_sha256(module.B_theta.weight),
        }
    return out


class Layer22InputCapture:
    def __init__(self, mixer: Any):
        self.mixer = mixer
        self.handle: Any | None = None
        self.captured: torch.Tensor | None = None

    def _hook(self, _module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        hidden = args[0] if args else kwargs.get("hidden_states")
        require(torch.is_tensor(hidden), "LAYER22_INPUT_MISSING")
        require(self.captured is None, "LAYER22_INPUT_MULTIPLE_CALLS")
        self.captured = hidden.detach().clone()

    def __enter__(self):
        self.handle = self.mixer.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        return False


def _capture_branch(
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    input_ids: torch.Tensor,
    anchor: int,
    direction: torch.Tensor,
    orientation: int,
    plus_branch: bool,
    pp3_planes: Mapping[str, torch.Tensor],
    native_upstream: bool,
    budget: Any,
) -> dict[str, Any]:
    target = int(anchor) + cuda_eq.core.TARGET_OFFSET
    handle, upstream_audit = restoration_cuda._install_upstream_hook(
        runtime_ctx=runtime_ctx,
        target=target,
        direction=direction,
        orientation=orientation,
        plus_branch=plus_branch,
        pp3_planes=pp3_planes,
        neutralized=not native_upstream,
    )
    mixer = model.mamba.layers[LAYER22].mixer
    scan_capture = cuda_eq.Layer22FastScanCapture(mixer)
    input_capture = Layer22InputCapture(mixer)
    budget.consume()
    try:
        with scan_capture.capture(), input_capture:
            with torch.inference_mode():
                _ = model.mamba(input_ids=input_ids.to("cuda:0"))
    finally:
        handle.remove()
    require(bool(upstream_audit), "UPSTREAM_HOOK_AUDIT")
    require(scan_capture.captured is not None, "LAYER22_SCAN_CAPTURE_MISSING")
    require(input_capture.captured is not None, "LAYER22_INPUT_CAPTURE_MISSING")
    return {
        "captured": scan_capture.captured,
        "mixer_input": input_capture.captured,
        "target_token_index": target,
        "upstream_condition": "native" if native_upstream else "pp3_neutralized",
        "upstream_audit": {
            "condition": str(upstream_audit["condition"]),
            "token_index": int(upstream_audit["token_index"]),
            "orientation": int(upstream_audit["orientation"]),
            "branch_sign": int(upstream_audit["branch_sign"]),
            "probe_correction_l2": float(upstream_audit["probe_correction_l2"]),
            "applied_correction_max_abs_residual":
                float(upstream_audit["applied_correction_max_abs_residual"]),
        },
    }


def _prepare_correction_sequence(
    correction: StateWriteCorrection,
    *,
    mixer_input: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_index: int,
) -> dict[str, Any]:
    mask = attention_mask.to(device=mixer_input.device)
    with torch.inference_mode():
        raw, effective = correction(
            mixer_input,
            attention_mask=mask,
            return_preproject=True,
        )
    require(tuple(raw.shape[:2]) == tuple(mixer_input.shape[:2]), "CORRECTION_RAW_SHAPE")
    require(raw.shape[-1] == STATE_WIDTH, "CORRECTION_RAW_WIDTH")
    require(tuple(effective.shape) == tuple(raw.shape), "CORRECTION_EFFECTIVE_SHAPE")
    require(bool(torch.isfinite(raw).all().item()), "CORRECTION_RAW_NONFINITE")
    require(bool(torch.isfinite(effective).all().item()), "CORRECTION_EFFECTIVE_NONFINITE")
    target_raw = raw[:, target_token_index, :].detach().clone()
    target_effective = effective[:, target_token_index, :].detach().clone()
    state_write = effective.reshape(
        effective.shape[0],
        effective.shape[1],
        STATE_SHAPE[1],
        STATE_SHAPE[2],
    ).contiguous()
    basis = None
    if correction.arm == "G5-M1":
        basis = correction.R22
    elif correction.arm == "G5-C1":
        basis = correction.C22
    if basis is None:
        projection_residual = 0.0
    else:
        projection_residual = float(
            torch.max(
                torch.abs(
                    target_effective.to(torch.float64)
                    @ basis.to(device=target_effective.device, dtype=torch.float64)
                )
            ).item()
        )
        require(projection_residual <= 1e-5, f"CORRECTION_PROJECTOR_RESIDUAL:{projection_residual}")
    return {
        "state_write": state_write,
        "target_raw": target_raw,
        "target_effective": target_effective,
        "target_raw_l2": float(torch.linalg.vector_norm(target_raw).item()),
        "target_effective_l2": float(torch.linalg.vector_norm(target_effective).item()),
        "projection_residual": projection_residual,
        "target_effective_sha256": tensor_sha256(target_effective),
    }


def ownership_write_plan(
    native_background_write: torch.Tensor,
    donor_native_write: torch.Tensor,
    correction_write: torch.Tensor,
    r22: torch.Tensor,
    c22: torch.Tensor,
    condition: str,
) -> dict[str, Any]:
    require(condition in CONDITIONS, f"CONDITION:{condition}")
    native_background = native_background_write.detach().cpu().to(torch.float64).reshape(-1).contiguous()
    donor_native = donor_native_write.detach().cpu().to(torch.float64).reshape(-1).contiguous()
    correction = correction_write.detach().cpu().to(torch.float64).reshape(-1).contiguous()
    require(native_background.numel() == donor_native.numel() == correction.numel() == STATE_WIDTH, "OWNERSHIP_WRITE_WIDTH")
    require(bool(torch.isfinite(native_background).all().item()), "NATIVE_BACKGROUND_NONFINITE")
    require(bool(torch.isfinite(donor_native).all().item()), "DONOR_NATIVE_NONFINITE")
    require(bool(torch.isfinite(correction).all().item()), "CORRECTION_WRITE_NONFINITE")
    corrected_background = (native_background + correction).contiguous()
    plan = restoration.restoration_vectors(
        corrected_background,
        donor_native,
        r22,
        c22,
        condition,
    )
    # The donor coefficient is computed only from the native donor. Correction
    # is added to the background before RR/RC restoration, never to the donor.
    expected_a_native = (r22.T @ donor_native).contiguous()
    require(
        torch.equal(plan["a_native"], expected_a_native),
        "DONOR_COEFFICIENT_NOT_NATIVE_ONLY",
    )
    return {
        **plan,
        "native_background": native_background,
        "correction_write": correction,
        "corrected_background": corrected_background,
    }


def _discrete_a_from_capture(
    delta_t: torch.Tensor,
    a_matrix: torch.Tensor,
    delta_bias: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Recover the Phase2 correction recurrence coefficient from frozen fast inputs."""
    require(delta_t.ndim == 2, "DELTA_T_RANK")
    require(a_matrix.ndim == 2, "A_MATRIX_RANK")
    require(delta_bias.ndim == 1, "DELTA_BIAS_RANK")
    require(delta_t.shape[1] == a_matrix.shape[0] == delta_bias.shape[0], "DISCRETE_A_INTERMEDIATE")
    discrete_time_step = torch.nn.functional.softplus(
        delta_t.float() + delta_bias.float().unsqueeze(0)
    )
    return torch.exp(
        a_matrix.float().unsqueeze(0)
        * discrete_time_step.unsqueeze(-1)
    ).to(dtype=dtype)


def _replay_ownership_window(
    captured: tuple[torch.Tensor, ...],
    *,
    anchor: int,
    condition: str,
    donor_native_write: torch.Tensor,
    correction_sequence: Mapping[str, Any],
    r22: torch.Tensor,
    c22: torch.Tensor,
    kernels: Mapping[str, Any],
) -> dict[str, Any]:
    require(condition in CONDITIONS, f"CONDITION:{condition}")
    require(len(captured) == 8, "REPLAY_CAPTURE_WIDTH")
    u, delta, a_matrix, b_scan, c_scan, d_vector, gate, delta_bias = captured
    target = int(anchor) + cuda_eq.core.TARGET_OFFSET
    require(anchor >= 1 and anchor + 4 < u.shape[-1], "REPLAY_RANGE")
    correction_write = correction_sequence["state_write"]
    require(correction_write.ndim == 4, "CORRECTION_SEQUENCE_RANK")
    require(correction_write.shape[0] == u.shape[0], "CORRECTION_SEQUENCE_BATCH")
    require(correction_write.shape[1] >= anchor + 5, "CORRECTION_SEQUENCE_LENGTH")

    kernel_scan = kernels["selective_scan_fn"]
    kernel_update = kernels["selective_state_update"]

    _, base_state = kernel_scan(
        cuda_eq._slice_last(u, anchor),
        cuda_eq._slice_last(delta, anchor),
        a_matrix,
        cuda_eq._slice_last(b_scan, anchor),
        cuda_eq._slice_last(c_scan, anchor),
        d_vector,
        cuda_eq._slice_last(gate, anchor),
        delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )
    require(torch.is_tensor(base_state), "PREFIX_STATE")

    # Keep the trained Phase2 correction recurrence separate from the frozen
    # native/restoration recurrence. This matches the production wrapper's
    # superposition semantics: native CUDA state + correction-only recurrence.
    corr_state = torch.zeros_like(base_state)
    for token in range(anchor):
        discrete_a_t = _discrete_a_from_capture(
            delta[..., token],
            a_matrix,
            delta_bias,
            dtype=corr_state.dtype,
        )
        corr_state = (
            discrete_a_t * corr_state
            + correction_write[:, token].to(
                device=corr_state.device,
                dtype=corr_state.dtype,
            )
        ).contiguous()

    post_states: dict[int, torch.Tensor] = {}
    native_target_write: torch.Tensor | None = None
    total_actual_write: torch.Tensor | None = None
    plan: dict[str, Any] | None = None

    for token in range(anchor, anchor + 5):
        corr_t = correction_write[:, token].to(
            device=corr_state.device,
            dtype=corr_state.dtype,
        )
        discrete_a_t = _discrete_a_from_capture(
            delta[..., token],
            a_matrix,
            delta_bias,
            dtype=corr_state.dtype,
        )

        if token != target:
            _ = kernel_update(
                base_state,
                u[..., token],
                delta[..., token],
                a_matrix,
                b_scan[..., token],
                c_scan[..., token],
                d_vector,
                gate[..., token],
                delta_bias,
                dt_softplus=True,
            )
            corr_state = (discrete_a_t * corr_state + corr_t).contiguous()
        else:
            pre = base_state.detach().clone()
            state_native = pre.clone()
            state_decay = pre.clone()
            _ = kernel_update(
                state_native,
                u[..., token],
                delta[..., token],
                a_matrix,
                b_scan[..., token],
                c_scan[..., token],
                d_vector,
                gate[..., token],
                delta_bias,
                dt_softplus=True,
            )
            _ = kernel_update(
                state_decay,
                torch.zeros_like(u[..., token]),
                delta[..., token],
                a_matrix,
                b_scan[..., token],
                c_scan[..., token],
                d_vector,
                gate[..., token],
                delta_bias,
                dt_softplus=True,
            )
            native_target_write = restoration_cuda._finite_tensor(
                state_native - state_decay,
                "OWNERSHIP_NATIVE_WRITE22",
            )
            plan = ownership_write_plan(
                native_target_write,
                donor_native_write,
                corr_t,
                r22,
                c22,
                condition,
            )

            # Restoration addition is part of the native/restoration component;
            # trained correction stays in its own recurrence component.
            restoration_add = (
                plan["applied"]
                .reshape(STATE_SHAPE)
                .to(device=state_decay.device, dtype=state_decay.dtype)
            )
            native_write_live = native_target_write.to(
                device=state_decay.device,
                dtype=state_decay.dtype,
            )
            base_state = (
                state_decay + native_write_live + restoration_add
            ).contiguous()
            corr_state = (discrete_a_t * corr_state + corr_t).contiguous()
            total_actual_write = (
                native_write_live + restoration_add + corr_t
            ).contiguous()
            planned_total = (
                plan["modified"]
                .reshape(STATE_SHAPE)
                .to(device=total_actual_write.device, dtype=total_actual_write.dtype)
            )
            total_write_residual = float(
                torch.max(torch.abs(total_actual_write - planned_total)).item()
            )
            require(
                total_write_residual <= MATCH_TOL,
                f"TOTAL_WRITE_PLAN_RESIDUAL:{total_write_residual}",
            )

        total_state = (base_state + corr_state).contiguous()
        post_states[token] = restoration_cuda._finite_tensor(
            total_state,
            f"OWNERSHIP_POST_STATE22:{token}",
        )

    require(
        native_target_write is not None
        and total_actual_write is not None
        and plan is not None,
        "OWNERSHIP_TARGET_WRITE_MISSING",
    )
    pe22 = restoration.necessity.q22_path_efficiency(post_states, int(anchor))
    return {
        "native_write": native_target_write,
        "actual_write": total_actual_write.detach().cpu().to(torch.float32).contiguous(),
        "correction_write": correction_sequence["target_effective"].detach().cpu().to(torch.float32).contiguous(),
        "plan": plan,
        "post_states": post_states,
        "pe22": float(pe22),
        "target_token_index": target,
    }

def _cell_role_audit(
    *,
    donor: Mapping[str, Any],
    b: Mapping[str, Any],
    rr: Mapping[str, Any],
    rc: Mapping[str, Any],
    correction_sequence: Mapping[str, Any],
    r22: torch.Tensor,
) -> dict[str, float | str]:
    b_native = b["native_write"].to(torch.float64)
    rr_native = rr["native_write"].to(torch.float64)
    rc_native = rc["native_write"].to(torch.float64)
    rr_res = float(torch.max(torch.abs(b_native - rr_native)).item())
    rc_res = float(torch.max(torch.abs(b_native - rc_native)).item())
    require(rr_res <= MATCH_TOL, f"RR_BACKGROUND_NATIVE_MISMATCH:{rr_res}")
    require(rc_res <= MATCH_TOL, f"RC_BACKGROUND_NATIVE_MISMATCH:{rc_res}")

    rr_a = rr["plan"]["a_native"].to(torch.float64)
    rc_a = rc["plan"]["a_native"].to(torch.float64)
    coeff_res = float(torch.max(torch.abs(rr_a - rc_a)).item())
    require(coeff_res <= MATCH_TOL, f"DONOR_COEFFICIENT_RR_RC:{coeff_res}")
    donor_native = donor["native_write"].to(torch.float64).reshape(-1)
    expected_a = r22.T @ donor_native
    donor_coeff_res = float(torch.max(torch.abs(rr_a - expected_a)).item())
    require(donor_coeff_res <= MATCH_TOL, f"DONOR_COEFFICIENT_NATIVE_SOURCE:{donor_coeff_res}")

    return {
        "background_native_rr_max_abs_residual": rr_res,
        "background_native_rc_max_abs_residual": rc_res,
        "donor_coefficient_rr_rc_max_abs_residual": coeff_res,
        "donor_coefficient_native_source_max_abs_residual": donor_coeff_res,
        "matched_addition_norm_residual": max(
            float(rr["plan"]["matched_addition_norm_residual"]),
            float(rc["plan"]["matched_addition_norm_residual"]),
        ),
        "correction_target_effective_sha256": str(correction_sequence["target_effective_sha256"]),
        "correction_target_raw_l2": float(correction_sequence["target_raw_l2"]),
        "correction_target_effective_l2": float(correction_sequence["target_effective_l2"]),
        "correction_projector_residual": float(correction_sequence["projection_residual"]),
    }


def _run_signed_coordinate_shared(
    seed_meta: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    orientation: int,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    kernels: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[Any, Any],
    pp3_planes: Mapping[str, torch.Tensor],
    r22: torch.Tensor,
    c22: torch.Tensor,
    corrections: Mapping[tuple[int, str], StateWriteCorrection],
    budget: Any,
) -> dict[str, dict[str, Any]]:
    pair = str(seed_meta["source_pair_id"])
    anchors = restoration.necessity.phase1._anchors_for_pair(pair, events)
    cells = restoration.necessity.phase1._cells()
    result = {
        cell_key(seed, arm): {"orientation": int(orientation), "branches": {}}
        for seed, arm in TRAINING_MATRIX
    }

    for role, plus_branch in (("tp", True), ("tm", False)):
        input_ids, attention_mask = _feature_row(encoded, row_index, pair, cells[role])
        common = dict(
            model=model,
            runtime_ctx=runtime_ctx,
            input_ids=input_ids,
            anchor=int(anchors[role]),
            direction=direction,
            orientation=int(orientation),
            plus_branch=plus_branch,
            pp3_planes=pp3_planes,
            budget=budget,
        )
        donor_capture = _capture_branch(native_upstream=True, **common)
        donor = restoration_cuda.replay_restoration_window(
            donor_capture["captured"],
            anchor=int(anchors[role]),
            condition="DONOR",
            donor_write=None,
            r22=r22,
            c22=c22,
            kernels=kernels,
        )

        condition_captures = {
            cond: _capture_branch(native_upstream=False, **common)
            for cond in CONDITIONS
        }
        b_input = condition_captures["B"]["mixer_input"]
        for cond in ("RR", "RC"):
            residual = float(
                torch.max(
                    torch.abs(b_input - condition_captures[cond]["mixer_input"])
                ).item()
            )
            require(residual <= MATCH_TOL, f"CORRECTION_INPUT_CONDITION_MISMATCH:{cond}:{residual}")

        for seed, arm in TRAINING_MATRIX:
            key = cell_key(seed, arm)
            module = corrections[(seed, arm)]
            target = int(donor_capture["target_token_index"])
            corr = _prepare_correction_sequence(
                module,
                mixer_input=b_input,
                attention_mask=attention_mask,
                target_token_index=target,
            )
            replayed = {
                cond: _replay_ownership_window(
                    condition_captures[cond]["captured"],
                    anchor=int(anchors[role]),
                    condition=cond,
                    donor_native_write=donor["native_write"],
                    correction_sequence=corr,
                    r22=r22,
                    c22=c22,
                    kernels=kernels,
                )
                for cond in CONDITIONS
            }
            audit = _cell_role_audit(
                donor=donor,
                b=replayed["B"],
                rr=replayed["RR"],
                rc=replayed["RC"],
                correction_sequence=corr,
                r22=r22,
            )
            audit.update({
                "B_pe22": float(replayed["B"]["pe22"]),
                "RR_pe22": float(replayed["RR"]["pe22"]),
                "RC_pe22": float(replayed["RC"]["pe22"]),
            })
            result[key]["branches"][role] = audit

    for key, row in result.items():
        for cond in CONDITIONS:
            row[f"F22_{cond}"] = (
                float(row["branches"]["tp"][f"{cond}_pe22"])
                - float(row["branches"]["tm"][f"{cond}_pe22"])
            )
    return result


def _run_direction_shared(
    seed_meta: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    family: str,
    basis_index: int,
    **kwargs,
) -> dict[str, dict[str, Any]]:
    pos = _run_signed_coordinate_shared(seed_meta, direction, orientation=1, **kwargs)
    neg = _run_signed_coordinate_shared(seed_meta, direction, orientation=-1, **kwargs)
    out: dict[str, dict[str, Any]] = {}
    for seed, arm in TRAINING_MATRIX:
        key = cell_key(seed, arm)
        row: dict[str, Any] = {
            "direction_key": f"{family}_{basis_index}",
            "basis_family": family,
            "basis_index": int(basis_index),
        }
        for cond in CONDITIONS:
            f_plus = float(pos[key][f"F22_{cond}"])
            f_minus = float(neg[key][f"F22_{cond}"])
            j = (f_plus - f_minus) / (2.0 * EPSILON)
            require(math.isfinite(j), f"J22_NONFINITE:{key}:{cond}:{family}:{basis_index}")
            row[f"J22_{cond}_squared"] = j * j
        # Persist maxima only; raw recurrent vectors are never persisted.
        audits = [
            pos[key]["branches"][role]
            for role in ("tp", "tm")
        ] + [
            neg[key]["branches"][role]
            for role in ("tp", "tm")
        ]
        for field in (
            "background_native_rr_max_abs_residual",
            "background_native_rc_max_abs_residual",
            "donor_coefficient_rr_rc_max_abs_residual",
            "donor_coefficient_native_source_max_abs_residual",
            "matched_addition_norm_residual",
            "correction_projector_residual",
            "correction_target_raw_l2",
            "correction_target_effective_l2",
        ):
            row[f"max_{field}"] = max(float(a[field]) for a in audits)
        out[key] = row
    return out


def endpoint(q_b: float, q_rr: float, q_rc: float) -> dict[str, float]:
    base = restoration.endpoint(q_b, q_rr, q_rc)
    return {**base, "I_A": float(base["D_SUF22"])}


def _seed_meta(pair_index: int, pair: str, events: Mapping[Any, Any]) -> dict[str, Any]:
    anchors = restoration.necessity.phase1._anchors_for_pair(pair, events)
    return {
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": int(pair_index),
        "target_plus_anchor": int(anchors["tp"]),
        "target_minus_anchor": int(anchors["tm"]),
        "reference_plus_anchor": int(anchors["rp"]),
        "reference_minus_anchor": int(anchors["rm"]),
    }


def run_pair_shared_capture(
    seed_meta: Mapping[str, Any],
    *,
    q_bases: Mapping[str, torch.Tensor],
    artifact_index: Mapping[tuple[int, str], CorrectionArtifact],
    **kwargs,
) -> dict[str, Any]:
    direction_rows: dict[str, list[dict[str, Any]]] = {
        cell_key(seed, arm): [] for seed, arm in TRAINING_MATRIX
    }
    for family in ("xg2", "xg4"):
        for basis_index in range(K):
            rows = _run_direction_shared(
                seed_meta,
                q_bases[family][:, basis_index],
                family=family,
                basis_index=basis_index,
                **kwargs,
            )
            for key, row in rows.items():
                direction_rows[key].append(row)

    cells: dict[str, Any] = {}
    for seed, arm in TRAINING_MATRIX:
        key = cell_key(seed, arm)
        directions = direction_rows[key]
        require([r["direction_key"] for r in directions] == list(DIRECTIONS), f"DIRECTION_ORDER:{key}")
        q: dict[str, float] = {}
        for cond in CONDITIONS:
            e2 = sum(float(r[f"J22_{cond}_squared"]) for r in directions[:K]) / K
            e4 = sum(float(r[f"J22_{cond}_squared"]) for r in directions[K:]) / K
            q[cond] = e2 - e4
            require(math.isfinite(q[cond]), f"Q22_NONFINITE:{key}:{cond}")
        ep = endpoint(q["B"], q["RR"], q["RC"])
        artifact = artifact_index[(seed, arm)]
        diagnostic_fields = [name for name in directions[0] if name.startswith("max_")]
        diagnostics = {
            field: max(float(row[field]) for row in directions)
            for field in diagnostic_fields
        }
        cells[key] = {
            "seed": seed,
            "arm": arm,
            "correction_checkpoint_sha256": artifact.checkpoint_sha256,
            "A_theta_sha256": artifact.a_sha256,
            "B_theta_sha256": artifact.b_sha256,
            **ep,
            "diagnostics": diagnostics,
        }

    return {
        **dict(seed_meta),
        "schema_version": ITEM_SCHEMA,
        "cells": cells,
        "scientific_model_forward_count_this_run": FORWARDS_PER_PAIR,
        "cpu_scientific_model_forward_count_this_run": 0,
        "shared_frozen_parent_capture_across_corrections": True,
        "row_dropped": False,
    }


def merge_shards(
    shard0: Sequence[Mapping[str, Any]],
    shard1: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    require([str(x["source_pair_id"]) for x in shard0] == list(shard_pairs(0)), "SHARD0_ORDER")
    require([str(x["source_pair_id"]) for x in shard1] == list(shard_pairs(1)), "SHARD1_ORDER")
    merged = [dict(x) for x in shard0] + [dict(x) for x in shard1]
    require([str(x["source_pair_id"]) for x in merged] == list(expected_pairs()), "MERGED_ORDER")
    require(len({str(x["source_pair_id"]) for x in merged}) == PAIR_COUNT, "PAIR_DUPLICATE")
    require(all(x.get("row_dropped") is False for x in merged), "ROW_DROPPING")
    require(
        sum(int(x["scientific_model_forward_count_this_run"]) for x in merged)
        == FULL_MODEL_FORWARD_BUDGET,
        "GLOBAL_FORWARD_BUDGET",
    )
    return merged


def aggregate_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(items) == PAIR_COUNT, "AGGREGATE_ITEM_COUNT")
    require([str(x["source_pair_id"]) for x in items] == list(expected_pairs()), "AGGREGATE_PAIR_ORDER")
    out: list[dict[str, Any]] = []
    for raw in items:
        cells = raw.get("cells")
        require(isinstance(cells, Mapping), "AGGREGATE_CELLS")
        require(set(cells) == {cell_key(seed, arm) for seed, arm in TRAINING_MATRIX}, "AGGREGATE_CELL_SET")
        seed_averaged: dict[str, dict[str, float]] = {}
        for arm in TRAINING_ARMS:
            rows = [cells[cell_key(seed, arm)] for seed in TRAINING_SEEDS]
            seed_averaged[arm] = {
                field: sum(float(row[field]) for row in rows) / len(TRAINING_SEEDS)
                for field in ("Q_B", "Q_RR", "Q_RC", "S_R", "S_C", "I_A")
            }
        d_own = seed_averaged["G5-M1"]["I_A"] - seed_averaged["G5-C1"]["I_A"]
        require(math.isfinite(d_own), "D_OWN_NONFINITE")
        out.append({
            "source_pair_id": str(raw["source_pair_id"]),
            "pair_index": int(raw["pair_index"]),
            "cells": {key: dict(value) for key, value in cells.items()},
            "seed_averaged": seed_averaged,
            "D_OWN": d_own,
            "row_dropped": False,
        })
    return out


def confirmatory_decision(
    aggregated: Sequence[Mapping[str, Any]],
    *,
    ttest_fn: Callable[[Sequence[float]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    require(len(aggregated) == PAIR_COUNT, "DECISION_ITEM_COUNT")
    require([str(x["source_pair_id"]) for x in aggregated] == list(expected_pairs()), "DECISION_PAIR_ORDER")
    require(all(x.get("row_dropped") is False for x in aggregated), "DECISION_ROW_DROPPING")
    d_own = [float(x["D_OWN"]) for x in aggregated]
    # The confirmatory test receives exactly 300 item-level seed-averaged
    # contrasts. Training seeds are never expanded into pseudo-replicates.
    require(len(d_own) == 300, "DECISION_SAMPLE_SIZE")
    if ttest_fn is None:
        ttest_fn = restoration.necessity.one_sided_one_sample_student_t
    test = dict(ttest_fn(d_own))
    mean_d = sum(d_own) / PAIR_COUNT
    mean_i_m1 = sum(float(x["seed_averaged"]["G5-M1"]["I_A"]) for x in aggregated) / PAIR_COUNT
    mean_q_rr_m1 = sum(float(x["seed_averaged"]["G5-M1"]["Q_RR"]) for x in aggregated) / PAIR_COUNT
    mean_s_r_m1 = sum(float(x["seed_averaged"]["G5-M1"]["S_R"]) for x in aggregated) / PAIR_COUNT
    supported = (
        mean_i_m1 > 0.0
        and mean_q_rr_m1 > 0.0
        and mean_s_r_m1 > 0.0
        and mean_d > 0.0
        and float(test["p_one_sided_greater"]) < 0.05
    )
    return {
        "mean_Ibar_M1": mean_i_m1,
        "mean_Q_RR_M1": mean_q_rr_m1,
        "mean_S_R_M1": mean_s_r_m1,
        "mean_D_OWN": mean_d,
        "confirmatory_test": {
            "test": "one_sided_one_sample_student_t",
            "alternative": "greater_than_zero",
            "endpoint": "D_OWN",
            "sample_size": 300,
            **test,
        },
        "confirmatory_p_value_count": 1,
        "label": LABEL_SUPPORTED if supported else LABEL_NOT_ESTABLISHED,
    }


def checksums_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{sha256_bytes(raw)}  {name}\n"
        for name, raw in sorted(files.items())
    ).encode("utf-8")


def build_output_bundle(
    *,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    worker_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    require(len(worker_manifests) == 2, "WORKER_MANIFEST_COUNT")
    primary = {
        ITEM_FILE: jsonl_bytes(items),
        SUMMARY_FILE: canonical_json_bytes(dict(summary)),
        WORKER0_FILE: canonical_json_bytes(dict(worker_manifests[0])),
        WORKER1_FILE: canonical_json_bytes(dict(worker_manifests[1])),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": summary["result"],
        "execution_head": summary["execution_head"],
        "confirmatory_p_value_count": 1,
        "raw_native_vectors_persisted": False,
        "raw_post_state_vectors_persisted": False,
        "files": {
            name: {"bytes": len(raw), "sha256": sha256_bytes(raw)}
            for name, raw in primary.items()
        },
    }
    files = {**primary, MANIFEST_FILE: canonical_json_bytes(manifest)}
    # Final checksums are generated after every mutable provenance/manifest byte
    # has been finalized. No file in `files` is rewritten after this point.
    files[OUTPUT_CHECKSUM_FILE] = checksums_bytes(files)
    return files


def write_outputs(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    worker_manifests: Sequence[Mapping[str, Any]],
) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    files = build_output_bundle(
        items=items,
        summary=summary,
        worker_manifests=worker_manifests,
    )
    expected = {
        ITEM_FILE,
        SUMMARY_FILE,
        WORKER0_FILE,
        WORKER1_FILE,
        MANIFEST_FILE,
        OUTPUT_CHECKSUM_FILE,
    }
    require(set(files) == expected, "ARTIFACT_BOUNDARY")
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".staging-", dir=str(output_dir.parent)) as tmp:
        staging = Path(tmp)
        for name, raw in files.items():
            (staging / name).write_bytes(raw)
        # Verify the final checksum file against bytes on disk before rename.
        listed = _parse_checksum_file(staging / OUTPUT_CHECKSUM_FILE)
        for rel, expected_sha in listed.items():
            require(sha256_file(staging / rel) == expected_sha, f"OUTPUT_CHECKSUM_VERIFY:{rel}")
        staging.rename(output_dir)


def run_static_verification(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=True)
    static = validate_static_inputs()
    artifacts = validate_training_artifacts(args.training_artifact_root)
    require(len(artifacts) == 9, "STATIC_CORRECTION_COUNT")
    print("RESULT=PASS_GEN5_PHASE2_FRESH_OWNERSHIP_ASSAY_IMPLEMENTATION_STATIC_VERIFICATION")
    print(f"XG1_PAIR_COUNT={static['pair_count']}")
    print(f"XG1_ROW_COUNT={static['row_count']}")
    print("TRAINING_CORRECTION_COUNT=9")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("TASK_EVALUATION_EXECUTED=False")
    print("SCIENTIFIC_P_VALUE_COUNT=0")
    print("SCIENTIFIC_EXECUTION=False")


def _worker_manifest_path(tmpdir: Path, shard_id: int) -> Path:
    return tmpdir / f"worker{shard_id}.manifest.json"


def _worker_items_path(tmpdir: Path, shard_id: int) -> Path:
    return tmpdir / f"worker{shard_id}.items.jsonl"


def run_worker(args: argparse.Namespace) -> None:
    require(args.shard_id in (0, 1), "WORKER_SHARD_ID")
    authenticate_repo(args.expected_head)
    validate_execution_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
        execution_authority_commit=args.execution_authority_commit,
    )
    validate_static_inputs()
    artifact_index = validate_training_artifacts(args.training_artifact_root)

    cuda_eq.backend.runtime_gate()
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() == 1, f"WORKER_VISIBLE_DEVICE_COUNT:{torch.cuda.device_count()}")
    require(torch.cuda.get_device_name(0) == "Tesla T4", "WORKER_DEVICE")
    require(not args.shard_output.exists(), "SHARD_OUTPUT_COLLISION")
    require(not args.shard_meta.exists(), "SHARD_META_COLLISION")

    r22, c22, basis_geometry = load_frozen_owner_bases(ROOT)
    q_bases = restoration.load_q_bases()
    pp3_planes = restoration.load_pp3_planes()

    with cuda_eq.backend.parent_runtime_rebind():
        rows, encoded, events, row_index, tokenizer_provenance = load_inputs(args.tokenizer_snapshot)
        del rows
        kernels = cuda_eq.kernel_compat.load_exact_fast_kernels()
        with cuda_eq.kernel_compat.exact_transformers_kernel_loader(kernels) as constructor_calls:
            model, checkpoint_sha = cuda_eq.parent.load_representative_model_external(
                model_snapshot=args.model_snapshot,
                checkpoint_path=args.checkpoint,
            )
            require(checkpoint_sha == PARENT_CHECKPOINT_SHA256, "CHECKPOINT_SHA256")
            runtime_ctx = cuda_eq.transport_runtime.validate_runtime_components(model)
        counts = Counter(constructor_calls)
        require(
            set(counts) == {"causal-conv1d", "mamba-ssm"}
            and counts["causal-conv1d"] == counts["mamba-ssm"]
            and counts["mamba-ssm"] > 0,
            f"CONSTRUCTOR_KERNEL_COUNTS:{dict(counts)}",
        )
        cuda_eq.kernel_compat.validate_transformers_kernel_bindings(kernels)
        model.to(torch.device("cuda:0"))
        model.eval()
        before_signature = cuda_eq.model_parameter_signature(model)

        corrections = load_corrections(
            artifact_index,
            r22=r22,
            c22=c22,
            device=torch.device("cuda:0"),
        )
        correction_before = correction_signatures(corrections)

        budget = cuda_eq.parent.ForwardBudget(SHARD_MODEL_FORWARD_BUDGET)
        items: list[dict[str, Any]] = []
        global_pairs = expected_pairs()
        pairs = shard_pairs(args.shard_id)
        start_index = 0 if args.shard_id == 0 else SHARD_PAIR_COUNT
        for local_index, pair in enumerate(pairs):
            global_index = start_index + local_index
            require(global_pairs[global_index] == pair, "GLOBAL_PAIR_INDEX")
            item = run_pair_shared_capture(
                _seed_meta(global_index, pair, events),
                q_bases=q_bases,
                artifact_index=artifact_index,
                model=model,
                runtime_ctx=runtime_ctx,
                kernels=kernels,
                encoded=encoded,
                row_index=row_index,
                events=events,
                pp3_planes=pp3_planes,
                r22=r22,
                c22=c22,
                corrections=corrections,
                budget=budget,
            )
            item["shard_id"] = int(args.shard_id)
            items.append(item)
            if (local_index + 1) % 10 == 0:
                gc.collect()
                print(
                    f"GEN5_PHASE2_OWNERSHIP_GPU{args.shard_id}_PROGRESS="
                    f"{local_index + 1}/{SHARD_PAIR_COUNT}",
                    flush=True,
                )
        budget.assert_exact()
        torch.cuda.synchronize()
        after_signature = cuda_eq.model_parameter_signature(model)
        require(before_signature == after_signature, "MODEL_PARAMETER_MUTATION")
        correction_after = correction_signatures(corrections)
        require(correction_before == correction_after, "CORRECTION_PARAMETER_MUTATION")

    require([str(x["source_pair_id"]) for x in items] == list(shard_pairs(args.shard_id)), "WORKER_PAIR_ORDER")
    require(
        sum(int(x["scientific_model_forward_count_this_run"]) for x in items)
        == SHARD_MODEL_FORWARD_BUDGET,
        "WORKER_FORWARD_BUDGET",
    )
    args.shard_output.parent.mkdir(parents=True, exist_ok=True)
    args.shard_output.write_bytes(jsonl_bytes(items))
    worker_manifest = {
        "schema_version": "gen5-phase2-fresh-ownership-worker-v1",
        "result": "PASS_GEN5_PHASE2_FRESH_OWNERSHIP_WORKER",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "shard_id": int(args.shard_id),
        "pair_first": items[0]["source_pair_id"],
        "pair_last": items[-1]["source_pair_id"],
        "pair_count": len(items),
        "scientific_model_forward_count": SHARD_MODEL_FORWARD_BUDGET,
        "cpu_scientific_model_forward_count": 0,
        "confirmatory_p_value_count": 0,
        "checkpoint_load_count": 1,
        "representative_checkpoint_sha256": checkpoint_sha,
        "training_correction_count": 9,
        "training_correction_sha256": {
            cell_key(seed, arm): artifact_index[(seed, arm)].checkpoint_sha256
            for seed, arm in TRAINING_MATRIX
        },
        "model_parameter_signature_before": before_signature,
        "model_parameter_signature_after": after_signature,
        "correction_signatures_before": correction_before,
        "correction_signatures_after": correction_after,
        "basis_geometry": basis_geometry,
        "tokenizer": tokenizer_provenance,
        "shared_frozen_parent_capture_across_corrections": True,
        "training_executed": False,
        "backward_executed": False,
        "task_evaluation_executed": False,
        "scientific_conclusion": None,
    }
    args.shard_meta.write_bytes(canonical_json_bytes(worker_manifest))
    print(f"RESULT=PASS_GEN5_PHASE2_FRESH_OWNERSHIP_WORKER_{args.shard_id}", flush=True)
    print("CONFIRMATORY_P_VALUE_COUNT=0", flush=True)


def run_coordinator(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head)
    validate_execution_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
        execution_authority_commit=args.execution_authority_commit,
    )
    validate_static_inputs()
    validate_training_artifacts(args.training_artifact_root)
    require(not args.output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() == 2, f"EXACT_TWO_GPU_REQUIRED:{torch.cuda.device_count()}")
    require(
        [torch.cuda.get_device_name(i) for i in range(2)] == ["Tesla T4", "Tesla T4"],
        "EXACT_TWO_T4_REQUIRED",
    )

    with tempfile.TemporaryDirectory(prefix="gen5-phase2-ownership-shards-") as tmp:
        tmpdir = Path(tmp)
        processes = []
        for shard_id in (0, 1):
            shard_output = _worker_items_path(tmpdir, shard_id)
            shard_meta = _worker_manifest_path(tmpdir, shard_id)
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "scripts.reason_router_gen5_phase2_fresh_ownership_assay",
                "--worker",
                "--shard-id",
                str(shard_id),
                "--expected-head",
                args.expected_head,
                "--implementation-freeze-commit",
                args.implementation_freeze_commit,
                "--execution-authority-commit",
                args.execution_authority_commit,
                "--model-snapshot",
                str(args.model_snapshot),
                "--tokenizer-snapshot",
                str(args.tokenizer_snapshot),
                "--checkpoint",
                str(args.checkpoint),
                "--training-artifact-root",
                str(args.training_artifact_root),
                "--shard-output",
                str(shard_output),
                "--shard-meta",
                str(shard_meta),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(shard_id)
            processes.append((
                shard_id,
                shard_output,
                shard_meta,
                subprocess.Popen(cmd, cwd=ROOT, env=env),
            ))

        failures = []
        for shard_id, _, _, process in processes:
            rc = process.wait()
            if rc != 0:
                failures.append((shard_id, rc))
        require(not failures, f"WORKER_FAILURES:{failures}")

        shard_rows: list[list[dict[str, Any]]] = []
        worker_manifests: list[dict[str, Any]] = []
        for shard_id, shard_output, shard_meta, _ in processes:
            require(shard_output.is_file(), f"SHARD_OUTPUT_MISSING:{shard_id}")
            require(shard_meta.is_file(), f"SHARD_META_MISSING:{shard_id}")
            rows = read_jsonl(shard_output)
            meta = json.loads(shard_meta.read_text(encoding="utf-8"))
            require(meta.get("result") == "PASS_GEN5_PHASE2_FRESH_OWNERSHIP_WORKER", "WORKER_RESULT")
            require(int(meta.get("shard_id", -1)) == shard_id, "WORKER_SHARD_ID")
            require(int(meta.get("confirmatory_p_value_count", -1)) == 0, "WORKER_PVALUE_FORBIDDEN")
            require(meta.get("scientific_conclusion") is None, "WORKER_CONCLUSION_FORBIDDEN")
            require(int(meta.get("scientific_model_forward_count", -1)) == SHARD_MODEL_FORWARD_BUDGET, "WORKER_FORWARD_COUNT")
            shard_rows.append(rows)
            worker_manifests.append(meta)

        merged = merge_shards(shard_rows[0], shard_rows[1])
        aggregated = aggregate_items(merged)
        decision = confirmatory_decision(aggregated)

        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "result": RESULT_PASS,
            "execution_head": args.expected_head,
            "implementation_freeze_commit": args.implementation_freeze_commit,
            "execution_authority_commit": args.execution_authority_commit,
            "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
            "phase2_design_commit": PHASE2_DESIGN_COMMIT,
            "training_execution_commit": TRAINING_EXECUTION_COMMIT,
            "source_pair_count": PAIR_COUNT,
            "pair_id_first": "xg1_fact_8701",
            "pair_id_last": "xg1_fact_9000",
            "training_seeds": list(TRAINING_SEEDS),
            "training_arms": list(TRAINING_ARMS),
            "primary_comparison": "G5-M1_MINUS_G5-C1",
            "seed_aggregation": "MEAN_WITHIN_ITEM_BEFORE_CONFIRMATORY_TEST",
            "confirmatory_sample_size": 300,
            "shard_topology": {
                "gpu0": ["xg1_fact_8701", "xg1_fact_8850"],
                "gpu1": ["xg1_fact_8851", "xg1_fact_9000"],
                "pairs_per_shard": SHARD_PAIR_COUNT,
                "physical_gpu_count": 2,
                "worker_visible_device_count": 1,
                "device_name": "Tesla T4",
            },
            "forwards_per_pair": FORWARDS_PER_PAIR,
            "scientific_model_forward_count_this_run": FULL_MODEL_FORWARD_BUDGET,
            "cuda_scientific_model_forward_count_this_run": FULL_MODEL_FORWARD_BUDGET,
            "cpu_scientific_model_forward_count_this_run": 0,
            "shared_frozen_parent_capture_across_corrections": True,
            "representative_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "r22_basis_sha256": R22_SHA256,
            "c22_basis_sha256": C22_SHA256,
            "q22_definition": "E_XG2_22-E_XG4_22",
            "conditions": list(CONDITIONS),
            "decision": decision,
            "confirmatory_p_value_count": 1,
            "training_executed": False,
            "backward_executed": False,
            "task_evaluation_executed": False,
            "row_dropping_executed": False,
            "raw_native_vectors_persisted": False,
            "raw_post_state_vectors_persisted": False,
            "scientific_conclusion": decision["label"],
            "next_stage": "GEN5_PHASE2_SCIENTIFIC_INTERPRETATION_AFTER_COLLECT_IMPORT",
        }
        write_outputs(
            args.output_dir,
            items=aggregated,
            summary=summary,
            worker_manifests=worker_manifests,
        )

    print("RESULT=" + RESULT_PASS)
    print("PAIR_ID_FIRST=xg1_fact_8701")
    print("PAIR_ID_LAST=xg1_fact_9000")
    print("CUDA_SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN=" + str(FULL_MODEL_FORWARD_BUDGET))
    print("CPU_SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN=0")
    print("CONFIRMATORY_SAMPLE_SIZE=300")
    print("CONFIRMATORY_P_VALUE_COUNT=1")
    print("MEAN_IBAR_M1=" + repr(decision["mean_Ibar_M1"]))
    print("MEAN_Q_RR_M1=" + repr(decision["mean_Q_RR_M1"]))
    print("MEAN_S_R_M1=" + repr(decision["mean_S_R_M1"]))
    print("MEAN_D_OWN=" + repr(decision["mean_D_OWN"]))
    print("P_ONE_SIDED_GREATER=" + repr(decision["confirmatory_test"]["p_one_sided_greater"]))
    print("SCIENTIFIC_CONCLUSION=" + decision["label"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-verify-only", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--run-assay", action="store_true")
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--execution-authority-commit")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shard-output", type=Path)
    parser.add_argument("--shard-meta", type=Path)
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    if args.static_verify_only:
        require(args.shard_id is None, "STATIC_SHARD_ID_FORBIDDEN")
        require(args.implementation_freeze_commit is None, "STATIC_IMPLEMENTATION_FREEZE_FORBIDDEN")
        require(args.execution_authority_commit is None, "STATIC_EXECUTION_AUTHORITY_FORBIDDEN")
        require(args.model_snapshot is None, "STATIC_MODEL_SNAPSHOT_FORBIDDEN")
        require(args.tokenizer_snapshot is None, "STATIC_TOKENIZER_SNAPSHOT_FORBIDDEN")
        require(args.checkpoint is None, "STATIC_CHECKPOINT_FORBIDDEN")
        require(args.output_dir is None, "STATIC_OUTPUT_FORBIDDEN")
        require(args.shard_output is None and args.shard_meta is None, "STATIC_SHARD_OUTPUT_FORBIDDEN")
        return

    require(args.implementation_freeze_commit is not None, "IMPLEMENTATION_FREEZE_REQUIRED")
    require(args.execution_authority_commit is not None, "EXECUTION_AUTHORITY_REQUIRED")
    require(args.model_snapshot is not None, "MODEL_SNAPSHOT_REQUIRED")
    require(args.tokenizer_snapshot is not None, "TOKENIZER_SNAPSHOT_REQUIRED")
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")
    if args.worker:
        require(args.shard_id in (0, 1), "WORKER_SHARD_ID_REQUIRED")
        require(args.shard_output is not None and args.shard_meta is not None, "WORKER_SHARD_PATHS_REQUIRED")
        require(args.output_dir is None, "WORKER_OUTPUT_DIR_FORBIDDEN")
    else:
        require(args.run_assay, "RUN_ASSAY_MODE_REQUIRED")
        require(args.shard_id is None, "COORDINATOR_SHARD_ID_FORBIDDEN")
        require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")
        require(args.shard_output is None and args.shard_meta is None, "COORDINATOR_SHARD_PATH_FORBIDDEN")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_mode_args(args)
    if args.static_verify_only:
        run_static_verification(args)
    elif args.worker:
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
