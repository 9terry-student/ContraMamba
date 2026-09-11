"""K3 scientific execution harness for frozen retention-vs-write causal decomposition.

This file is intentionally separate from the frozen structural-replay module.
It may execute only when a one-file K3 execution-authority commit is the exact
current HEAD and is the immediate child of the commit that first added this
harness + its focused tests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_BRANCH = "longterm-k-series-native-state-kinematics"

K3_PREREG_COMMIT = "20032bb53d77416bb7eb25411eb4f77b008648b4"
K3_PREREG_REL = "reports/longterm_k3_selective_ssm_retention_write_causal_decomposition_prereg_candidate.md"
K3_PREREG_SHA256 = "7561f4188921b645eba3d7108bcd007b7c223389b5e1abc512cc8d61af976c84"

K3_REPLAY_IMPLEMENTATION_COMMIT = "a2f33796903466680091ccbede9f6728022e678f"
K3_REPLAY_REL = "scripts/longterm_k3_selective_ssm_retention_write_causal_decomposition.py"
K3_REPLAY_SHA256 = "116e9daae8a4d62b19d8ab4e7a4d4171df8ad7ae99b08953fc1f2ba49563a60e"
K3_REPLAY_GIT_BLOB = "24278583af0132f4ec5cf8610d2cb85972c1bd51"
K3_REPLAY_TEST_REL = "tests/test_longterm_k3_selective_ssm_retention_write_causal_decomposition.py"
K3_REPLAY_TEST_SHA256 = "392ad7c6a60b2cff996b44bb93943ad303974e5301222e5b1221744b26633565"

SCRIPT_REL = "scripts/longterm_k3_selective_ssm_retention_write_causal_execution.py"
TEST_REL = "tests/test_longterm_k3_selective_ssm_retention_write_causal_execution.py"
AUTHORITY_REL = "reports/longterm_k3_selective_ssm_retention_write_causal_execution_authority_spec_candidate.md"
AUTHORITY_SCHEMA = "k3-scientific-execution-authority-v1"

K2R_RESULT_ARCHIVE_COMMIT = "ba0ad9052a3b8a5eb80fef46dea461f372f95ee8"
K2R_ARCHIVE_DIR = "reports/longterm_k2r_claim_disjoint_replication_52bd363_v1"
K2R_CANDIDATE_REL = f"{K2R_ARCHIVE_DIR}/candidate_pool.jsonl"
K2R_CANDIDATE_SHA256 = "00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4"
K2R_BLOCK_REL = f"{K2R_ARCHIVE_DIR}/block_metrics.jsonl"
K2R_BLOCK_SHA256 = "e4982e8a57e15863227d17f080e7a7a699fb14100387fc36c7c47ab68962c8de"
K2R_MANIFEST_REL = f"{K2R_ARCHIVE_DIR}/manifest.json"
K2R_MANIFEST_SHA256 = "fecf202165256de086d49d60e0b8613765bb869c4988744b92aabed3f1931919"

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
MAMBA_SOURCE_SHA256 = "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"

PRIMARY_LAYER = 23
W = 8
N_ITEMS = 300
N_BLOCKS = 150
METRICS = ("R", "D", "DISP", "P")
PRIMARY_TEST_ORDER = (
    "R_ATT", "R_SEL",
    "D_ATT", "D_SEL",
    "DISP_ATT", "DISP_SEL",
    "P_ATT", "P_SEL",
)

HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}

ITEM_SCHEMA = "k3-causal-item-v1"
BLOCK_SCHEMA = "k3-causal-block-v1"
INTEGRITY_SCHEMA = "k3-causal-integrity-v1"
MANIFEST_SCHEMA = "k3-causal-manifest-v1"
PRIMARY_SCHEMA = "k3-causal-primary-stats-v1"


class ContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(dict(row)) + b"\n" for row in rows)


def parse_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    require(not raw.startswith(b"\xef\xbb\xbf"), "JSONL_BOM_FORBIDDEN")
    require(b"\r" not in raw, "JSONL_CR_FORBIDDEN")
    require(raw.endswith(b"\n"), "JSONL_FINAL_LF_REQUIRED")
    lines = raw[:-1].split(b"\n")
    require(bool(lines) and all(lines), "JSONL_BLANK_LINE_FORBIDDEN")
    out: list[dict[str, Any]] = []
    for line in lines:
        value = json.loads(line.decode("utf-8", "strict"))
        require(isinstance(value, dict), "JSONL_ROW_NOT_OBJECT")
        out.append(value)
    return out


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"GIT_FAILURE:{' '.join(args)}") from exc


def _changed_files(root: Path, commit: str) -> list[str]:
    raw = _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
    return [] if not raw else raw.splitlines()


def _addition_commit(root: Path, rel: str) -> str:
    raw = _git(root, "log", "--diff-filter=A", "--format=%H", "--", rel)
    commits = [line for line in raw.splitlines() if line]
    require(len(commits) == 1, f"ADDITION_COMMIT_AMBIGUOUS:{rel}")
    return commits[0]


def parse_authority_markers(text: str) -> dict[str, str]:
    keys = {
        "K3_EXECUTION_AUTHORITY_SCHEMA",
        "K3_SCIENTIFIC_INTERVENTION_EXECUTION_AUTHORIZED",
        "K3_ONE_SCIENTIFIC_EXECUTION",
        "K3_EXECUTION_IMPLEMENTATION_COMMIT",
        "K3_EXECUTION_RUNNER_SHA256",
        "K3_EXECUTION_TEST_SHA256",
        "K3_REPLAY_IMPLEMENTATION_COMMIT",
        "K3_REPLAY_MODULE_SHA256",
        "K3_PREREG_COMMIT",
    }
    found: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        key = left.strip().strip("`")
        if key not in keys:
            continue
        value = right.strip().strip("`").strip()
        require(key not in found, f"AUTHORITY_MARKER_DUPLICATE:{key}")
        found[key] = value
    require(set(found) == keys, "AUTHORITY_MARKER_SET_MISMATCH")
    return found


def authority_provenance(root: Path) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == REPO_BRANCH, "GIT_BRANCH_MISMATCH")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        require(line[:2] == "??" and path in HISTORICAL_K1_UNTRACKED, "GIT_DIRTY_CONTRACT_MISMATCH")

    harness_commit = _addition_commit(root, SCRIPT_REL)
    test_commit = _addition_commit(root, TEST_REL)
    require(harness_commit == test_commit, "EXECUTION_HARNESS_TEST_COMMIT_MISMATCH")
    require(_git(root, "rev-parse", f"{harness_commit}^") == K3_REPLAY_IMPLEMENTATION_COMMIT, "EXECUTION_HARNESS_PARENT_MISMATCH")
    require(set(_changed_files(root, harness_commit)) == {SCRIPT_REL, TEST_REL}, "EXECUTION_HARNESS_COMMIT_SCOPE_MISMATCH")

    script_path = root / SCRIPT_REL
    test_path = root / TEST_REL
    require(script_path.is_file() and test_path.is_file(), "EXECUTION_HARNESS_FILES_MISSING")
    script_sha = file_sha256(script_path)
    test_sha = file_sha256(test_path)
    require(_git(root, "rev-parse", f"{head}:{SCRIPT_REL}") == _git(root, "rev-parse", f"{harness_commit}:{SCRIPT_REL}"), "EXECUTION_RUNNER_BLOB_DRIFT")
    require(_git(root, "rev-parse", f"{head}:{TEST_REL}") == _git(root, "rev-parse", f"{harness_commit}:{TEST_REL}"), "EXECUTION_TEST_BLOB_DRIFT")

    authority_commit = _addition_commit(root, AUTHORITY_REL)
    require(_git(root, "rev-parse", f"{authority_commit}^") == harness_commit, "EXECUTION_AUTHORITY_PARENT_MISMATCH")
    require(_changed_files(root, authority_commit) == [AUTHORITY_REL], "EXECUTION_AUTHORITY_COMMIT_SCOPE_MISMATCH")
    require(head == authority_commit, "RUNTIME_HEAD_MUST_EQUAL_EXECUTION_AUTHORITY_COMMIT")
    require(
        _git(root, "rev-parse", f"{head}:{AUTHORITY_REL}")
        == _git(root, "rev-parse", f"{authority_commit}:{AUTHORITY_REL}"),
        "EXECUTION_AUTHORITY_BLOB_DRIFT",
    )

    authority_path = root / AUTHORITY_REL
    require(authority_path.is_file(), "EXECUTION_AUTHORITY_FILE_MISSING")
    authority_raw = authority_path.read_bytes()
    require(not authority_raw.startswith(b"\xef\xbb\xbf") and b"\r" not in authority_raw, "EXECUTION_AUTHORITY_BYTE_CONTRACT")
    markers = parse_authority_markers(authority_raw.decode("utf-8", "strict"))
    expected = {
        "K3_EXECUTION_AUTHORITY_SCHEMA": AUTHORITY_SCHEMA,
        "K3_SCIENTIFIC_INTERVENTION_EXECUTION_AUTHORIZED": "YES",
        "K3_ONE_SCIENTIFIC_EXECUTION": "YES",
        "K3_EXECUTION_IMPLEMENTATION_COMMIT": harness_commit,
        "K3_EXECUTION_RUNNER_SHA256": script_sha,
        "K3_EXECUTION_TEST_SHA256": test_sha,
        "K3_REPLAY_IMPLEMENTATION_COMMIT": K3_REPLAY_IMPLEMENTATION_COMMIT,
        "K3_REPLAY_MODULE_SHA256": K3_REPLAY_SHA256,
        "K3_PREREG_COMMIT": K3_PREREG_COMMIT,
    }
    require(markers == expected, "EXECUTION_AUTHORITY_MARKERS_MISMATCH")

    exact_files = {
        K3_PREREG_REL: K3_PREREG_SHA256,
        K3_REPLAY_REL: K3_REPLAY_SHA256,
        K3_REPLAY_TEST_REL: K3_REPLAY_TEST_SHA256,
        K2R_CANDIDATE_REL: K2R_CANDIDATE_SHA256,
        K2R_BLOCK_REL: K2R_BLOCK_SHA256,
        K2R_MANIFEST_REL: K2R_MANIFEST_SHA256,
    }
    for rel, expected_sha in exact_files.items():
        path = root / rel
        require(path.is_file(), f"REQUIRED_FILE_MISSING:{rel}")
        require(file_sha256(path) == expected_sha, f"REQUIRED_FILE_SHA_MISMATCH:{rel}")
    require(_git(root, "rev-parse", f"{head}:{K3_REPLAY_REL}") == K3_REPLAY_GIT_BLOB, "K3_REPLAY_BLOB_DRIFT")

    return {
        "runtime_branch": branch,
        "runtime_git_head": head,
        "runtime_dirty_contract": status,
        "execution_implementation_commit": harness_commit,
        "execution_runner_sha256": script_sha,
        "execution_test_sha256": test_sha,
        "execution_authority_commit": authority_commit,
        "execution_authority_sha256": sha256_bytes(authority_raw),
        "execution_authority_schema": AUTHORITY_SCHEMA,
        "k3_replay_implementation_commit": K3_REPLAY_IMPLEMENTATION_COMMIT,
        "k3_replay_sha256": K3_REPLAY_SHA256,
        "k3_replay_git_blob": K3_REPLAY_GIT_BLOB,
        "k3_prereg_commit": K3_PREREG_COMMIT,
        "k3_prereg_sha256": K3_PREREG_SHA256,
    }


def load_modules(root: Path) -> tuple[Any, Any, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    k3 = importlib.import_module("scripts.longterm_k3_selective_ssm_retention_write_causal_decomposition")
    require(Path(k3.__file__).resolve() == (root / K3_REPLAY_REL).resolve(), "K3_REPLAY_IMPORT_PATH_MISMATCH")
    require(file_sha256(Path(k3.__file__)) == K3_REPLAY_SHA256, "K3_REPLAY_IMPORT_BYTE_MISMATCH")
    k2r, k2s = k3.load_frozen_dependencies(root)
    require(k3.PRIMARY_LAYER == PRIMARY_LAYER and k3.W == W, "K3_GEOMETRY_DRIFT")
    require(tuple(k3.METRICS) == METRICS, "K3_METRIC_DRIFT")
    require(tuple(k3.PRIMARY_TEST_ORDER) == PRIMARY_TEST_ORDER, "K3_PRIMARY_TEST_DRIFT")
    return k3, k2r, k2s


def _runtime_versions() -> dict[str, str]:
    import torch
    import transformers
    import tokenizers
    import huggingface_hub
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
    }


def output_target(root: Path, supplied: Path, authority_commit: str) -> Path:
    target = supplied.resolve()
    repo = root.resolve()
    require(target != repo and repo not in target.parents, "OUTPUT_DIRECTORY_MUST_BE_OUTSIDE_REPO")
    require(not target.exists(), "OUTPUT_DIRECTORY_ALREADY_EXISTS")
    require(target.name == f"k3-retention-write-{authority_commit[:12]}", "OUTPUT_DIRECTORY_NAME_MISMATCH")
    return target


def _tensor_sequence_sha(mapping: Mapping[int, Any], indices: Sequence[int]) -> str:
    chunks = []
    for index in indices:
        require(index in mapping, "TENSOR_SEQUENCE_COORDINATE_MISSING")
        chunks.append(mapping[index].detach().cpu().contiguous().numpy().tobytes())
    return sha256_bytes(b"".join(chunks))


def _exact_natural_and_sham(k3: Any, capture: Mapping[str, Any], d: int, end: int) -> tuple[dict[int, Any], bool]:
    import torch
    natural = k3.structural_replay(capture["post_state"][d - 1], capture["G"], capture["W"], d, end)
    for t in range(d, end + 1):
        require(torch.equal(natural[t], capture["post_state"][t]), "SCIENTIFIC_NATURAL_REPLAY_MISMATCH")
    sham_g = {t: capture["G"][t].clone() for t in range(d, end + 1)}
    sham_w = {t: capture["W"][t].clone() for t in range(d, end + 1)}
    sham = k3.structural_replay(capture["post_state"][d - 1], sham_g, sham_w, d, end)
    for t in range(d, end + 1):
        require(torch.equal(sham[t], capture["post_state"][t]), "SCIENTIFIC_SHAM_REPLAY_MISMATCH")
    return natural, True


def _pair_condition_metrics(
    k3: Any,
    k2s: Any,
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    d: int,
    p: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import torch
    end = p + W
    require(torch.equal(corr["post_state"][d - 1], ctrl["post_state"][d - 1]), "SCIENTIFIC_PAIR_D_MINUS_1_STATE_MISMATCH")

    natural_corr, _ = _exact_natural_and_sham(k3, corr, d, end)
    natural_ctrl, _ = _exact_natural_and_sham(k3, ctrl, d, end)

    outputs: dict[str, dict[str, Any]] = {}
    base_corr_states = k3.combine_natural_and_replay(corr["post_state"], natural_corr, d, p)
    base_ctrl_states = k3.combine_natural_and_replay(ctrl["post_state"], natural_ctrl, d, p)
    outputs["BASE"] = {
        "corr": k3.branch_metrics(k2s, base_corr_states, p),
        "ctrl": k3.branch_metrics(k2s, base_ctrl_states, p),
    }

    for condition in ("W_EQ", "G_EQ"):
        a, b = k3.replay_pair(corr, ctrl, d, end, condition)
        for t in range(d, end + 1):
            require(bool(torch.isfinite(a[t]).all().item()) and bool(torch.isfinite(b[t]).all().item()), f"{condition}_SCIENTIFIC_NONFINITE")
        corr_states = k3.combine_natural_and_replay(corr["post_state"], a, d, p)
        ctrl_states = k3.combine_natural_and_replay(ctrl["post_state"], b, d, p)
        outputs[condition] = {
            "corr": k3.branch_metrics(k2s, corr_states, p),
            "ctrl": k3.branch_metrics(k2s, ctrl_states, p),
        }

    gw_a, gw_b = k3.replay_pair(corr, ctrl, d, end, "GW_EQ")
    for t in range(d, end + 1):
        require(torch.equal(gw_a[t], gw_b[t]), "SCIENTIFIC_GW_EQ_PAIR_COLLAPSE_FAILURE")

    integrity = {
        "d": d,
        "end": end,
        "natural_replay": "PASS_EXACT",
        "sham_replay": "PASS_EXACT",
        "gw_eq_pair_state_collapse": "PASS_EXACT",
    }
    return outputs, integrity


def _branch_capture_digest(capture: Mapping[str, Any], d: int, p: int) -> dict[str, str]:
    component_indices = list(range(d, p + W + 1))
    state_indices = list(range(p - 1, p + W + 1))
    return {
        "G_sequence_sha256": _tensor_sequence_sha(capture["G"], component_indices),
        "W_sequence_sha256": _tensor_sequence_sha(capture["W"], component_indices),
        "natural_state_window_sha256": _tensor_sequence_sha(capture["post_state"], state_indices),
    }


def _item_scientific(
    k3: Any,
    k2s: Any,
    model: Any,
    binding: Any,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    import torch
    p = int(contract["p"])
    end = p + W
    d_by_assignment = {
        "matched": p + int(contract["matched_d_minus_p"]),
        "swapped": p + int(contract["swapped_d_minus_p"]),
    }
    ids = contract["branch_token_ids"]
    captures: dict[str, Any] = {}

    for assignment in ("matched", "swapped"):
        d = d_by_assignment[assignment]
        component_indices = range(d, end + 1)
        state_indices = range(p - 1, end + 1)
        for suffix in ("corr", "ctrl"):
            name = f"{assignment}_{suffix}"
            capture = k3._branch_capture(model, ids[name], binding, component_indices, state_indices)
            require(set(capture["G"]) == set(component_indices), "SCIENTIFIC_G_CAPTURE_INCOMPLETE")
            require(set(capture["W"]) == set(component_indices), "SCIENTIFIC_W_CAPTURE_INCOMPLETE")
            require(set(capture["post_state"]) == set(state_indices), "SCIENTIFIC_STATE_CAPTURE_INCOMPLETE")
            for t in component_indices:
                require(torch.equal(capture["pre_state"][t], capture["post_state"][t - 1]), "SCIENTIFIC_PRE_POST_COORDINATE_MISMATCH")
            captures[name] = capture

    # All four branches are identical through the literal prefix, including p-1 and p.
    for t in (p - 1, p):
        hashes = {k3.tensor_sha256(captures[name]["post_state"][t]) for name in captures}
        require(len(hashes) == 1, "SCIENTIFIC_COMMON_PREFIX_STATE_MISMATCH")

    condition_branches: dict[str, dict[str, Any]] = {condition: {} for condition in ("BASE", "W_EQ", "G_EQ")}
    pair_integrity: dict[str, Any] = {}
    branch_digests: dict[str, Any] = {}

    for assignment in ("matched", "swapped"):
        d = d_by_assignment[assignment]
        pair_metrics, integ = _pair_condition_metrics(
            k3,
            k2s,
            captures[f"{assignment}_corr"],
            captures[f"{assignment}_ctrl"],
            d,
            p,
        )
        pair_integrity[assignment] = integ
        for condition in ("BASE", "W_EQ", "G_EQ"):
            condition_branches[condition][f"{assignment}_corr"] = pair_metrics[condition]["corr"]
            condition_branches[condition][f"{assignment}_ctrl"] = pair_metrics[condition]["ctrl"]
        branch_digests[f"{assignment}_corr"] = _branch_capture_digest(captures[f"{assignment}_corr"], d, p)
        branch_digests[f"{assignment}_ctrl"] = _branch_capture_digest(captures[f"{assignment}_ctrl"], d, p)

    x = {condition: k3.x_pair_specificity(condition_branches[condition]) for condition in condition_branches}
    row = {
        "schema_version": ITEM_SCHEMA,
        "item_index": int(contract["item_index"]),
        "block_index": int(contract["block_index"]),
        "block_id": str(contract["block_id"]),
        "stable_item_id": str(contract["stable_item_id"]),
        "pair_id": str(contract["pair_id"]),
        "donor_index": int(contract["donor_index"]),
        "donor_stable_item_id": str(contract["donor_stable_item_id"]),
        "p": p,
        "matched_d": d_by_assignment["matched"],
        "swapped_d": d_by_assignment["swapped"],
        "matched_d_minus_p": int(contract["matched_d_minus_p"]),
        "swapped_d_minus_p": int(contract["swapped_d_minus_p"]),
        "prefix_token_ids": list(contract["prefix_token_ids"]),
        "branch_token_ids": {name: list(values) for name, values in sorted(ids.items())},
        "capture_digests": branch_digests,
        "pair_integrity": pair_integrity,
        "conditions": {
            condition: {
                "branches": condition_branches[condition],
                "X_pair_specificity": x[condition],
            }
            for condition in ("BASE", "W_EQ", "G_EQ")
        },
    }
    del captures
    return row, {
        "natural_branch_exact": 4,
        "sham_branch_exact": 4,
        "gw_pair_exact": 2,
    }


def build_block_rows(k3: Any, item_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(item_rows) == N_ITEMS, "ITEM_ROW_COUNT_MISMATCH")
    rows: list[dict[str, Any]] = []
    for block_index in range(N_BLOCKS):
        a = item_rows[2 * block_index]
        b = item_rows[2 * block_index + 1]
        require(a["block_index"] == block_index and b["block_index"] == block_index, "BLOCK_ITEM_ALIGNMENT_FAILURE")
        row: dict[str, Any] = {
            "schema_version": BLOCK_SCHEMA,
            "block_index": block_index,
            "block_id": f"k2r-block-{block_index:03d}",
            "item_a_stable_id": a["stable_item_id"],
            "item_b_stable_id": b["stable_item_id"],
        }
        for condition in ("BASE", "W_EQ", "G_EQ"):
            for metric in METRICS:
                xa = a["conditions"][condition]["X_pair_specificity"][metric]
                xb = b["conditions"][condition]["X_pair_specificity"][metric]
                if xa is None or xb is None:
                    value = None
                else:
                    value = (float(xa) + float(xb)) / 2.0
                    require(math.isfinite(value), "BLOCK_VALUE_NONFINITE")
                row[f"B_{condition}_{metric}"] = value
                row[f"Z_{condition}_{metric}"] = k3.aligned_signal(metric, value)

        for metric in METRICS:
            att = k3.attenuation_and_selectivity(
                metric,
                row[f"Z_BASE_{metric}"],
                row[f"Z_W_EQ_{metric}"],
                row[f"Z_G_EQ_{metric}"],
            )
            for key, value in att.items():
                row[f"{key}_{metric}"] = value
        rows.append(row)
    return rows


def verify_archived_baseline(block_rows: Sequence[Mapping[str, Any]], archived_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(block_rows) == N_BLOCKS and len(archived_rows) == N_BLOCKS, "BASELINE_BLOCK_COUNT_MISMATCH")
    comparisons = 0
    for new, old in zip(block_rows, archived_rows):
        require(int(new["block_index"]) == int(old["block_index"]), "BASELINE_BLOCK_INDEX_MISMATCH")
        require(new["block_id"] == old["block_id"], "BASELINE_BLOCK_ID_MISMATCH")
        require(new["item_a_stable_id"] == old["item_a_stable_id"], "BASELINE_ITEM_A_MISMATCH")
        require(new["item_b_stable_id"] == old["item_b_stable_id"], "BASELINE_ITEM_B_MISMATCH")
        for metric in METRICS:
            observed = new[f"B_BASE_{metric}"]
            expected = old[f"B_{metric}"]
            require(observed == expected, f"BASELINE_BLOCK_VALUE_MISMATCH:{metric}:{new['block_index']}")
            comparisons += 1
    return {
        "status": "PASS_EXACT",
        "archived_block_sha256": K2R_BLOCK_SHA256,
        "block_count": N_BLOCKS,
        "metric_value_comparisons": comparisons,
    }


def primary_from_blocks(k3: Any, block_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float | None]] = {name: [] for name in PRIMARY_TEST_ORDER}
    for row in block_rows:
        for metric in METRICS:
            values[f"{metric}_ATT"].append(row[f"ATT_DOM_{metric}"])
            values[f"{metric}_SEL"].append(row[f"SEL_{metric}"])
    primary = k3.k3_primary_statistics(values)
    primary["schema_version"] = PRIMARY_SCHEMA
    return primary


def _report(primary: Mapping[str, Any], integrity: Mapping[str, Any], authority: Mapping[str, Any]) -> str:
    lines = [
        "# K3 Retention-vs-Write Causal Decomposition Scientific Report",
        "",
        f"Scientific verdict: `{primary['scientific_verdict']}`",
        "",
        f"Full support: `{primary['full_support']}`",
        "",
        f"Execution authority commit: `{authority['execution_authority_commit']}`",
        "",
        "## Integrity",
        "",
        f"- natural structural replay: `{integrity['natural_structural_replay']}`",
        f"- sham replay: `{integrity['sham_replay']}`",
        f"- GW equalization pair collapse: `{integrity['gw_eq_pair_state_collapse']}`",
        f"- archived K2R baseline reproduction: `{integrity['archived_k2r_baseline_reproduction']['status']}`",
        "",
        "## Confirmatory tests",
        "",
    ]
    for name in PRIMARY_TEST_ORDER:
        test = primary["tests"][name]
        lines.append(
            f"- {name}: +{test['positive_count']} / -{test['negative_count']} / "
            f"0={test['zero_count']} / U={test['undefined_count']}; "
            f"effect={test['rank_biserial_sign_effect']}; "
            f"Holm p={test['holm_adjusted_p']}; match={test['direction_match']}"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "This execution tests only layer-23 native recurrence component specialization under the frozen G/W replay interventions.",
        "It does not establish task-decision causality, authorization causality, external-distribution generalization, or K4 claims.",
        "",
    ])
    return "\n".join(lines)


def _write_artifacts(
    target: Path,
    item_rows: Sequence[Mapping[str, Any]],
    block_rows: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
    integrity: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
) -> dict[str, str]:
    require(not target.exists(), "OUTPUT_DIRECTORY_ALREADY_EXISTS_AT_WRITE")
    target.mkdir(parents=False)

    payloads: dict[str, bytes] = {
        "item_metrics.jsonl": canonical_jsonl(item_rows),
        "block_metrics.jsonl": canonical_jsonl(block_rows),
        "primary_stats.json": canonical_json(primary) + b"\n",
        "integrity.json": canonical_json(integrity) + b"\n",
    }
    for name, raw in payloads.items():
        (target / name).write_bytes(raw)

    report = _report(primary, integrity, manifest_base["authority_provenance"]).encode("utf-8")
    if not report.endswith(b"\n"):
        report += b"\n"
    (target / "report.md").write_bytes(report)

    non_manifest = {
        name: file_sha256(target / name)
        for name in sorted([*payloads.keys(), "report.md"])
    }
    manifest = {
        **dict(manifest_base),
        "schema_version": MANIFEST_SCHEMA,
        "primary_statistics": primary,
        "scientific_verdict": primary["scientific_verdict"],
        "artifact_hash_binding_note": "all non-self scientific artifacts are bound here; manifest.json is bound by SHA256SUMS.txt",
        "artifact_sha256": non_manifest,
    }
    (target / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")

    sums_targets = [*sorted(non_manifest), "manifest.json"]
    sums = b"".join(
        f"{file_sha256(target / name)}  {name}\n".encode("utf-8")
        for name in sums_targets
    )
    (target / "SHA256SUMS.txt").write_bytes(sums)
    return {name: file_sha256(path) for name, path in sorted((p.name, p) for p in target.iterdir() if p.is_file())}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    authority = authority_provenance(root)
    target = output_target(root, Path(args.output_dir), authority["execution_authority_commit"])

    k3, k2r, k2s = load_modules(root)
    # Re-run the frozen implementation provenance gate in scientific-clean mode.
    replay_runtime = k3.git_provenance(root, replay_preflight=False)
    population_audit = k3.audit_population_without_outcomes(root)

    snapshot, hf = k2s.resolve_hf_snapshot(args.hf_revision)
    require(hf["hf_model_id"] == HF_MODEL and hf["resolved_hf_revision"] == HF_REVISION, "HF_IDENTITY_MISMATCH")
    require(hf["transformers_version"] == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")

    handoff = k2s.audit_handoff(Path(args.seed180_handoff))
    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    encoder = k2s.encoder_fingerprint(checkpoint["model_state_dict"])
    model = k2s.build_a0_model(root, snapshot, checkpoint)
    model.eval()
    require(next(model.parameters()).device.type == "cpu", "K3_CPU_REQUIRED")

    binding = k3.resolve_component_binding(k2s)
    require(binding.source_sha256 == MAMBA_SOURCE_SHA256, "RECURRENCE_SOURCE_SHA_MISMATCH")
    replay_preflight = k3.run_replay_preflight(root, k2s, model, hf["tokenizer"], binding)
    require(replay_preflight["status"] == "PASS_K3_REPLAY_PREFLIGHT", "K3_REPLAY_PREFLIGHT_FAILED")
    require(replay_preflight["scientific_population_intervention_executed"] is False, "PREFLIGHT_BOUNDARY_BROKEN")

    candidate_raw = (root / K2R_CANDIDATE_REL).read_bytes()
    require(sha256_bytes(candidate_raw) == K2R_CANDIDATE_SHA256, "K2R_CANDIDATE_SHA_MISMATCH")
    candidates = parse_jsonl_bytes(candidate_raw)
    require(len(candidates) == N_ITEMS, "K2R_CANDIDATE_COUNT_MISMATCH")
    contracts, feasibility = k2r.build_input_contracts(candidates, hf["tokenizer"])
    require(len(contracts) == N_ITEMS, "K3_INPUT_CONTRACT_COUNT_MISMATCH")

    # Scientific outcomes are read only now, after authority + replay gates pass.
    archived_raw = (root / K2R_BLOCK_REL).read_bytes()
    require(sha256_bytes(archived_raw) == K2R_BLOCK_SHA256, "K2R_BLOCK_SHA_MISMATCH")
    archived_blocks = parse_jsonl_bytes(archived_raw)

    item_rows: list[dict[str, Any]] = []
    exact_counts = {"natural_branch_exact": 0, "sham_branch_exact": 0, "gw_pair_exact": 0}
    for index, contract in enumerate(contracts):
        row, counts = _item_scientific(k3, k2s, model, binding, contract)
        item_rows.append(row)
        for key in exact_counts:
            exact_counts[key] += counts[key]
        if index == 0 or (index + 1) % 10 == 0:
            print(f"K3_PROGRESS items={index + 1}/{N_ITEMS}", flush=True)

    block_rows = build_block_rows(k3, item_rows)
    baseline = verify_archived_baseline(block_rows, archived_blocks)
    primary = primary_from_blocks(k3, block_rows)

    require(exact_counts == {
        "natural_branch_exact": 1200,
        "sham_branch_exact": 1200,
        "gw_pair_exact": 600,
    }, "SCIENTIFIC_REPLAY_INTEGRITY_COUNT_MISMATCH")

    integrity = {
        "schema_version": INTEGRITY_SCHEMA,
        "natural_structural_replay": "PASS_EXACT",
        "sham_replay": "PASS_EXACT",
        "gw_eq_pair_state_collapse": "PASS_EXACT",
        "natural_branch_exact_count": exact_counts["natural_branch_exact"],
        "sham_branch_exact_count": exact_counts["sham_branch_exact"],
        "gw_pair_exact_count": exact_counts["gw_pair_exact"],
        "archived_k2r_baseline_reproduction": baseline,
        "synthetic_replay_preflight": replay_preflight,
    }

    manifest_base = {
        "authority_provenance": authority,
        "replay_runtime": replay_runtime,
        "population": {
            **population_audit,
            "input_feasibility": feasibility,
            "candidate_pool_path": K2R_CANDIDATE_REL,
            "candidate_pool_sha256": K2R_CANDIDATE_SHA256,
            "archived_k2r_block_path": K2R_BLOCK_REL,
            "archived_k2r_block_sha256": K2R_BLOCK_SHA256,
        },
        "handoff": {**handoff, "encoder": encoder, "strict_load": "PASS"},
        "hf": {key: value for key, value in hf.items() if key not in {"config", "tokenizer"}},
        "runtime_versions": _runtime_versions(),
        "recurrence": {
            "source_sha256": binding.source_sha256,
            "source_bytes": binding.source_bytes,
            "qualname": binding.qualname,
            "recurrence_update_line": binding.recurrence_update_line,
            "post_update_line": binding.post_update_line,
            "discrete_A_line": binding.discrete_A_line,
            "deltaB_u_line": binding.deltaB_u_line,
            "capture_layer": PRIMARY_LAYER,
            "window_W": W,
            "device": "cpu",
        },
        "intervention": {
            "conditions": ["BASE", "W_EQ", "G_EQ"],
            "integrity_control": "GW_EQ",
            "onset": "first_corr_ctrl_token_divergence_d",
            "end": "p_plus_8",
            "midpoint": "arithmetic_0.5_times_sum",
            "dominant_component": dict(k3.DOMINANT_COMPONENT),
            "expected_direction": dict(k3.EXPECTED_DIRECTION),
            "primary_test_order": list(PRIMARY_TEST_ORDER),
            "holm_m": 8,
            "holm_alpha": 0.05,
        },
        "integrity": integrity,
        "exact_command": [
            SCRIPT_REL,
            "--seed180-handoff", str(Path(args.seed180_handoff).resolve()),
            "--hf-revision", args.hf_revision,
            "--output-dir", str(target),
        ],
    }

    artifact_hashes = _write_artifacts(target, item_rows, block_rows, primary, integrity, manifest_base)
    return {
        "scientific_execution": "PASS",
        "output_dir": str(target),
        "scientific_verdict": primary["scientific_verdict"],
        "full_support": primary["full_support"],
        "primary_statistics": primary,
        "artifact_sha256": artifact_hashes,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Authorized K3 retention-vs-write scientific execution harness")
    p.add_argument("--seed180-handoff", required=True)
    p.add_argument("--hf-revision", required=True)
    p.add_argument("--output-dir", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require(args.hf_revision == HF_REVISION, "HF_REVISION_MISMATCH")
    result = execute(args)
    print(canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
