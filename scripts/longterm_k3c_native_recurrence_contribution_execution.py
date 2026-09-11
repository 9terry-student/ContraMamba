"""K3C scientific execution harness for native recurrence contribution decomposition.

This file is intentionally separate from the frozen K3C replay module. It may
execute scientific recurrent-state work only when a one-file K3C execution
authority commit is the exact current HEAD and is the immediate child of the
commit that first added this harness and its focused tests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_BRANCH = "longterm-k-series-native-state-kinematics"

K3C_PREREG_COMMIT = "b85272d88d0bb57db45fdc963d313714529e7975"
K3C_PREREG_REL = (
    "reports/longterm_k3c_native_recurrence_contribution_decomposition_"
    "prereg_candidate.md"
)
K3C_PREREG_SHA256 = (
    "0a9de28237e107ce3a62d4fa9e0bb7f230d2e0289e019d3864edf491dd271436"
)

K3C_REPLAY_IMPLEMENTATION_COMMIT = "e91116b4b5837a16de1d6eabfa2503be5bfe1d3d"
K3C_REPLAY_REL = (
    "scripts/longterm_k3c_native_recurrence_contribution_decomposition.py"
)
K3C_REPLAY_SHA256 = (
    "ab07c6052a07af354043e0aa38bf24fd73c2fde53fccec125c25f7db2718f233"
)
K3C_REPLAY_GIT_BLOB = "633ca87b365f50ac27dbec5a8595dbdad42a7dfe"
K3C_REPLAY_TEST_REL = (
    "tests/test_longterm_k3c_native_recurrence_contribution_decomposition.py"
)
K3C_REPLAY_TEST_SHA256 = (
    "a1e1eafe80806d62103e2a1445b4264451785099a6ac27385ea95afdb3a4058d"
)
K3C_REPLAY_TEST_GIT_BLOB = "6734cbccf34107cba160ac1da05137deb55bcc52"

SCRIPT_REL = (
    "scripts/longterm_k3c_native_recurrence_contribution_execution.py"
)
TEST_REL = (
    "tests/test_longterm_k3c_native_recurrence_contribution_execution.py"
)
AUTHORITY_REL = (
    "reports/longterm_k3c_native_recurrence_contribution_"
    "execution_authority_spec_candidate.md"
)
AUTHORITY_SCHEMA = "k3c-scientific-execution-authority-v1"

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

PRIMARY_LAYER = 23
W = 8
N_ITEMS = 300
N_BLOCKS = 150
METRICS = ("R", "D", "DISP", "P")
CONDITIONS = ("BASE", "W_EQ", "H_EQ", "W_SEED_H_CARRY")
PRIMARY_TEST_ORDER = (
    "R_DOM", "R_CARRY",
    "D_DOM", "D_CARRY",
    "DISP_DOM", "DISP_CARRY",
    "P_DOM", "P_CARRY",
)

GENERATED_SOURCE_CANONICAL_SHA256 = (
    "33bff5a0b657d1ceb38ae9c651e1cadfc8308286398cc1b8c4245c47f1c42000"
)
CANDIDATE_POOL_CANONICAL_SHA256 = (
    "9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e"
)
RECIPROCAL_MAPPING_CANONICAL_SHA256 = (
    "4fbc0f6642db3b2c3cca148fdc03cdc738dd6e8718cffb3fcdee73cd5b7f9acc"
)

HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}

ITEM_SCHEMA = "k3c-contribution-item-v1"
BLOCK_SCHEMA = "k3c-contribution-block-v1"
INTEGRITY_SCHEMA = "k3c-contribution-integrity-v1"
MANIFEST_SCHEMA = "k3c-contribution-manifest-v1"
BASE_SCHEMA = "k3c-base-replication-stats-v1"
PRIMARY_SCHEMA = "k3c-contribution-primary-stats-v1"


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


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"GIT_FAILURE:{' '.join(args)}") from exc


def _changed_files(root: Path, commit: str) -> list[str]:
    raw = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return [] if not raw else raw.splitlines()


def _addition_commit(root: Path, rel: str) -> str:
    raw = _git(root, "log", "--diff-filter=A", "--format=%H", "--", rel)
    commits = [line for line in raw.splitlines() if line]
    require(len(commits) == 1, f"ADDITION_COMMIT_AMBIGUOUS:{rel}")
    return commits[0]


def parse_authority_markers(text: str) -> dict[str, str]:
    keys = {
        "K3C_EXECUTION_AUTHORITY_SCHEMA",
        "K3C_SCIENTIFIC_RECURRENT_STATE_EXECUTION_AUTHORIZED",
        "K3C_ONE_SCIENTIFIC_EXECUTION",
        "K3C_EXECUTION_IMPLEMENTATION_COMMIT",
        "K3C_EXECUTION_RUNNER_SHA256",
        "K3C_EXECUTION_TEST_SHA256",
        "K3C_REPLAY_IMPLEMENTATION_COMMIT",
        "K3C_REPLAY_MODULE_SHA256",
        "K3C_PREREG_COMMIT",
        "K3C_PREREG_SHA256",
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
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    ).splitlines()
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        require(
            line[:2] == "??" and path in HISTORICAL_K1_UNTRACKED,
            "GIT_DIRTY_CONTRACT_MISMATCH",
        )

    harness_commit = _addition_commit(root, SCRIPT_REL)
    test_commit = _addition_commit(root, TEST_REL)
    require(
        harness_commit == test_commit,
        "EXECUTION_HARNESS_TEST_COMMIT_MISMATCH",
    )
    require(
        _git(root, "rev-parse", f"{harness_commit}^")
        == K3C_REPLAY_IMPLEMENTATION_COMMIT,
        "EXECUTION_HARNESS_PARENT_MISMATCH",
    )
    require(
        set(_changed_files(root, harness_commit)) == {SCRIPT_REL, TEST_REL},
        "EXECUTION_HARNESS_COMMIT_SCOPE_MISMATCH",
    )

    script_path = root / SCRIPT_REL
    test_path = root / TEST_REL
    require(
        script_path.is_file() and test_path.is_file(),
        "EXECUTION_HARNESS_FILES_MISSING",
    )
    script_sha = file_sha256(script_path)
    test_sha = file_sha256(test_path)
    require(
        _git(root, "rev-parse", f"{head}:{SCRIPT_REL}")
        == _git(root, "rev-parse", f"{harness_commit}:{SCRIPT_REL}"),
        "EXECUTION_RUNNER_BLOB_DRIFT",
    )
    require(
        _git(root, "rev-parse", f"{head}:{TEST_REL}")
        == _git(root, "rev-parse", f"{harness_commit}:{TEST_REL}"),
        "EXECUTION_TEST_BLOB_DRIFT",
    )

    authority_commit = _addition_commit(root, AUTHORITY_REL)
    require(
        _git(root, "rev-parse", f"{authority_commit}^") == harness_commit,
        "EXECUTION_AUTHORITY_PARENT_MISMATCH",
    )
    require(
        _changed_files(root, authority_commit) == [AUTHORITY_REL],
        "EXECUTION_AUTHORITY_COMMIT_SCOPE_MISMATCH",
    )
    require(
        head == authority_commit,
        "RUNTIME_HEAD_MUST_EQUAL_EXECUTION_AUTHORITY_COMMIT",
    )
    require(
        _git(root, "rev-parse", f"{head}:{AUTHORITY_REL}")
        == _git(root, "rev-parse", f"{authority_commit}:{AUTHORITY_REL}"),
        "EXECUTION_AUTHORITY_BLOB_DRIFT",
    )

    authority_path = root / AUTHORITY_REL
    require(authority_path.is_file(), "EXECUTION_AUTHORITY_FILE_MISSING")
    authority_raw = authority_path.read_bytes()
    require(
        not authority_raw.startswith(b"\xef\xbb\xbf")
        and b"\r" not in authority_raw,
        "EXECUTION_AUTHORITY_BYTE_CONTRACT",
    )
    markers = parse_authority_markers(
        authority_raw.decode("utf-8", "strict")
    )
    expected = {
        "K3C_EXECUTION_AUTHORITY_SCHEMA": AUTHORITY_SCHEMA,
        "K3C_SCIENTIFIC_RECURRENT_STATE_EXECUTION_AUTHORIZED": "YES",
        "K3C_ONE_SCIENTIFIC_EXECUTION": "YES",
        "K3C_EXECUTION_IMPLEMENTATION_COMMIT": harness_commit,
        "K3C_EXECUTION_RUNNER_SHA256": script_sha,
        "K3C_EXECUTION_TEST_SHA256": test_sha,
        "K3C_REPLAY_IMPLEMENTATION_COMMIT":
            K3C_REPLAY_IMPLEMENTATION_COMMIT,
        "K3C_REPLAY_MODULE_SHA256": K3C_REPLAY_SHA256,
        "K3C_PREREG_COMMIT": K3C_PREREG_COMMIT,
        "K3C_PREREG_SHA256": K3C_PREREG_SHA256,
    }
    require(
        markers == expected,
        "EXECUTION_AUTHORITY_MARKERS_MISMATCH",
    )

    exact_files = {
        K3C_PREREG_REL: K3C_PREREG_SHA256,
        K3C_REPLAY_REL: K3C_REPLAY_SHA256,
        K3C_REPLAY_TEST_REL: K3C_REPLAY_TEST_SHA256,
    }
    for rel, expected_sha in exact_files.items():
        path = root / rel
        require(path.is_file(), f"REQUIRED_FILE_MISSING:{rel}")
        require(
            file_sha256(path) == expected_sha,
            f"REQUIRED_FILE_SHA_MISMATCH:{rel}",
        )

    require(
        _git(root, "rev-parse", f"{head}:{K3C_REPLAY_REL}")
        == K3C_REPLAY_GIT_BLOB,
        "K3C_REPLAY_BLOB_DRIFT",
    )
    require(
        _git(root, "rev-parse", f"{head}:{K3C_REPLAY_TEST_REL}")
        == K3C_REPLAY_TEST_GIT_BLOB,
        "K3C_REPLAY_TEST_BLOB_DRIFT",
    )

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
        "k3c_replay_implementation_commit":
            K3C_REPLAY_IMPLEMENTATION_COMMIT,
        "k3c_replay_sha256": K3C_REPLAY_SHA256,
        "k3c_replay_git_blob": K3C_REPLAY_GIT_BLOB,
        "k3c_prereg_commit": K3C_PREREG_COMMIT,
        "k3c_prereg_sha256": K3C_PREREG_SHA256,
    }


def load_modules(root: Path) -> tuple[Any, Any, Any, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    k3c = importlib.import_module(
        "scripts.longterm_k3c_native_recurrence_contribution_decomposition"
    )
    require(
        Path(k3c.__file__).resolve() == (root / K3C_REPLAY_REL).resolve(),
        "K3C_REPLAY_IMPORT_PATH_MISMATCH",
    )
    require(
        file_sha256(Path(k3c.__file__)) == K3C_REPLAY_SHA256,
        "K3C_REPLAY_IMPORT_BYTE_MISMATCH",
    )
    generator, k2s, k3 = k3c.load_frozen_dependencies(root)
    require(
        k3c.PRIMARY_LAYER == PRIMARY_LAYER and k3c.W == W,
        "K3C_GEOMETRY_DRIFT",
    )
    require(tuple(k3c.METRICS) == METRICS, "K3C_METRIC_DRIFT")
    require(
        tuple(k3c.PRIMARY_TEST_ORDER) == PRIMARY_TEST_ORDER,
        "K3C_PRIMARY_TEST_DRIFT",
    )
    return k3c, generator, k2s, k3


def _runtime_versions() -> dict[str, str]:
    import huggingface_hub
    import tokenizers
    import torch
    import transformers

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
    }


def output_target(
    root: Path,
    supplied: Path,
    authority_commit: str,
) -> Path:
    target = supplied.resolve()
    repo = root.resolve()
    require(
        target != repo and repo not in target.parents,
        "OUTPUT_DIRECTORY_MUST_BE_OUTSIDE_REPO",
    )
    require(not target.exists(), "OUTPUT_DIRECTORY_ALREADY_EXISTS")
    require(
        target.name == f"k3c-contribution-{authority_commit[:12]}",
        "OUTPUT_DIRECTORY_NAME_MISMATCH",
    )
    return target


def reciprocal_mapping(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    require(len(candidates) == N_ITEMS, "RECIPROCAL_CANDIDATE_COUNT")
    rows: list[dict[str, Any]] = []
    for block_index in range(N_BLOCKS):
        a = 2 * block_index
        b = a + 1
        rows.append(
            {
                "block_index": block_index,
                "item_a_stable_id": candidates[a]["stable_item_id"],
                "item_b_stable_id": candidates[b]["stable_item_id"],
                "item_a_donor_index": a ^ 1,
                "item_b_donor_index": b ^ 1,
            }
        )
    require(
        sha256_bytes(canonical_json(rows))
        == RECIPROCAL_MAPPING_CANONICAL_SHA256,
        "RECIPROCAL_MAPPING_CANONICAL_SHA_MISMATCH",
    )
    return rows


def scientific_input_contracts(
    k3c: Any,
    candidates: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, feasibility = k3c.build_input_contracts(candidates, tokenizer)
    contracts: list[dict[str, Any]] = []

    for i, row in enumerate(candidates):
        donor = candidates[i ^ 1]
        prefix = str(row["prefix_text"])
        prefix_ids = k3c._token_ids(tokenizer, prefix)
        p = len(prefix_ids) - 1
        texts = {
            "matched_corr": prefix + str(row["correction_text"]),
            "matched_ctrl": prefix + str(row["control_text"]),
            "swapped_corr": prefix + str(donor["correction_text"]),
            "swapped_ctrl": prefix + str(donor["control_text"]),
        }
        arrays = {
            name: k3c._token_ids(tokenizer, text)
            for name, text in texts.items()
        }
        for ids in arrays.values():
            require(
                ids[:len(prefix_ids)] == prefix_ids,
                "SCIENTIFIC_EXACT_PREFIX_FAILURE",
            )
            require(
                len(ids) - len(prefix_ids) >= W,
                "SCIENTIFIC_WINDOW8_FAILURE",
            )

        md = k3c.first_divergence_after_prefix(
            arrays["matched_corr"],
            arrays["matched_ctrl"],
            len(prefix_ids),
        )
        sd = k3c.first_divergence_after_prefix(
            arrays["swapped_corr"],
            arrays["swapped_ctrl"],
            len(prefix_ids),
        )
        require(md - p in {2, 3}, "SCIENTIFIC_MATCHED_D_MINUS_P")
        require(sd - p in {2, 3}, "SCIENTIFIC_SWAPPED_D_MINUS_P")

        contracts.append(
            {
                "item_index": i,
                "block_index": i // 2,
                "block_id": f"k3c-block-{i // 2:03d}",
                "stable_item_id": str(row["stable_item_id"]),
                "pair_id": str(row["pair_id"]),
                "donor_index": i ^ 1,
                "donor_stable_item_id": str(donor["stable_item_id"]),
                "p": p,
                "matched_d_minus_p": md - p,
                "swapped_d_minus_p": sd - p,
                "prefix_token_ids": list(prefix_ids),
                "branch_token_ids": {
                    name: list(ids)
                    for name, ids in sorted(arrays.items())
                },
            }
        )
    require(len(contracts) == N_ITEMS, "SCIENTIFIC_CONTRACT_COUNT")
    return contracts, feasibility


def _tensor_sequence_sha(
    mapping: Mapping[int, Any],
    indices: Sequence[int],
) -> str:
    chunks = []
    for index in indices:
        require(index in mapping, "TENSOR_SEQUENCE_COORDINATE_MISSING")
        chunks.append(
            mapping[index]
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
    return sha256_bytes(b"".join(chunks))


def _exact_natural_and_sham(
    k3c: Any,
    capture: Mapping[str, Any],
    d: int,
    end: int,
) -> dict[int, Any]:
    import torch

    natural = k3c.structural_replay(
        capture["post_state"][d - 1],
        capture["G"],
        capture["W"],
        d,
        end,
    )
    for t in range(d, end + 1):
        require(
            torch.equal(natural[t], capture["post_state"][t]),
            "SCIENTIFIC_NATURAL_REPLAY_MISMATCH",
        )
        h = k3c.retained_contribution(
            capture["G"][t],
            capture["post_state"][t - 1],
        )
        require(
            torch.equal(h + capture["W"][t], capture["post_state"][t]),
            "SCIENTIFIC_NATURAL_H_PLUS_W_MISMATCH",
        )

    sham_g = {
        t: capture["G"][t].clone()
        for t in range(d, end + 1)
    }
    sham_w = {
        t: capture["W"][t].clone()
        for t in range(d, end + 1)
    }
    sham = k3c.structural_replay(
        capture["post_state"][d - 1],
        sham_g,
        sham_w,
        d,
        end,
    )
    for t in range(d, end + 1):
        require(
            torch.equal(sham[t], capture["post_state"][t]),
            "SCIENTIFIC_SHAM_REPLAY_MISMATCH",
        )
    return natural


def _pair_condition_metrics(
    k3c: Any,
    k2s: Any,
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    d: int,
    p: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import torch

    end = p + W
    require(
        torch.equal(
            corr["post_state"][d - 1],
            ctrl["post_state"][d - 1],
        ),
        "SCIENTIFIC_PAIR_D_MINUS_1_STATE_MISMATCH",
    )

    natural_corr = _exact_natural_and_sham(k3c, corr, d, end)
    natural_ctrl = _exact_natural_and_sham(k3c, ctrl, d, end)

    outputs: dict[str, dict[str, Any]] = {}
    base_corr_states = k3c.combine_natural_and_replay(
        corr["post_state"],
        natural_corr,
        d,
        p,
    )
    base_ctrl_states = k3c.combine_natural_and_replay(
        ctrl["post_state"],
        natural_ctrl,
        d,
        p,
    )
    outputs["BASE"] = {
        "corr": k2s.compute_branch_metrics(base_corr_states, p),
        "ctrl": k2s.compute_branch_metrics(base_ctrl_states, p),
    }

    intervention_states: dict[
        str,
        tuple[dict[int, Any], dict[int, Any]],
    ] = {}
    for condition in ("W_EQ", "H_EQ", "W_SEED_H_CARRY"):
        a, b = k3c.replay_pair_contribution(
            corr,
            ctrl,
            d,
            end,
            condition,
        )
        intervention_states[condition] = (a, b)
        for t in range(d, end + 1):
            require(
                bool(torch.isfinite(a[t]).all().item())
                and bool(torch.isfinite(b[t]).all().item()),
                f"{condition}_SCIENTIFIC_NONFINITE",
            )
        corr_states = k3c.combine_natural_and_replay(
            corr["post_state"], a, d, p
        )
        ctrl_states = k3c.combine_natural_and_replay(
            ctrl["post_state"], b, d, p
        )
        outputs[condition] = {
            "corr": k2s.compute_branch_metrics(corr_states, p),
            "ctrl": k2s.compute_branch_metrics(ctrl_states, p),
        }

    seed_corr, seed_ctrl = intervention_states["W_SEED_H_CARRY"]
    c_h_d = k3c.retained_contribution(
        corr["G"][d],
        corr["post_state"][d - 1],
    )
    n_h_d = k3c.retained_contribution(
        ctrl["G"][d],
        ctrl["post_state"][d - 1],
    )
    hbar_d = k3c.arithmetic_midpoint(c_h_d, n_h_d)
    require(
        torch.equal(seed_corr[d], hbar_d + corr["W"][d])
        and torch.equal(seed_ctrl[d], hbar_d + ctrl["W"][d]),
        "SCIENTIFIC_W_SEED_D_SEMANTICS_FAILURE",
    )

    wh_a, wh_b = k3c.replay_pair_contribution(
        corr,
        ctrl,
        d,
        end,
        "WH_EQ",
    )
    for t in range(d, end + 1):
        require(
            torch.equal(wh_a[t], wh_b[t]),
            "SCIENTIFIC_WH_EQ_PAIR_COLLAPSE_FAILURE",
        )

    integrity = {
        "d": d,
        "end": end,
        "natural_h_plus_w_identity": "PASS_EXACT",
        "natural_replay": "PASS_EXACT",
        "sham_replay": "PASS_EXACT",
        "w_seed_h_carry_d_semantics": "PASS_EXACT",
        "wh_eq_pair_state_collapse": "PASS_EXACT",
    }
    return outputs, integrity


def _branch_capture_digest(
    capture: Mapping[str, Any],
    d: int,
    p: int,
) -> dict[str, str]:
    component_indices = list(range(d, p + W + 1))
    state_indices = list(range(p - 1, p + W + 1))
    return {
        "G_sequence_sha256":
            _tensor_sequence_sha(capture["G"], component_indices),
        "W_sequence_sha256":
            _tensor_sequence_sha(capture["W"], component_indices),
        "natural_state_window_sha256":
            _tensor_sequence_sha(capture["post_state"], state_indices),
    }


def _item_scientific(
    k3c: Any,
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
            capture = k3._branch_capture(
                model,
                ids[name],
                binding,
                component_indices,
                state_indices,
            )
            require(
                set(capture["G"]) == set(component_indices),
                "SCIENTIFIC_G_CAPTURE_INCOMPLETE",
            )
            require(
                set(capture["W"]) == set(component_indices),
                "SCIENTIFIC_W_CAPTURE_INCOMPLETE",
            )
            require(
                set(capture["post_state"]) == set(state_indices),
                "SCIENTIFIC_STATE_CAPTURE_INCOMPLETE",
            )
            for t in component_indices:
                require(
                    torch.equal(
                        capture["pre_state"][t],
                        capture["post_state"][t - 1],
                    ),
                    "SCIENTIFIC_PRE_POST_COORDINATE_MISMATCH",
                )
            captures[name] = capture

    for t in (p - 1, p):
        hashes = {
            k3.tensor_sha256(captures[name]["post_state"][t])
            for name in captures
        }
        require(
            len(hashes) == 1,
            "SCIENTIFIC_COMMON_PREFIX_STATE_MISMATCH",
        )

    condition_branches: dict[str, dict[str, Any]] = {
        condition: {}
        for condition in CONDITIONS
    }
    pair_integrity: dict[str, Any] = {}
    branch_digests: dict[str, Any] = {}

    for assignment in ("matched", "swapped"):
        d = d_by_assignment[assignment]
        pair_metrics, integ = _pair_condition_metrics(
            k3c,
            k2s,
            captures[f"{assignment}_corr"],
            captures[f"{assignment}_ctrl"],
            d,
            p,
        )
        pair_integrity[assignment] = integ
        for condition in CONDITIONS:
            condition_branches[condition][f"{assignment}_corr"] = (
                pair_metrics[condition]["corr"]
            )
            condition_branches[condition][f"{assignment}_ctrl"] = (
                pair_metrics[condition]["ctrl"]
            )
        branch_digests[f"{assignment}_corr"] = _branch_capture_digest(
            captures[f"{assignment}_corr"], d, p
        )
        branch_digests[f"{assignment}_ctrl"] = _branch_capture_digest(
            captures[f"{assignment}_ctrl"], d, p
        )

    x = {
        condition: k3c.x_pair_specificity(
            condition_branches[condition]
        )
        for condition in CONDITIONS
    }

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
        "branch_token_ids": {
            name: list(values)
            for name, values in sorted(ids.items())
        },
        "capture_digests": branch_digests,
        "pair_integrity": pair_integrity,
        "conditions": {
            condition: {
                "branches": condition_branches[condition],
                "X_pair_specificity": x[condition],
            }
            for condition in CONDITIONS
        },
    }
    del captures
    return row, {
        "natural_branch_exact": 4,
        "sham_branch_exact": 4,
        "wh_pair_exact": 2,
        "w_seed_pair_exact": 2,
    }


def build_block_rows(
    k3c: Any,
    item_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    require(len(item_rows) == N_ITEMS, "ITEM_ROW_COUNT_MISMATCH")
    rows: list[dict[str, Any]] = []

    for block_index in range(N_BLOCKS):
        a = item_rows[2 * block_index]
        b = item_rows[2 * block_index + 1]
        require(
            a["block_index"] == block_index
            and b["block_index"] == block_index,
            "BLOCK_ITEM_ALIGNMENT_FAILURE",
        )
        row: dict[str, Any] = {
            "schema_version": BLOCK_SCHEMA,
            "block_index": block_index,
            "block_id": f"k3c-block-{block_index:03d}",
            "item_a_stable_id": a["stable_item_id"],
            "item_b_stable_id": b["stable_item_id"],
        }

        for condition in CONDITIONS:
            for metric in METRICS:
                xa = a["conditions"][condition]["X_pair_specificity"][metric]
                xb = b["conditions"][condition]["X_pair_specificity"][metric]
                if xa is None or xb is None:
                    value = None
                else:
                    value = (float(xa) + float(xb)) / 2.0
                    require(
                        math.isfinite(value),
                        "BLOCK_VALUE_NONFINITE",
                    )
                row[f"B_{condition}_{metric}"] = value
                row[f"Z_{condition}_{metric}"] = (
                    k3c.aligned_signal(metric, value)
                )

        for metric in METRICS:
            derived = k3c.mechanism_values(
                row[f"Z_BASE_{metric}"],
                row[f"Z_W_EQ_{metric}"],
                row[f"Z_H_EQ_{metric}"],
                row[f"Z_W_SEED_H_CARRY_{metric}"],
            )
            for key, value in derived.items():
                row[f"{key}_{metric}"] = value
        rows.append(row)

    return rows


def scientific_statistics_from_blocks(
    k3c: Any,
    block_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_values: dict[str, list[float | None]] = {
        metric: []
        for metric in METRICS
    }
    mechanism_values: dict[str, list[float | None]] = {
        name: []
        for name in PRIMARY_TEST_ORDER
    }

    for row in block_rows:
        for metric in METRICS:
            base_values[metric].append(row[f"B_BASE_{metric}"])
            mechanism_values[f"{metric}_DOM"].append(row[f"DOM_{metric}"])
            mechanism_values[f"{metric}_CARRY"].append(
                row[f"CARRY_{metric}"]
            )

    result = k3c.k3c_scientific_statistics(
        base_values,
        mechanism_values,
    )
    result["schema_version"] = PRIMARY_SCHEMA
    result["base_replication"]["schema_version"] = BASE_SCHEMA
    result["full_support"] = bool(
        result["base_replication"]["base_replication_verdict"] == "PASS"
        and result["mechanism"]["full_support"]
    )
    return result


def _report(
    primary: Mapping[str, Any],
    integrity: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> str:
    base = primary["base_replication"]
    mechanism = primary["mechanism"]
    lines = [
        "# K3C Native Recurrence Contribution Decomposition Scientific Report",
        "",
        f"Scientific verdict: `{primary['scientific_verdict']}`",
        "",
        f"BASE replication gate: `{base['base_replication_verdict']}`",
        "",
        f"Full K3C support: `{primary['full_support']}`",
        "",
        f"Execution authority commit: `{authority['execution_authority_commit']}`",
        "",
        "## Integrity",
        "",
        f"- natural H+W identity: `{integrity['natural_h_plus_w_identity']}`",
        f"- natural structural replay: `{integrity['natural_structural_replay']}`",
        f"- sham replay: `{integrity['sham_replay']}`",
        f"- W-seed d-step semantics: `{integrity['w_seed_h_carry_d_semantics']}`",
        f"- WH equalization pair collapse: `{integrity['wh_eq_pair_state_collapse']}`",
        "",
        "## BASE replication gate",
        "",
    ]
    for metric in METRICS:
        test = base["endpoints"][metric]
        lines.append(
            f"- {metric}: +{test['positive_count']} / "
            f"-{test['negative_count']} / "
            f"0={test['zero_count']} / U={test['undefined_count']}; "
            f"effect={test['rank_biserial_sign_effect']}; "
            f"Holm p={test['holm_adjusted_p']}; "
            f"match={test['direction_match']}"
        )

    lines.extend([
        "",
        "## Confirmatory mechanism tests",
        "",
    ])
    for name in PRIMARY_TEST_ORDER:
        test = mechanism["tests"][name]
        lines.append(
            f"- {name}: +{test['positive_count']} / "
            f"-{test['negative_count']} / "
            f"0={test['zero_count']} / U={test['undefined_count']}; "
            f"effect={test['rank_biserial_sign_effect']}; "
            f"Holm p={test['holm_adjusted_p']}; "
            f"match={test['direction_match']}; "
            f"contradiction={test['directional_contradiction']}"
        )

    lines.extend([
        "",
        "## Claim boundary",
        "",
        "This execution tests only the preregistered layer-23 W-vs-H "
        "contribution decomposition and retained-carry hypothesis on the "
        "prospectively frozen K3C controlled population.",
        "",
        "It does not establish task-decision causality, authorization "
        "causality, external-distribution generalization, or K4 claims.",
        "",
    ])
    return "\n".join(lines)


def _write_artifacts(
    target: Path,
    source_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, Any]],
    item_rows: Sequence[Mapping[str, Any]],
    block_rows: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
    integrity: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not target.exists(),
        "OUTPUT_DIRECTORY_ALREADY_EXISTS_AT_WRITE",
    )
    target.mkdir(parents=False)

    generated_source_raw = canonical_jsonl(source_rows)
    candidate_raw = canonical_jsonl(candidates)
    require(
        sha256_bytes(generated_source_raw)
        == GENERATED_SOURCE_CANONICAL_SHA256,
        "OUTPUT_GENERATED_SOURCE_SHA_MISMATCH",
    )
    require(
        sha256_bytes(candidate_raw)
        == CANDIDATE_POOL_CANONICAL_SHA256,
        "OUTPUT_CANDIDATE_POOL_SHA_MISMATCH",
    )

    payloads: dict[str, bytes] = {
        "generated_source.jsonl": generated_source_raw,
        "candidate_pool.jsonl": candidate_raw,
        "reciprocal_mapping.json":
            canonical_json(list(mapping_rows)) + b"\n",
        "item_metrics.jsonl": canonical_jsonl(item_rows),
        "block_metrics.jsonl": canonical_jsonl(block_rows),
        "base_replication_stats.json":
            canonical_json(primary["base_replication"]) + b"\n",
        "primary_stats.json": canonical_json(primary) + b"\n",
        "integrity.json": canonical_json(integrity) + b"\n",
    }
    for name, raw in payloads.items():
        (target / name).write_bytes(raw)

    report = _report(
        primary,
        integrity,
        manifest_base["authority_provenance"],
    ).encode("utf-8")
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
        "base_replication_statistics": primary["base_replication"],
        "primary_statistics": primary,
        "scientific_verdict": primary["scientific_verdict"],
        "artifact_hash_binding_note":
            "all non-self scientific artifacts are bound here; "
            "manifest.json is bound by SHA256SUMS.txt",
        "artifact_sha256": non_manifest,
    }
    (target / "manifest.json").write_bytes(
        canonical_json(manifest) + b"\n"
    )

    sums_targets = [*sorted(non_manifest), "manifest.json"]
    sums = b"".join(
        f"{file_sha256(target / name)}  {name}\n".encode("utf-8")
        for name in sums_targets
    )
    (target / "SHA256SUMS.txt").write_bytes(sums)

    return {
        name: file_sha256(path)
        for name, path in sorted(
            (p.name, p)
            for p in target.iterdir()
            if p.is_file()
        )
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    authority = authority_provenance(root)
    target = output_target(
        root,
        Path(args.output_dir),
        authority["execution_authority_commit"],
    )

    k3c, generator, k2s, k3 = load_modules(root)

    # Re-run the frozen implementation provenance gate in scientific-clean
    # mode. This gate permits only the historical K1 untracked files.
    replay_runtime = k3c.git_provenance(
        root,
        replay_preflight=False,
    )

    # Prospective population is regenerated state-blind from the frozen
    # generator before any scientific recurrent-state read.
    source_rows, candidates, population_audit = (
        k3c.materialize_population(root, generator)
    )
    require(
        population_audit["scientific_population_recurrent_state_read"]
        is False,
        "POPULATION_AUDIT_BOUNDARY_BROKEN",
    )
    mapping_rows = reciprocal_mapping(candidates)

    snapshot, hf = k2s.resolve_hf_snapshot(args.hf_revision)
    require(
        hf["hf_model_id"] == HF_MODEL
        and hf["resolved_hf_revision"] == HF_REVISION,
        "HF_IDENTITY_MISMATCH",
    )
    require(
        hf["transformers_version"] == TRANSFORMERS_VERSION,
        "TRANSFORMERS_VERSION_MISMATCH",
    )
    contracts, feasibility = scientific_input_contracts(
        k3c,
        candidates,
        hf["tokenizer"],
    )

    handoff = k2s.audit_handoff(Path(args.seed180_handoff))
    require(
        handoff["zip_sha256"] == k3c.EXPECTED_ZIP_SHA256,
        "HANDOFF_ZIP_SHA_MISMATCH",
    )
    require(
        handoff["checkpoint_sha256"] == k3c.EXPECTED_CHECKPOINT_SHA256,
        "CHECKPOINT_SHA_MISMATCH",
    )
    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    encoder = k2s.encoder_fingerprint(checkpoint["model_state_dict"])
    require(
        encoder["canonical_digest"]
        == k3c.COMMON_ENCODER_CANONICAL_SHA256,
        "ENCODER_CANONICAL_SHA_MISMATCH",
    )
    require(
        encoder["raw_concat_digest"]
        == k3c.COMMON_ENCODER_RAW_CONCAT_SHA256,
        "ENCODER_RAW_SHA_MISMATCH",
    )
    model = k2s.build_a0_model(root, snapshot, checkpoint)
    model.eval()
    require(
        next(model.parameters()).device.type == "cpu",
        "K3C_CPU_REQUIRED",
    )

    binding = k3.resolve_component_binding(k2s)
    require(
        binding.source_sha256 == MAMBA_SOURCE_SHA256,
        "RECURRENCE_SOURCE_SHA_MISMATCH",
    )

    # Re-run synthetic-only implementation preflight before population state
    # capture. The frozen preflight itself asserts that no scientific
    # population recurrent state was read.
    replay_preflight = k3c.run_replay_preflight(
        root,
        k2s,
        k3,
        model,
        hf["tokenizer"],
        binding,
    )
    require(
        replay_preflight["status"] == "PASS_K3C_REPLAY_PREFLIGHT",
        "K3C_REPLAY_PREFLIGHT_FAILED",
    )
    require(
        replay_preflight["scientific_population_recurrent_state_read"]
        is False
        and replay_preflight["scientific_population_intervention_executed"]
        is False,
        "PREFLIGHT_BOUNDARY_BROKEN",
    )

    # Scientific recurrent-state read starts only here, after authority,
    # provenance, population, runtime, and synthetic replay gates pass.
    item_rows: list[dict[str, Any]] = []
    exact_counts = {
        "natural_branch_exact": 0,
        "sham_branch_exact": 0,
        "wh_pair_exact": 0,
        "w_seed_pair_exact": 0,
    }

    for index, contract in enumerate(contracts):
        row, counts = _item_scientific(
            k3c,
            k3,
            k2s,
            model,
            binding,
            contract,
        )
        item_rows.append(row)
        for key in exact_counts:
            exact_counts[key] += counts[key]
        if index == 0 or (index + 1) % 10 == 0:
            print(
                f"K3C_PROGRESS items={index + 1}/{N_ITEMS}",
                flush=True,
            )

    require(
        exact_counts == {
            "natural_branch_exact": 1200,
            "sham_branch_exact": 1200,
            "wh_pair_exact": 600,
            "w_seed_pair_exact": 600,
        },
        "SCIENTIFIC_REPLAY_INTEGRITY_COUNT_MISMATCH",
    )

    block_rows = build_block_rows(k3c, item_rows)
    primary = scientific_statistics_from_blocks(k3c, block_rows)

    integrity = {
        "schema_version": INTEGRITY_SCHEMA,
        "natural_h_plus_w_identity": "PASS_EXACT",
        "natural_structural_replay": "PASS_EXACT",
        "sham_replay": "PASS_EXACT",
        "w_seed_h_carry_d_semantics": "PASS_EXACT",
        "wh_eq_pair_state_collapse": "PASS_EXACT",
        "natural_branch_exact_count":
            exact_counts["natural_branch_exact"],
        "sham_branch_exact_count":
            exact_counts["sham_branch_exact"],
        "wh_pair_exact_count":
            exact_counts["wh_pair_exact"],
        "w_seed_pair_exact_count":
            exact_counts["w_seed_pair_exact"],
        "synthetic_replay_preflight": replay_preflight,
    }

    manifest_base = {
        "authority_provenance": authority,
        "replay_runtime": replay_runtime,
        "population": {
            **population_audit,
            "input_feasibility": feasibility,
            "generated_source_path": "generated_source.jsonl",
            "generated_source_canonical_sha256":
                GENERATED_SOURCE_CANONICAL_SHA256,
            "candidate_pool_path": "candidate_pool.jsonl",
            "candidate_pool_canonical_sha256":
                CANDIDATE_POOL_CANONICAL_SHA256,
            "reciprocal_mapping_path": "reciprocal_mapping.json",
            "reciprocal_mapping_canonical_sha256":
                RECIPROCAL_MAPPING_CANONICAL_SHA256,
        },
        "handoff": {
            **handoff,
            "encoder": encoder,
            "strict_load": "PASS",
        },
        "hf": {
            key: value
            for key, value in hf.items()
            if key not in {"config", "tokenizer"}
        },
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
            "state_timing": "post_consumption_s_t",
        },
        "intervention": {
            "conditions": list(CONDITIONS),
            "integrity_control": "WH_EQ",
            "onset": "first_corr_ctrl_token_divergence_d",
            "end": "p_plus_8",
            "midpoint": "arithmetic_0.5_times_sum",
            "retained_contribution":
                "H_t=G_t_elementwise_times_S_t_minus_1_replay",
            "h_eq":
                "dynamic_pair_midpoint_of_current_replay_retained_contributions",
            "w_seed_h_carry":
                "H_equalized_at_d_then_W_equalized_from_d_plus_1_through_p_plus_8",
            "expected_direction": dict(k3c.EXPECTED_DIRECTION),
            "base_test_order": list(k3c.BASE_ORDER),
            "base_holm_m": 4,
            "primary_test_order": list(PRIMARY_TEST_ORDER),
            "mechanism_holm_m": 8,
            "holm_alpha": 0.05,
        },
        "integrity": integrity,
        "exact_command": [
            SCRIPT_REL,
            "--seed180-handoff",
            str(Path(args.seed180_handoff).resolve()),
            "--hf-revision",
            args.hf_revision,
            "--output-dir",
            str(target),
        ],
    }

    artifact_hashes = _write_artifacts(
        target,
        source_rows,
        candidates,
        mapping_rows,
        item_rows,
        block_rows,
        primary,
        integrity,
        manifest_base,
    )
    return {
        "scientific_execution": "PASS",
        "output_dir": str(target),
        "base_replication_verdict":
            primary["base_replication"]["base_replication_verdict"],
        "scientific_verdict": primary["scientific_verdict"],
        "full_support": primary["full_support"],
        "primary_statistics": primary,
        "artifact_sha256": artifact_hashes,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Authorized K3C native recurrence contribution "
            "scientific execution harness"
        )
    )
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
