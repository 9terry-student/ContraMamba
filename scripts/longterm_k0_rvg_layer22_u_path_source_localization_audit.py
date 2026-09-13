"""K0-RVG layer-22 U-path source-localization audit.

Frozen scientific question:
    Within layer 22, is the corr-k2 U separation already present at the
    mixer-input / hidden-branch receptive field, or is it selectively
    amplified across hidden in-projection, fixed depthwise causal convolution,
    or SiLU activation?

Authenticated native path:
    X_RF(t) -> H_RF(t) -> C_t -> U_t -> W_t

where:
    X_t  = layer-22 mixer input before in_proj
    H_t  = hidden half of the bias-free in_proj
    C_t  = pre-activation depthwise causal-convolution output
    U_t  = SiLU(C_t), bridged to the recurrence-frame hidden_states operand
    W_t  = deltaB_u, retained only to reproduce the frozen parent trajectory

Primary raw measurements:
    ||delta_X_RF||, ||delta_H_RF||, ||delta_C||, ||delta_U||

Stage transfers:
    T_XH = ||delta_H_RF|| / ||delta_X_RF||
    T_HC = ||delta_C||    / ||delta_H_RF||
    T_CU = ||delta_U||    / ||delta_C||

Common-330 k2 corr-vs-ctrl enrichment:
    E_B  = log(delta_B_corr / delta_B_ctrl)
    G_XH = E_H - E_X
    G_HC = E_C - E_H
    G_CU = E_U - E_C

No epsilon is introduced for zero denominators. Undefined ratios/logs are
stored as JSON null and counted explicitly.

Observational/algebraic only. No tokenizer, logits, task heads, training,
intervention, PCA/SVD/whitening, learned probe, or post-hoc layer/window search.
Raw vectors are never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

AUTHORITY_FREEZE_COMMIT = "497598f22aeccf628d1e0ada4a2e5f9aa67f4c94"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_u_path_source_localization_static_design_candidate.md"
)
AUTHORITY_SHA256 = "35ca426397aa5b95fc4439c8fd967a30cefb37a9a89103c6d1ad8cd4eb438e29"

PARENT_EVIDENCE_FREEZE_COMMIT = "2228d4601b55d08310392812ef8cc6fb1feb1ba0"
PARENT_RUNNER_REL = (
    "scripts/longterm_k0_rvg_layer22_write_factor_decomposition_audit.py"
)
PARENT_RUNNER_SHA256 = (
    "96e5a7911a26d589a693c3e26b2b6889f152238c3da6c868820544786663c4d0"
)
PARENT_RUNNER_BLOB = "f51e9dd471d27f5b60ee0fc67663b8fa68e29ee2"

PARENT_SUMMARY_REL = (
    "reports/longterm_k0_rvg_layer22_write_factor_decomposition_bd2331b_v1/"
    "summary.json"
)
PARENT_SUMMARY_SHA256 = (
    "99f917f77b6950d099b0a14004e9c001aa89dde8ef7366b1f43878458d916ab4"
)
PARENT_MANIFEST_REL = (
    "reports/longterm_k0_rvg_layer22_write_factor_decomposition_bd2331b_v1/"
    "execution_manifest.json"
)
PARENT_MANIFEST_SHA256 = (
    "fd370690205d1d3245e24fff251534c143885ae912daa372226fc9aad573c433"
)
PARENT_METRICS_REL = (
    "reports/longterm_k0_rvg_layer22_write_factor_decomposition_bd2331b_v1/"
    "layer22_write_factor_decomposition_metrics.jsonl"
)
PARENT_METRICS_SHA256 = (
    "68ad2fb9354a176d63391735d33d102f5393ce3fb6f04b1a7690fa33fb1d1f8b"
)
PARENT_EXECUTION_HEAD = "bd2331b41d2eeb653040c05d98a285050747acf7"

POSTCONV_PRECEDENT_REL = "scripts/longterm_k0_rvg_postconv_u_factorization_audit.py"
POSTCONV_PRECEDENT_SHA256 = (
    "0f3e9527ff3d5a3c91e646547b1ab17e134e799130833100446bd976e4a4d052"
)
POSTCONV_PRECEDENT_BLOB = "acab0c14e7e7ceb7216c244e3f99a93333309b05"

FOUR_TAP_PRECEDENT_REL = (
    "scripts/longterm_k0_rvg_four_tap_convolution_decomposition_audit.py"
)
FOUR_TAP_PRECEDENT_SHA256 = (
    "49cd2708bf7ba3ef9f6beb5987f845d585cb9f991e459939c86cb303e0ce5070"
)
FOUR_TAP_PRECEDENT_BLOB = "83928f80349aab50f0c077fb4b2d1f4349a08d41"

HIDDEN_PRECEDENT_REL = "scripts/longterm_k0_rvg_hidden_inproj_transfer_audit.py"
HIDDEN_PRECEDENT_SHA256 = (
    "65810137bebb768c12bb11fd279a630d6a6a1f1a6116b3b12f00a0c286ea8d64"
)
HIDDEN_PRECEDENT_BLOB = "4a87718b919d3164a87c9d9d704751f5389f897d"

RUNNER_REL = "scripts/longterm_k0_rvg_layer22_u_path_source_localization_audit.py"
K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344
EXPECTED_LAYER_COUNT = 24

SOURCE_LAYER = 22
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536
STATE_SIZE = 16
CONV_KERNEL_SIZE = 4
RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6
EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

INPROJ_RECON_REL_TOL = 1e-6
CONV_RECON_REL_TOL = 2e-5
ACTIVATION_REL_TOL = 1e-6
PARENT_U_BRIDGE_REL_TOL = 1e-6
PARENT_TRAJECTORY_REL_TOL = 1e-13
PARENT_TRAJECTORY_ABS_TOL = 1e-13
ENRICHMENT_IDENTITY_ABS_TOL = 1e-12

QUESTION = (
    "Within layer 22, is the corr-k2 U separation already present at the "
    "mixer input / hidden-branch receptive field, or is it selectively "
    "amplified across hidden in-projection, fixed depthwise causal "
    "convolution, or SiLU activation?"
)


class Layer22UPathError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Layer22UPathError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def import_module(path: Path, name: str):
    require(path.is_file(), f"MODULE_MISSING:{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(
        spec is not None and spec.loader is not None,
        f"MODULE_SPEC_FAILURE:{path}",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Layer22UPathError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Layer22UPathError(f"GIT_SHOW_FAILURE:{spec}") from exc


def _status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"').replace("\\", "/")


def authenticate_repo(root: Path, runtime_mode: bool) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"GIT_BRANCH_MISMATCH:{branch}")

    for ancestor, label in (
        (AUTHORITY_FREEZE_COMMIT, "AUTHORITY"),
        (PARENT_EVIDENCE_FREEZE_COMMIT, "PARENT_EVIDENCE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_FREEZE_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    ).splitlines()

    for line in status:
        path = _status_path(line)
        xy = line[:2]
        if path in K1_UNTRACKED:
            require(xy == "??", f"K1_STATE_CHANGED:{line}")
            continue
        if not runtime_mode and path == RUNNER_REL:
            require(xy == "??", f"STATIC_RUNNER_STATE_UNEXPECTED:{line}")
            continue
        raise Layer22UPathError(f"UNEXPECTED_WORKTREE_CHANGE:{line}")

    runner = root / RUNNER_REL
    require(runner.is_file(), "RUNNER_FILE_MISSING")

    if runtime_mode:
        tracked = subprocess.call(
            ["git", "ls-files", "--error-unmatch", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(tracked == 0, "RUNTIME_REQUIRES_TRACKED_RUNNER")
        require(
            subprocess.call(
                ["git", "diff", "--quiet", "--", RUNNER_REL],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0,
            "RUNTIME_RUNNER_WORKTREE_DRIFT",
        )

    return {
        "branch": branch,
        "head": head,
        "status": status,
        "runtime_mode": runtime_mode,
    }


def authenticate_authority(root: Path) -> None:
    frozen = git_bytes(root, f"{AUTHORITY_FREEZE_COMMIT}:{AUTHORITY_REL}")
    require(
        sha256_bytes(frozen) == AUTHORITY_SHA256,
        "AUTHORITY_SHA256_MISMATCH",
    )
    current = root / AUTHORITY_REL
    require(current.is_file(), "AUTHORITY_FILE_MISSING")
    require(current.read_bytes() == frozen, "AUTHORITY_WORKTREE_DRIFT")


def load_parent(root: Path):
    path = root / PARENT_RUNNER_REL
    require(path.is_file(), "PARENT_RUNNER_MISSING")
    frozen = git_bytes(
        root,
        f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{PARENT_RUNNER_REL}",
    )
    require(
        sha256_bytes(frozen) == PARENT_RUNNER_SHA256,
        "PARENT_RUNNER_SHA256_MISMATCH",
    )
    require(
        git(
            root,
            "rev-parse",
            f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{PARENT_RUNNER_REL}",
        )
        == PARENT_RUNNER_BLOB,
        "PARENT_RUNNER_BLOB_MISMATCH",
    )
    require(path.read_bytes() == frozen, "PARENT_RUNNER_WORKTREE_DRIFT")
    return import_module(path, "k0_rvg_layer22_write_factor_parent")


def authenticate_parent_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    files = (
        (PARENT_SUMMARY_REL, PARENT_SUMMARY_SHA256, "SUMMARY"),
        (PARENT_MANIFEST_REL, PARENT_MANIFEST_SHA256, "MANIFEST"),
        (PARENT_METRICS_REL, PARENT_METRICS_SHA256, "METRICS"),
    )
    raws = {}
    for rel, expected_sha, label in files:
        raw = git_bytes(root, f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{rel}")
        require(
            sha256_bytes(raw) == expected_sha,
            f"PARENT_{label}_SHA256_MISMATCH",
        )
        current = root / rel
        require(current.is_file(), f"PARENT_{label}_MISSING")
        require(current.read_bytes() == raw, f"PARENT_{label}_WORKTREE_DRIFT")
        raws[label] = raw

    summary = json.loads(raws["SUMMARY"])
    manifest = json.loads(raws["MANIFEST"])

    require(
        summary.get("schema_version")
        == "k0-rvg-layer22-write-factor-decomposition-summary-v1",
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("schema_version")
        == "k0-rvg-layer22-write-factor-decomposition-execution-manifest-v1",
        "PARENT_MANIFEST_SCHEMA_MISMATCH",
    )
    require(
        summary.get("pair_role_count") == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_PAIR_ROLE_COUNT_MISMATCH",
    )
    require(
        summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT,
        "PARENT_COMMON_COUNT_MISMATCH",
    )
    require(
        summary.get("source_layer") == SOURCE_LAYER,
        "PARENT_SOURCE_LAYER_MISMATCH",
    )
    require(
        summary.get("state_size") == STATE_SIZE,
        "PARENT_STATE_SIZE_MISMATCH",
    )
    require(
        summary.get("parent_delta_w_trajectory_match") is True,
        "PARENT_W_REPRODUCTION_NOT_TRUE",
    )
    require(
        manifest.get("runtime_git_head") == PARENT_EXECUTION_HEAD,
        "PARENT_EXECUTION_HEAD_MISMATCH",
    )
    require(
        manifest.get("model_forward_count") == EXPECTED_FORWARD_COUNT,
        "PARENT_FORWARD_COUNT_MISMATCH",
    )
    return summary, manifest


def authenticate_precedents(root: Path) -> dict[str, str]:
    specs = (
        (
            POSTCONV_PRECEDENT_REL,
            POSTCONV_PRECEDENT_SHA256,
            POSTCONV_PRECEDENT_BLOB,
            "POSTCONV",
        ),
        (
            FOUR_TAP_PRECEDENT_REL,
            FOUR_TAP_PRECEDENT_SHA256,
            FOUR_TAP_PRECEDENT_BLOB,
            "FOUR_TAP",
        ),
        (
            HIDDEN_PRECEDENT_REL,
            HIDDEN_PRECEDENT_SHA256,
            HIDDEN_PRECEDENT_BLOB,
            "HIDDEN",
        ),
    )
    out = {}
    for rel, expected_sha, expected_blob, label in specs:
        path = root / rel
        require(path.is_file(), f"{label}_PRECEDENT_MISSING")
        frozen = git_bytes(root, f"{AUTHORITY_FREEZE_COMMIT}:{rel}")
        require(
            sha256_bytes(frozen) == expected_sha,
            f"{label}_PRECEDENT_SHA256_MISMATCH",
        )
        blob = git(root, "rev-parse", f"{AUTHORITY_FREEZE_COMMIT}:{rel}")
        require(blob == expected_blob, f"{label}_PRECEDENT_BLOB_MISMATCH")
        current_blob = git(root, "rev-parse", f"HEAD:{rel}")
        require(
            current_blob == expected_blob,
            f"{label}_PRECEDENT_HEAD_BLOB_MISMATCH",
        )
        require(path.read_bytes() == frozen, f"{label}_PRECEDENT_WORKTREE_DRIFT")
        out[label.lower()] = blob
    return out


def build_plan(root: Path, parent: Any):
    carry_parent = parent.load_parent(root)
    values = parent.build_plan(root, carry_parent)
    require(len(values) == 20, "PARENT_BUILD_PLAN_ARITY_MISMATCH")
    (
        recurrent_parent,
        gate_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        four_tap,
        postconv,
        time_step,
        dt_projection,
        operating,
        secant,
        discrete_b,
        write,
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    ) = values
    require(len(plan) == EXPECTED_PAIR_ROLE_COUNT, "PLAN_COUNT_MISMATCH")
    require(len(cohort) == EXPECTED_COMMON_COUNT, "COMMON_COHORT_COUNT_MISMATCH")
    return (
        carry_parent,
        recurrent_parent,
        gate_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        four_tap,
        postconv,
        time_step,
        dt_projection,
        operating,
        secant,
        discrete_b,
        write,
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    )


def _snapshot(value: Any, role: str):
    import torch

    require(isinstance(value, torch.Tensor), f"{role}_NOT_TENSOR")
    out = value.detach().cpu().contiguous().clone()
    require(out.device.type == "cpu", f"{role}_NOT_CPU")
    require(out.dtype == torch.float32, f"{role}_DTYPE_MISMATCH:{out.dtype}")
    require(bool(torch.isfinite(out).all().item()), f"{role}_NONFINITE")
    return out


def resolve_runtime_bundle(
    root: Path,
    parent: Any,
    carry_parent: Any,
    recurrent_parent: Any,
    output_parent: Any,
    residual_parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    postconv: Any,
    base: Any,
    handoff_path: Path,
):
    bundle = parent.resolve_runtime_bundle(
        root,
        carry_parent,
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        handoff_path,
    )
    require(len(bundle) == 14, "PARENT_RUNTIME_BUNDLE_ARITY_MISMATCH")
    (
        observer,
        k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        snapshot_info,
        layer22,
        layer23,
        norm23,
        mixer23,
        mixer22,
        out_proj,
    ) = bundle
    require(len(layer_map) == EXPECTED_LAYER_COUNT, "LAYER_MAP_COUNT_MISMATCH")
    require(SOURCE_LAYER in set(layer_map.values()), "SOURCE_LAYER_NOT_REGISTERED")
    require(bool(mixer22.use_mambapy) is False, "USE_MAMBAPY_NOT_FALSE")
    require(mixer22.training is False, "MIXER22_NOT_EVAL")
    return bundle


def resolve_path_operators(mixer: Any):
    import torch

    require(
        isinstance(mixer.in_proj, torch.nn.Linear),
        "IN_PROJ_NOT_LINEAR",
    )
    require(int(mixer.hidden_size) == HIDDEN_SIZE, "HIDDEN_SIZE_MISMATCH")
    require(
        int(mixer.intermediate_size) == INTERMEDIATE_SIZE,
        "INTERMEDIATE_SIZE_MISMATCH",
    )
    require(
        tuple(mixer.in_proj.weight.shape)
        == (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "IN_PROJ_WEIGHT_SHAPE_MISMATCH",
    )
    require(mixer.in_proj.bias is None, "IN_PROJ_BIAS_PRESENT")

    require(
        isinstance(mixer.conv1d, torch.nn.Conv1d),
        "CONV1D_NOT_CONV1D",
    )
    require(
        mixer.conv1d.in_channels == INTERMEDIATE_SIZE,
        "CONV1D_IN_CHANNEL_MISMATCH",
    )
    require(
        mixer.conv1d.out_channels == INTERMEDIATE_SIZE,
        "CONV1D_OUT_CHANNEL_MISMATCH",
    )
    require(
        mixer.conv1d.groups == INTERMEDIATE_SIZE,
        "CONV1D_NOT_DEPTHWISE",
    )
    require(
        tuple(mixer.conv1d.kernel_size) == (CONV_KERNEL_SIZE,),
        "CONV_KERNEL_SIZE_MISMATCH",
    )
    require(
        tuple(mixer.conv1d.padding) == (CONV_KERNEL_SIZE - 1,),
        "CONV_PADDING_MISMATCH",
    )
    require(str(mixer.activation).lower() == "silu", "ACTIVATION_NOT_SILU")

    w_hidden = (
        mixer.in_proj.weight[:INTERMEDIATE_SIZE, :]
        .detach()
        .cpu()
        .contiguous()
        .clone()
    )
    kernel = (
        mixer.conv1d.weight.detach()
        .cpu()
        .contiguous()
        .clone()[:, 0, :]
    )
    bias = (
        None
        if mixer.conv1d.bias is None
        else mixer.conv1d.bias.detach().cpu().contiguous().clone()
    )

    require(
        tuple(w_hidden.shape) == (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "HIDDEN_WEIGHT_SHAPE_MISMATCH",
    )
    require(
        tuple(kernel.shape) == (INTERMEDIATE_SIZE, CONV_KERNEL_SIZE),
        "CONV_KERNEL_SHAPE_MISMATCH",
    )
    if bias is not None:
        require(
            tuple(bias.shape) == (INTERMEDIATE_SIZE,),
            "CONV_BIAS_SHAPE_MISMATCH",
        )
    return w_hidden, kernel, bias


def _relative_residual(reconstructed, observed) -> float:
    import torch

    residual = reconstructed.to(torch.float64) - observed.to(torch.float64)
    numerator = float(torch.linalg.vector_norm(residual).item())
    denominator = max(
        float(torch.linalg.vector_norm(observed.to(torch.float64)).item()),
        1e-12,
    )
    return numerator / denominator


def capture_path(
    parent: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    mixer: Any,
    w_hidden: Any,
    kernel: Any,
    bias: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    import torch

    targets = tuple(int(i) for i in targets)
    x_rf_records = {}
    h_rf_records = {}
    c_records = {}
    counts = {"inproj_pre": 0, "conv_pre": 0, "conv_post": 0}

    def inproj_pre_hook(module, args):
        counts["inproj_pre"] += 1
        require(counts["inproj_pre"] == 1, "DUPLICATE_INPROJ_PRE_HOOK")
        require(len(args) == 1, "INPROJ_PRE_HOOK_ARG_COUNT_MISMATCH")
        x_full = _snapshot(args[0], "LAYER22_INPUT_X_FULL")
        require(len(x_full.shape) == 3, "LAYER22_X_RANK_MISMATCH")
        require(x_full.shape[0] == 1, "LAYER22_X_BATCH_MISMATCH")
        require(x_full.shape[2] == HIDDEN_SIZE, "LAYER22_X_WIDTH_MISMATCH")

        for i in targets:
            require(
                0 <= i < x_full.shape[1],
                f"LAYER22_X_TARGET_OUT_OF_RANGE:{i}",
            )
            lag_vectors = []
            for lag in range(CONV_KERNEL_SIZE):
                pos = i - lag
                if pos >= 0:
                    vector = x_full[0, pos, :].contiguous().clone()
                else:
                    vector = torch.zeros(
                        HIDDEN_SIZE,
                        dtype=x_full.dtype,
                        device=x_full.device,
                    )
                lag_vectors.append(vector)
            x_rf_records[i] = torch.stack(lag_vectors, dim=0).contiguous()

    def conv_pre_hook(module, args):
        counts["conv_pre"] += 1
        require(counts["conv_pre"] == 1, "DUPLICATE_CONV_PRE_HOOK")
        require(len(args) == 1, "CONV_PRE_HOOK_ARG_COUNT_MISMATCH")
        h_full = _snapshot(args[0], "LAYER22_PRECONV_H_FULL")
        require(len(h_full.shape) == 3, "LAYER22_H_RANK_MISMATCH")
        require(h_full.shape[0] == 1, "LAYER22_H_BATCH_MISMATCH")
        require(
            h_full.shape[1] == INTERMEDIATE_SIZE,
            "LAYER22_H_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < h_full.shape[2],
                f"LAYER22_H_TARGET_OUT_OF_RANGE:{i}",
            )
            lag_vectors = []
            for lag in range(CONV_KERNEL_SIZE):
                pos = i - lag
                if pos >= 0:
                    vector = h_full[0, :, pos].contiguous().clone()
                else:
                    vector = torch.zeros(
                        INTERMEDIATE_SIZE,
                        dtype=h_full.dtype,
                        device=h_full.device,
                    )
                lag_vectors.append(vector)
            h_rf_records[i] = torch.stack(lag_vectors, dim=0).contiguous()

    def conv_post_hook(module, args, output):
        counts["conv_post"] += 1
        require(counts["conv_post"] == 1, "DUPLICATE_CONV_POST_HOOK")
        c_full = _snapshot(output, "LAYER22_CONV_PREACT_FULL")
        require(len(c_full.shape) == 3, "LAYER22_C_RANK_MISMATCH")
        require(c_full.shape[0] == 1, "LAYER22_C_BATCH_MISMATCH")
        require(
            c_full.shape[1] == INTERMEDIATE_SIZE,
            "LAYER22_C_WIDTH_MISMATCH",
        )
        for i in targets:
            require(
                0 <= i < c_full.shape[2],
                f"LAYER22_C_TARGET_OUT_OF_RANGE:{i}",
            )
            c_records[i] = c_full[:, :, i].contiguous().clone()

    handles = [
        mixer.in_proj.register_forward_pre_hook(inproj_pre_hook),
        mixer.conv1d.register_forward_pre_hook(conv_pre_hook),
        mixer.conv1d.register_forward_hook(conv_post_hook),
    ]

    try:
        records = parent.capture_factors(
            base,
            model,
            binding,
            layer_map,
            token_ids,
            targets,
        )
    finally:
        for handle in reversed(handles):
            handle.remove()

    require(counts["inproj_pre"] == 1, "INPROJ_PRE_HOOK_COUNT_FAILURE")
    require(counts["conv_pre"] == 1, "CONV_PRE_HOOK_COUNT_FAILURE")
    require(counts["conv_post"] == 1, "CONV_POST_HOOK_COUNT_FAILURE")
    require(set(x_rf_records) == set(targets), "X_RF_TARGET_SET_MISMATCH")
    require(set(h_rf_records) == set(targets), "H_RF_TARGET_SET_MISMATCH")
    require(set(c_records) == set(targets), "C_TARGET_SET_MISMATCH")
    require(set(records) == set(targets), "PARENT_TARGET_SET_MISMATCH")

    for i in targets:
        x_rf = x_rf_records[i]
        h_rf = h_rf_records[i]
        c = c_records[i]
        require(
            tuple(x_rf.shape) == (CONV_KERNEL_SIZE, HIDDEN_SIZE),
            f"X_RF_SHAPE_MISMATCH:{i}",
        )
        require(
            tuple(h_rf.shape) == (CONV_KERNEL_SIZE, INTERMEDIATE_SIZE),
            f"H_RF_SHAPE_MISMATCH:{i}",
        )
        require(
            tuple(c.shape) == (1, INTERMEDIATE_SIZE),
            f"C_SHAPE_MISMATCH:{i}",
        )
        require("U32" in records[i], f"PARENT_U_MISSING:{i}")
        require("W32" in records[i], f"PARENT_W_MISSING:{i}")

        u = records[i]["U32"]
        require(
            tuple(u.shape) == (1, INTERMEDIATE_SIZE),
            f"PARENT_U_SHAPE_MISMATCH:{i}",
        )

        inproj_residuals = []
        with torch.inference_mode():
            for lag in range(CONV_KERNEL_SIZE):
                x = x_rf[lag, :].unsqueeze(0)
                h_observed = h_rf[lag, :].unsqueeze(0)
                h_recon = torch.nn.functional.linear(
                    x,
                    w_hidden,
                    bias=None,
                )
                rel = _relative_residual(h_recon, h_observed)
                inproj_residuals.append(rel)

            c_recon = torch.zeros(
                (1, INTERMEDIATE_SIZE),
                dtype=h_rf.dtype,
                device=h_rf.device,
            )
            if bias is not None:
                c_recon = c_recon + bias.unsqueeze(0)
            for lag in range(CONV_KERNEL_SIZE):
                kernel_index = CONV_KERNEL_SIZE - 1 - lag
                c_recon = (
                    c_recon
                    + h_rf[lag, :].unsqueeze(0)
                    * kernel[:, kernel_index].unsqueeze(0)
                )

            u_from_c = (
                mixer.act(c)
                .detach()
                .cpu()
                .contiguous()
            )

        max_inproj_rel = max(inproj_residuals)
        conv_rel = _relative_residual(c_recon, c)
        activation_rel = _relative_residual(u_from_c, u)
        bridge_rel = activation_rel

        require(
            max_inproj_rel <= INPROJ_RECON_REL_TOL,
            f"INPROJ_RECONSTRUCTION_FAILURE:{i}:{max_inproj_rel}",
        )
        require(
            conv_rel <= CONV_RECON_REL_TOL,
            f"CONV_RECONSTRUCTION_FAILURE:{i}:{conv_rel}",
        )
        require(
            activation_rel <= ACTIVATION_REL_TOL,
            f"ACTIVATION_RECONSTRUCTION_FAILURE:{i}:{activation_rel}",
        )
        require(
            bridge_rel <= PARENT_U_BRIDGE_REL_TOL,
            f"PARENT_U_BRIDGE_FAILURE:{i}:{bridge_rel}",
        )

        records[i]["X_RF32"] = x_rf
        records[i]["H_RF32"] = h_rf
        records[i]["C32"] = c
        records[i]["inproj_reconstruction_relative_residual"] = max_inproj_rel
        records[i]["conv_reconstruction_relative_residual"] = conv_rel
        records[i]["activation_reconstruction_relative_residual"] = activation_rel
        records[i]["parent_u_bridge_relative_residual"] = bridge_rel
        records[i]["parent_u_bridge_exact_equal"] = bool(
            torch.equal(u_from_c, u)
        )

    return records


def _prefixes(parent: Any, carry_parent: Any, recurrent_parent: Any, row):
    matched, swapped = parent._prefixes(carry_parent, recurrent_parent, row)
    return tuple(matched), tuple(swapped)


def _safe_transfer(numerator: float, denominator: float):
    if denominator > 0.0:
        return numerator / denominator
    return None


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    cohort: frozenset[int],
    signature: str,
):
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])
    anchor = int(row["anchor"])
    out = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k
        m = matched[token]
        s = swapped[token]

        xm = m["X_RF32"]
        xs = s["X_RF32"]
        hm = m["H_RF32"]
        hs = s["H_RF32"]
        cm = m["C32"]
        cs = s["C32"]
        um = m["U32"]
        us = s["U32"]
        wm = m["W32"]
        ws = s["W32"]

        x_equal = bool(torch.equal(xm, xs))
        h_equal = bool(torch.equal(hm, hs))
        c_equal = bool(torch.equal(cm, cs))
        u_equal = bool(torch.equal(um, us))

        dx = xm.to(torch.float64) - xs.to(torch.float64)
        dh = hm.to(torch.float64) - hs.to(torch.float64)
        dc = cm.to(torch.float64) - cs.to(torch.float64)
        du = um.to(torch.float64) - us.to(torch.float64)
        dw = wm.to(torch.float64) - ws.to(torch.float64)

        dx_rf_l2 = float(torch.linalg.vector_norm(dx).item())
        dh_rf_l2 = float(torch.linalg.vector_norm(dh).item())
        dc_l2 = float(torch.linalg.vector_norm(dc).item())
        du_l2 = float(torch.linalg.vector_norm(du).item())
        dw_l2 = float(torch.linalg.vector_norm(dw).item())

        dx_current_l2 = float(
            torch.linalg.vector_norm(dx[0, :]).item()
        )
        dh_current_l2 = float(
            torch.linalg.vector_norm(dh[0, :]).item()
        )

        t_xh = _safe_transfer(dh_rf_l2, dx_rf_l2)
        t_hc = _safe_transfer(dc_l2, dh_rf_l2)
        t_cu = _safe_transfer(du_l2, dc_l2)

        if k == -1:
            require(
                x_equal and h_equal and c_equal and u_equal,
                f"K_MINUS_1_IDENTITY_FAILURE:{idx}:{role}",
            )
            for value, label in (
                (dx_rf_l2, "DELTA_X_RF"),
                (dh_rf_l2, "DELTA_H_RF"),
                (dc_l2, "DELTA_C"),
                (du_l2, "DELTA_U"),
                (dx_current_l2, "DELTA_X_CURRENT"),
                (dh_current_l2, "DELTA_H_CURRENT"),
            ):
                require(
                    value == 0.0,
                    f"K_MINUS_1_{label}_NONZERO:{idx}:{role}:{value}",
                )
            require(t_xh is None, f"K_MINUS_1_T_XH_DEFINED:{idx}:{role}")
            require(t_hc is None, f"K_MINUS_1_T_HC_DEFINED:{idx}:{role}")
            require(t_cu is None, f"K_MINUS_1_T_CU_DEFINED:{idx}:{role}")

        values = {
            "schema_version": "k0-rvg-layer22-u-path-source-localization-row-v1",
            "local_template_index": idx,
            "stable_item_id": row["stable_item_id"],
            "role": role,
            "relative_coordinate": k,
            "token_index": token,
            "divergence_anchor_token_index": anchor,
            "token_equality_signature_k0_to_k6": signature,
            "in_common_ddsssss_cohort": idx in cohort,
            "x_rf_exact_equal": x_equal,
            "h_rf_exact_equal": h_equal,
            "c_exact_equal": c_equal,
            "u_exact_equal": u_equal,
            "delta_x_rf_l2": dx_rf_l2,
            "delta_h_rf_l2": dh_rf_l2,
            "delta_c_l2": dc_l2,
            "delta_u_l2": du_l2,
            "delta_x_current_l2": dx_current_l2,
            "delta_h_current_l2": dh_current_l2,
            "t_xh": t_xh,
            "t_hc": t_hc,
            "t_cu": t_cu,
            "delta_w_l2": dw_l2,
            "inproj_reconstruction_relative_residual": max(
                float(m["inproj_reconstruction_relative_residual"]),
                float(s["inproj_reconstruction_relative_residual"]),
            ),
            "conv_reconstruction_relative_residual": max(
                float(m["conv_reconstruction_relative_residual"]),
                float(s["conv_reconstruction_relative_residual"]),
            ),
            "activation_reconstruction_relative_residual": max(
                float(m["activation_reconstruction_relative_residual"]),
                float(s["activation_reconstruction_relative_residual"]),
            ),
            "parent_u_bridge_relative_residual": max(
                float(m["parent_u_bridge_relative_residual"]),
                float(s["parent_u_bridge_relative_residual"]),
            ),
            "parent_u_bridge_exact_equal_both": bool(
                m["parent_u_bridge_exact_equal"]
                and s["parent_u_bridge_exact_equal"]
            ),
            "parent_write_branch_reconstruction_relative_residual": max(
                float(m["branch_reconstruction_relative_residual"]),
                float(s["branch_reconstruction_relative_residual"]),
            ),
        }

        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                require(math.isfinite(float(value)), f"NONFINITE_METRIC:{key}")

        out.append(values)

    return out


RAW_FIELDS = (
    "delta_x_rf_l2",
    "delta_h_rf_l2",
    "delta_c_l2",
    "delta_u_l2",
)
DIAGNOSTIC_FIELDS = (
    "delta_x_current_l2",
    "delta_h_current_l2",
    "delta_w_l2",
)
TRANSFER_FIELDS = ("t_xh", "t_hc", "t_cu")
RESIDUAL_FIELDS = (
    "inproj_reconstruction_relative_residual",
    "conv_reconstruction_relative_residual",
    "activation_reconstruction_relative_residual",
    "parent_u_bridge_relative_residual",
    "parent_write_branch_reconstruction_relative_residual",
)
SUMMARY_FIELDS = RAW_FIELDS + DIAGNOSTIC_FIELDS + TRANSFER_FIELDS + RESIDUAL_FIELDS


def aggregate_nullable(values):
    vals = list(values)
    defined = [float(v) for v in vals if v is not None]
    require(
        all(math.isfinite(v) for v in defined),
        "NONFINITE_AGGREGATE",
    )
    result = {
        "count": len(vals),
        "defined_count": len(defined),
        "undefined_count": len(vals) - len(defined),
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
    }
    if defined:
        result.update(
            {
                "mean": float(statistics.fmean(defined)),
                "median": float(statistics.median(defined)),
                "min": float(min(defined)),
                "max": float(max(defined)),
            }
        )
    return result


def aggregate_trajectory(rows):
    result = {"corr": {}, "ctrl": {}}
    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            bucket = [
                row
                for row in rows
                if row["role"] == role
                and int(row["relative_coordinate"]) == k
            ]
            require(bool(bucket), f"EMPTY_BUCKET:{role}:{k}")
            result[role][str(k)] = {
                field: aggregate_nullable(row[field] for row in bucket)
                for field in SUMMARY_FIELDS
            }
    return result


def _aligned_common_k2(rows):
    corr = {
        int(r["local_template_index"]): r
        for r in rows
        if r["role"] == "corr"
        and int(r["relative_coordinate"]) == 2
        and bool(r["in_common_ddsssss_cohort"])
    }
    ctrl = {
        int(r["local_template_index"]): r
        for r in rows
        if r["role"] == "ctrl"
        and int(r["relative_coordinate"]) == 2
        and bool(r["in_common_ddsssss_cohort"])
    }
    require(set(corr) == set(ctrl), "COMMON_K2_ROLE_ALIGNMENT_MISMATCH")
    require(len(corr) == EXPECTED_COMMON_COUNT, "COMMON_K2_ALIGNMENT_COUNT_MISMATCH")
    return [(corr[i], ctrl[i]) for i in sorted(corr)]


def _paired_count_nullable(pairs, field: str):
    gt = lt = eq = undefined = 0
    for corr, ctrl in pairs:
        a = corr[field]
        b = ctrl[field]
        if a is None or b is None:
            undefined += 1
        elif float(a) > float(b):
            gt += 1
        elif float(a) < float(b):
            lt += 1
        else:
            eq += 1
    return {
        "count": len(pairs),
        "defined_pair_count": len(pairs) - undefined,
        "undefined_pair_count": undefined,
        "corr_gt_ctrl": gt,
        "corr_lt_ctrl": lt,
        "equal": eq,
    }


def _log_ratio(a, b):
    if a is None or b is None:
        return None
    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        return None
    return math.log(a / b)


def common_k2_enrichment(pairs):
    fields = {
        "e_x": [],
        "e_h": [],
        "e_c": [],
        "e_u": [],
        "g_xh": [],
        "g_hc": [],
        "g_cu": [],
    }
    max_identity_residual = 0.0

    for corr, ctrl in pairs:
        e_x = _log_ratio(corr["delta_x_rf_l2"], ctrl["delta_x_rf_l2"])
        e_h = _log_ratio(corr["delta_h_rf_l2"], ctrl["delta_h_rf_l2"])
        e_c = _log_ratio(corr["delta_c_l2"], ctrl["delta_c_l2"])
        e_u = _log_ratio(corr["delta_u_l2"], ctrl["delta_u_l2"])

        g_xh = None if e_x is None or e_h is None else e_h - e_x
        g_hc = None if e_h is None or e_c is None else e_c - e_h
        g_cu = None if e_c is None or e_u is None else e_u - e_c

        alt_xh = _log_ratio(corr["t_xh"], ctrl["t_xh"])
        alt_hc = _log_ratio(corr["t_hc"], ctrl["t_hc"])
        alt_cu = _log_ratio(corr["t_cu"], ctrl["t_cu"])

        for g, alt, label in (
            (g_xh, alt_xh, "G_XH"),
            (g_hc, alt_hc, "G_HC"),
            (g_cu, alt_cu, "G_CU"),
        ):
            if g is not None and alt is not None:
                residual = abs(float(g) - float(alt))
                max_identity_residual = max(max_identity_residual, residual)
                require(
                    residual <= ENRICHMENT_IDENTITY_ABS_TOL,
                    f"TRANSFER_ENRICHMENT_IDENTITY_FAILURE:{label}:{residual}",
                )

        if (
            g_xh is not None
            and g_hc is not None
            and g_cu is not None
            and e_x is not None
            and e_u is not None
        ):
            residual = abs((g_xh + g_hc + g_cu) - (e_u - e_x))
            max_identity_residual = max(max_identity_residual, residual)
            require(
                residual <= ENRICHMENT_IDENTITY_ABS_TOL,
                f"TELESCOPING_ENRICHMENT_IDENTITY_FAILURE:{residual}",
            )

        for key, value in (
            ("e_x", e_x),
            ("e_h", e_h),
            ("e_c", e_c),
            ("e_u", e_u),
            ("g_xh", g_xh),
            ("g_hc", g_hc),
            ("g_cu", g_cu),
        ):
            fields[key].append(value)

    return (
        {key: aggregate_nullable(values) for key, values in fields.items()},
        max_identity_residual,
    )


def make_summary(rows, cohort):
    require(
        len(rows) == EXPECTED_PAIR_ROLE_COUNT * len(RELATIVE_COORDINATES),
        "ROW_COUNT_MISMATCH",
    )
    common_rows = [r for r in rows if bool(r["in_common_ddsssss_cohort"])]
    require(
        len(common_rows)
        == EXPECTED_COMMON_COUNT * 2 * len(RELATIVE_COORDINATES),
        "COMMON_ROW_COUNT_MISMATCH",
    )

    km1 = [r for r in rows if int(r["relative_coordinate"]) == -1]
    require(len(km1) == EXPECTED_PAIR_ROLE_COUNT, "K_MINUS_1_ROW_COUNT_MISMATCH")
    require(
        all(
            r["x_rf_exact_equal"]
            and r["h_rf_exact_equal"]
            and r["c_exact_equal"]
            and r["u_exact_equal"]
            and float(r["delta_x_rf_l2"]) == 0.0
            and float(r["delta_h_rf_l2"]) == 0.0
            and float(r["delta_c_l2"]) == 0.0
            and float(r["delta_u_l2"]) == 0.0
            and r["t_xh"] is None
            and r["t_hc"] is None
            and r["t_cu"] is None
            for r in km1
        ),
        "K_MINUS_1_SUMMARY_IDENTITY_FAILURE",
    )

    pairs = _aligned_common_k2(rows)
    enrichment, max_enrichment_residual = common_k2_enrichment(pairs)

    return {
        "schema_version": "k0-rvg-layer22-u-path-source-localization-summary-v1",
        "scientific_question": QUESTION,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": len(cohort),
        "trajectory_row_count": len(rows),
        "source_layer": SOURCE_LAYER,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "relative_coordinates": list(RELATIVE_COORDINATES),
        "execution_protocol": EXECUTION_PROTOCOL,
        "path_identity": "X_RF -> H_RF -> C -> U -> W",
        "transfer_definitions": {
            "t_xh": "||delta_H_RF|| / ||delta_X_RF||",
            "t_hc": "||delta_C|| / ||delta_H_RF||",
            "t_cu": "||delta_U|| / ||delta_C||",
        },
        "undefined_ratio_policy": "no epsilon; JSON null with explicit undefined count",
        "enrichment_definitions": {
            "e_x": "log(delta_X_RF_corr / delta_X_RF_ctrl)",
            "e_h": "log(delta_H_RF_corr / delta_H_RF_ctrl)",
            "e_c": "log(delta_C_corr / delta_C_ctrl)",
            "e_u": "log(delta_U_corr / delta_U_ctrl)",
            "g_xh": "E_H - E_X",
            "g_hc": "E_C - E_H",
            "g_cu": "E_U - E_C",
        },
        "full_336_trajectory": aggregate_trajectory(rows),
        "common_330_ddsssss_trajectory": aggregate_trajectory(common_rows),
        "common_330_k2_role_counts": {
            field: _paired_count_nullable(pairs, field)
            for field in RAW_FIELDS + TRANSFER_FIELDS
        },
        "common_330_k2_enrichment": enrichment,
        "max_enrichment_identity_abs_residual": max_enrichment_residual,
        "max_inproj_reconstruction_relative_residual": max(
            float(r["inproj_reconstruction_relative_residual"]) for r in rows
        ),
        "max_conv_reconstruction_relative_residual": max(
            float(r["conv_reconstruction_relative_residual"]) for r in rows
        ),
        "max_activation_reconstruction_relative_residual": max(
            float(r["activation_reconstruction_relative_residual"]) for r in rows
        ),
        "max_parent_u_bridge_relative_residual": max(
            float(r["parent_u_bridge_relative_residual"]) for r in rows
        ),
        "max_parent_write_branch_reconstruction_relative_residual": max(
            float(r["parent_write_branch_reconstruction_relative_residual"])
            for r in rows
        ),
        "parent_delta_w_trajectory_match": False,
        "raw_vectors_persisted": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_whitening_or_learned_geometry_executed": False,
        "posthoc_layer_or_window_search_executed": False,
    }


def validate_parent_reproduction(summary, parent_summary):
    for trajectory in ("full_336_trajectory", "common_330_ddsssss_trajectory"):
        current = summary[trajectory]
        expected = parent_summary[trajectory]
        for role in ("corr", "ctrl"):
            for k in RELATIVE_COORDINATES:
                got_stats = current[role][str(k)]["delta_w_l2"]
                exp_stats = expected[role][str(k)]["delta_w_l2"]
                require(
                    int(got_stats["defined_count"]) == int(exp_stats["count"]),
                    f"PARENT_W_COUNT_MISMATCH:{trajectory}:{role}:{k}",
                )
                for stat in ("mean", "median", "min", "max"):
                    got = float(got_stats[stat])
                    exp = float(exp_stats[stat])
                    require(
                        math.isclose(
                            got,
                            exp,
                            rel_tol=PARENT_TRAJECTORY_REL_TOL,
                            abs_tol=PARENT_TRAJECTORY_ABS_TOL,
                        ),
                        (
                            f"PARENT_W_TRAJECTORY_MISMATCH:"
                            f"{trajectory}:{role}:{k}:{stat}:{got}:{exp}"
                        ),
                    )
    summary["parent_delta_w_trajectory_match"] = True


def json_bytes(obj):
    return (
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows):
    return b"".join(json_bytes(row) for row in rows)


def runtime_preflight(
    root,
    parent,
    carry_parent,
    recurrent_parent,
    output_parent,
    residual_parent,
    rms_parent,
    hidden_parent,
    postconv,
    base,
    plan,
    cohort,
    signatures,
    handoff_path,
):
    (
        _observer,
        _k2s,
        model,
        binding,
        layer_map,
        _handoff,
        _encoder,
        _snapshot_info,
        _layer22,
        _layer23,
        _norm23,
        _mixer23,
        mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
        carry_parent,
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        handoff_path,
    )
    w_hidden, kernel, bias = resolve_path_operators(mixer22)

    row = plan[0]
    idx = int(row["local_template_index"])
    role = str(row["role"])
    matched_prefix, swapped_prefix = _prefixes(
        parent,
        carry_parent,
        recurrent_parent,
        row,
    )

    matched = capture_path(
        parent,
        base,
        model,
        binding,
        layer_map,
        mixer22,
        w_hidden,
        kernel,
        bias,
        matched_prefix,
        row["targets"],
    )
    swapped = capture_path(
        parent,
        base,
        model,
        binding,
        layer_map,
        mixer22,
        w_hidden,
        kernel,
        bias,
        swapped_prefix,
        row["targets"],
    )
    rows = metric_rows_for_pair(
        row,
        matched,
        swapped,
        cohort,
        signatures[(idx, role)],
    )

    print("PASS_LAYER22_U_PATH_SOURCE_LOCALIZATION_RUNTIME_PREFLIGHT")
    print("pair_role =", idx, role)
    print("model_forward_count = 2")
    print("scientific_population_accessed = True")
    print("scientific_evidence_emitted = False")
    print("raw_vectors_persisted = False")
    print(
        "max_inproj_reconstruction_relative_residual =",
        max(float(r["inproj_reconstruction_relative_residual"]) for r in rows),
    )
    print(
        "max_conv_reconstruction_relative_residual =",
        max(float(r["conv_reconstruction_relative_residual"]) for r in rows),
    )
    print(
        "max_activation_reconstruction_relative_residual =",
        max(float(r["activation_reconstruction_relative_residual"]) for r in rows),
    )
    print(
        "max_parent_u_bridge_relative_residual =",
        max(float(r["parent_u_bridge_relative_residual"]) for r in rows),
    )
    print(
        "max_parent_write_branch_reconstruction_relative_residual =",
        max(
            float(r["parent_write_branch_reconstruction_relative_residual"])
            for r in rows
        ),
    )


def execute(
    root,
    parent,
    carry_parent,
    recurrent_parent,
    output_parent,
    residual_parent,
    rms_parent,
    hidden_parent,
    postconv,
    base,
    repo,
    parent_summary,
    precedent_blobs,
    plan,
    cohort,
    signatures,
    handoff_path,
    output_dir,
):
    final_dir = output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")
    require(not final_dir.exists(), f"OUTPUT_DIR_EXISTS:{final_dir}")
    require(not partial_dir.exists(), f"PARTIAL_OUTPUT_EXISTS:{partial_dir}")

    (
        _observer,
        _k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        _snapshot_info,
        _layer22,
        _layer23,
        _norm23,
        _mixer23,
        mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
        carry_parent,
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        handoff_path,
    )
    w_hidden, kernel, bias = resolve_path_operators(mixer22)

    rows = []
    forward_count = 0
    for n, row in enumerate(plan, start=1):
        idx = int(row["local_template_index"])
        role = str(row["role"])
        matched_prefix, swapped_prefix = _prefixes(
            parent,
            carry_parent,
            recurrent_parent,
            row,
        )

        matched = capture_path(
            parent,
            base,
            model,
            binding,
            layer_map,
            mixer22,
            w_hidden,
            kernel,
            bias,
            matched_prefix,
            row["targets"],
        )
        forward_count += 1
        swapped = capture_path(
            parent,
            base,
            model,
            binding,
            layer_map,
            mixer22,
            w_hidden,
            kernel,
            bias,
            swapped_prefix,
            row["targets"],
        )
        forward_count += 1

        rows.extend(
            metric_rows_for_pair(
                row,
                matched,
                swapped,
                cohort,
                signatures[(idx, role)],
            )
        )

        if n % 16 == 0 or n == len(plan):
            print(
                f"PROGRESS pair_roles={n}/{len(plan)} "
                f"model_forwards={forward_count}",
                flush=True,
            )

    require(forward_count == EXPECTED_FORWARD_COUNT, "FORWARD_COUNT_MISMATCH")
    summary = make_summary(rows, cohort)
    validate_parent_reproduction(summary, parent_summary)

    require(
        summary["max_inproj_reconstruction_relative_residual"]
        <= INPROJ_RECON_REL_TOL,
        "SUMMARY_INPROJ_RECON_TOLERANCE_FAILURE",
    )
    require(
        summary["max_conv_reconstruction_relative_residual"]
        <= CONV_RECON_REL_TOL,
        "SUMMARY_CONV_RECON_TOLERANCE_FAILURE",
    )
    require(
        summary["max_activation_reconstruction_relative_residual"]
        <= ACTIVATION_REL_TOL,
        "SUMMARY_ACTIVATION_RECON_TOLERANCE_FAILURE",
    )
    require(
        summary["max_parent_u_bridge_relative_residual"]
        <= PARENT_U_BRIDGE_REL_TOL,
        "SUMMARY_PARENT_U_BRIDGE_TOLERANCE_FAILURE",
    )
    require(
        summary["max_parent_write_branch_reconstruction_relative_residual"] == 0.0,
        "SUMMARY_PARENT_WRITE_RECON_NOT_EXACT",
    )
    require(
        summary["max_enrichment_identity_abs_residual"]
        <= ENRICHMENT_IDENTITY_ABS_TOL,
        "SUMMARY_ENRICHMENT_IDENTITY_FAILURE",
    )

    partial_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = (
        partial_dir / "layer22_u_path_source_localization_metrics.jsonl"
    )
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"

    metrics_path.write_bytes(jsonl_bytes(rows))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version": (
            "k0-rvg-layer22-u-path-source-localization-execution-manifest-v1"
        ),
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "authority_freeze_commit": AUTHORITY_FREEZE_COMMIT,
        "authority_sha256": AUTHORITY_SHA256,
        "parent_evidence_freeze_commit": PARENT_EVIDENCE_FREEZE_COMMIT,
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_runner_blob": PARENT_RUNNER_BLOB,
        "parent_summary_sha256": PARENT_SUMMARY_SHA256,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_metrics_sha256": PARENT_METRICS_SHA256,
        "structural_precedent_blobs": precedent_blobs,
        "scientific_question": QUESTION,
        "path_identity": "X_RF -> H_RF -> C -> U -> W",
        "execution_protocol": EXECUTION_PROTOCOL,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": EXPECTED_COMMON_COUNT,
        "model_forward_count": forward_count,
        "source_layer": SOURCE_LAYER,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "state_size": STATE_SIZE,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "mamba_source_sha256": binding.source_sha256,
        "inproj_reconstruction_rel_tol": INPROJ_RECON_REL_TOL,
        "conv_reconstruction_rel_tol": CONV_RECON_REL_TOL,
        "activation_reconstruction_rel_tol": ACTIVATION_REL_TOL,
        "parent_u_bridge_rel_tol": PARENT_U_BRIDGE_REL_TOL,
        "parent_trajectory_rel_tol": PARENT_TRAJECTORY_REL_TOL,
        "parent_trajectory_abs_tol": PARENT_TRAJECTORY_ABS_TOL,
        "enrichment_identity_abs_tol": ENRICHMENT_IDENTITY_ABS_TOL,
        "undefined_ratio_policy": "no epsilon; JSON null",
        "handoff_zip_sha256": handoff["zip_sha256"],
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "encoder_canonical_digest": encoder["canonical_digest"],
        "encoder_raw_concat_digest": encoder["raw_concat_digest"],
        "parent_delta_w_trajectory_match": True,
        "capture_method": (
            "read-only forward-pre hook on layer-22 in_proj captures X; "
            "read-only pre/post hooks on layer-22 depthwise conv capture H and C; "
            "frozen parent recurrence-frame trace captures U and W in the same "
            "forward; no model parameter or forward semantic is modified"
        ),
        "parent_u_bridge_definition": (
            "relative L2 residual between SiLU(direct captured C) and the "
            "same-token layer-22 recurrence-frame U operand"
        ),
        "raw_vectors_persisted": False,
        "scientific_model_forward_executed": True,
        "scientific_x_read": True,
        "scientific_h_read": True,
        "scientific_c_read": True,
        "scientific_u_read": True,
        "scientific_w_read_for_parent_reproduction": True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_whitening_or_learned_geometry_executed": False,
        "posthoc_layer_or_window_search_executed": False,
        "runner_rel": RUNNER_REL,
        "runner_sha256": sha256_bytes((root / RUNNER_REL).read_bytes()),
        "outputs": {
            "layer22_u_path_source_localization_metrics.jsonl": sha256_bytes(
                metrics_path.read_bytes()
            ),
            "summary.json": sha256_bytes(summary_path.read_bytes()),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    os.replace(partial_dir, final_dir)

    common = summary["common_330_ddsssss_trajectory"]
    enrich = summary["common_330_k2_enrichment"]
    counts = summary["common_330_k2_role_counts"]

    print("PASS_LAYER22_U_PATH_SOURCE_LOCALIZATION_EXECUTION")
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("parent_delta_w_trajectory_match = True")
    print(
        "max_inproj_reconstruction_relative_residual =",
        summary["max_inproj_reconstruction_relative_residual"],
    )
    print(
        "max_conv_reconstruction_relative_residual =",
        summary["max_conv_reconstruction_relative_residual"],
    )
    print(
        "max_activation_reconstruction_relative_residual =",
        summary["max_activation_reconstruction_relative_residual"],
    )
    print(
        "max_parent_u_bridge_relative_residual =",
        summary["max_parent_u_bridge_relative_residual"],
    )

    for role in ("corr", "ctrl"):
        t = common[role]["2"]
        print(
            f"{role}_k2_median "
            f"XRF={t['delta_x_rf_l2']['median']} "
            f"HRF={t['delta_h_rf_l2']['median']} "
            f"C={t['delta_c_l2']['median']} "
            f"U={t['delta_u_l2']['median']} "
            f"T_XH={t['t_xh']['median']} "
            f"T_HC={t['t_hc']['median']} "
            f"T_CU={t['t_cu']['median']}"
        )

    for field in RAW_FIELDS + TRANSFER_FIELDS:
        c = counts[field]
        print(
            f"k2_role_count {field} "
            f"defined={c['defined_pair_count']} "
            f"undefined={c['undefined_pair_count']} "
            f"corr_gt_ctrl={c['corr_gt_ctrl']} "
            f"corr_lt_ctrl={c['corr_lt_ctrl']} "
            f"equal={c['equal']}"
        )

    print(
        "k2_enrichment_medians "
        f"E_X={enrich['e_x']['median']} "
        f"E_H={enrich['e_h']['median']} "
        f"E_C={enrich['e_c']['median']} "
        f"E_U={enrich['e_u']['median']} "
        f"G_XH={enrich['g_xh']['median']} "
        f"G_HC={enrich['g_hc']['median']} "
        f"G_CU={enrich['g_cu']['median']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--runtime-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    require(
        sum(
            bool(v)
            for v in (
                args.static_preflight,
                args.runtime_preflight,
                args.execute,
            )
        )
        == 1,
        "SELECT_EXACTLY_ONE_MODE",
    )

    root = Path.cwd().resolve()
    runtime_mode = bool(args.runtime_preflight or args.execute)
    repo = authenticate_repo(root, runtime_mode=runtime_mode)
    authenticate_authority(root)
    parent = load_parent(root)
    parent_summary, _parent_manifest = authenticate_parent_evidence(root)
    precedent_blobs = authenticate_precedents(root)

    (
        carry_parent,
        recurrent_parent,
        gate_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        four_tap,
        postconv,
        time_step,
        dt_projection,
        operating,
        secant,
        discrete_b,
        write,
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    ) = build_plan(root, parent)

    source = secant.validate_frozen_source_semantics()
    require(
        source["source_sha256"] == output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )
    _observer, observer_binding = carry_parent.authenticate_observer_static(
        root,
        base,
        output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
    )

    print("=== LAYER22 U-PATH SOURCE LOCALIZATION AUDIT PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("authority_freeze_commit =", AUTHORITY_FREEZE_COMMIT)
    print("authority_sha256 =", AUTHORITY_SHA256)
    print("parent_evidence_freeze_commit =", PARENT_EVIDENCE_FREEZE_COMMIT)
    print("parent_runner_sha256 =", PARENT_RUNNER_SHA256)
    print("parent_summary_sha256 =", PARENT_SUMMARY_SHA256)
    print("parent_manifest_sha256 =", PARENT_MANIFEST_SHA256)
    print("parent_metrics_sha256 =", PARENT_METRICS_SHA256)
    print("postconv_precedent_blob =", precedent_blobs["postconv"])
    print("four_tap_precedent_blob =", precedent_blobs["four_tap"])
    print("hidden_precedent_blob =", precedent_blobs["hidden"])
    print("pair_role_count =", len(plan))
    print("common_ddsssss_item_count =", len(cohort))
    print("source_layer =", SOURCE_LAYER)
    print("hidden_size =", HIDDEN_SIZE)
    print("intermediate_size =", INTERMEDIATE_SIZE)
    print("conv_kernel_size =", CONV_KERNEL_SIZE)
    print("scientific_question =", QUESTION)
    print("path_identity = X_RF -> H_RF -> C -> U -> W")
    print("inproj_reconstruction_rel_tol =", INPROJ_RECON_REL_TOL)
    print("conv_reconstruction_rel_tol =", CONV_RECON_REL_TOL)
    print("activation_reconstruction_rel_tol =", ACTIVATION_REL_TOL)
    print("parent_u_bridge_rel_tol =", PARENT_U_BRIDGE_REL_TOL)
    print("enrichment_identity_abs_tol =", ENRICHMENT_IDENTITY_ABS_TOL)
    print("observer_update_line =", observer_binding.update_line)
    print("observer_readout_line =", observer_binding.readout_line)
    print("mamba_source_sha256 =", source["source_sha256"])
    print("raw_vectors_persisted = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")
    print("causal_intervention_executed = False")
    print("posthoc_layer_or_window_search_executed = False")

    if args.static_preflight:
        print("scientific_model_forward_executed = False")
        print("PASS_LAYER22_U_PATH_SOURCE_LOCALIZATION_STATIC_PREFLIGHT")
        return 0

    require(args.handoff is not None, "RUNTIME_MODE_REQUIRES_HANDOFF")
    handoff_path = args.handoff.resolve()
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")

    if args.runtime_preflight:
        runtime_preflight(
            root,
            parent,
            carry_parent,
            recurrent_parent,
            output_parent,
            residual_parent,
            rms_parent,
            hidden_parent,
            postconv,
            base,
            plan,
            cohort,
            signatures,
            handoff_path,
        )
        return 0

    require(args.output_dir is not None, "EXECUTE_REQUIRES_OUTPUT_DIR")
    execute(
        root,
        parent,
        carry_parent,
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        repo,
        parent_summary,
        precedent_blobs,
        plan,
        cohort,
        signatures,
        handoff_path,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Layer22UPathError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(
            f"BLOCKED_UNEXPECTED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
